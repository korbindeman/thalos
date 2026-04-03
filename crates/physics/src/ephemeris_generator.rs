//! Streaming ephemeris generator with checkpoint/resume support.
//!
//! Writes samples to per-body temp files as they're committed, keeping only
//! the integration state and last sample per body in memory.  A checkpoint
//! file allows generation to be stopped and resumed.

use crate::ephemeris::{
    build_initial_states, rk4_step,
    EphemerisSample, NBodyState, Rk4Scratch,
    CURVATURE_THRESHOLD, DT,
};
use crate::types::SolarSystemDefinition;
use glam::DVec3;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Size of one sample record in the raw binary files (7 × f64).
const SAMPLE_BYTES: usize = 7 * 8;

/// Steps between automatic checkpoints during `advance`.
const CHECKPOINT_INTERVAL: u64 = 100_000;

// ---------------------------------------------------------------------------
// Checkpoint (serialised with bincode)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct Checkpoint {
    current_step: u64,
    total_steps: u64,
    time_span: f64,
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    gms: Vec<f64>,
    masses: Vec<f64>,
    star_id: usize,
    sample_counts: Vec<u64>,
    // Last committed sample per body (for adaptive sampling decision).
    last_sample_times: Vec<f64>,
    last_sample_positions: Vec<[f64; 3]>,
    last_sample_velocities: Vec<[f64; 3]>,
}

// ---------------------------------------------------------------------------
// Raw sample I/O
// ---------------------------------------------------------------------------

fn write_sample(w: &mut impl Write, s: &EphemerisSample) -> io::Result<()> {
    w.write_all(&s.time.to_le_bytes())?;
    w.write_all(&s.position.x.to_le_bytes())?;
    w.write_all(&s.position.y.to_le_bytes())?;
    w.write_all(&s.position.z.to_le_bytes())?;
    w.write_all(&s.velocity.x.to_le_bytes())?;
    w.write_all(&s.velocity.y.to_le_bytes())?;
    w.write_all(&s.velocity.z.to_le_bytes())?;
    Ok(())
}

fn body_file_path(work_dir: &Path, body_id: usize) -> PathBuf {
    work_dir.join(format!("body_{body_id:04}.samples"))
}

fn checkpoint_path(work_dir: &Path) -> PathBuf {
    work_dir.join("checkpoint.bin")
}

// ---------------------------------------------------------------------------
// Public generator
// ---------------------------------------------------------------------------

/// Streaming ephemeris generator.
///
/// Holds only the current integration state in memory.  Samples are flushed
/// to per-body files on disk.  Call [`advance`](Self::advance) in a loop,
/// then [`finalize`](Self::finalize) to assemble the final `ephemeris.bin`.
pub struct EphemerisGenerator {
    work_dir: PathBuf,
    state: NBodyState,
    scratch: Rk4Scratch,
    gms: Vec<f64>,
    masses: Vec<f64>,
    star_id: usize,
    current_step: u64,
    total_steps: u64,
    time_span: f64,
    // Per-body: last committed sample + buffered writer + count.
    last_samples: Vec<EphemerisSample>,
    writers: Vec<BufWriter<File>>,
    sample_counts: Vec<u64>,
    steps_since_checkpoint: u64,
}

impl EphemerisGenerator {
    /// Open (or resume) a generation in `work_dir`.
    ///
    /// If `work_dir` contains a valid checkpoint, generation resumes.  If the
    /// requested `time_span` is larger than the checkpoint's, the target is
    /// extended — existing samples are kept and integration continues.
    pub fn open(
        system: &SolarSystemDefinition,
        time_span: f64,
        work_dir: &Path,
    ) -> io::Result<Self> {
        let cp_path = checkpoint_path(work_dir);

        if cp_path.exists() {
            return Self::resume(system, time_span, work_dir);
        }

        Self::start_fresh(system, time_span, work_dir)
    }

    /// Start a brand new generation.
    fn start_fresh(
        system: &SolarSystemDefinition,
        time_span: f64,
        work_dir: &Path,
    ) -> io::Result<Self> {
        fs::create_dir_all(work_dir)?;

        let n = system.bodies.len();
        let (init_positions, init_velocities, star_id) = build_initial_states(system);
        let gms: Vec<f64> = system.bodies.iter().map(|b| b.gm).collect();
        let masses: Vec<f64> = system.bodies.iter().map(|b| b.mass_kg).collect();
        let total_steps = (time_span / DT).ceil() as u64;

        // Seed each body with a t=0 sample written to its file.
        let mut writers = Vec::with_capacity(n);
        let mut last_samples = Vec::with_capacity(n);
        let mut sample_counts = vec![0u64; n];

        for i in 0..n {
            let path = body_file_path(work_dir, i);
            let file = File::create(&path)?;
            let mut w = BufWriter::new(file);

            let sample = EphemerisSample {
                time: 0.0,
                position: init_positions[i],
                velocity: init_velocities[i],
            };

            if i != star_id {
                write_sample(&mut w, &sample)?;
                sample_counts[i] = 1;
            }

            writers.push(w);
            last_samples.push(sample);
        }

        let generator = Self {
            work_dir: work_dir.to_path_buf(),
            state: NBodyState {
                positions: init_positions,
                velocities: init_velocities,
            },
            scratch: Rk4Scratch::new(n),
            gms,
            masses,
            star_id,
            current_step: 0,
            total_steps,
            time_span,
            last_samples,
            writers,
            sample_counts,
            steps_since_checkpoint: 0,
        };

        Ok(generator)
    }

    /// Resume from a checkpoint, optionally extending to a new `time_span`.
    fn resume(
        system: &SolarSystemDefinition,
        time_span: f64,
        work_dir: &Path,
    ) -> io::Result<Self> {
        let cp_bytes = fs::read(checkpoint_path(work_dir))?;
        let cp: Checkpoint = bincode::deserialize(&cp_bytes)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let n = cp.positions.len();

        if n != system.bodies.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Checkpoint has {} bodies but system has {}",
                    n,
                    system.bodies.len()
                ),
            ));
        }

        // Use the larger of the checkpoint's span and the requested span.
        let effective_span = time_span.max(cp.time_span);
        let new_total_steps = (effective_span / DT).ceil() as u64;

        let positions: Vec<DVec3> = cp.positions.iter().map(|p| DVec3::from_array(*p)).collect();
        let velocities: Vec<DVec3> = cp.velocities.iter().map(|v| DVec3::from_array(*v)).collect();

        let last_samples: Vec<EphemerisSample> = (0..n)
            .map(|i| EphemerisSample {
                time: cp.last_sample_times[i],
                position: DVec3::from_array(cp.last_sample_positions[i]),
                velocity: DVec3::from_array(cp.last_sample_velocities[i]),
            })
            .collect();

        // Open body files for appending, truncating to match checkpoint counts.
        let mut writers = Vec::with_capacity(n);
        for i in 0..n {
            let path = body_file_path(work_dir, i);
            let expected_bytes = cp.sample_counts[i] as usize * SAMPLE_BYTES;

            let file = fs::OpenOptions::new().write(true).open(&path)?;
            file.set_len(expected_bytes as u64)?;
            drop(file);

            let file = fs::OpenOptions::new().append(true).open(&path)?;
            writers.push(BufWriter::new(file));
        }

        let cp_years = cp.time_span / 3.156e7;
        let target_years = effective_span / 3.156e7;
        if effective_span > cp.time_span {
            eprintln!(
                "  Extending from {:.0} to {:.0} years (step {}/{})",
                cp_years, target_years, cp.current_step, new_total_steps,
            );
        } else {
            eprintln!(
                "  Resuming at {:.1}% ({:.0}-year target)",
                cp.current_step as f64 / new_total_steps as f64 * 100.0,
                target_years,
            );
        }

        Ok(Self {
            work_dir: work_dir.to_path_buf(),
            state: NBodyState { positions, velocities },
            scratch: Rk4Scratch::new(n),
            gms: cp.gms,
            masses: cp.masses,
            star_id: cp.star_id,
            current_step: cp.current_step,
            total_steps: new_total_steps,
            time_span: effective_span,
            last_samples,
            writers,
            sample_counts: cp.sample_counts,
            steps_since_checkpoint: 0,
        })
    }

    /// `true` when all integration steps have been computed.
    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }

    /// `(current_step, total_steps)`.
    pub fn progress(&self) -> (u64, u64) {
        (self.current_step, self.total_steps)
    }

    /// Advance by up to `batch_size` steps.  Returns steps actually taken.
    ///
    /// Automatically checkpoints every [`CHECKPOINT_INTERVAL`] steps.
    pub fn advance(&mut self, batch_size: u64) -> io::Result<u64> {
        let end_step = (self.current_step + batch_size).min(self.total_steps);
        let steps_to_take = end_step - self.current_step;

        for _ in 0..steps_to_take {
            self.current_step += 1;
            rk4_step(
                &mut self.state,
                &mut self.scratch,
                &self.gms,
                DT,
                self.star_id,
            );

            let t = self.current_step as f64 * DT;
            let is_last = self.current_step == self.total_steps;
            let n = self.state.positions.len();

            for i in 0..n {
                if i == self.star_id {
                    continue;
                }

                let last = &self.last_samples[i];
                let dt_since = t - last.time;
                let extrapolated = last.position + last.velocity * dt_since;
                let deviation = (self.state.positions[i] - extrapolated).length();

                if deviation > CURVATURE_THRESHOLD || is_last {
                    let sample = EphemerisSample {
                        time: t,
                        position: self.state.positions[i],
                        velocity: self.state.velocities[i],
                    };
                    write_sample(&mut self.writers[i], &sample)?;
                    self.last_samples[i] = sample;
                    self.sample_counts[i] += 1;
                }
            }

            self.steps_since_checkpoint += 1;
            if self.steps_since_checkpoint >= CHECKPOINT_INTERVAL {
                self.checkpoint()?;
            }
        }

        Ok(steps_to_take)
    }

    /// Flush writers and write a checkpoint file.
    pub fn checkpoint(&mut self) -> io::Result<()> {
        // Flush all writers first.
        for w in &mut self.writers {
            w.flush()?;
        }

        let n = self.state.positions.len();
        let cp = Checkpoint {
            current_step: self.current_step,
            total_steps: self.total_steps,
            time_span: self.time_span,
            positions: self.state.positions.iter().map(|v| v.to_array()).collect(),
            velocities: self.state.velocities.iter().map(|v| v.to_array()).collect(),
            gms: self.gms.clone(),
            masses: self.masses.clone(),
            star_id: self.star_id,
            sample_counts: self.sample_counts.clone(),
            last_sample_times: (0..n).map(|i| self.last_samples[i].time).collect(),
            last_sample_positions: (0..n)
                .map(|i| self.last_samples[i].position.to_array())
                .collect(),
            last_sample_velocities: (0..n)
                .map(|i| self.last_samples[i].velocity.to_array())
                .collect(),
        };

        let bytes = bincode::serialize(&cp)
            .map_err(io::Error::other)?;

        // Write to a temp file then rename for atomicity.
        let tmp_path = self.work_dir.join("checkpoint.tmp");
        fs::write(&tmp_path, &bytes)?;
        fs::rename(&tmp_path, checkpoint_path(&self.work_dir))?;

        self.steps_since_checkpoint = 0;
        Ok(())
    }

    /// Total bytes that will be written during finalize (for progress reporting).
    pub fn finalize_total_bytes(&self) -> u64 {
        let n = self.state.positions.len() as u64;
        let header = 8; // track count
        let per_track_overhead = 8 + 8; // mass_kg + sample_count
        let sample_bytes = self.sample_counts.iter().sum::<u64>() * SAMPLE_BYTES as u64;
        let star_samples = SAMPLE_BYTES as u64; // star has 1 sample
        let footer = 8; // time_span
        header + n * per_track_overhead + sample_bytes + star_samples + footer
    }

    /// Assemble the final ephemeris file from per-body sample files.
    ///
    /// Streams samples directly from per-body temp files into the output,
    /// writing the bincode wire format without loading everything into memory.
    /// Calls `on_progress(bytes_written, total_bytes)` periodically.
    pub fn finalize(
        mut self,
        output: &Path,
        mut on_progress: impl FnMut(u64, u64),
    ) -> io::Result<()> {
        // Final checkpoint + flush.
        self.checkpoint()?;

        let n = self.state.positions.len();
        let total_bytes = self.finalize_total_bytes();

        // Drop writers so files are closed.
        drop(self.writers);
        let mut bytes_written = 0u64;

        // Write bincode wire format directly:
        //   u64  track_count
        //   for each track:
        //     f64  mass_kg
        //     u64  sample_count
        //     for each sample: 7 × f64
        //   f64  time_span
        let tmp_output = output.with_extension("bin.tmp");
        let out_file = File::create(&tmp_output)?;
        let mut out = BufWriter::new(out_file);

        out.write_all(&(n as u64).to_le_bytes())?;
        bytes_written += 8;

        for i in 0..n {
            out.write_all(&self.masses[i].to_le_bytes())?;
            bytes_written += 8;

            if i == self.star_id {
                out.write_all(&1u64.to_le_bytes())?;
                let zero = 0.0_f64;
                for _ in 0..7 {
                    out.write_all(&zero.to_le_bytes())?;
                }
                bytes_written += 8 + SAMPLE_BYTES as u64;
            } else {
                let count = self.sample_counts[i];
                out.write_all(&count.to_le_bytes())?;
                bytes_written += 8;

                // Stream samples from the body file in chunks.
                let path = body_file_path(&self.work_dir, i);
                let mut reader = BufReader::new(File::open(&path)?);
                let mut buf = [0u8; 64 * 1024]; // 64 KB copy buffer
                let body_total_bytes = count as usize * SAMPLE_BYTES;
                let mut remaining = body_total_bytes;

                while remaining > 0 {
                    let to_read = remaining.min(buf.len());
                    reader.read_exact(&mut buf[..to_read])?;
                    out.write_all(&buf[..to_read])?;
                    remaining -= to_read;
                    bytes_written += to_read as u64;
                    on_progress(bytes_written, total_bytes);
                }
            }
        }

        out.write_all(&self.time_span.to_le_bytes())?;
        out.flush()?;
        drop(out);

        on_progress(total_bytes, total_bytes);

        // Atomic rename.
        fs::rename(&tmp_output, output)?;

        // Clean up work directory.
        fs::remove_dir_all(&self.work_dir)?;

        Ok(())
    }

    /// Number of bodies.
    pub fn body_count(&self) -> usize {
        self.state.positions.len()
    }

    /// Sample counts per body so far.
    pub fn sample_counts(&self) -> &[u64] {
        &self.sample_counts
    }

    /// Total samples across all bodies.
    pub fn total_sample_count(&self) -> u64 {
        self.sample_counts.iter().sum()
    }

    /// Access the current integration state (for energy validation etc.).
    pub fn current_state(&self) -> (&[DVec3], &[DVec3]) {
        (&self.state.positions, &self.state.velocities)
    }

    /// The GM values for all bodies.
    pub fn gms(&self) -> &[f64] {
        &self.gms
    }

    /// The star body index.
    pub fn star_id(&self) -> usize {
        self.star_id
    }

    /// Time span in seconds.
    pub fn time_span(&self) -> f64 {
        self.time_span
    }
}
