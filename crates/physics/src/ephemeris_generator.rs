//! Streaming ephemeris generator with checkpoint/resume support.
//!
//! Writes Chebyshev segments to per-body temp files while keeping only the
//! current integration state and the in-progress segment buffers in memory.

use crate::ephemeris::{
    build_initial_states, fit_chebyshev_segment, read_segment, rk4_step, write_segment,
    BodyTrack, Ephemeris, EphemerisSample, NBodyState, Rk4Scratch, DT,
    FILE_HEADER_BYTES, SEGMENT_BYTES, SEGMENT_STEPS, TRACK_HEADER_BYTES,
};
use crate::types::SolarSystemDefinition;
use glam::DVec3;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Steps between automatic checkpoints during `advance`.
const CHECKPOINT_INTERVAL: u64 = 100_000;

// ---------------------------------------------------------------------------
// Checkpoint
// ---------------------------------------------------------------------------

struct Checkpoint {
    current_step: u64,
    total_steps: u64,
    time_span: f64,
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    gms: Vec<f64>,
    masses: Vec<f64>,
    star_id: usize,
    segment_counts: Vec<u64>,
    segment_buffers: Vec<Vec<EphemerisSample>>,
}

// ---------------------------------------------------------------------------
// Raw segment I/O
// ---------------------------------------------------------------------------

fn body_file_path(work_dir: &Path, body_id: usize) -> PathBuf {
    work_dir.join(format!("body_{body_id:04}.segments"))
}

fn checkpoint_path(work_dir: &Path) -> PathBuf {
    work_dir.join("checkpoint.bin")
}

fn write_u64(w: &mut impl Write, value: u64) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn write_f64(w: &mut impl Write, value: f64) -> io::Result<()> {
    w.write_all(&value.to_le_bytes())
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut bytes = [0u8; 8];
    r.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn serialize_checkpoint(cp: &Checkpoint) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"THCPSEG1");
    write_u64(&mut bytes, cp.current_step)?;
    write_u64(&mut bytes, cp.total_steps)?;
    write_f64(&mut bytes, cp.time_span)?;

    write_u64(&mut bytes, cp.positions.len() as u64)?;
    for values in [&cp.positions, &cp.velocities] {
        for xyz in values.iter() {
            for &v in xyz.iter() {
                write_f64(&mut bytes, v)?;
            }
        }
    }

    for scalars in [&cp.gms, &cp.masses] {
        write_u64(&mut bytes, scalars.len() as u64)?;
        for &v in scalars.iter() {
            write_f64(&mut bytes, v)?;
        }
    }

    write_u64(&mut bytes, cp.star_id as u64)?;
    write_u64(&mut bytes, cp.segment_counts.len() as u64)?;
    for &count in &cp.segment_counts {
        write_u64(&mut bytes, count)?;
    }

    write_u64(&mut bytes, cp.segment_buffers.len() as u64)?;
    for buffer in &cp.segment_buffers {
        write_u64(&mut bytes, buffer.len() as u64)?;
        for sample in buffer {
            write_f64(&mut bytes, sample.time)?;
            for v in [
                sample.position.x,
                sample.position.y,
                sample.position.z,
                sample.velocity.x,
                sample.velocity.y,
                sample.velocity.z,
            ] {
                write_f64(&mut bytes, v)?;
            }
        }
    }

    Ok(bytes)
}

fn deserialize_checkpoint(bytes: &[u8]) -> io::Result<Checkpoint> {
    let mut reader = io::Cursor::new(bytes);
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != b"THCPSEG1" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Unrecognized checkpoint format",
        ));
    }

    let current_step = read_u64(&mut reader)?;
    let total_steps = read_u64(&mut reader)?;
    let time_span = read_f64(&mut reader)?;

    let body_count = read_u64(&mut reader)? as usize;
    let mut read_vec3s = || -> io::Result<Vec<[f64; 3]>> {
        let mut out = Vec::with_capacity(body_count);
        for _ in 0..body_count {
            out.push([
                read_f64(&mut reader)?,
                read_f64(&mut reader)?,
                read_f64(&mut reader)?,
            ]);
        }
        Ok(out)
    };
    let positions = read_vec3s()?;
    let velocities = read_vec3s()?;

    let mut read_scalars = || -> io::Result<Vec<f64>> {
        let len = read_u64(&mut reader)? as usize;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(read_f64(&mut reader)?);
        }
        Ok(out)
    };
    let gms = read_scalars()?;
    let masses = read_scalars()?;

    let star_id = read_u64(&mut reader)? as usize;
    let segment_count_len = read_u64(&mut reader)? as usize;
    let mut segment_counts = Vec::with_capacity(segment_count_len);
    for _ in 0..segment_count_len {
        segment_counts.push(read_u64(&mut reader)?);
    }

    let buffer_count = read_u64(&mut reader)? as usize;
    let mut segment_buffers = Vec::with_capacity(buffer_count);
    for _ in 0..buffer_count {
        let len = read_u64(&mut reader)? as usize;
        let mut buffer = Vec::with_capacity(len);
        for _ in 0..len {
            let time = read_f64(&mut reader)?;
            let position = DVec3::new(
                read_f64(&mut reader)?,
                read_f64(&mut reader)?,
                read_f64(&mut reader)?,
            );
            let velocity = DVec3::new(
                read_f64(&mut reader)?,
                read_f64(&mut reader)?,
                read_f64(&mut reader)?,
            );
            buffer.push(EphemerisSample { time, position, velocity });
        }
        segment_buffers.push(buffer);
    }

    Ok(Checkpoint {
        current_step,
        total_steps,
        time_span,
        positions,
        velocities,
        gms,
        masses,
        star_id,
        segment_counts,
        segment_buffers,
    })
}

// ---------------------------------------------------------------------------
// Public generator
// ---------------------------------------------------------------------------

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
    segment_buffers: Vec<Vec<EphemerisSample>>,
    writers: Vec<BufWriter<File>>,
    segment_counts: Vec<u64>,
    steps_since_checkpoint: u64,
}

impl EphemerisGenerator {
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

        let mut writers = Vec::with_capacity(n);
        let mut segment_buffers = Vec::with_capacity(n);
        let mut segment_counts = vec![0u64; n];

        for i in 0..n {
            let path = body_file_path(work_dir, i);
            let file = File::create(&path)?;
            let mut writer = BufWriter::new(file);
            let initial = EphemerisSample {
                time: 0.0,
                position: init_positions[i],
                velocity: init_velocities[i],
            };

            if i == star_id {
                let segment = fit_chebyshev_segment(&[initial]);
                write_segment(&mut writer, &segment)?;
                segment_counts[i] = 1;
            }

            writers.push(writer);
            segment_buffers.push(vec![initial]);
        }

        Ok(Self {
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
            segment_buffers,
            writers,
            segment_counts,
            steps_since_checkpoint: 0,
        })
    }

    fn resume(
        system: &SolarSystemDefinition,
        time_span: f64,
        work_dir: &Path,
    ) -> io::Result<Self> {
        let cp = deserialize_checkpoint(&fs::read(checkpoint_path(work_dir))?)?;
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

        let effective_span = time_span.max(cp.time_span);
        let new_total_steps = (effective_span / DT).ceil() as u64;

        let positions: Vec<DVec3> = cp.positions.iter().map(|p| DVec3::from_array(*p)).collect();
        let velocities: Vec<DVec3> = cp.velocities.iter().map(|v| DVec3::from_array(*v)).collect();

        let mut writers = Vec::with_capacity(n);
        for i in 0..n {
            let path = body_file_path(work_dir, i);
            let expected_bytes = cp.segment_counts[i] as usize * SEGMENT_BYTES;

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
            segment_buffers: cp.segment_buffers,
            writers,
            segment_counts: cp.segment_counts,
            steps_since_checkpoint: 0,
        })
    }

    pub fn is_done(&self) -> bool {
        self.current_step >= self.total_steps
    }

    pub fn progress(&self) -> (u64, u64) {
        (self.current_step, self.total_steps)
    }

    pub fn advance(&mut self, batch_size: u64) -> io::Result<u64> {
        let end_step = (self.current_step + batch_size).min(self.total_steps);
        let steps_to_take = end_step - self.current_step;

        for _ in 0..steps_to_take {
            self.current_step += 1;
            rk4_step(&mut self.state, &mut self.scratch, &self.gms, DT, self.star_id);

            let t = self.current_step as f64 * DT;
            let is_last = self.current_step == self.total_steps;
            let n = self.state.positions.len();

            for i in 0..n {
                if i == self.star_id {
                    continue;
                }

                self.segment_buffers[i].push(EphemerisSample {
                    time: t,
                    position: self.state.positions[i],
                    velocity: self.state.velocities[i],
                });

                let completed_steps = self.segment_buffers[i].len() - 1;
                if completed_steps >= SEGMENT_STEPS || is_last {
                    let segment = fit_chebyshev_segment(&self.segment_buffers[i]);
                    write_segment(&mut self.writers[i], &segment)?;
                    self.segment_counts[i] += 1;

                    let last_sample = *self.segment_buffers[i].last().unwrap();
                    self.segment_buffers[i].clear();
                    self.segment_buffers[i].push(last_sample);
                }
            }

            self.steps_since_checkpoint += 1;
            if self.steps_since_checkpoint >= CHECKPOINT_INTERVAL {
                self.checkpoint()?;
            }
        }

        Ok(steps_to_take)
    }

    pub fn checkpoint(&mut self) -> io::Result<()> {
        for w in &mut self.writers {
            w.flush()?;
        }

        let cp = Checkpoint {
            current_step: self.current_step,
            total_steps: self.total_steps,
            time_span: self.time_span,
            positions: self.state.positions.iter().map(|v| v.to_array()).collect(),
            velocities: self.state.velocities.iter().map(|v| v.to_array()).collect(),
            gms: self.gms.clone(),
            masses: self.masses.clone(),
            star_id: self.star_id,
            segment_counts: self.segment_counts.clone(),
            segment_buffers: self.segment_buffers.clone(),
        };

        let bytes = serialize_checkpoint(&cp)?;
        let tmp_path = self.work_dir.join("checkpoint.tmp");
        fs::write(&tmp_path, &bytes)?;
        fs::rename(&tmp_path, checkpoint_path(&self.work_dir))?;

        self.steps_since_checkpoint = 0;
        Ok(())
    }

    pub fn finalize_total_bytes(&self) -> u64 {
        FILE_HEADER_BYTES
            + self.state.positions.len() as u64 * TRACK_HEADER_BYTES
            + self.segment_counts.iter().sum::<u64>() * SEGMENT_BYTES as u64
    }

    pub fn finalize(
        mut self,
        output: &Path,
        mut on_progress: impl FnMut(u64, u64),
    ) -> io::Result<()> {
        self.checkpoint()?;

        let n = self.state.positions.len();
        let total_bytes = self.finalize_total_bytes();

        drop(self.writers);
        let tmp_output = output.with_extension("bin.tmp");
        let out_file = File::create(&tmp_output)?;
        let mut out = BufWriter::new(out_file);

        let mut tracks = Vec::with_capacity(n);
        for i in 0..n {
            let count = self.segment_counts[i] as usize;
            let mut segments = Vec::with_capacity(count);
            let path = body_file_path(&self.work_dir, i);
            let mut reader = BufReader::new(File::open(&path)?);
            for _ in 0..count {
                segments.push(read_segment(&mut reader)?);
            }
            tracks.push(BodyTrack {
                mass_kg: self.masses[i],
                segments,
            });
        }

        let ephemeris = Ephemeris {
            tracks,
            time_span: self.time_span,
        };
        ephemeris.write_to(&mut out)?;
        out.flush()?;
        drop(out);

        on_progress(total_bytes, total_bytes);

        fs::rename(&tmp_output, output)?;
        fs::remove_dir_all(&self.work_dir)?;

        Ok(())
    }

    pub fn body_count(&self) -> usize {
        self.state.positions.len()
    }

    pub fn segment_counts(&self) -> &[u64] {
        &self.segment_counts
    }

    pub fn total_segment_count(&self) -> u64 {
        self.segment_counts.iter().sum()
    }

    pub fn current_state(&self) -> (&[DVec3], &[DVec3]) {
        (&self.state.positions, &self.state.velocities)
    }

    pub fn gms(&self) -> &[f64] {
        &self.gms
    }

    pub fn star_id(&self) -> usize {
        self.star_id
    }

    pub fn time_span(&self) -> f64 {
        self.time_span
    }
}
