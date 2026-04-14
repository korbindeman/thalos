//! Bridge between the physics simulation and Bevy ECS.
//!
//! All simulation state lives in [`Simulation`] (physics crate). This module
//! is a thin adapter that:
//!
//! 1. Calls [`Simulation::step`] each frame to advance the ship.
//! 2. Dispatches prediction refreshes to a worker thread on every maneuver
//!    edit. The worker runs a single unbudgeted pass per job — stable-orbit
//!    termination keeps the typical pass well under one frame (~4 ms), so
//!    the result lands on the next frame and the drag feels live. No
//!    synchronous prediction work on the main thread.
//! 3. Maps keyboard input to warp controls.

use std::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender, TryRecvError},
};

use bevy::prelude::*;
use thalos_physics::{
    maneuver::ManeuverNode,
    simulation::PredictionRequest,
    trajectory::{TrajectoryPrediction, propagate_trajectory_budgeted},
};

use crate::SimStage;
use crate::maneuver::ManeuverPlan;
use crate::rendering::SimulationState;

/// Maximum real-world seconds before the trajectory prediction is re-submitted
/// without a maneuver edit — keeps the trail origin close to the ship while it
/// drifts during idle time.
const PREDICTION_REAL_STALE_SECS: f64 = 2.0;

// ---------------------------------------------------------------------------
// Prediction worker types
// ---------------------------------------------------------------------------

struct PredictionJob {
    request: PredictionRequest,
}

struct PredictionResult {
    prediction: TrajectoryPrediction,
    epoch: u64,
    predicted_at_sim_time: f64,
}

#[derive(Resource)]
struct PredictionWorker {
    request_tx: Sender<PredictionJob>,
    result_rx: Mutex<Receiver<PredictionResult>>,
    latest_requested_epoch: Option<u64>,
    /// Real time (from `Time::elapsed_secs_f64`) when the last prediction job
    /// was submitted.
    last_submit_real_time: f64,
}

impl PredictionWorker {
    fn new() -> Self {
        let (request_tx, request_rx) = mpsc::channel::<PredictionJob>();
        let (result_tx, result_rx) = mpsc::channel::<PredictionResult>();

        std::thread::Builder::new()
            .name("trajectory-prediction".to_string())
            .spawn(move || prediction_worker_loop(request_rx, result_tx))
            .expect("failed to spawn trajectory prediction worker thread");

        Self {
            request_tx,
            result_rx: Mutex::new(result_rx),
            latest_requested_epoch: None,
            last_submit_real_time: 0.0,
        }
    }
}

fn setup_prediction_worker(mut commands: Commands) {
    commands.insert_resource(PredictionWorker::new());
}

pub fn advance_simulation(time: Res<Time>, mut sim: ResMut<SimulationState>) {
    let _span = tracing::info_span!("advance_simulation").entered();
    sim.simulation.step(time.delta_secs_f64());
}

/// Drain pending results and return the one with the newest epoch. Older
/// results are thrown away — they've already been superseded by a fresh edit.
fn drain_latest_result(worker: &PredictionWorker) -> Option<PredictionResult> {
    let Ok(rx) = worker.result_rx.lock() else {
        return None;
    };

    let mut latest: Option<PredictionResult> = None;
    loop {
        match rx.try_recv() {
            Ok(result) => {
                let accept = latest
                    .as_ref()
                    .map(|prev| result.epoch > prev.epoch)
                    .unwrap_or(true);
                if accept {
                    latest = Some(result);
                }
            }
            Err(TryRecvError::Empty) => return latest,
            Err(TryRecvError::Disconnected) => return latest,
        }
    }
}

fn submit_prediction_job(
    worker: &mut PredictionWorker,
    request: PredictionRequest,
    submit_real_time: f64,
) {
    let epoch = request.epoch;
    let job = PredictionJob { request };

    if worker.request_tx.send(job).is_err() {
        warn!("[bridge] trajectory worker is unavailable; prediction refresh skipped");
        return;
    }

    worker.latest_requested_epoch = Some(epoch);
    worker.last_submit_real_time = submit_real_time;
}

fn update_prediction(
    time: Res<Time>,
    mut sim: ResMut<SimulationState>,
    mut worker: ResMut<PredictionWorker>,
) {
    let _span = tracing::info_span!("update_prediction").entered();

    // Observation mode: freeze prediction work entirely. Still drain any
    // stray result so the channel doesn't back up.
    if sim.simulation.is_observation_mode() {
        let _ = drain_latest_result(&worker);
        return;
    }

    if let Some(result) = drain_latest_result(&worker) {
        sim.simulation.apply_prediction(
            result.prediction,
            result.predicted_at_sim_time,
            result.epoch,
        );
    }

    let real_now = time.elapsed_secs_f64();
    let current_epoch = sim.simulation.prediction_epoch();
    let has_new_epoch = worker.latest_requested_epoch != Some(current_epoch);
    let dirty = sim.simulation.prediction_needs_refresh();
    let real_stale = (real_now - worker.last_submit_real_time) >= PREDICTION_REAL_STALE_SECS;

    if dirty && (has_new_epoch || real_stale) {
        let request = sim.simulation.prediction_request();
        sim.simulation.clear_prediction_dirty();
        submit_prediction_job(&mut worker, request, real_now);
    }
}

/// Handle keyboard input to adjust the warp multiplier.
///
/// - `.`      -- increase to next warp level
/// - `,`      -- decrease to previous warp level (0x = paused)
/// - `\`      -- reset to 1x
/// - `Space`  -- toggle pause (0x) / resume previous level
pub fn handle_warp_controls(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<SimulationState>) {
    let prev = sim.simulation.warp_speed();

    if keys.just_pressed(KeyCode::Period) {
        sim.simulation.increase_warp();
    } else if keys.just_pressed(KeyCode::Comma) {
        sim.simulation.decrease_warp();
    } else if keys.just_pressed(KeyCode::Backslash) {
        sim.simulation.reset_warp();
    } else if keys.just_pressed(KeyCode::Space) {
        sim.simulation.toggle_pause();
    }

    let new = sim.simulation.warp_speed();
    if (new - prev).abs() > 0.5 {
        info!("[bridge] warp speed: {}", sim.simulation.warp_label());
    }
}

/// Sync the UI-side [`ManeuverPlan`] to the physics `ManeuverSequence` when
/// dirty. `maneuvers_mut()` marks the prediction dirty; `update_prediction`
/// runs after this system and dispatches the fresh job to the worker.
///
/// Also drops UI nodes whose execution time has passed — the physics side
/// auto-consumes them as burns, and leaving stale nodes in the UI would
/// re-inject them on the next edit.
fn sync_maneuver_plan(mut plan: ResMut<ManeuverPlan>, mut sim: ResMut<SimulationState>) {
    let sim_time = sim.simulation.sim_time();
    let before = plan.nodes.len();
    plan.nodes.retain(|n| n.time > sim_time);
    if plan.nodes.len() != before {
        plan.dirty = true;
    }

    if !plan.dirty {
        return;
    }
    let _span = tracing::info_span!("sync_maneuver_plan").entered();
    plan.dirty = false;

    let seq = sim.simulation.maneuvers_mut();
    seq.nodes.clear();
    for node in &plan.nodes {
        seq.nodes.push(ManeuverNode {
            time: node.time,
            delta_v: node.delta_v,
            reference_body: node.reference_body,
        });
    }
    seq.nodes.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

pub struct BridgePlugin;

impl Plugin for BridgePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_prediction_worker);
        app.add_systems(
            Update,
            (
                handle_warp_controls,
                advance_simulation,
                sync_maneuver_plan,
                update_prediction,
            )
                .chain()
                .in_set(SimStage::Physics),
        );
    }
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

/// Single-pass worker loop. Each job runs one unbudgeted
/// `propagate_flight_plan` call — early termination (stable orbit, collision,
/// cone fade) keeps typical passes at ~4 ms, so results land on the next
/// frame from the user's edit.
///
/// If multiple jobs accumulated while a pass was running, only the newest is
/// honored — earlier submissions are stale the moment a new one arrives.
fn prediction_worker_loop(
    request_rx: Receiver<PredictionJob>,
    result_tx: Sender<PredictionResult>,
) {
    loop {
        let mut current = match request_rx.recv() {
            Ok(job) => job,
            Err(_) => return,
        };

        // Drain to the newest queued job; anything older is already stale.
        while let Ok(newer) = request_rx.try_recv() {
            current = newer;
        }

        let prediction = run_prediction(&current.request);
        if result_tx
            .send(PredictionResult {
                prediction,
                epoch: current.request.epoch,
                predicted_at_sim_time: current.request.sim_time,
            })
            .is_err()
        {
            return;
        }
    }
}

fn run_prediction(request: &PredictionRequest) -> TrajectoryPrediction {
    propagate_trajectory_budgeted(
        request.ship_state,
        request.sim_time,
        &request.maneuvers,
        request.active_burns.clone(),
        request.ephemeris.as_ref(),
        &request.bodies,
        &request.prediction_config,
        request.integrator_config.clone(),
        request.ship_thrust_acceleration,
        None,
    )
}
