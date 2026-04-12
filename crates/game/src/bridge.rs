//! Bridge between the physics simulation and Bevy ECS.
//!
//! All simulation state lives in [`Simulation`] (physics crate). This module
//! is a thin adapter that:
//!
//! 1. Calls [`Simulation::step`] each frame to advance the ship.
//! 2. Refreshes prediction asynchronously when it is dirty or stale.
//! 3. Maps keyboard input to warp controls.

use std::sync::{
    Mutex,
    mpsc::{self, Receiver, Sender, TryRecvError},
};

use bevy::prelude::*;
use thalos_physics::{
    maneuver::ManeuverNode,
    simulation::PredictionRequest,
    trajectory::{PropagationBudget, TrajectoryPrediction, propagate_trajectory_budgeted},
};

use crate::SimStage;
use crate::maneuver::ManeuverPlan;
use crate::rendering::SimulationState;

/// Progressive prediction budgets (total integrator steps per pass).
///
/// Worker passes run from coarse to fine so a first usable path appears before
/// the final full-resolution prediction. Kept for future interplanetary use
/// where full predictions may take hundreds of milliseconds.
const PROGRESSIVE_BUDGETS: [usize; 3] = [2_000, 10_000, 40_000];

/// Maximum real-world seconds before the trajectory prediction is re-submitted,
/// even when nothing has changed. Keeps the trail origin close to the ship.
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
    is_final: bool,
}

#[derive(Resource)]
struct PredictionWorker {
    request_tx: Sender<PredictionJob>,
    result_rx: Mutex<Receiver<PredictionResult>>,
    in_flight: bool,
    latest_requested_epoch: Option<u64>,
    /// Real time (from `Time::elapsed_secs_f64`) when the last prediction job was submitted.
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
            in_flight: false,
            latest_requested_epoch: None,
            last_submit_real_time: 0.0,
        }
    }
}

fn setup_prediction_worker(mut commands: Commands) {
    commands.insert_resource(PredictionWorker::new());
}

pub fn advance_simulation(time: Res<Time>, mut sim: ResMut<SimulationState>) {
    sim.simulation.step(time.delta_secs_f64());
}

/// Drain all pending results from the worker, returning the final result if
/// one is available. Intermediate progressive results are discarded — only
/// the full-resolution final prediction is applied to avoid visual flashing.
fn drain_final_result(worker: &PredictionWorker) -> Option<PredictionResult> {
    let Ok(rx) = worker.result_rx.lock() else {
        return None;
    };

    let mut final_result: Option<PredictionResult> = None;
    loop {
        match rx.try_recv() {
            Ok(result) => {
                if result.is_final {
                    final_result = Some(result);
                }
                // Non-final results are silently discarded.
            }
            Err(TryRecvError::Empty) => return final_result,
            Err(TryRecvError::Disconnected) => return final_result,
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

    worker.in_flight = true;
    worker.latest_requested_epoch = Some(epoch);
    worker.last_submit_real_time = submit_real_time;
}

fn update_prediction(
    time: Res<Time>,
    mut sim: ResMut<SimulationState>,
    mut worker: ResMut<PredictionWorker>,
) {
    // 1. Receive any pending results from the worker.
    //    Only apply the final full-resolution result. Intermediate progressive
    //    passes are silently discarded to avoid visual flashing from coarse
    //    trajectories replacing full ones.
    if let Some(result) = drain_final_result(&worker) {
        let _accepted = sim.simulation.apply_prediction(
            result.prediction,
            result.predicted_at_sim_time,
            result.epoch,
        );
        if worker.latest_requested_epoch == Some(result.epoch) {
            worker.in_flight = false;
        }
    }

    // 2. Submit a replacement request immediately when maneuver edits advance
    // the prediction epoch, even if a worker pass is already running. The
    // worker thread will interrupt its progressive passes and restart from the
    // newest queued request.
    let real_now = time.elapsed_secs_f64();
    let current_epoch = sim.simulation.prediction_epoch();
    let needs_new_epoch = worker.latest_requested_epoch != Some(current_epoch);
    if sim.simulation.prediction_needs_refresh() && needs_new_epoch {
        let request = sim.simulation.prediction_request();
        sim.simulation.clear_prediction_dirty();
        submit_prediction_job(&mut worker, request, real_now);
        return;
    }

    // 3. Otherwise only resubmit on staleness when no request is already in flight.
    if !worker.in_flight {
        let real_stale = (real_now - worker.last_submit_real_time) >= PREDICTION_REAL_STALE_SECS;
        if sim.simulation.prediction_needs_refresh() || real_stale {
            let request = sim.simulation.prediction_request();
            sim.simulation.clear_prediction_dirty();
            submit_prediction_job(&mut worker, request, real_now);
        }
    }
}

/// Handle keyboard input to adjust the warp multiplier.
///
/// - `.`  -- increase to next warp level
/// - `,`  -- decrease to previous warp level (0x = paused)
/// - `\`  -- reset to 1x
pub fn handle_warp_controls(keys: Res<ButtonInput<KeyCode>>, mut sim: ResMut<SimulationState>) {
    let prev = sim.simulation.warp_speed();

    if keys.just_pressed(KeyCode::Period) {
        sim.simulation.increase_warp();
    } else if keys.just_pressed(KeyCode::Comma) {
        sim.simulation.decrease_warp();
    } else if keys.just_pressed(KeyCode::Backslash) {
        sim.simulation.reset_warp();
    }

    let new = sim.simulation.warp_speed();
    if (new - prev).abs() > 0.5 {
        info!("[bridge] warp speed: {}", sim.simulation.warp_label());
    }
}

/// Sync the UI-side ManeuverPlan to the physics ManeuverSequence when dirty.
///
/// Must run before `update_prediction` so the new nodes are included in the
/// next prediction request. `maneuvers_mut()` marks the prediction dirty internally.
fn sync_maneuver_plan(mut plan: ResMut<ManeuverPlan>, mut sim: ResMut<SimulationState>) {
    if !plan.dirty {
        return;
    }
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
    seq.nodes
        .sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
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

fn prediction_worker_loop(
    request_rx: Receiver<PredictionJob>,
    result_tx: Sender<PredictionResult>,
) {
    let mut pending: Option<PredictionJob> = None;

    loop {
        // Get next job: either carried over from an interrupted pass, or blocking recv.
        let mut current = if let Some(job) = pending.take() {
            job
        } else {
            match request_rx.recv() {
                Ok(job) => job,
                Err(_) => return,
            }
        };

        // Drain to the latest queued job — if multiple accumulated while we were
        // busy, only the most recent state matters.
        while let Ok(newer) = request_rx.try_recv() {
            current = newer;
        }

        // Run progressive passes (coarse → fine), sending intermediate results
        // so the renderer can cross-fade to a usable preview quickly.
        let coarse_budgets =
            progressive_budgets(current.request.prediction_config.max_steps_per_segment);
        let mut interrupted = false;
        for budget_steps in &coarse_budgets {
            let prediction =
                run_prediction(&current.request, Some(PropagationBudget::new(*budget_steps)));
            if result_tx
                .send(PredictionResult {
                    prediction,
                    epoch: current.request.epoch,
                    predicted_at_sim_time: current.request.sim_time,
                    is_final: false,
                })
                .is_err()
            {
                return;
            }

            // Check for a newer job after each pass completes.
            if let Ok(newer) = request_rx.try_recv() {
                pending = Some(newer);
                while let Ok(even_newer) = request_rx.try_recv() {
                    pending = Some(even_newer);
                }
                interrupted = true;
                break;
            }
        }

        if interrupted {
            continue;
        }

        // Check once more before the expensive final pass.
        if let Ok(newer) = request_rx.try_recv() {
            pending = Some(newer);
            while let Ok(even_newer) = request_rx.try_recv() {
                pending = Some(even_newer);
            }
            continue;
        }

        // Final unbounded pass.
        let prediction = run_prediction(&current.request, None);
        if result_tx
            .send(PredictionResult {
                prediction,
                epoch: current.request.epoch,
                predicted_at_sim_time: current.request.sim_time,
                is_final: true,
            })
            .is_err()
        {
            return;
        }
    }
}

fn progressive_budgets(max_steps_per_segment: usize) -> Vec<usize> {
    let mut budgets: Vec<usize> = PROGRESSIVE_BUDGETS
        .iter()
        .map(|b| (*b).min(max_steps_per_segment))
        .filter(|b| *b > 0)
        .collect();
    budgets.sort_unstable();
    budgets.dedup();
    budgets
}

fn run_prediction(
    request: &PredictionRequest,
    budget: Option<PropagationBudget>,
) -> TrajectoryPrediction {
    propagate_trajectory_budgeted(
        request.ship_state,
        request.sim_time,
        &request.maneuvers,
        request.ephemeris.as_ref(),
        &request.bodies,
        &request.prediction_config,
        request.integrator_config.clone(),
        request.ship_thrust_acceleration,
        budget,
    )
}
