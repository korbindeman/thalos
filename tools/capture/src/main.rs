use std::{
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    process::{Command, ExitCode},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thalos_capture_protocol::{
    CAPTURE_PRESETS, CAPTURE_PROTOCOL_SCHEMA, CaptureAction, CaptureRequest, CaptureResponse,
    CaptureServerState,
};
use uuid::Uuid;

mod compare;

const REQUEST_FILE: &str = "visual_capture_request.json";
const RESPONSE_FILE: &str = "visual_capture_response.json";
const STATE_FILE: &str = "visual_capture_server.json";
const LAUNCHER_FILE: &str = "visual_capture_launcher.json";
const LOG_FILE: &str = "visual_capture_server.log";

const OVERRIDE_KEYS: &[&str] = &[
    "THALOS_SCREENSHOT_OUT",
    "THALOS_SCREENSHOT_REPORT",
    "THALOS_SCREENSHOT_SIZE",
    "THALOS_SCREENSHOT_AZIMUTH",
    "THALOS_SCREENSHOT_ELEVATION",
    "THALOS_SCREENSHOT_DISTANCE",
    "THALOS_SCREENSHOT_WARMUP",
    "THALOS_SCREENSHOT_HUD",
    "THALOS_SCREENSHOT_CAMERA_ALTITUDE",
    "THALOS_SCREENSHOT_LOOK_ELEVATION",
    "THALOS_SCREENSHOT_SUN_ELEVATION",
    "THALOS_SCREENSHOT_CLOUD_QUALITY",
    "THALOS_SCREENSHOT_CLOUD_TEMPORAL",
    "THALOS_SCREENSHOT_CLOUD_COVERAGE",
    "THALOS_SCREENSHOT_CLOUD_RECONSTRUCTION",
    "THALOS_SCREENSHOT_CLOUD_DENSITY_COUPLING",
    "THALOS_SCREENSHOT_CLOUD_TIER",
    "THALOS_SCREENSHOT_CLOUD_FAR_FILTER",
    "THALOS_SCREENSHOT_CLOUD_FAR_AGGREGATION",
    "THALOS_SCREENSHOT_OCEAN_TIME",
    "THALOS_PLUME_PRESSURE",
    "THALOS_CONTACT_SHADOW",
    "THALOS_SSAO",
    "THALOS_TERRAIN_INSPECTION",
    "THALOS_TERRAIN_CULL",
    "THALOS_TERRAIN",
    "THALOS_TILE_RENDERER",
    "THALOS_TILE_CACHE",
    "THALOS_RUNWAY_SITE",
    "THALOS_WGPU_BACKEND",
];

/// Request inputs that shape boot-time world or renderer construction rather
/// than a live capture resource. A change in this subset requires a restart.
const STARTUP_OVERRIDE_KEYS: &[&str] = &[
    "THALOS_SCREENSHOT_SIZE",
    "THALOS_TERRAIN_CULL",
    "THALOS_TERRAIN",
    "THALOS_TILE_RENDERER",
    "THALOS_TILE_CACHE",
    "THALOS_RUNWAY_SITE",
    "THALOS_WGPU_BACKEND",
];

#[derive(Debug, Deserialize, Serialize)]
struct LauncherState {
    schema_version: u32,
    pid: u32,
    preset: String,
    launched_unix_ms: u128,
    #[serde(default)]
    log_start_bytes: u64,
    #[serde(default)]
    startup_overrides: BTreeMap<String, String>,
    command: Vec<String>,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("visual capture failed: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "shot".to_owned());
    match command.as_str() {
        "shot" | "capture" => {
            let ShotArgs {
                presets,
                output,
                report,
                assignments,
            } = parse_capture_options(args)?;
            if presets.len() > 1 && (output.is_some() || report.is_some()) {
                return Err("--out and --report require exactly one preset".into());
            }
            if presets.len() == 1 {
                capture(&presets[0], output, report, assignments)?;
            } else {
                capture_batch(presets, assignments)?;
            }
        }
        "compare" => compare::run_cli(args)?,
        "status" => status(),
        "stop" => stop_server(false)?,
        "reset" => reset_build_state()?,
        "-h" | "--help" | "help" => print_help(),
        other => return Err(format!("unknown command {other:?}; use --help")),
    }
    Ok(())
}

/// Capture every requested scene while minimizing full host boots. The host is
/// the authority on compatibility: after the first scene in a boot context,
/// pull every remaining compatible scene forward before crossing into the next
/// context.
fn capture_batch(mut pending: Vec<String>, assignments: Vec<String>) -> Result<(), String> {
    let total = pending.len();
    let mut completed = 0;
    while !pending.is_empty() {
        let first = pending.remove(0);
        completed += 1;
        println!("[{completed}/{total}] scene={first}");
        capture(&first, None, None, assignments.clone())?;

        let compatible = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE))
            .map(|state| state.compatible_presets)
            .unwrap_or_default();
        let (same_context, later) = pending
            .into_iter()
            .partition(|preset| compatible.contains(preset));
        pending = later;
        for preset in same_context {
            completed += 1;
            println!("[{completed}/{total}] scene={preset} (reusing world)");
            capture(&preset, None, None, assignments.clone())?;
        }
    }
    Ok(())
}

#[derive(Debug, Eq, PartialEq)]
struct ShotArgs {
    presets: Vec<String>,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    assignments: Vec<String>,
}

fn parse_capture_options(mut args: impl Iterator<Item = String>) -> Result<ShotArgs, String> {
    let mut presets = Vec::new();
    let mut output = None;
    let mut report = None;
    let mut assignments = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => output = Some(PathBuf::from(args.next().ok_or("--out requires a path")?)),
            "--report" => {
                report = Some(PathBuf::from(
                    args.next().ok_or("--report requires a path")?,
                ))
            }
            "--set" => assignments.push(args.next().ok_or("--set requires KEY=VALUE")?),
            option if option.starts_with('-') => {
                return Err(format!("unknown shot option {option:?}"));
            }
            preset => presets.push(canonical_preset(preset)?),
        }
    }
    if presets.is_empty() {
        presets.push("spaceport-aerial".to_owned());
    }
    Ok(ShotArgs {
        presets,
        output,
        report,
        assignments,
    })
}

pub(crate) fn capture(
    preset: &str,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    assignments: Vec<String>,
) -> Result<PathBuf, String> {
    let preset = canonical_preset(preset)?;
    let mut overrides = OVERRIDE_KEYS
        .iter()
        .filter_map(|key| env::var(key).ok().map(|value| ((*key).to_owned(), value)))
        .collect::<BTreeMap<_, _>>();
    for assignment in assignments {
        let (key, value) = assignment
            .split_once('=')
            .ok_or_else(|| format!("--set expects KEY=VALUE, got {assignment:?}"))?;
        if !OVERRIDE_KEYS.contains(&key) {
            return Err(format!("unsupported capture override {key:?}"));
        }
        overrides.insert(key.to_owned(), value.to_owned());
    }
    if let Some(path) = output {
        overrides.insert(
            "THALOS_SCREENSHOT_OUT".into(),
            absolute(path).display().to_string(),
        );
    }
    if let Some(path) = report {
        overrides.insert(
            "THALOS_SCREENSHOT_REPORT".into(),
            absolute(path).display().to_string(),
        );
    }

    let mut state = ensure_server(&preset, &overrides)?;
    match capture_once(&preset, &overrides, &state) {
        Ok(path) => Ok(path),
        Err(error) if error.recoverable => {
            eprintln!(
                "capture host became unhealthy ({}); restarting once and retrying scene={preset}",
                error.message.lines().next().unwrap_or("unknown error")
            );
            stop_server(true)?;
            state = start_server(&preset, &overrides)?;
            capture_once(&preset, &overrides, &state).map_err(|error| error.message)
        }
        Err(error) => Err(error.message),
    }
}

#[derive(Debug)]
struct CaptureFailure {
    message: String,
    recoverable: bool,
}

fn capture_once(
    preset: &str,
    overrides: &BTreeMap<String, String>,
    state: &CaptureServerState,
) -> Result<PathBuf, CaptureFailure> {
    let log_start = if state.completed_captures == 0 {
        read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
            .map_or(0, |launcher| launcher.log_start_bytes)
    } else {
        file_len(&diagnostic_path(LOG_FILE))
    };
    let request = CaptureRequest {
        schema_version: CAPTURE_PROTOCOL_SCHEMA,
        id: Uuid::new_v4().simple().to_string(),
        action: CaptureAction::Capture,
        preset: preset.to_owned(),
        overrides: overrides.clone(),
    };
    write_json_atomic(&diagnostic_path(REQUEST_FILE), &request).map_err(|message| {
        CaptureFailure {
            message,
            recoverable: false,
        }
    })?;
    // Software-Vulkan (llvmpipe) fallback boxes render warmup at seconds per
    // frame, so the response wait must cover a full preset warmup there; a
    // hardware run answers long before either limit.
    let deadline = Instant::now() + Duration::from_secs(capture_timeout_secs());
    while Instant::now() < deadline {
        if let Some(response) = read_json::<CaptureResponse>(&diagnostic_path(RESPONSE_FILE))
            && response.id == request.id
        {
            if response.schema_version != CAPTURE_PROTOCOL_SCHEMA {
                return Err(CaptureFailure {
                    message: format!(
                        "capture response schema {} does not match client schema {}",
                        response.schema_version, CAPTURE_PROTOCOL_SCHEMA
                    ),
                    recoverable: true,
                });
            }
            if !response.ok {
                return Err(CaptureFailure {
                    recoverable: response.message.contains("different boot world"),
                    message: response.message,
                });
            }
            let path = response
                .output
                .map(PathBuf::from)
                .ok_or_else(|| CaptureFailure {
                    message: "capture succeeded without an output path".into(),
                    recoverable: true,
                })?;
            validate_capture_output(&path)?;
            validate_render_log(log_start)?;
            println!("captured {}", path.display());
            return Ok(path);
        }
        if !process_alive(state.pid) {
            return Err(CaptureFailure {
                message: format!("capture server exited\n{}", log_tail(50)),
                recoverable: true,
            });
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(CaptureFailure {
        message: format!("capture timed out\n{}", log_tail(50)),
        recoverable: true,
    })
}

fn ensure_server(
    preset: &str,
    overrides: &BTreeMap<String, String>,
) -> Result<CaptureServerState, String> {
    let state = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE));
    let state = if compatible_state(state.as_ref(), preset, requested_size(overrides))
        && launcher_matches(overrides)
        && !host_sources_stale()
    {
        state.expect("compatible state exists")
    } else {
        // Stale Rust/Cargo sources restart the host through a rebuild: there
        // is no in-process code reload (Rust hot-patching was retired —
        // ADR-20260724T153619Z; an applied subsecond patch crashed the app,
        // INC-20260724T044418Z). `cargo run` below recompiles whatever
        // changed and relaunches, so a Rust edit can never leave a silently
        // stale or crashed server behind.
        start_server(preset, overrides)?
    };
    wait_for_shader_reload(state)
}

/// True when any workspace Rust/manifest source is newer than the running
/// host's launch time — the host binary can no longer match the tree.
fn host_sources_stale() -> bool {
    let launched = read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        .map_or(0, |launcher| launcher.launched_unix_ms);
    let roots = [
        workspace_root().join("apps"),
        workspace_root().join("crates"),
        workspace_root().join("tools/capture_host"),
    ];
    let tree_mtime = newest_mtime_ms(&roots, "rs").max(newest_mtime_ms(&roots, "toml"));
    let root_mtime = ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"]
        .iter()
        .map(|path| modified_millis(&workspace_root().join(path)))
        .max()
        .unwrap_or(0);
    tree_mtime.max(root_mtime) > launched
}

/// Boot the host, recovering once from a self-inconsistent build tree.
///
/// A dev-lane build can fail for two very different reasons: the code is wrong
/// (the agent must read the error), or the *artifacts* disagree with each other
/// (nobody's error — the tree just needs its incremental cache dropped). The
/// second class used to end the run with a raw log tail, and the obvious manual
/// remedy — `cargo clean -p bevy_dylib` — is what produces it in the first
/// place. Recover automatically instead, so a stale-artifact link failure costs
/// a rebuild rather than a diagnosis session. See
/// INC-20260724T182642Z-stale-dylib-incremental-link-corruption.
fn start_server(
    preset: &str,
    overrides: &BTreeMap<String, String>,
) -> Result<CaptureServerState, String> {
    let offset = file_len(&diagnostic_path(LOG_FILE));
    match start_server_once(preset, overrides) {
        Ok(state) => Ok(state),
        Err(_) if toolchain_corruption(&log_from(offset)) => {
            eprintln!(
                "build failed on stale-artifact corruption (objects reference internal symbols \
                 from a previous bevy_dylib link, not a code error); dropping \
                 target/debug/incremental and rebuilding once"
            );
            purge_incremental_cache()?;
            let retry_offset = file_len(&diagnostic_path(LOG_FILE));
            start_server_once(preset, overrides).map_err(|retry| {
                if toolchain_corruption(&log_from(retry_offset)) {
                    format!(
                        "{retry}\n\nstill corrupt after dropping the incremental cache — \
                         run `just build-reset` for a full lane reset"
                    )
                } else {
                    retry
                }
            })
        }
        Err(error) => Err(error),
    }
}

/// Link/metadata failures that mean "the build tree disagrees with itself"
/// rather than "this code does not compile".
///
/// The observed signature is `undefined symbol: anon.<cgu>.N.llvm.<hash>`
/// referenced from a workspace rlib: LLVM-internalized symbols shared with
/// `bevy_dylib` that an incremental CGU still names after the dylib was rebuilt
/// or partially cleaned underneath it.
pub(crate) fn toolchain_corruption(log: &str) -> bool {
    let lower = log.to_ascii_lowercase();
    lower.contains("undefined symbol: anon.")
        || (lower.contains("undefined symbol") && lower.contains(".llvm."))
        || lower.contains("found invalid metadata files")
        || (lower.contains("incremental compilation") && lower.contains("corrupt"))
}

fn purge_incremental_cache() -> Result<(), String> {
    let cache = workspace_root().join("target/debug/incremental");
    match fs::remove_dir_all(&cache) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!("purge {}: {error}", cache.display())),
    }
}

/// Full dev-lane reset: stop the host, drop the incremental cache, and clean
/// the dynamic-linking set **as one unit**.
///
/// The set matters. Cleaning `bevy_dylib` on its own (or any subset of the
/// crates that link against it) is exactly the partial state that produces
/// dangling internal-symbol references — the reset exists so nobody has to
/// hand-roll `cargo clean -p` again.
fn reset_build_state() -> Result<(), String> {
    let _ = stop_server(true);
    purge_incremental_cache()?;
    println!("dropped target/debug/incremental");
    let status = Command::new("cargo")
        .args([
            "clean",
            "-p",
            "bevy_dylib",
            "-p",
            "thalos_runtime",
            "-p",
            "thalos_capture_runtime",
            "-p",
            "thalos_capture_host",
            "-p",
            "thalos_game",
            "-p",
            "thalos_body_render",
        ])
        .current_dir(workspace_root())
        .status()
        .map_err(|error| format!("cargo clean: {error}"))?;
    if !status.success() {
        return Err(format!("cargo clean exited with {status}"));
    }
    println!("cleaned the dynamic-linking crate set; the next capture rebuilds it");
    Ok(())
}

fn start_server_once(
    preset: &str,
    overrides: &BTreeMap<String, String>,
) -> Result<CaptureServerState, String> {
    stop_server(true)?;
    fs::create_dir_all(diagnostics_dir()).map_err(|error| error.to_string())?;
    for filename in [REQUEST_FILE, RESPONSE_FILE, STATE_FILE] {
        let _ = fs::remove_file(diagnostic_path(filename));
    }

    // Plain cargo run on the shared dynamic dev fingerprint: cargo rebuilds
    // whatever changed, sets up the bevy_dylib search path for its child
    // (the INC-0008 contract), and stays resident as the launcher process
    // whose tree `stop` terminates.
    let command = vec![
        "run",
        "-p",
        "thalos_capture_host",
        "--features",
        "dev-renderer",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();

    let log_path = diagnostic_path(LOG_FILE);
    let log_start_bytes = file_len(&log_path);
    let startup_overrides = startup_overrides(overrides);
    let mut environment = startup_overrides.clone();
    environment.insert("THALOS_SCREENSHOT".into(), preset.to_owned());
    environment.insert("THALOS_CAPTURE_SERVER".into(), "1".into());
    environment.insert(
        "BEVY_ASSET_ROOT".into(),
        workspace_root().display().to_string(),
    );
    let mut launcher_process = spawn_detached_launcher(&command, &environment, &log_path)?;
    let launcher = LauncherState {
        schema_version: CAPTURE_PROTOCOL_SCHEMA,
        pid: launcher_process.pid,
        preset: preset.to_owned(),
        launched_unix_ms: timestamp_millis(),
        log_start_bytes,
        startup_overrides,
        command: std::iter::once("cargo".to_owned()).chain(command).collect(),
    };
    write_json_atomic(&diagnostic_path(LAUNCHER_FILE), &launcher)?;
    println!(
        "starting capture renderer for {preset} (rebuilding changed sources; a cold build may take a while)"
    );
    let deadline = Instant::now() + Duration::from_secs(1800);
    while Instant::now() < deadline {
        if !launcher_process.is_running() {
            return Err(format!("capture launcher exited\n{}", log_tail(50)));
        }
        if let Some(state) = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE))
            && compatible_state(Some(&state), preset, requested_size(overrides))
            && state.ready
        {
            return Ok(state);
        }
        thread::sleep(Duration::from_millis(250));
    }
    Err(format!(
        "capture server did not become ready\n{}",
        log_tail(50)
    ))
}

/// Block until the running host has picked up any WGSL edit newer than its
/// launch (Bevy's `embedded_watcher` reloads both `assets/shaders/*` and
/// crate-embedded shaders in place). Rust staleness never reaches here — it
/// restarts the host in `ensure_server` instead.
fn wait_for_shader_reload(mut state: CaptureServerState) -> Result<CaptureServerState, String> {
    let launched = read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        .map_or(0, |launcher| launcher.launched_unix_ms);
    let shader_source = newest_mtime_ms(
        &[
            workspace_root().join("assets/shaders"),
            workspace_root().join("crates"),
        ],
        "wgsl",
    );
    let deadline = Instant::now() + Duration::from_secs(180);
    let mut announced = false;
    loop {
        state = read_json(&diagnostic_path(STATE_FILE)).unwrap_or(state);
        let need_shader = shader_source > launched && state.shader_reload_unix_ms < shader_source;
        if !need_shader {
            thread::sleep(Duration::from_millis(750));
            return Ok(state);
        }
        if !announced {
            println!("waiting for shader hot reload");
            announced = true;
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out waiting for shader hot reload\n{}",
                log_tail(50)
            ));
        }
        if !process_alive(state.pid) {
            return Err(format!(
                "capture server exited during reload\n{}",
                log_tail(50)
            ));
        }
        thread::sleep(Duration::from_millis(250));
    }
}

fn stop_server(quiet: bool) -> Result<(), String> {
    if let Some(state) = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE))
        && process_alive(state.pid)
    {
        let request = CaptureRequest {
            schema_version: CAPTURE_PROTOCOL_SCHEMA,
            id: Uuid::new_v4().simple().to_string(),
            action: CaptureAction::Shutdown,
            preset: String::new(),
            overrides: BTreeMap::new(),
        };
        write_json_atomic(&diagnostic_path(REQUEST_FILE), &request)?;
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline && process_alive(state.pid) {
            thread::sleep(Duration::from_millis(100));
        }
    }
    if let Some(launcher) = read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        && process_alive(launcher.pid)
    {
        terminate_process_tree(launcher.pid);
    }
    let _ = fs::remove_file(diagnostic_path(STATE_FILE));
    let _ = fs::remove_file(diagnostic_path(LAUNCHER_FILE));
    if !quiet {
        println!("visual capture server stopped");
    }
    Ok(())
}

fn status() {
    match read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE)) {
        Some(state) if process_alive(state.pid) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&state).unwrap_or_default()
            )
        }
        _ => println!("visual capture server is not running"),
    }
}

fn compatible_state(
    state: Option<&CaptureServerState>,
    preset: &str,
    expected_size: Option<(u32, u32)>,
) -> bool {
    state.is_some_and(|state| {
        state.schema_version == CAPTURE_PROTOCOL_SCHEMA
            && state
                .compatible_presets
                .iter()
                .any(|compatible| compatible == preset)
            && process_alive(state.pid)
            && expected_size.is_none_or(|size| size == (state.width, state.height))
            && timestamp_millis().saturating_sub(state.heartbeat_unix_ms) < 300_000
            && state.ready
    })
}

fn requested_size(overrides: &BTreeMap<String, String>) -> Option<(u32, u32)> {
    let raw = overrides.get("THALOS_SCREENSHOT_SIZE")?;
    let (width, height) = raw.split_once(['x', 'X', '*'])?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
}

fn startup_overrides(overrides: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    STARTUP_OVERRIDE_KEYS
        .iter()
        .filter_map(|key| {
            overrides
                .get(*key)
                .map(|value| ((*key).to_owned(), value.clone()))
        })
        .collect()
}

fn launcher_matches(overrides: &BTreeMap<String, String>) -> bool {
    read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        .is_some_and(|launcher| launcher.startup_overrides == startup_overrides(overrides))
}

fn canonical_preset(raw: &str) -> Result<String, String> {
    let slug = raw.trim().to_ascii_lowercase().replace('_', "-");
    let canonical = match slug.as_str() {
        "latest" | "perspective" => "latest-perspective",
        "spaceport" | "aerial" | "base" => "spaceport-aerial",
        "runway-sky" | "surface-atmosphere" => "runway-atmosphere",
        "space-center" | "spacecenter" | "play" => "hub",
        "dry" | "drybelt" | "desert" | "biome" => "dry-belt",
        "earth-ref" | "atmosphere" | "atmo" => "earth-reference",
        "open-ocean" | "sea" | "water" => "ocean",
        "sea-slopes" | "slope-field" => "ocean-slopes",
        "mira" => "mira-orbit",
        "regolith" => "mira-surface",
        "regolith-eva" => "mira-eva",
        "mira-full" | "mira-globe" => "mira-disc",
        "mira-oblique" | "mira-limb" => "mira-approach",
        "mira-crater" | "crater-rim" => "mira-rim",
        "clouds-runway" => "cloud-runway",
        "clouds-motion" => "cloud-motion",
        "clouds-cruise" | "cloud-deck" => "cloud-cruise",
        "inside-cloud" | "inside-clouds" => "cloud-interior",
        "cloud-orbit" | "clouds-orbit" => "cloud-limb",
        "cloud-globe" | "cloud-disc" | "full-planet" | "planet-disc" => "cloud-planet",
        "clouds-sunset" => "cloud-sunset",
        "engine" | "exhaust" | "rocket" => "plume",
        "massif" | "mountains" => "massif-aerial",
        "ridge" => "massif-ridge",
        "valley" => "massif-valley",
        value => value,
    };
    if CAPTURE_PRESETS.contains(&canonical) {
        Ok(canonical.to_owned())
    } else {
        Err(format!(
            "unknown capture scene {raw:?}; expected one of {}",
            CAPTURE_PRESETS.join(", ")
        ))
    }
}

fn capture_timeout_secs() -> u64 {
    env::var("THALOS_CAPTURE_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .unwrap_or(1800)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("capture tool lives under <workspace>/tools/capture")
        .to_path_buf()
}

fn diagnostics_dir() -> PathBuf {
    workspace_root().join("artifacts/diagnostics")
}

fn diagnostic_path(filename: &str) -> PathBuf {
    diagnostics_dir().join(filename)
}

fn absolute(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        path
    } else {
        workspace_root().join(path)
    }
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Option<T> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn validate_capture_output(path: &Path) -> Result<(), CaptureFailure> {
    let metadata = fs::metadata(path).map_err(|error| CaptureFailure {
        message: format!(
            "capture reported success but {} is unavailable: {error}",
            path.display()
        ),
        recoverable: true,
    })?;
    if metadata.len() == 0 {
        return Err(CaptureFailure {
            message: format!("capture wrote an empty file at {}", path.display()),
            recoverable: true,
        });
    }
    image::ImageReader::open(path)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(|error| CaptureFailure {
            message: format!("could not inspect {}: {error}", path.display()),
            recoverable: true,
        })?
        .decode()
        .map_err(|error| CaptureFailure {
            message: format!(
                "capture wrote an invalid image at {}: {error}",
                path.display()
            ),
            recoverable: true,
        })?;
    Ok(())
}

fn validate_render_log(offset: u64) -> Result<(), CaptureFailure> {
    let log = log_from(offset);
    let Some(message) = render_log_failure(&log) else {
        return Ok(());
    };
    Err(CaptureFailure {
        message,
        recoverable: false,
    })
}

pub(crate) fn render_log_failure(log: &str) -> Option<String> {
    let lower = log.to_ascii_lowercase();
    let fatal_markers = [
        "shaderprocessorerror",
        "shader validation error",
        "pipeline validation error",
        "error occurred when trying to process shader",
        "wgpu error: validation error",
        "device::create_shader_module",
        "gpu is lost",
        "device lost",
        "panicked at",
    ];
    let broad_render_error = lower.lines().find(|line| {
        line.contains("error")
            && (line.contains("shader")
                || line.contains("pipeline")
                || line.contains("wgpu_core")
                || line.contains("wgpu_hal"))
    });
    let marker = fatal_markers
        .iter()
        .copied()
        .find(|marker| lower.contains(*marker))
        .or_else(|| broad_render_error.map(|_| "render error log"));
    if let Some(marker) = marker {
        return Some(format!(
            "capture produced an image after a fatal render error ({marker}); refusing invalid evidence\n{}",
            log.lines()
                .rev()
                .take(50)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        ));
    }
    None
}

fn log_from(offset: u64) -> String {
    let Ok(mut file) = OpenOptions::new()
        .read(true)
        .open(diagnostic_path(LOG_FILE))
    else {
        return String::new();
    };
    if file.seek(SeekFrom::Start(offset)).is_err() {
        return String::new();
    }
    let mut text = String::new();
    let _ = file.read_to_string(&mut text);
    text
}

fn write_json_atomic(path: &Path, value: &impl Serialize) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let temporary = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("capture"),
        std::process::id()
    ));
    fs::write(
        &temporary,
        serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let _ = fs::remove_file(path);
    fs::rename(&temporary, path).map_err(|error| error.to_string())
}

fn newest_mtime_ms(roots: &[PathBuf], extension: &str) -> u128 {
    roots
        .iter()
        .flat_map(|root| recursive_files(root))
        .filter(|path| path.extension().is_some_and(|value| value == extension))
        .filter_map(|path| fs::metadata(path).ok()?.modified().ok())
        .filter_map(|time| {
            time.duration_since(UNIX_EPOCH)
                .ok()
                .map(|value| value.as_millis())
        })
        .max()
        .unwrap_or(0)
}

fn modified_millis(path: &Path) -> u128 {
    fs::metadata(path)
        .ok()
        .and_then(|metadata| metadata.modified().ok())
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_millis())
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map_or(0, |metadata| metadata.len())
}

fn recursive_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(root) else {
        return files;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(recursive_files(&path));
        } else {
            files.push(path);
        }
    }
    files
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_millis())
}

fn log_tail(lines: usize) -> String {
    fs::read_to_string(diagnostic_path(LOG_FILE))
        .map(|text| {
            text.lines()
                .rev()
                .take(lines)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_else(|_| "(capture-server log is unavailable)".to_owned())
}

fn process_alive(pid: u32) -> bool {
    #[cfg(windows)]
    {
        Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"])
            .output()
            .is_ok_and(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
    }
    #[cfg(not(windows))]
    {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .status()
            .is_ok_and(|status| status.success())
    }
}

/// A launched host process, however the platform detaches it.
struct LauncherProcess {
    pid: u32,
    /// Present only when we own a real child handle (Unix). On Windows the
    /// host is deliberately *not* our child, so liveness is queried by pid.
    child: Option<std::process::Child>,
}

impl LauncherProcess {
    fn is_running(&mut self) -> bool {
        match &mut self.child {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => process_alive(self.pid),
        }
    }
}

/// Start the persistent host so that it shares **no handle** with the shell
/// that invoked us.
///
/// The host outlives its client by design, so anything it inherits it holds
/// forever. On Windows that broke every piped or background-captured run: a
/// `CreateProcess(bInheritHandles = TRUE)` spawn duplicates *every* inheritable
/// handle into the child — including inheritable copies of the caller's stdout
/// pipe that never appear as our own std handles — so `just capture … | tail`
/// (and any agent harness that captures output) waited for an EOF that could
/// not arrive until the host died, minutes after the captures had succeeded.
/// Clearing `HANDLE_FLAG_INHERIT` on our std handles was measured to be
/// insufficient for exactly that reason.
///
/// The fix is to launch through `Start-Process`, i.e. `ShellExecuteEx`, which
/// creates the process with `bInheritHandles = FALSE` — a guaranteed-empty
/// inherited handle table, rather than an enumeration of leaks to plug. The
/// log redirection moves into a small `.cmd` shim so no handle needs to be
/// passed at all, and `cargo run` still fronts the host, preserving the
/// bevy_dylib search-path contract (INC-0008). See
/// INC-20260724T185500Z-persistent-host-inherits-caller-pipe.
#[cfg(windows)]
fn spawn_detached_launcher(
    command: &[String],
    environment: &BTreeMap<String, String>,
    log_path: &Path,
) -> Result<LauncherProcess, String> {
    let script_path = diagnostic_path("visual_capture_launch.cmd");
    let script = format!(
        "@echo off\r\ncd /d \"{root}\"\r\ncargo {args} 1>>\"{log}\" 2>&1\r\n",
        root = workspace_root().display(),
        args = command.join(" "),
        log = log_path.display(),
    );
    fs::write(&script_path, script)
        .map_err(|error| format!("write {}: {error}", script_path.display()))?;

    // `Start-Process` keeps `UseShellExecute = true` as long as no stream
    // redirection is requested — which is why the redirection lives in the
    // shim above. `-PassThru` hands back the shim's pid so `stop` can still
    // kill the whole tree.
    let quoted = script_path.display().to_string().replace('\'', "''");
    let ps = format!("$p = Start-Process -FilePath '{quoted}' -WindowStyle Hidden -PassThru; $p.Id");
    let output = Command::new("powershell")
        .args(["-NoProfile", "-NonInteractive", "-Command", &ps])
        .current_dir(workspace_root())
        .envs(environment)
        .output()
        .map_err(|error| format!("launch capture host: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "launch capture host: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let pid = String::from_utf8_lossy(&output.stdout)
        .trim()
        .parse::<u32>()
        .map_err(|error| format!("capture host launcher returned no pid: {error}"))?;
    Ok(LauncherProcess { pid, child: None })
}

/// Unix needs no such surgery: `std` sets `CLOEXEC` on every descriptor it does
/// not explicitly pass, so redirecting the host's stdio to the log is already
/// enough to keep the caller's pipe out of its hands.
#[cfg(unix)]
fn spawn_detached_launcher(
    command: &[String],
    environment: &BTreeMap<String, String>,
    log_path: &Path,
) -> Result<LauncherProcess, String> {
    use std::os::unix::process::CommandExt;

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|error| format!("open capture log: {error}"))?;
    let stderr = log.try_clone().map_err(|error| error.to_string())?;
    let child = Command::new("cargo")
        .args(command)
        .current_dir(workspace_root())
        .envs(environment)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::from(log))
        .stderr(std::process::Stdio::from(stderr))
        .process_group(0)
        .spawn()
        .map_err(|error| format!("launch cargo run (capture host): {error}"))?;
    Ok(LauncherProcess {
        pid: child.id(),
        child: Some(child),
    })
}

fn terminate_process_tree(pid: u32) {
    #[cfg(windows)]
    let _ = Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .output();
    #[cfg(not(windows))]
    let _ = Command::new("kill")
        .args(["-TERM", "--", &format!("-{pid}")])
        .output();
}

fn print_help() {
    println!(
        "thalos_capture shot [PRESET ...] [--out PATH] [--report PATH] [--set KEY=VALUE]\n\
         thalos_capture compare [PRESET] [AXIS] [--out DIR] [--cold]\n\
         thalos_capture status\n\
         thalos_capture stop\n\
         thalos_capture reset   (stop + drop incremental + clean the dylib set)\n\n\
         Multiple compatible presets reuse one booted world; incompatible presets\n\
         are restarted automatically. --out/--report are single-preset options."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shot_parser_accepts_multiple_scenes_and_aliases() {
        let parsed = parse_capture_options(
            ["spaceport", "runway-atmosphere", "--set", "THALOS_SSAO=off"]
                .into_iter()
                .map(str::to_owned),
        )
        .expect("parse");
        assert_eq!(
            parsed,
            ShotArgs {
                presets: vec!["spaceport-aerial".into(), "runway-atmosphere".into()],
                output: None,
                report: None,
                assignments: vec!["THALOS_SSAO=off".into()],
            }
        );
    }

    #[test]
    fn unknown_scene_is_rejected_instead_of_falling_back() {
        assert!(canonical_preset("typo-scene").is_err());
    }

    #[test]
    fn fatal_render_errors_are_promoted() {
        assert!(render_log_failure("wgpu error: Validation Error").is_some());
        assert!(render_log_failure("capture complete").is_none());
    }

    #[test]
    fn stale_artifact_link_failures_are_recognized() {
        // The observed INC-20260724T182642Z signature.
        assert!(toolchain_corruption(
            "lld-link: error: undefined symbol: anon.7d2b28def3d478c7dd7908d0ff8f73af.72.llvm.11501338756853688745"
        ));
        assert!(toolchain_corruption(
            "error: undefined symbol: <bevy_window::monitor::PrimaryMonitor as Component>\
             ::register_required_components (.llvm.17986031716569828249)"
        ));
    }

    #[test]
    fn ordinary_build_errors_are_not_treated_as_corruption() {
        // A real code error must reach the agent verbatim, not trigger a purge
        // and a silent rebuild.
        assert!(!toolchain_corruption(
            "error[E0425]: cannot find value `foo` in this scope"
        ));
        assert!(!toolchain_corruption(
            "lld-link: error: undefined symbol: my_missing_extern_fn"
        ));
    }
}
