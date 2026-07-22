use std::{
    collections::BTreeMap,
    env, fs,
    fs::OpenOptions,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thalos_capture_protocol::{
    CAPTURE_PROTOCOL_SCHEMA, CaptureAction, CaptureRequest, CaptureResponse, CaptureServerState,
};
use uuid::Uuid;

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
    "THALOS_SCREENSHOT_OCEAN_TIME",
    "THALOS_SSAO",
    "THALOS_TERRAIN_INSPECTION",
    "THALOS_TERRAIN_CULL",
];

#[derive(Debug, Deserialize, Serialize)]
struct LauncherState {
    schema_version: u32,
    pid: u32,
    preset: String,
    launched_unix_ms: u128,
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
            let remaining = args.collect::<Vec<_>>();
            let (preset, option_start) = remaining
                .first()
                .filter(|value| !value.starts_with('-'))
                .map_or_else(
                    || ("spaceport-aerial".to_owned(), 0),
                    |value| (value.clone(), 1),
                );
            let (output, report, assignments) =
                parse_capture_options(remaining.into_iter().skip(option_start))?;
            capture(&preset, output, report, assignments)?;
        }
        "status" => status(),
        "stop" => stop_server(false)?,
        "-h" | "--help" | "help" => print_help(),
        other => return Err(format!("unknown command {other:?}; use --help")),
    }
    Ok(())
}

fn parse_capture_options(
    mut args: impl Iterator<Item = String>,
) -> Result<(Option<PathBuf>, Option<PathBuf>, Vec<String>), String> {
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
            other => return Err(format!("unknown shot option {other:?}")),
        }
    }
    Ok((output, report, assignments))
}

fn capture(
    preset: &str,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    assignments: Vec<String>,
) -> Result<PathBuf, String> {
    let state = ensure_server(preset)?;
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

    let request = CaptureRequest {
        schema_version: CAPTURE_PROTOCOL_SCHEMA,
        id: Uuid::new_v4().simple().to_string(),
        action: CaptureAction::Capture,
        preset: preset.to_owned(),
        overrides,
    };
    write_json_atomic(&diagnostic_path(REQUEST_FILE), &request)?;
    // Software-Vulkan (llvmpipe) fallback boxes render warmup at seconds per
    // frame, so the response wait must cover a full preset warmup there; a
    // hardware run answers long before either limit.
    let deadline = Instant::now() + Duration::from_secs(capture_timeout_secs());
    while Instant::now() < deadline {
        if let Some(response) = read_json::<CaptureResponse>(&diagnostic_path(RESPONSE_FILE))
            && response.id == request.id
        {
            if !response.ok {
                return Err(response.message);
            }
            let path = response
                .output
                .map(PathBuf::from)
                .ok_or("capture succeeded without an output path")?;
            println!("captured {}", path.display());
            return Ok(path);
        }
        if !process_alive(state.pid) {
            return Err(format!("capture server exited\n{}", log_tail(50)));
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err(format!("capture timed out\n{}", log_tail(50)))
}

fn ensure_server(preset: &str) -> Result<CaptureServerState, String> {
    let state = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE));
    let state = if compatible_state(state.as_ref(), preset) {
        state.expect("compatible state exists")
    } else {
        start_server(preset)?
    };
    wait_for_reloads(state)
}

fn start_server(preset: &str) -> Result<CaptureServerState, String> {
    Command::new("dx").arg("--version").output().map_err(|_| {
        "Dioxus CLI (`dx`) is not installed; run `cargo binstall dioxus-cli@0.7.9` once".to_owned()
    })?;
    stop_server(true)?;
    fs::create_dir_all(diagnostics_dir()).map_err(|error| error.to_string())?;
    for filename in [REQUEST_FILE, RESPONSE_FILE, STATE_FILE] {
        let _ = fs::remove_file(diagnostic_path(filename));
    }

    let command = vec![
        "serve",
        "--hot-patch",
        "--interactive",
        "false",
        "--open",
        "false",
        "--windows-subsystem",
        "CONSOLE",
        "--session-cache-dir",
        "target/dx/visual_capture",
        "--verbose",
        "--trace",
        "--package",
        "thalos_capture_host",
        "--features",
        "dev-iteration",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();

    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(diagnostic_path(LOG_FILE))
        .map_err(|error| format!("open capture log: {error}"))?;
    let stderr = log.try_clone().map_err(|error| error.to_string())?;
    let mut process = Command::new("dx");
    process
        .args(&command)
        .current_dir(workspace_root())
        .env("THALOS_SCREENSHOT", preset)
        .env("THALOS_CAPTURE_SERVER", "1")
        .env("BEVY_ASSET_ROOT", workspace_root())
        .stdin(Stdio::null())
        .stdout(Stdio::from(log))
        .stderr(Stdio::from(stderr));
    configure_background_process(&mut process)?;
    let mut child = process
        .spawn()
        .map_err(|error| format!("launch dx: {error}"))?;
    let launcher = LauncherState {
        schema_version: CAPTURE_PROTOCOL_SCHEMA,
        pid: child.id(),
        preset: preset.to_owned(),
        launched_unix_ms: timestamp_millis(),
        command: std::iter::once("dx".to_owned()).chain(command).collect(),
    };
    write_json_atomic(&diagnostic_path(LAUNCHER_FILE), &launcher)?;
    println!("starting hot-patched renderer for {preset} (first build may take a while)");
    let deadline = Instant::now() + Duration::from_secs(1800);
    while Instant::now() < deadline {
        if let Some(status) = child.try_wait().map_err(|error| error.to_string())? {
            return Err(format!(
                "capture launcher exited with {status}\n{}",
                log_tail(50)
            ));
        }
        if let Some(state) = read_json::<CaptureServerState>(&diagnostic_path(STATE_FILE))
            && compatible_state(Some(&state), preset)
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

fn wait_for_reloads(mut state: CaptureServerState) -> Result<CaptureServerState, String> {
    let launched = read_json::<LauncherState>(&diagnostic_path(LAUNCHER_FILE))
        .map_or(0, |launcher| launcher.launched_unix_ms);
    let rust_source = newest_mtime_ms(
        &[
            workspace_root().join("apps"),
            workspace_root().join("crates"),
        ],
        "rs",
    );
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
        let need_code = rust_source > launched && state.code_reload_unix_ms < rust_source;
        let need_shader = shader_source > launched && state.shader_reload_unix_ms < shader_source;
        if !need_code && !need_shader {
            thread::sleep(Duration::from_millis(750));
            return Ok(state);
        }
        if !announced {
            println!("waiting for renderer hot reload");
            announced = true;
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "timed out waiting for hot reload\n{}",
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

fn compatible_state(state: Option<&CaptureServerState>, preset: &str) -> bool {
    state.is_some_and(|state| {
        state.schema_version == CAPTURE_PROTOCOL_SCHEMA
            && state.preset == preset
            && process_alive(state.pid)
            && expected_size().is_none_or(|size| size == (state.width, state.height))
            && state.ready
    })
}

fn expected_size() -> Option<(u32, u32)> {
    let raw = env::var("THALOS_SCREENSHOT_SIZE").ok()?;
    let (width, height) = raw.split_once(['x', 'X', '*'])?;
    Some((width.trim().parse().ok()?, height.trim().parse().ok()?))
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

fn configure_background_process(command: &mut Command) -> Result<(), String> {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let sysroot = Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .map_err(|error| format!("query rustc sysroot: {error}"))?;
        let sysroot = String::from_utf8_lossy(&sysroot.stdout).trim().to_owned();
        let hotpatch_deps = workspace_root().join("target/x86_64-pc-windows-msvc/desktop-dev/deps");
        let mut paths = vec![hotpatch_deps, PathBuf::from(sysroot).join("bin")];
        if let Some(existing) = env::var_os("PATH") {
            paths.extend(env::split_paths(&existing));
        }
        let path = env::join_paths(paths).map_err(|error| error.to_string())?;
        command
            .env("PATH", path)
            .creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    Ok(())
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
        "thalos_capture shot [preset] [--out PATH] [--report PATH] [--set KEY=VALUE]\n\
         thalos_capture status\n\
         thalos_capture stop"
    );
}
