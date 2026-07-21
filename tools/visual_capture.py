#!/usr/bin/env python3
"""Client/launcher for Thalos's persistent headless visual-iteration server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTICS = ROOT / "tools" / "diagnostics"
REQUEST = DIAGNOSTICS / "visual_capture_request.json"
RESPONSE = DIAGNOSTICS / "visual_capture_response.json"
STATE = DIAGNOSTICS / "visual_capture_server.json"
LAUNCHER = DIAGNOSTICS / "visual_capture_launcher.json"
LOG = DIAGNOSTICS / "visual_capture_server.log"
SCHEMA = 1

OVERRIDE_KEYS = (
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
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def log_tail(lines: int = 50) -> str:
    try:
        return "\n".join(LOG.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "(capture-server log is unavailable)"


def stop_server(quiet: bool = False) -> None:
    state = read_json(STATE)
    if state and process_alive(int(state.get("pid", 0))):
        request_id = uuid.uuid4().hex
        write_json_atomic(
            REQUEST,
            {
                "schema_version": SCHEMA,
                "id": request_id,
                "action": "shutdown",
            },
        )
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and process_alive(int(state["pid"])):
            time.sleep(0.1)

    launcher = read_json(LAUNCHER)
    launcher_pid = int((launcher or {}).get("pid", 0))
    if process_alive(launcher_pid):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(launcher_pid), "/T", "/F"],
                capture_output=True,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            try:
                os.killpg(launcher_pid, signal.SIGTERM)
            except OSError:
                pass
    for path in (STATE, LAUNCHER):
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    if not quiet:
        print("visual capture server stopped")


def expected_size() -> tuple[int, int] | None:
    raw = os.environ.get("THALOS_SCREENSHOT_SIZE", "")
    for separator in ("x", "X", "*"):
        if separator in raw:
            left, right = raw.split(separator, 1)
            try:
                return int(left.strip()), int(right.strip())
            except ValueError:
                return None
    return None


def compatible_state(state: dict | None, preset: str) -> bool:
    if not state or state.get("schema_version") != SCHEMA:
        return False
    if state.get("preset") != preset or not process_alive(int(state.get("pid", 0))):
        return False
    size = expected_size()
    return size is None or size == (state.get("width"), state.get("height"))


def start_server(preset: str) -> dict:
    dx = shutil.which("dx")
    if not dx:
        raise RuntimeError(
            "Dioxus CLI (`dx`) is not installed; run `cargo binstall dioxus-cli` once"
        )
    stop_server(quiet=True)
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    for path in (REQUEST, RESPONSE, STATE):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    environment = os.environ.copy()
    environment.update(
        {
            "THALOS_SCREENSHOT": preset,
            "THALOS_CAPTURE_SERVER": "1",
            "BEVY_ASSET_ROOT": str(ROOT),
        }
    )
    if os.name == "nt":
        # Subsecond's linker wrapper and the patched executable both resolve
        # Rust/Bevy DLLs through PATH on Windows. Keep this in the controller so
        # developers do not need a machine-specific .env.just incantation.
        sysroot = subprocess.run(
            ["rustc", "--print", "sysroot"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        hotpatch_deps = ROOT / "target" / "x86_64-pc-windows-msvc" / "desktop-dev" / "deps"
        environment["PATH"] = os.pathsep.join(
            (str(hotpatch_deps), str(Path(sysroot) / "bin"), environment.get("PATH", ""))
        )
    command = [
        dx,
        "serve",
        "--hot-patch",
        "--interactive",
        "false",
        "--open",
        "false",
        "--windows-subsystem",
        "CONSOLE",
        "--session-cache-dir",
        str(ROOT / "target" / "dx" / "visual_capture"),
        "--verbose",
        "--trace",
        "--package",
        "thalos_game",
        "--features",
        "dev-iteration",
    ]
    creationflags = 0
    popen_options: dict = {}
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        popen_options["start_new_session"] = True
    with LOG.open("a", encoding="utf-8") as output:
        output.write(f"\n=== launcher {time.strftime('%Y-%m-%d %H:%M:%S')} preset={preset} ===\n")
        output.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            **popen_options,
        )
    launched_ms = int(time.time() * 1000)
    write_json_atomic(
        LAUNCHER,
        {
            "schema_version": SCHEMA,
            "pid": process.pid,
            "preset": preset,
            "launched_unix_ms": launched_ms,
            "command": command,
        },
    )
    print(f"starting hot-patched renderer for {preset} (first build may take a while)")
    deadline = time.monotonic() + 1800.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"capture-server launcher exited with {process.returncode}\n{log_tail()}"
            )
        state = read_json(STATE)
        if compatible_state(state, preset) and state.get("ready"):
            return state
        time.sleep(0.25)
    raise RuntimeError(f"capture server did not become ready\n{log_tail()}")


def newest_mtime_ms(suffix: str) -> int:
    roots = (ROOT / "crates", ROOT / "assets" / "shaders")
    newest = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(f"*{suffix}"):
            try:
                newest = max(newest, int(path.stat().st_mtime * 1000))
            except OSError:
                pass
    return newest


def wait_for_reloads(state: dict) -> dict:
    launcher = read_json(LAUNCHER) or {}
    launched = int(launcher.get("launched_unix_ms", 0))
    rust_source = newest_mtime_ms(".rs")
    shader_source = newest_mtime_ms(".wgsl")
    need_code = rust_source > launched
    need_shader = shader_source > launched
    deadline = time.monotonic() + 180.0
    announced = False
    while need_code or need_shader:
        state = read_json(STATE) or state
        need_code = rust_source > launched and int(state.get("code_reload_unix_ms", 0)) < rust_source
        need_shader = shader_source > launched and int(state.get("shader_reload_unix_ms", 0)) < shader_source
        if not (need_code or need_shader):
            # Asset events precede render-pipeline recompilation by a few frames.
            time.sleep(0.75)
            return state
        if not announced:
            waiting = ", ".join(
                item for item, needed in (("Rust patch", need_code), ("shader reload", need_shader)) if needed
            )
            print(f"waiting for {waiting}")
            announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for hot reload\n{log_tail()}")
        if not process_alive(int(state.get("pid", 0))):
            raise RuntimeError(f"capture server exited during hot reload\n{log_tail()}")
        time.sleep(0.25)
    return state


def ensure_server(preset: str) -> dict:
    state = read_json(STATE)
    if not compatible_state(state, preset) or not state.get("ready"):
        state = start_server(preset)
    return wait_for_reloads(state)


def capture(preset: str, output: Path | None, report: Path | None, assignments: list[str]) -> Path:
    state = ensure_server(preset)
    overrides = {key: os.environ[key] for key in OVERRIDE_KEYS if key in os.environ}
    for assignment in assignments:
        if "=" not in assignment:
            raise RuntimeError(f"--set expects KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        if key not in OVERRIDE_KEYS:
            raise RuntimeError(f"unsupported capture override {key!r}")
        overrides[key] = value
    if output is not None:
        output = output if output.is_absolute() else ROOT / output
        overrides["THALOS_SCREENSHOT_OUT"] = str(output)
    if report is not None:
        report = report if report.is_absolute() else ROOT / report
        overrides["THALOS_SCREENSHOT_REPORT"] = str(report)

    request_id = uuid.uuid4().hex
    write_json_atomic(
        REQUEST,
        {
            "schema_version": SCHEMA,
            "id": request_id,
            "action": "capture",
            "preset": preset,
            "overrides": overrides,
        },
    )
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        response = read_json(RESPONSE)
        if response and response.get("id") == request_id:
            if not response.get("ok"):
                raise RuntimeError(str(response.get("message", "capture failed")))
            result = Path(response["output"])
            print(f"captured {result}")
            return result
        state = read_json(STATE) or state
        if not process_alive(int(state.get("pid", 0))):
            raise RuntimeError(f"capture server exited\n{log_tail()}")
        time.sleep(0.1)
    raise RuntimeError(f"capture timed out\n{log_tail()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    shot = subcommands.add_parser("capture")
    shot.add_argument("preset", nargs="?", default="spaceport-aerial")
    shot.add_argument("--out", type=Path)
    shot.add_argument("--report", type=Path)
    shot.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    subcommands.add_parser("status")
    subcommands.add_parser("stop")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "stop":
            stop_server()
        elif args.command == "status":
            state = read_json(STATE)
            if state and process_alive(int(state.get("pid", 0))):
                print(json.dumps(state, indent=2))
            else:
                print("visual capture server is not running")
        else:
            capture(args.preset, args.out, args.report, args.set)
        return 0
    except (OSError, RuntimeError) as error:
        print(f"visual capture failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
