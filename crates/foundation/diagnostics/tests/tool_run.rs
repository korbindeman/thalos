//! End-to-end shape of a developer-tool run record.
//!
//! Its own test binary, and one test in it, because the process sink is a
//! one-shot global: this is the only place a test may install it, and
//! installing it is the point — the capture client's records go through
//! exactly this path.

use std::time::Duration;

use serde_json::Value;

#[test]
fn tool_runs_record_phases_counters_outcome_and_abandonment() {
    let path = std::env::temp_dir().join(format!(
        "thalos-tool-run-{}-{}.jsonl",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    // SAFETY: single-threaded, before any other thread exists in this binary.
    unsafe {
        std::env::set_var(thalos_diagnostics::TOOL_LOG_ENV, &path);
    }
    assert!(
        thalos_diagnostics::install_tool_lane("capture"),
        "the tool lane must install in a fresh process"
    );

    let mut run = thalos_diagnostics::ToolRun::start("capture", "shot mira-orbit");
    run.field("preset", "mira-orbit");
    run.field("host_action", "restart_stale_source");
    run.phase("host_start", Duration::from_millis(90_000));
    run.phase("render", Duration::from_millis(400));
    // Repeated phases accumulate: three snapshots, one reported cost.
    run.phase("source_snapshot", Duration::from_millis(300));
    run.phase("source_snapshot", Duration::from_millis(200));
    run.count("retry");
    run.ok();

    // A run that is never finished still leaves a record — a killed or
    // panicking tool is exactly the case worth diagnosing later.
    drop(thalos_diagnostics::ToolRun::start(
        "capture",
        "shot dropped",
    ));

    let lines: Vec<Value> = std::fs::read_to_string(&path)
        .expect("read sink")
        .lines()
        .map(|line| serde_json::from_str(line).expect("valid JSON line"))
        .collect();
    std::fs::remove_file(&path).ok();
    let find = |event: &str, command: &str| -> Value {
        lines
            .iter()
            .find(|line| line["fields"]["event"] == event && line["fields"]["command"] == command)
            .unwrap_or_else(|| panic!("no {event} record for {command:?}"))
            .clone()
    };

    let start = find("tool_run_start", "shot mira-orbit");
    assert_eq!(start["fields"]["tool"], "capture");

    let record = find("tool_run", "shot mira-orbit");
    let fields = &record["fields"];
    assert_eq!(fields["outcome"], "ok");
    assert_eq!(fields["preset"], "mira-orbit");
    assert_eq!(fields["host_action"], "restart_stale_source");
    assert_eq!(fields["phase_host_start_ms"], 90_000);
    assert_eq!(fields["phase_render_ms"], 400);
    assert_eq!(
        fields["phase_source_snapshot_ms"], 500,
        "repeated phases report the total cost the caller paid"
    );
    assert_eq!(fields["retry_count"], 1);
    assert!(
        fields["total_ms"].is_number(),
        "every run reports its own wall time"
    );
    // Same envelope as the game's runtime stream, so one reader serves both.
    assert_eq!(record["schema"], thalos_diagnostics::SCHEMA);
    assert_eq!(record["session"], thalos_diagnostics::session_id());
    assert!(record["pid"].is_number());

    assert_eq!(
        find("tool_run", "shot dropped")["fields"]["outcome"],
        "abandoned"
    );
}
