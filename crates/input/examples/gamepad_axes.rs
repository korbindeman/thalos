//! HOTAS raw-axis discovery tool.
//!
//! Bevy's `bevy_gilrs` converter drops every axis gilrs labels `Unknown`
//! (e.g. a flight-stick twist / throttle slider), so they never reach Bevy's
//! gamepad state. Thalos reads them through a private `gilrs::Gilrs` instance
//! instead (see `crates/input/src/joystick.rs`), binding axes by raw platform
//! `Code` (`u32`). This probe runs the same kind of private instance alongside
//! a focused window (which Windows rawinput needs) and prints the raw code +
//! value range for every axis — use it to find the codes for your hardware,
//! then put them in the `game.hotas.axes` block of `assets/input.ron`.
//!
//! Run with `cargo run -p thalos_input --example gamepad_axes`, focus the
//! window, exercise every control for ~25s, then read the summary on exit
//! (also written to gamepad_axes.txt). Move one control at a time to label it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;

use bevy::app::AppExit;
use bevy::prelude::*;
use gilrs::{Axis, Gilrs};

const WINDOW_SECS: f32 = 25.0;

/// Private gilrs instance + accumulated raw-axis ranges. Stored as a non-send
/// resource because `gilrs::Gilrs` is `!Sync`.
struct Probe {
    gilrs: Gilrs,
    /// device label -> (raw code u32 -> (gilrs Axis label, min, max)).
    ranges: BTreeMap<String, BTreeMap<u32, (String, f32, f32)>>,
}

fn main() {
    let gilrs = Gilrs::new().expect("create private gilrs instance");
    App::new()
        .add_plugins(DefaultPlugins)
        .insert_non_send_resource(Probe {
            gilrs,
            ranges: BTreeMap::new(),
        })
        .add_systems(Update, (pump, finish))
        .run();
}

fn axis_label(axis: Axis) -> &'static str {
    match axis {
        Axis::LeftStickX => "LeftStickX",
        Axis::LeftStickY => "LeftStickY",
        Axis::LeftZ => "LeftZ",
        Axis::RightStickX => "RightStickX",
        Axis::RightStickY => "RightStickY",
        Axis::RightZ => "RightZ",
        Axis::DPadX => "DPadX",
        Axis::DPadY => "DPadY",
        Axis::Unknown => "Unknown",
    }
}

const KNOWN_AXES: [Axis; 8] = [
    Axis::LeftStickX,
    Axis::LeftStickY,
    Axis::LeftZ,
    Axis::RightStickX,
    Axis::RightStickY,
    Axis::RightZ,
    Axis::DPadX,
    Axis::DPadY,
];

fn pump(mut probe: NonSendMut<Probe>) {
    // Drain events so gilrs updates its internal state.
    while probe.gilrs.next_event().is_some() {}

    // (device, raw code, gilrs Axis label, value) snapshot this frame.
    let mut samples: Vec<(String, u32, String, f32)> = Vec::new();
    for (_id, gamepad) in probe.gilrs.gamepads() {
        let name = gamepad.name().to_string();
        // Reverse map: raw code -> named gilrs Axis, for any axis gilrs mapped.
        let mut named: BTreeMap<u32, String> = BTreeMap::new();
        for a in KNOWN_AXES {
            if let Some(c) = gamepad.axis_code(a) {
                named.insert(c.into_u32(), axis_label(a).to_string());
            }
        }
        for (code, data) in gamepad.state().axes() {
            let code = code.into_u32();
            let label = named
                .get(&code)
                .cloned()
                .unwrap_or_else(|| "Unknown".to_string());
            samples.push((name.clone(), code, label, data.value()));
        }
    }

    for (name, code, label, value) in samples {
        let dev = probe.ranges.entry(name).or_default();
        let entry = dev.entry(code).or_insert((label, value, value));
        entry.1 = entry.1.min(value);
        entry.2 = entry.2.max(value);
    }
}

fn finish(time: Res<Time>, probe: NonSend<Probe>, mut exit: MessageWriter<AppExit>) {
    if time.elapsed_secs() < WINDOW_SECS {
        return;
    }
    let mut out = String::new();
    for (dev, axes) in &probe.ranges {
        let _ = writeln!(out, "device: {dev}");
        for (code, (label, lo, hi)) in axes {
            let span = hi - lo;
            let moved = if span > 0.2 { "  <-- MOVED" } else { "" };
            let _ = writeln!(
                out,
                "  code {code:<6} [{label:<11}] {lo:+.2} .. {hi:+.2}  (span {span:.2}){moved}"
            );
        }
    }
    if probe.ranges.is_empty() {
        let _ = writeln!(out, "no devices seen by private gilrs instance");
    }
    print!("{out}");
    let _ = std::io::stdout().flush();
    let _ = std::fs::write("gamepad_axes.txt", &out);
    exit.write(AppExit::Success);
}
