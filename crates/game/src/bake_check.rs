//! Startup pre-flight: validate every procedural body's bake before
//! the Bevy app boots. If anything is missing or stale, shell out to
//! `thalos_bake_dump all` to auto-repair it, then re-validate.
//!
//! The game loads pre-baked terrain from `target/bakes/<name>.bin`. If
//! a bake is missing, stale, or corrupt the older design panicked from
//! an `AsyncComputeTaskPool` task — surfaced to the user as a confusing
//! Bevy/winit stack with the actionable line buried near the top. This
//! module runs a synchronous, pre-Bevy validation pass in `main` and
//! auto-bakes whatever's needed before launching Bevy, so the iterate
//! loop after editing `crates/terrain_gen/src/` is: re-run `just game`,
//! wait, play.
//!
//! The cache-key + path computation here is the single source of truth
//! for the game's view of a body's bake. `rendering::spawn` calls the
//! same helpers when it dispatches the async load, so pre-flight and
//! load can never disagree on what they're looking for.

use std::path::PathBuf;
use std::process::Command;

use bevy::math::Vec3;
use thalos_terrain::{TerrainCompileContext, TerrainCompileOptions, cache};
use thalos_world::{BodyDefinition, BodyKind};

/// Crater-count scale baked into the cache key. Local bakes always use
/// 1.0 (full crater authoring); any other value would mismatch the
/// stored hash on load.
const BAKE_CRATER_COUNT_SCALE: f32 = 1.0;

/// Directory holding local bake artifacts. Mirror of
/// `crates/bake_dump/src/main.rs::local_bake_dir` — both must resolve
/// to the same workspace-relative path so producer and consumer agree.
pub fn local_bake_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/bakes")
}

pub fn terrain_compile_options() -> TerrainCompileOptions {
    TerrainCompileOptions {
        crater_count_scale: BAKE_CRATER_COUNT_SCALE,
        cubemap_resolution_override: None,
    }
}

pub fn terrain_compile_context(body: &BodyDefinition) -> TerrainCompileContext {
    let body_radius_m = body.radius_m as f32;
    let gravity_m_s2 = (body.gm / (body.radius_m * body.radius_m)) as f32;
    let tidal_axis = matches!(body.kind, BodyKind::Moon).then_some(Vec3::Z);
    let axial_tilt_rad = body.axial_tilt_rad as f32;
    TerrainCompileContext {
        body_name: body.name.clone(),
        radius_m: body_radius_m,
        gravity_m_s2,
        rotation_hours: None,
        obliquity_deg: Some(axial_tilt_rad.to_degrees()),
        tidal_axis,
        axial_tilt_rad,
    }
}

pub fn expected_cache_key(body: &BodyDefinition) -> u64 {
    cache::terrain_cache_key(
        &body.terrain,
        body.tectonics.as_ref(),
        &terrain_compile_context(body),
        terrain_compile_options(),
    )
}

pub fn bake_path_for(body: &BodyDefinition) -> PathBuf {
    cache::cache_path(&local_bake_dir(), &body.name)
}

#[derive(Debug)]
pub enum BakeIssueKind {
    Missing,
    Stale { stored: u64, expected: u64 },
    Corrupt(String),
}

#[derive(Debug)]
pub struct BakeIssue {
    pub body: String,
    pub path: PathBuf,
    pub kind: BakeIssueKind,
}

/// Validate every body that requires a bake. Returns one entry per
/// affected body; an empty result means startup can proceed.
pub fn validate_bakes(bodies: &[BodyDefinition]) -> Vec<BakeIssue> {
    let mut issues = Vec::new();
    for body in bodies {
        if !body.terrain.is_some() {
            continue;
        }
        let expected = expected_cache_key(body);
        let path = bake_path_for(body);
        match cache::peek_key(&path) {
            Ok(stored) if stored == expected => {}
            Ok(stored) => issues.push(BakeIssue {
                body: body.name.clone(),
                path,
                kind: BakeIssueKind::Stale { stored, expected },
            }),
            Err(cache::LoadError::Missing { .. }) => issues.push(BakeIssue {
                body: body.name.clone(),
                path,
                kind: BakeIssueKind::Missing,
            }),
            Err(cache::LoadError::Decode { message, .. }) => issues.push(BakeIssue {
                body: body.name.clone(),
                path,
                kind: BakeIssueKind::Corrupt(message),
            }),
            // peek_key cannot return HashMismatch (it does not know the
            // expected key). Left explicit so an API change can't slip
            // through silently.
            Err(cache::LoadError::HashMismatch { .. }) => {
                unreachable!("peek_key does not return HashMismatch")
            }
        }
    }
    issues
}

/// Validate every body's bake; if any are missing or stale, shell out
/// to `thalos_bake_dump all` to rebake them in place, then re-validate.
/// Exits with a clean error block if auto-baking fails or leaves issues.
pub fn ensure_bakes_or_exit(bodies: &[BodyDefinition]) {
    let issues = validate_bakes(bodies);
    if issues.is_empty() {
        return;
    }

    eprintln!();
    eprintln!(
        "Auto-baking {} terrain bake(s) before startup:",
        issues.len()
    );
    for issue in &issues {
        let detail = match &issue.kind {
            BakeIssueKind::Missing => "missing",
            BakeIssueKind::Stale { .. } => "stale",
            BakeIssueKind::Corrupt(_) => "corrupt",
        };
        eprintln!("  {} — {}", issue.body, detail);
    }
    eprintln!();

    // Shell out to `cargo run -p thalos_bake_dump -- all`. Inheriting
    // stdio is what makes indicatif's progress bars animate: bake_dump
    // sees a TTY on stderr and draws normally. `--quiet` suppresses
    // cargo's own Compiling/Finished/Running banner so the user sees
    // only the bake UI.
    //
    // `all` (not the explicit broken-body list) is intentional: the
    // bake_dump binary checks each body's stored cache key against the
    // recomputed expected key and skips the no-ops itself, so a single
    // `all` invocation is both simpler and matches the `just bake all`
    // path users already know.
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let status = Command::new("cargo")
        .current_dir(&workspace_root)
        .args([
            "run",
            "--quiet",
            "--release",
            "-p",
            "thalos_bake_dump",
            "--",
            "all",
        ])
        .status()
        .unwrap_or_else(|e| {
            eprintln!("Failed to spawn `cargo run -p thalos_bake_dump`: {e}");
            eprintln!("Run `just bake all` manually and try again.");
            std::process::exit(1);
        });

    if !status.success() {
        eprintln!();
        eprintln!(
            "Auto-bake failed (exit {:?}). Run `just bake all` to investigate.",
            status.code(),
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    let remaining = validate_bakes(bodies);
    if !remaining.is_empty() {
        eprintln!();
        eprintln!("Auto-bake completed but bakes are still invalid:");
        report_and_exit(&remaining);
    }
}

/// Print a clean error block listing every issue and exit with status 1.
/// Used as the final fallback when auto-bake reports success but the
/// bakes still don't pass validation — a bug worth surfacing loudly.
pub fn report_and_exit(issues: &[BakeIssue]) -> ! {
    let names: Vec<&str> = issues.iter().map(|i| i.body.as_str()).collect();
    eprintln!();
    eprintln!(
        "Cannot start: {} terrain bake(s) missing or stale.",
        issues.len()
    );
    eprintln!();
    for issue in issues {
        let detail = match &issue.kind {
            BakeIssueKind::Missing => "missing".to_string(),
            BakeIssueKind::Stale { stored, expected } => {
                format!("stale (stored {stored:016x}, expected {expected:016x})")
            }
            BakeIssueKind::Corrupt(message) => format!("corrupt: {message}"),
        };
        eprintln!("  {} — {}", issue.body, detail);
        eprintln!("      {}", issue.path.display());
    }
    eprintln!();
    eprintln!("Fix: just bake all");
    eprintln!("     just bake <body>   (per body: {})", names.join(" "));
    eprintln!();
    std::process::exit(1);
}
