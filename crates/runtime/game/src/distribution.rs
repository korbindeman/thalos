//! Build capability reporting and non-rendering package verification.

use thalos_input::settings::InputSettings;
use thalos_shipyard::{PartCatalog, ShipBlueprint};
use thalos_world::parsing::load_solar_system_from_dir;

use crate::content::ContentRoot;
use crate::terrain_registry::{BodySurfaceRegistry, thalos_diffusion_enabled};

pub const NEURAL_TERRAIN_AVAILABLE: bool = cfg!(feature = "neural-terrain");
pub const NEURAL_TERRAIN_DEFAULT: bool = cfg!(feature = "neural-terrain-default");

pub fn build_info() -> String {
    format!(
        "package=thalos_game\nversion={}\nneural_terrain_available={}\nneural_terrain_default={}\n",
        env!("CARGO_PKG_VERSION"),
        NEURAL_TERRAIN_AVAILABLE,
        NEURAL_TERRAIN_DEFAULT,
    )
}

/// Validate everything needed to start and change between the shipped game
/// scenarios without creating a window or GPU device.
pub fn verify_install() -> Result<String, String> {
    let content = ContentRoot::discover()?;
    let assets = content.assets();
    let system = load_solar_system_from_dir(&assets)?;

    InputSettings::load_from_path(assets.join("input.ron"))
        .map_err(|error| format!("input bindings: {error}"))?;
    PartCatalog::load_from_path(assets.join("parts.ron"))
        .map_err(|error| format!("part catalog: {error}"))?;

    let ship_paths = ["apollo.ron", "meridian.ron", "saturn.ron"];
    for filename in ship_paths {
        let path = content.ships().join(filename);
        let text = std::fs::read_to_string(&path)
            .map_err(|error| format!("ship blueprint {}: {error}", path.display()))?;
        ShipBlueprint::from_ron(&text)
            .map_err(|error| format!("ship blueprint {}: {error}", path.display()))?;
    }

    let surfaces = BodySurfaceRegistry::load(&system.bodies, &assets.join("terrain_packages"))?;
    let degraded = surfaces
        .degraded_bodies()
        .map(|body| format!("{} ({})", body.body_name, body.reason))
        .collect::<Vec<_>>();
    if !degraded.is_empty() {
        return Err(format!(
            "terrain packages failed verification: {}",
            degraded.join(", ")
        ));
    }

    #[cfg(feature = "neural-terrain")]
    {
        let neural_dir = assets.join("terrain_packages/thalos_diffusion");
        verify_neural_payloads(&neural_dir)?;
        if !thalos_diffusion_enabled() {
            let thalos = system
                .bodies
                .iter()
                .find(|body| body.name == "Thalos")
                .ok_or_else(|| "authored system has no Thalos body".to_string())?;
            thalos_terrain::DiffusionSurface::load(
                &neural_dir,
                thalos.radius_m as f32,
                thalos.id as u32,
            )
            .map_err(|error| format!("neural terrain payload: {error}"))?;
        }
    }

    crate::viewpoints::load_catalog().map_err(|error| format!("viewpoint catalog: {error}"))?;

    Ok(format!(
        "Thalos install verified\nroot={}\nversion={}\nbodies={}\nships={}\nneural_terrain_available={}\nneural_terrain_default={}\nneural_terrain_active={}\n",
        content.root().display(),
        env!("CARGO_PKG_VERSION"),
        system.bodies.len(),
        ship_paths.len(),
        NEURAL_TERRAIN_AVAILABLE,
        NEURAL_TERRAIN_DEFAULT,
        thalos_diffusion_enabled(),
    ))
}

#[cfg(feature = "neural-terrain")]
fn verify_neural_payloads(dir: &std::path::Path) -> Result<(), String> {
    let chart = dir.join("thalos_chart_elev.json");
    verify_raster_payload(&chart)?;

    let mut details = std::fs::read_dir(dir)
        .map_err(|error| format!("neural terrain directory {}: {error}", dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("thalos_site_detail_") && name.ends_with("_90m.json")
                })
        })
        .collect::<Vec<_>>();
    details.sort();
    for sidecar in details {
        verify_raster_payload(&sidecar)?;
    }
    Ok(())
}

#[cfg(feature = "neural-terrain")]
fn verify_raster_payload(sidecar: &std::path::Path) -> Result<(), String> {
    use std::io::Read;

    use sha2::{Digest, Sha256};

    let sidecar_text = std::fs::read_to_string(sidecar)
        .map_err(|error| format!("neural sidecar {}: {error}", sidecar.display()))?;
    let metadata: serde_json::Value = serde_json::from_str(&sidecar_text)
        .map_err(|error| format!("neural sidecar {}: {error}", sidecar.display()))?;
    let width = metadata["width"]
        .as_u64()
        .ok_or_else(|| format!("neural sidecar {} has no width", sidecar.display()))?;
    let height = metadata["height"]
        .as_u64()
        .ok_or_else(|| format!("neural sidecar {} has no height", sidecar.display()))?;
    let expected_hash = metadata["sha256_le_f32"]
        .as_str()
        .ok_or_else(|| format!("neural sidecar {} has no sha256_le_f32", sidecar.display()))?
        .to_ascii_lowercase();
    let payload = sidecar.with_extension("f32");
    let expected_bytes = width
        .checked_mul(height)
        .and_then(|samples| samples.checked_mul(4))
        .ok_or_else(|| format!("neural sidecar {} dimensions overflow", sidecar.display()))?;
    let actual_bytes = std::fs::metadata(&payload)
        .map_err(|error| format!("neural payload {}: {error}", payload.display()))?
        .len();
    if actual_bytes != expected_bytes {
        return Err(format!(
            "neural payload {} is incomplete: expected {expected_bytes} bytes, found {actual_bytes}",
            payload.display()
        ));
    }

    let mut file = std::fs::File::open(&payload)
        .map_err(|error| format!("neural payload {}: {error}", payload.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("neural payload {}: {error}", payload.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    let actual_hash = format!("{:x}", hasher.finalize());
    if actual_hash != expected_hash {
        return Err(format!(
            "neural payload {} hash mismatch: expected {expected_hash}, found {actual_hash}",
            payload.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_neural_feature_implies_capability() {
        assert!(!NEURAL_TERRAIN_DEFAULT || NEURAL_TERRAIN_AVAILABLE);
    }

    #[test]
    fn build_info_is_machine_readable() {
        let info = build_info();
        assert!(info.contains("package=thalos_game\n"));
        assert!(info.contains("neural_terrain_available="));
        assert!(info.contains("neural_terrain_default="));
    }
}
