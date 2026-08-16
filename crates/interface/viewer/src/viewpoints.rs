//! Shared saved-viewpoint store and application adapter seam.

use std::{
    fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use bevy::{math::DQuat, prelude::*};
use serde::Deserialize;
use thalos_render_model::{
    CAPTURE_PRESETS, CameraOptics as CameraOpticsSpec, ScriptedViewpoint, Viewpoint,
    ViewpointCatalog, ViewpointFrame, viewpoint_id_from_name,
};

const ID_STEM_MAX: usize = 58;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewpointStartupSet {
    Defaults,
    Load,
    Ui,
}

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ViewpointSet {
    Snapshot,
    Input,
    Apply,
    Ui,
}

/// Optional application-authored entries used when no catalog exists.
#[derive(Resource, Debug, Default, Clone)]
pub struct ViewpointFallbacks(pub Vec<Viewpoint>);

/// Migration information for the old projected-local yaw/pitch file.
#[derive(Debug, Clone)]
pub struct LegacyProjectedViewpoints {
    pub reference: String,
    pub optics: CameraOpticsSpec,
}

/// One shared catalog and write path.
#[derive(Resource, Debug)]
pub struct ViewpointStore {
    path: PathBuf,
    catalog: ViewpointCatalog,
    revision: u64,
    load_error: Option<String>,
    legacy_projected: Option<LegacyProjectedViewpoints>,
}

impl ViewpointStore {
    fn new(path: PathBuf, legacy_projected: Option<LegacyProjectedViewpoints>) -> Self {
        Self {
            path,
            catalog: ViewpointCatalog::default(),
            revision: 0,
            load_error: None,
            legacy_projected,
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn catalog(&self) -> &ViewpointCatalog {
        &self.catalog
    }

    pub fn entries(&self) -> &[Viewpoint] {
        &self.catalog.viewpoints
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn load_error(&self) -> Option<&str> {
        self.load_error.as_deref()
    }

    pub fn find(&self, id_or_name: &str) -> Option<&Viewpoint> {
        self.catalog.find(id_or_name).or_else(|| {
            self.catalog
                .viewpoints
                .iter()
                .find(|viewpoint| viewpoint.name.eq_ignore_ascii_case(id_or_name))
        })
    }

    pub fn resolve(&self, raw: &str) -> Option<&Viewpoint> {
        let requested = raw.trim().strip_prefix("viewpoint:").unwrap_or(raw.trim());
        if matches!(
            requested.to_ascii_lowercase().as_str(),
            "latest" | "perspective" | "latest-perspective" | "latest_perspective"
        ) {
            self.catalog.latest()
        } else {
            self.find(requested)
        }
    }

    pub fn reload(&mut self, fallbacks: &[Viewpoint]) -> Result<(), String> {
        match load_catalog(&self.path) {
            Ok(catalog) => {
                self.catalog = catalog;
                self.load_error = None;
            }
            Err(error) if error.kind == CatalogLoadErrorKind::NotFound => {
                self.catalog = catalog_from_fallbacks(fallbacks)?;
                self.load_error = None;
            }
            Err(canonical_error) => {
                let migrated = self
                    .legacy_projected
                    .as_ref()
                    .and_then(|migration| migrate_legacy_projected(&self.path, migration).ok());
                match migrated {
                    Some(catalog) => {
                        self.catalog = catalog;
                        self.load_error = None;
                    }
                    None => {
                        self.catalog = catalog_from_fallbacks(fallbacks)?;
                        self.load_error = Some(canonical_error.message.clone());
                        self.revision = self.revision.wrapping_add(1);
                        return Err(canonical_error.message);
                    }
                }
            }
        }
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }

    pub fn append_snapshot(
        &mut self,
        snapshot: &ViewpointSnapshot,
        requested_name: &str,
        description: &str,
    ) -> Result<String, String> {
        let name = requested_name.trim();
        if name.is_empty() {
            return Err("give the viewpoint a name".into());
        }
        let id = unique_id(&self.catalog, &viewpoint_id_from_name(name));
        let viewpoint = snapshot.to_viewpoint(id.clone(), name.into(), description.trim().into());
        viewpoint.validate()?;
        self.catalog.viewpoints.push(viewpoint);
        self.persist()?;
        Ok(id)
    }

    pub fn replace_from_snapshot(
        &mut self,
        selected_id: &str,
        snapshot: &ViewpointSnapshot,
        name: &str,
        description: &str,
    ) -> Result<String, String> {
        let replacement = snapshot.to_viewpoint(
            selected_id.to_owned(),
            name.trim().to_owned(),
            description.trim().to_owned(),
        );
        replacement.validate()?;
        if let Some(entry) = self
            .catalog
            .viewpoints
            .iter_mut()
            .find(|viewpoint| viewpoint.id == selected_id)
        {
            *entry = replacement;
        } else {
            let before = self.catalog.scripted_viewpoints.len();
            self.catalog
                .scripted_viewpoints
                .retain(|viewpoint| viewpoint.id != selected_id);
            if before == self.catalog.scripted_viewpoints.len() {
                return Err(format!("viewpoint {selected_id:?} no longer exists"));
            }
            self.catalog.viewpoints.push(replacement);
        }
        self.persist()?;
        Ok(selected_id.to_owned())
    }

    pub fn update_metadata(
        &mut self,
        selected_id: &str,
        name: &str,
        description: &str,
    ) -> Result<(), String> {
        let name = name.trim();
        if let Some(viewpoint) = self
            .catalog
            .viewpoints
            .iter_mut()
            .find(|viewpoint| viewpoint.id == selected_id)
        {
            viewpoint.name = name.to_owned();
            viewpoint.description = description.trim().to_owned();
            viewpoint.validate()?;
        } else if let Some(viewpoint) = self
            .catalog
            .scripted_viewpoints
            .iter_mut()
            .find(|viewpoint| viewpoint.id == selected_id)
        {
            viewpoint.name = name.to_owned();
            viewpoint.description = description.trim().to_owned();
            viewpoint.validate()?;
        } else {
            return Err(format!("viewpoint {selected_id:?} no longer exists"));
        }
        self.persist()
    }

    pub fn delete(&mut self, id: &str) -> Result<(), String> {
        let before = self.catalog.viewpoints.len() + self.catalog.scripted_viewpoints.len();
        self.catalog
            .viewpoints
            .retain(|viewpoint| viewpoint.id != id);
        self.catalog
            .scripted_viewpoints
            .retain(|viewpoint| viewpoint.id != id);
        if before == self.catalog.viewpoints.len() + self.catalog.scripted_viewpoints.len() {
            return Err(format!("viewpoint {id:?} no longer exists"));
        }
        self.persist()
    }

    fn persist(&mut self) -> Result<(), String> {
        write_catalog(&self.path, &self.catalog)?;
        self.catalog
            .viewpoints
            .sort_by(|left, right| left.id.cmp(&right.id));
        self.catalog
            .scripted_viewpoints
            .sort_by(|left, right| left.id.cmp(&right.id));
        self.revision = self.revision.wrapping_add(1);
        self.load_error = None;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CatalogLoadErrorKind {
    NotFound,
    Invalid,
}

#[derive(Debug)]
struct CatalogLoadError {
    kind: CatalogLoadErrorKind,
    message: String,
}

fn load_catalog(path: &Path) -> Result<ViewpointCatalog, CatalogLoadError> {
    let bytes = fs::read(path).map_err(|error| CatalogLoadError {
        kind: if error.kind() == ErrorKind::NotFound {
            CatalogLoadErrorKind::NotFound
        } else {
            CatalogLoadErrorKind::Invalid
        },
        message: format!("could not read {}: {error}", path.display()),
    })?;
    let catalog: ViewpointCatalog =
        serde_json::from_slice(&bytes).map_err(|error| CatalogLoadError {
            kind: CatalogLoadErrorKind::Invalid,
            message: format!("could not parse {}: {error}", path.display()),
        })?;
    catalog.validate().map_err(|error| CatalogLoadError {
        kind: CatalogLoadErrorKind::Invalid,
        message: format!("invalid viewpoint catalog {}: {error}", path.display()),
    })?;
    Ok(catalog)
}

/// Read and validate a canonical catalog without installing the Bevy plugin.
pub fn read_viewpoint_catalog(path: &Path) -> Result<ViewpointCatalog, String> {
    load_catalog(path).map_err(|error| error.message)
}

/// Atomically write a canonical catalog without installing the Bevy plugin.
pub fn write_viewpoint_catalog(path: &Path, catalog: &ViewpointCatalog) -> Result<(), String> {
    write_catalog(path, catalog)
}

fn write_catalog(path: &Path, catalog: &ViewpointCatalog) -> Result<(), String> {
    catalog.validate()?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create {}: {error}", parent.display()))?;
    }
    let mut canonical = catalog.clone();
    canonical
        .viewpoints
        .sort_by(|left, right| left.id.cmp(&right.id));
    canonical
        .scripted_viewpoints
        .sort_by(|left, right| left.id.cmp(&right.id));
    let mut json = serde_json::to_vec_pretty(&canonical)
        .map_err(|error| format!("could not encode viewpoint catalog: {error}"))?;
    json.push(b'\n');
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, json)
        .map_err(|error| format!("could not write {}: {error}", temporary.display()))?;
    if let Err(error) = fs::rename(&temporary, path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("could not replace {}: {error}", path.display()));
    }
    Ok(())
}

fn catalog_from_fallbacks(fallbacks: &[Viewpoint]) -> Result<ViewpointCatalog, String> {
    let catalog = ViewpointCatalog {
        viewpoints: fallbacks.to_vec(),
        ..Default::default()
    };
    catalog.validate()?;
    Ok(catalog)
}

#[derive(Deserialize)]
struct LegacyProjectedFile {
    version: u32,
    viewpoints: Vec<LegacyProjectedViewpoint>,
}

#[derive(Deserialize)]
struct LegacyProjectedViewpoint {
    name: String,
    position_m: [f64; 3],
    yaw_degrees: f64,
    pitch_degrees: f64,
}

fn migrate_legacy_projected(
    path: &Path,
    migration: &LegacyProjectedViewpoints,
) -> Result<ViewpointCatalog, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let legacy: LegacyProjectedFile =
        serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    if legacy.version != 1 {
        return Err(format!(
            "unsupported projected viewpoint version {}",
            legacy.version
        ));
    }
    let mut catalog = ViewpointCatalog::default();
    for (index, viewpoint) in legacy.viewpoints.into_iter().enumerate() {
        let rotation = DQuat::from_euler(
            EulerRot::YXZ,
            viewpoint.yaw_degrees.to_radians(),
            viewpoint.pitch_degrees.to_radians(),
            0.0,
        );
        let id = unique_id(&catalog, &viewpoint_id_from_name(&viewpoint.name));
        catalog.viewpoints.push(Viewpoint {
            id,
            name: viewpoint.name,
            description: String::new(),
            saved_unix_ms: index as u128,
            frame: ViewpointFrame::ProjectedLocal {
                reference: migration.reference.clone(),
            },
            camera_position_m: viewpoint.position_m,
            camera_rotation_xyzw: rotation.to_array(),
            optics: migration.optics,
        });
    }
    catalog.validate()?;
    Ok(catalog)
}

/// The application-projected camera state frozen by F9 or manager actions.
#[derive(Resource, Debug, Default, Clone)]
pub struct CurrentViewpoint(pub Option<ViewpointSnapshot>);

#[derive(Debug, Clone, PartialEq)]
pub struct ViewpointSnapshot {
    pub frame: ViewpointFrame,
    pub camera_position_m: [f64; 3],
    pub camera_rotation_xyzw: [f64; 4],
    pub optics: CameraOpticsSpec,
    pub suggested_name: String,
}

impl ViewpointSnapshot {
    fn to_viewpoint(&self, id: String, name: String, description: String) -> Viewpoint {
        Viewpoint {
            id,
            name,
            description,
            saved_unix_ms: timestamp_millis(),
            frame: self.frame.clone(),
            camera_position_m: self.camera_position_m,
            camera_rotation_xyzw: self.camera_rotation_xyzw,
            optics: self.optics,
        }
    }
}

#[derive(Debug, Clone)]
pub enum ViewpointApplyTarget {
    Saved(Viewpoint),
    Scripted(ScriptedViewpoint),
}

/// Single-slot apply request consumed by the application's spatial adapter.
#[derive(Resource, Debug, Default)]
pub struct PendingViewpointApply(pub Option<ViewpointApplyTarget>);

impl PendingViewpointApply {
    pub fn take(&mut self) -> Option<ViewpointApplyTarget> {
        self.0.take()
    }
}

/// Shared manager/quick-save visibility and application result channel.
#[derive(Resource, Debug, Default)]
pub struct ViewpointUiState {
    pub(crate) manager_open: bool,
    pub(crate) quick_open: bool,
    pub(crate) selected: Option<String>,
    pub(crate) status: Option<(bool, String)>,
}

impl ViewpointUiState {
    pub fn is_open(&self) -> bool {
        self.manager_open || self.quick_open
    }

    pub fn report(&mut self, result: Result<String, String>) {
        self.status = Some(match result {
            Ok(message) => (true, message),
            Err(error) => (false, error),
        });
    }
}

pub struct ViewpointPlugin {
    path: PathBuf,
    interactive: bool,
    legacy_projected: Option<LegacyProjectedViewpoints>,
}

impl ViewpointPlugin {
    pub fn new(path: PathBuf, interactive: bool) -> Self {
        Self {
            path,
            interactive,
            legacy_projected: None,
        }
    }

    pub fn with_legacy_projected(mut self, reference: impl Into<String>) -> Self {
        self.legacy_projected = Some(LegacyProjectedViewpoints {
            reference: reference.into(),
            optics: CameraOpticsSpec::default(),
        });
        self
    }
}

impl Plugin for ViewpointPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(ViewpointStore::new(
            self.path.clone(),
            self.legacy_projected.clone(),
        ))
        .init_resource::<ViewpointFallbacks>()
        .init_resource::<CurrentViewpoint>()
        .init_resource::<PendingViewpointApply>()
        .init_resource::<ViewpointUiState>()
        .configure_sets(
            Startup,
            (
                ViewpointStartupSet::Defaults,
                ViewpointStartupSet::Load,
                ViewpointStartupSet::Ui,
            )
                .chain(),
        )
        .configure_sets(
            Update,
            (
                ViewpointSet::Snapshot,
                ViewpointSet::Input,
                ViewpointSet::Apply,
                ViewpointSet::Ui,
            )
                .chain(),
        )
        .add_systems(Startup, load_store.in_set(ViewpointStartupSet::Load));

        if self.interactive {
            app.add_plugins(super::viewpoint_ui::ViewpointUiPlugin);
        }
    }
}

fn load_store(fallbacks: Res<ViewpointFallbacks>, mut store: ResMut<ViewpointStore>) {
    if let Err(error) = store.reload(&fallbacks.0) {
        error!("{error}");
    } else {
        info!(
            "loaded {} saved viewpoints and {} scripted viewpoints from {}",
            store.catalog.viewpoints.len(),
            store.catalog.scripted_viewpoints.len(),
            store.path.display()
        );
    }
}

pub fn unique_id(catalog: &ViewpointCatalog, stem: &str) -> String {
    let mut stem = if stem.is_empty() { "viewpoint" } else { stem };
    if stem.len() > ID_STEM_MAX {
        stem = stem[..ID_STEM_MAX].trim_end_matches('-');
    }
    if id_free(catalog, stem) {
        return stem.to_owned();
    }
    (2..1000)
        .map(|number| format!("{stem}-{number}"))
        .find(|candidate| id_free(catalog, candidate))
        .unwrap_or_else(|| stem.to_owned())
}

fn id_free(catalog: &ViewpointCatalog, id: &str) -> bool {
    !id.is_empty()
        && !catalog.contains(id)
        && !CAPTURE_PRESETS.contains(&id)
        && !matches!(id, "latest" | "perspective" | "latest-perspective")
}

fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temporary_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "thalos-viewer-{name}-{}-{}.json",
            std::process::id(),
            timestamp_millis()
        ))
    }

    #[test]
    fn legacy_projected_file_migrates_roll_free_rotation_and_ids() {
        let path = temporary_path("legacy");
        fs::write(
            &path,
            br#"{"version":1,"viewpoints":[{"name":"West Point","position_m":[1.0,2.0,3.0],"yaw_degrees":10.0,"pitch_degrees":-20.0}]}"#,
        )
        .unwrap();
        let migration = LegacyProjectedViewpoints {
            reference: "EPSG:32619".into(),
            optics: CameraOpticsSpec::default(),
        };

        let catalog = migrate_legacy_projected(&path, &migration).unwrap();

        assert_eq!(catalog.viewpoints[0].id, "west-point");
        assert!(matches!(
            &catalog.viewpoints[0].frame,
            ViewpointFrame::ProjectedLocal { reference } if reference == "EPSG:32619"
        ));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn store_writes_one_canonical_catalog_atomically() {
        let path = temporary_path("store");
        let snapshot = ViewpointSnapshot {
            frame: ViewpointFrame::ProjectedLocal {
                reference: "EPSG:32619".into(),
            },
            camera_position_m: [1.0, 2.0, 3.0],
            camera_rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            optics: CameraOpticsSpec::default(),
            suggested_name: "West Point".into(),
        };
        let mut store = ViewpointStore::new(path.clone(), None);
        store.append_snapshot(&snapshot, "West Point", "").unwrap();

        let loaded = read_viewpoint_catalog(&path).unwrap();
        assert_eq!(loaded.schema, thalos_render_model::VIEWPOINT_CATALOG_SCHEMA);
        assert_eq!(loaded.viewpoints[0].id, "west-point");
        assert!(!path.with_extension("json.tmp").exists());
        let _ = fs::remove_file(path);
    }
}
