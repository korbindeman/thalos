//! Runtime content-root discovery for developer and packaged builds.
//!
//! A distributed game keeps `assets/` and `ships/` beside the executable.
//! Developer launchers run from a workspace and capture tools may provide an
//! explicit root. Every direct filesystem consumer and Bevy's `AssetPlugin`
//! must use the same answer or a package can boot with one class of content
//! while silently failing to find another.

use std::path::{Path, PathBuf};

use bevy::prelude::Resource;

const REQUIRED_CONTENT: &[&str] = &[
    "assets/solar_system.ron",
    "assets/parts.ron",
    "assets/input.ron",
    "ships/apollo.ron",
    "ships/meridian.ron",
    "ships/saturn.ron",
];

#[derive(Resource, Clone, Debug, PartialEq, Eq)]
pub struct ContentRoot {
    root: PathBuf,
}

impl ContentRoot {
    /// Resolve the one directory containing both `assets/` and `ships/`.
    ///
    /// Explicit overrides are authoritative and fail when invalid. Automatic
    /// discovery prefers the executable's directory (the portable package),
    /// then the process working directory, then this crate's compile-time
    /// workspace for developer binaries launched from elsewhere.
    pub fn discover() -> Result<Self, String> {
        for variable in ["THALOS_CONTENT_ROOT", "BEVY_ASSET_ROOT"] {
            if let Some(value) = std::env::var_os(variable) {
                let root = PathBuf::from(value);
                return Self::from_explicit(variable, root);
            }
        }

        let mut candidates = Vec::new();
        if let Ok(executable) = std::env::current_exe()
            && let Some(parent) = executable.parent()
        {
            candidates.push(parent.to_path_buf());
        }
        if let Ok(current) = std::env::current_dir() {
            candidates.push(current);
        }
        candidates.push(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .to_path_buf(),
        );

        if let Some(root) = first_valid_root(candidates.iter()) {
            return Ok(Self {
                root: canonical_or_original(root),
            });
        }

        let searched = candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        Err(format!(
            "could not locate Thalos runtime content; expected assets/ and ships/ in one of: {searched}. \
             Keep them beside the executable or set THALOS_CONTENT_ROOT."
        ))
    }

    fn from_explicit(variable: &str, root: PathBuf) -> Result<Self, String> {
        let missing = missing_content(&root);
        if !missing.is_empty() {
            return Err(format!(
                "{variable}={} is not a complete Thalos content root; missing: {}",
                root.display(),
                missing.join(", ")
            ));
        }
        Ok(Self {
            root: canonical_or_original(&root),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn assets(&self) -> PathBuf {
        self.root.join("assets")
    }

    pub fn ships(&self) -> PathBuf {
        self.root.join("ships")
    }

    pub fn resolve(&self, path: impl AsRef<Path>) -> PathBuf {
        self.root.join(path)
    }
}

fn first_valid_root<'a>(candidates: impl IntoIterator<Item = &'a PathBuf>) -> Option<&'a Path> {
    candidates
        .into_iter()
        .map(PathBuf::as_path)
        .find(|candidate| missing_content(candidate).is_empty())
}

fn missing_content(root: &Path) -> Vec<&'static str> {
    REQUIRED_CONTENT
        .iter()
        .copied()
        .filter(|relative| !root.join(relative).is_file())
        .collect()
}

fn canonical_or_original(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repository_is_a_complete_content_root() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..");
        assert!(missing_content(&root).is_empty());
    }

    #[test]
    fn resolved_paths_share_one_root() {
        let content = ContentRoot {
            root: PathBuf::from("package"),
        };
        assert_eq!(content.assets(), PathBuf::from("package/assets"));
        assert_eq!(content.ships(), PathBuf::from("package/ships"));
        assert_eq!(
            content.resolve("ships/meridian.ron"),
            PathBuf::from("package/ships/meridian.ron")
        );
    }
}
