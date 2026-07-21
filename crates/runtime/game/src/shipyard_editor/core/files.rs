//! Ship blueprint file I/O helpers shared by every editor front-end:
//! name slugging, `ships/` paths, and the saved-ship listing.

use serde::Deserialize;
use std::path::PathBuf;

pub const SHIPS_DIR: &str = "ships";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedShip {
    pub slug: String,
    pub name: String,
}

#[derive(Deserialize)]
struct ShipFileHeader {
    name: String,
}

pub fn schema_ship_name(name: &str) -> String {
    let name = name.trim();
    if name.is_empty() {
        "Unnamed".into()
    } else {
        name.into()
    }
}

pub fn slugify_ship_name(name: &str) -> String {
    let mut slug = String::new();
    let mut pending_separator = false;

    for c in name.trim().chars() {
        if c.is_ascii_alphanumeric() {
            if pending_separator && !slug.is_empty() {
                slug.push('-');
            }
            slug.push(c.to_ascii_lowercase());
            pending_separator = false;
        } else {
            pending_separator = !slug.is_empty();
        }
    }

    if slug.is_empty() {
        "unnamed".into()
    } else {
        slug
    }
}

pub fn ship_path_for_name(name: &str) -> PathBuf {
    ship_path_for_slug(&slugify_ship_name(name))
}

pub fn ship_path_for_slug(slug: &str) -> PathBuf {
    PathBuf::from(SHIPS_DIR).join(format!("{slug}.ron"))
}

pub fn ship_name_from_ron(text: &str) -> Option<String> {
    ron::from_str::<ShipFileHeader>(text)
        .ok()
        .map(|header| schema_ship_name(&header.name))
}

pub fn list_ships() -> Vec<SavedShip> {
    let dir = PathBuf::from(SHIPS_DIR);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<SavedShip> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) != Some("ron") {
                return None;
            }
            let slug = p
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())?;
            let name = std::fs::read_to_string(&p)
                .ok()
                .and_then(|text| ship_name_from_ron(&text))
                .unwrap_or_else(|| slug.clone());
            Some(SavedShip { slug, name })
        })
        .collect();
    out.sort_by_key(|ship| (ship.name.to_ascii_lowercase(), ship.slug.clone()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugifies_ship_names_for_ron_filenames() {
        assert_eq!(
            slugify_ship_name("  Lunar Transfer Mk II  "),
            "lunar-transfer-mk-ii"
        );
        assert_eq!(slugify_ship_name("A__B / C"), "a-b-c");
        assert_eq!(slugify_ship_name("***"), "unnamed");
    }

    #[test]
    fn reads_ship_name_from_schema_header() {
        let text = r#"(
            name: "Lunar Transfer Vehicle",
            root: 0,
            parts: [],
            connections: [],
        )"#;

        assert_eq!(
            ship_name_from_ron(text).as_deref(),
            Some("Lunar Transfer Vehicle")
        );
    }

    #[test]
    fn schema_ship_names_are_trimmed_and_non_empty() {
        assert_eq!(schema_ship_name("  Apollo  "), "Apollo");
        assert_eq!(schema_ship_name("   "), "Unnamed");
    }
}
