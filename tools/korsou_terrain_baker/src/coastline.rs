use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use super::{UTM_ZONE, utm_forward, utm_to_local};

pub const DISTANCE_DECIMETRES_PER_METRE: f32 = 10.0;
pub const DISTANCE_CLAMP_M: f64 = 120.0;

pub struct Coastline {
    pub source_path: PathBuf,
    pub source_timestamp: String,
    pub way_count: usize,
    pub node_count: usize,
    pub segment_count: usize,
    rings: Vec<Vec<[f64; 2]>>,
}

impl Coastline {
    pub fn read(path: &Path, local_origin_utm_m: [f64; 2]) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        let value: serde_json::Value =
            serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
        if value.get("format").and_then(|value| value.as_str()) == Some("korsou.coastline-rings.v1")
        {
            Self::from_rings(path, &value, local_origin_utm_m)
        } else {
            Self::from_overpass(path, &bytes, local_origin_utm_m)
        }
    }

    fn from_rings(
        path: &Path,
        value: &serde_json::Value,
        local_origin_utm_m: [f64; 2],
    ) -> Result<Self> {
        let document: RingsDocument = serde_json::from_value(value.clone())
            .with_context(|| format!("parse coastline rings {}", path.display()))?;
        ensure!(
            !document.rings.is_empty(),
            "coastline rings source is empty"
        );
        let mut rings = Vec::with_capacity(document.rings.len());
        for (index, source) in document.rings.iter().enumerate() {
            if source.len() < 4 {
                eprintln!(
                    "skipping coastline ring {index}: {} vertices is not a closed polygon",
                    source.len()
                );
                continue;
            }
            let mut ring = Vec::with_capacity(source.len());
            for &[lon, lat] in source {
                let (easting, northing) = utm_forward(lat, lon, UTM_ZONE);
                ring.push(utm_to_local(easting, northing, local_origin_utm_m));
            }
            ensure!(
                ring.first() == ring.last(),
                "coastline ring {index} is open"
            );
            rings.push(ring);
        }
        ensure!(
            !rings.is_empty(),
            "coastline rings source has no closed polygons"
        );
        let node_count = rings.iter().map(|ring| ring.len().saturating_sub(1)).sum();
        let segment_count = rings.iter().map(|ring| ring.len() - 1).sum();
        Ok(Self {
            source_path: path.to_owned(),
            source_timestamp: document.osm_timestamp,
            way_count: rings.len(),
            node_count,
            segment_count,
            rings,
        })
    }

    fn from_overpass(path: &Path, bytes: &[u8], local_origin_utm_m: [f64; 2]) -> Result<Self> {
        let document: OverpassDocument =
            serde_json::from_slice(bytes).with_context(|| format!("parse {}", path.display()))?;

        let mut nodes = HashMap::new();
        let mut ways = Vec::new();
        for element in document.elements {
            match element {
                OverpassElement::Node { id, lat, lon } => {
                    nodes.insert(id, [lon, lat]);
                }
                OverpassElement::Way { id, nodes, tags } => {
                    if tags
                        .get("natural")
                        .is_some_and(|value| value == "coastline")
                    {
                        ensure!(
                            nodes.len() >= 2,
                            "OSM coastline way {id} has fewer than 2 nodes"
                        );
                        ways.push(nodes);
                    }
                }
            }
        }
        ensure!(!nodes.is_empty(), "OSM coastline source has no nodes");
        ensure!(
            !ways.is_empty(),
            "OSM coastline source has no coastline ways"
        );

        let way_count = ways.len();
        let node_count = nodes.len();
        let node_rings = assemble_rings(ways)?;
        let mut rings = Vec::with_capacity(node_rings.len());
        for node_ring in node_rings {
            let mut ring = Vec::with_capacity(node_ring.len());
            for node_id in node_ring {
                let [lon, lat] = *nodes
                    .get(&node_id)
                    .with_context(|| format!("OSM coastline references missing node {node_id}"))?;
                let (easting, northing) = utm_forward(lat, lon, UTM_ZONE);
                ring.push(utm_to_local(easting, northing, local_origin_utm_m));
            }
            ensure!(ring.len() >= 4 && ring.first() == ring.last());
            rings.push(ring);
        }
        let segment_count = rings.iter().map(|ring| ring.len() - 1).sum();

        Ok(Self {
            source_path: path.to_owned(),
            source_timestamp: document.osm3s.timestamp_osm_base,
            way_count,
            node_count,
            segment_count,
            rings,
        })
    }

    pub fn write_polylines(&self, path: &Path) -> Result<()> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"KSH1");
        bytes.extend_from_slice(&(self.rings.len() as u32).to_le_bytes());
        for ring in &self.rings {
            bytes.extend_from_slice(&(ring.len() as u32).to_le_bytes());
            for &[x, z] in ring {
                bytes.extend_from_slice(&x.to_le_bytes());
                bytes.extend_from_slice(&z.to_le_bytes());
            }
        }
        fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
    }

    pub fn rasterize_land(
        &self,
        bounds: [f64; 4],
        width: usize,
        height: usize,
        spacing_m: f64,
    ) -> Vec<u8> {
        let mut mask = vec![0; width * height];
        let mut crossings = Vec::new();
        for z in 0..height {
            let world_z = bounds[1] + z as f64 * spacing_m;
            crossings.clear();
            for ring in &self.rings {
                for edge in ring.windows(2) {
                    let [a, b] = [edge[0], edge[1]];
                    if (a[1] > world_z) != (b[1] > world_z) {
                        let t = (world_z - a[1]) / (b[1] - a[1]);
                        crossings.push(a[0] + (b[0] - a[0]) * t);
                    }
                }
            }
            crossings.sort_by(f64::total_cmp);
            for pair in crossings.chunks_exact(2) {
                let first = ((pair[0] - bounds[0]) / spacing_m).ceil() as isize;
                let last = ((pair[1] - bounds[0]) / spacing_m).floor() as isize;
                let first = first.clamp(0, width as isize) as usize;
                let last = last.clamp(-1, width as isize - 1);
                if last >= first as isize {
                    mask[z * width + first..=z * width + last as usize].fill(255);
                }
            }
        }
        mask
    }

    pub fn signed_distance_field(
        &self,
        mask: &[u8],
        bounds: [f64; 4],
        width: usize,
        height: usize,
        spacing_m: f64,
    ) -> Vec<i16> {
        assert_eq!(mask.len(), width * height);
        let mut distances = vec![DISTANCE_CLAMP_M as f32; mask.len()];
        for ring in &self.rings {
            for edge in ring.windows(2) {
                let [a, b] = [edge[0], edge[1]];
                let min_x = a[0].min(b[0]) - DISTANCE_CLAMP_M;
                let max_x = a[0].max(b[0]) + DISTANCE_CLAMP_M;
                let min_z = a[1].min(b[1]) - DISTANCE_CLAMP_M;
                let max_z = a[1].max(b[1]) + DISTANCE_CLAMP_M;
                let x0 = (((min_x - bounds[0]) / spacing_m).floor() as isize)
                    .clamp(0, width as isize - 1) as usize;
                let x1 = (((max_x - bounds[0]) / spacing_m).ceil() as isize)
                    .clamp(0, width as isize - 1) as usize;
                let z0 = (((min_z - bounds[1]) / spacing_m).floor() as isize)
                    .clamp(0, height as isize - 1) as usize;
                let z1 = (((max_z - bounds[1]) / spacing_m).ceil() as isize)
                    .clamp(0, height as isize - 1) as usize;
                for z in z0..=z1 {
                    let world_z = bounds[1] + z as f64 * spacing_m;
                    for x in x0..=x1 {
                        let world_x = bounds[0] + x as f64 * spacing_m;
                        let distance = point_segment_distance([world_x, world_z], a, b) as f32;
                        let index = z * width + x;
                        distances[index] = distances[index].min(distance);
                    }
                }
            }
        }

        distances
            .into_iter()
            .zip(mask)
            .map(|(distance, land)| {
                let magnitude = (distance.min(DISTANCE_CLAMP_M as f32)
                    * DISTANCE_DECIMETRES_PER_METRE)
                    .round()
                    .max(1.0) as i16;
                if *land == 0 { -magnitude } else { magnitude }
            })
            .collect()
    }
}

fn assemble_rings(ways: Vec<Vec<i64>>) -> Result<Vec<Vec<i64>>> {
    let mut unused: Vec<Option<Vec<i64>>> = ways.into_iter().map(Some).collect();
    let mut rings = Vec::new();
    while let Some(index) = unused.iter().position(Option::is_some) {
        let mut ring = unused[index].take().unwrap();
        while ring.first() != ring.last() {
            let end = *ring.last().unwrap();
            let next_index = unused.iter().position(|way| {
                way.as_ref()
                    .is_some_and(|nodes| nodes.first() == Some(&end) || nodes.last() == Some(&end))
            });
            let next_index = next_index.with_context(|| {
                format!("OSM coastline is not a closed ring; no way continues node {end}")
            })?;
            let mut next = unused[next_index].take().unwrap();
            if next.last() == Some(&end) {
                next.reverse();
            }
            ring.extend(next.into_iter().skip(1));
        }
        rings.push(ring);
    }
    Ok(rings)
}

fn point_segment_distance(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length_squared = ab[0] * ab[0] + ab[1] * ab[1];
    if length_squared == 0.0 {
        return (point[0] - a[0]).hypot(point[1] - a[1]);
    }
    let t =
        (((point[0] - a[0]) * ab[0] + (point[1] - a[1]) * ab[1]) / length_squared).clamp(0.0, 1.0);
    (point[0] - (a[0] + ab[0] * t)).hypot(point[1] - (a[1] + ab[1] * t))
}

#[derive(Deserialize)]
struct RingsDocument {
    osm_timestamp: String,
    rings: Vec<Vec<[f64; 2]>>,
}

#[derive(Deserialize)]
struct OverpassDocument {
    osm3s: OverpassMetadata,
    elements: Vec<OverpassElement>,
}

#[derive(Deserialize)]
struct OverpassMetadata {
    timestamp_osm_base: String,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum OverpassElement {
    Node {
        id: i64,
        lat: f64,
        lon: f64,
    },
    Way {
        id: i64,
        nodes: Vec<i64>,
        #[serde(default)]
        tags: HashMap<String, String>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_reversed_ways_into_a_closed_ring() {
        let rings = assemble_rings(vec![vec![1, 2], vec![3, 2], vec![3, 1]]).unwrap();
        assert_eq!(rings, vec![vec![1, 2, 3, 1]]);
    }

    #[test]
    fn skips_degenerate_lonlat_rings() {
        let json = serde_json::json!({
            "format": "korsou.coastline-rings.v1",
            "osm_timestamp": "2026-08-07T20:31:21Z",
            "rings": [
                [[-69.0, 12.1], [-68.99, 12.1], [-68.99, 12.11], [-69.0, 12.11], [-69.0, 12.1]],
                [[-69.1, 12.2], [-69.1, 12.2]]
            ]
        });
        let path = std::env::temp_dir().join("korsou-coast-rings-degenerate.json");
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let coastline = Coastline::read(&path, [500_000.0, 1_350_000.0]).unwrap();
        assert_eq!(coastline.way_count, 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_closed_lonlat_rings() {
        let json = serde_json::json!({
            "format": "korsou.coastline-rings.v1",
            "osm_timestamp": "2026-08-07T20:31:21Z",
            "rings": [[[-69.0, 12.1], [-68.99, 12.1], [-68.99, 12.11], [-69.0, 12.11], [-69.0, 12.1]]]
        });
        let path = std::env::temp_dir().join("korsou-coast-rings-test.json");
        fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
        let coastline = Coastline::read(&path, [500_000.0, 1_350_000.0]).unwrap();
        assert_eq!(coastline.way_count, 1);
        assert_eq!(coastline.rings[0].first(), coastline.rings[0].last());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn signed_distance_has_land_and_water_signs() {
        let coastline = Coastline {
            source_path: PathBuf::new(),
            source_timestamp: String::new(),
            way_count: 1,
            node_count: 4,
            segment_count: 4,
            rings: vec![vec![
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
                [0.0, 0.0],
            ]],
        };
        let bounds = [-5.0, -5.0, 15.0, 15.0];
        let mask = coastline.rasterize_land(bounds, 5, 5, 5.0);
        let distance = coastline.signed_distance_field(&mask, bounds, 5, 5, 5.0);
        assert!(distance[2 * 5 + 2] > 0);
        assert!(distance[0] < 0);
    }
}
