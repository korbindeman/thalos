use std::{collections::HashMap, fs, path::Path};

use anyhow::{Context, Result, ensure};

const POLYLINE_MAGIC: &[u8; 4] = b"KSH1";
const CELL_M: f64 = 32.0;
const SEARCH_CELLS: i32 = 8;

pub struct CoastPolylines {
    rings: Vec<Vec<[f64; 2]>>,
    cells: HashMap<(i32, i32), Vec<(u16, u32)>>,
}

#[derive(Clone, Copy)]
struct CoastHit {
    ring: usize,
    segment: usize,
    t: f64,
    point: [f64; 2],
}

impl CoastPolylines {
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
        ensure!(
            bytes.len() >= 8,
            "{} is too short for coastline polylines",
            path.display()
        );
        ensure!(
            bytes[..4] == POLYLINE_MAGIC[..],
            "{} is not a Kòrsou coastline polyline file",
            path.display()
        );
        let ring_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let mut offset = 8;
        let mut rings = Vec::with_capacity(ring_count);
        for index in 0..ring_count {
            ensure!(
                offset + 4 <= bytes.len(),
                "polyline ring {index} is truncated"
            );
            let count = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4;
            let byte_len = count
                .checked_mul(16)
                .context("polyline ring is too large")?;
            ensure!(
                offset + byte_len <= bytes.len(),
                "polyline ring {index} is truncated"
            );
            let mut ring = Vec::with_capacity(count);
            for _ in 0..count {
                let x = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
                let z = f64::from_le_bytes(bytes[offset + 8..offset + 16].try_into().unwrap());
                ring.push([x, z]);
                offset += 16;
            }
            ensure!(
                ring.len() >= 4 && ring.first() == ring.last(),
                "polyline ring {index} is open"
            );
            rings.push(ring);
        }
        ensure!(offset == bytes.len(), "polyline file has trailing bytes");
        Ok(Self::from_rings(rings))
    }

    fn from_rings(rings: Vec<Vec<[f64; 2]>>) -> Self {
        let mut cells: HashMap<(i32, i32), Vec<(u16, u32)>> = HashMap::new();
        for (ring_index, ring) in rings.iter().enumerate() {
            for (segment, edge) in ring.windows(2).enumerate() {
                let [a, b] = [edge[0], edge[1]];
                let min_x = cell(a[0].min(b[0]));
                let max_x = cell(a[0].max(b[0]));
                let min_z = cell(a[1].min(b[1]));
                let max_z = cell(a[1].max(b[1]));
                for z in min_z..=max_z {
                    for x in min_x..=max_x {
                        cells
                            .entry((x, z))
                            .or_default()
                            .push((ring_index as u16, segment as u32));
                    }
                }
            }
        }
        Self { rings, cells }
    }

    pub fn vertex_count(&self) -> usize {
        self.rings
            .iter()
            .map(|ring| ring.len().saturating_sub(1))
            .sum()
    }

    pub fn nearest_point(&self, x: f64, z: f64) -> [f64; 2] {
        self.nearest_hit([x, z]).point
    }

    pub fn distance_m(&self, x: f64, z: f64) -> f64 {
        let point = self.nearest_point(x, z);
        (point[0] - x).hypot(point[1] - z)
    }

    pub fn path(&self, start: [f64; 2], end: [f64; 2], max_edge_m: f64) -> Vec<[f64; 2]> {
        let start_hit = self.nearest_hit(start);
        let end_hit = self.nearest_hit(end);
        let raw = if start_hit.ring == end_hit.ring {
            let forward = self.walk(start_hit, end_hit, 1);
            let backward = self.walk(start_hit, end_hit, -1);
            if path_length(&forward) <= path_length(&backward) {
                forward
            } else {
                backward
            }
        } else {
            vec![start_hit.point, end_hit.point]
        };
        densify_path(&raw, max_edge_m)
    }

    fn nearest_hit(&self, point: [f64; 2]) -> CoastHit {
        let origin = (cell(point[0]), cell(point[1]));
        let mut best: Option<CoastHit> = None;
        let mut best_distance = f64::INFINITY;
        for radius in 0..=SEARCH_CELLS {
            for dz in -radius..=radius {
                for dx in -radius..=radius {
                    if radius > 0 && dx.abs() != radius && dz.abs() != radius {
                        continue;
                    }
                    let Some(segments) = self.cells.get(&(origin.0 + dx, origin.1 + dz)) else {
                        continue;
                    };
                    for &(ring_index, segment) in segments {
                        let ring = &self.rings[ring_index as usize];
                        let a = ring[segment as usize];
                        let b = ring[segment as usize + 1];
                        let (t, closest, distance) = closest_on_segment(point, a, b);
                        if distance < best_distance {
                            best_distance = distance;
                            best = Some(CoastHit {
                                ring: ring_index as usize,
                                segment: segment as usize,
                                t,
                                point: closest,
                            });
                        }
                    }
                }
            }
            if best_distance <= f64::from(radius) * CELL_M {
                break;
            }
        }
        if let Some(hit) = best {
            return hit;
        }
        for (ring_index, ring) in self.rings.iter().enumerate() {
            for (segment, edge) in ring.windows(2).enumerate() {
                let (t, closest, distance) = closest_on_segment(point, edge[0], edge[1]);
                if distance < best_distance {
                    best_distance = distance;
                    best = Some(CoastHit {
                        ring: ring_index,
                        segment,
                        t,
                        point: closest,
                    });
                }
            }
        }
        best.expect("coastline polylines must contain at least one segment")
    }

    fn walk(&self, start: CoastHit, end: CoastHit, direction: i32) -> Vec<[f64; 2]> {
        let ring = &self.rings[start.ring];
        let segment_count = ring.len() - 1;
        let mut points = vec![start.point];
        let same_segment = start.segment == end.segment;
        let direct = same_segment
            && ((direction > 0 && start.t <= end.t + 1.0e-9)
                || (direction < 0 && start.t + 1.0e-9 >= end.t));
        if direct {
            if hypot(start.point, end.point) > 1.0e-4 {
                points.push(end.point);
            }
            return points;
        }

        if direction > 0 {
            let mut segment = start.segment;
            for _ in 0..=segment_count {
                let next_vertex = (segment + 1) % segment_count;
                let vertex = ring[next_vertex];
                if hypot(*points.last().unwrap(), vertex) > 1.0e-4 {
                    points.push(vertex);
                }
                segment = next_vertex;
                if segment == end.segment {
                    if hypot(*points.last().unwrap(), end.point) > 1.0e-4 {
                        points.push(end.point);
                    }
                    return points;
                }
            }
        } else {
            let mut segment = start.segment;
            for _ in 0..=segment_count {
                let vertex = ring[segment];
                if hypot(*points.last().unwrap(), vertex) > 1.0e-4 {
                    points.push(vertex);
                }
                segment = (segment + segment_count - 1) % segment_count;
                if segment == end.segment {
                    if hypot(*points.last().unwrap(), end.point) > 1.0e-4 {
                        points.push(end.point);
                    }
                    return points;
                }
            }
        }
        if hypot(*points.last().unwrap(), end.point) > 1.0e-4 {
            points.push(end.point);
        }
        points
    }
}

fn cell(value: f64) -> i32 {
    (value / CELL_M).floor() as i32
}

fn closest_on_segment(point: [f64; 2], a: [f64; 2], b: [f64; 2]) -> (f64, [f64; 2], f64) {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length_squared = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if length_squared == 0.0 {
        0.0
    } else {
        (((point[0] - a[0]) * ab[0] + (point[1] - a[1]) * ab[1]) / length_squared).clamp(0.0, 1.0)
    };
    let closest = [a[0] + ab[0] * t, a[1] + ab[1] * t];
    let distance = hypot(point, closest);
    (t, closest, distance)
}

fn hypot(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn path_length(points: &[[f64; 2]]) -> f64 {
    points.windows(2).map(|edge| hypot(edge[0], edge[1])).sum()
}

fn densify_path(points: &[[f64; 2]], max_edge_m: f64) -> Vec<[f64; 2]> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut densified = vec![points[0]];
    for edge in points.windows(2) {
        let length = hypot(edge[0], edge[1]);
        let segments = (length / max_edge_m).ceil().max(1.0) as usize;
        for step in 1..=segments {
            let t = step as f64 / segments as f64;
            densified.push([
                edge[0][0] + (edge[1][0] - edge[0][0]) * t,
                edge[0][1] + (edge[1][1] - edge[0][1]) * t,
            ]);
        }
    }
    densified
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> CoastPolylines {
        CoastPolylines::from_rings(vec![vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [0.0, 0.0],
        ]])
    }

    #[test]
    fn snaps_off_segment_queries_onto_the_polyline() {
        let coast = square();
        let point = coast.nearest_point(5.0, -3.0);
        assert!((point[0] - 5.0).abs() < 1.0e-9);
        assert!(point[1].abs() < 1.0e-9);
    }

    #[test]
    fn walks_the_shorter_ring_arc_and_keeps_corners() {
        let coast = square();
        let path = coast.path([1.0, 0.0], [10.0, 1.0], 3.0);
        assert!(
            path.iter()
                .any(|point| (point[0] - 10.0).abs() < 1.0e-6 && point[1].abs() < 1.0e-6)
        );
        assert!(path_length(&path) < 20.0);
        assert!(
            path.windows(2)
                .all(|edge| hypot(edge[0], edge[1]) <= 3.0 + 1.0e-6)
        );
    }
}
