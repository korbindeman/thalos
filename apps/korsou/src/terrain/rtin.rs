#[derive(Debug)]
pub struct Rtin {
    size: usize,
    errors: Vec<f32>,
}

impl Rtin {
    pub fn new(size: usize, heights: &[f32], constrained: &[bool]) -> Self {
        assert!(size >= 3 && (size - 1).is_power_of_two());
        assert_eq!(heights.len(), size * size);
        assert_eq!(constrained.len(), heights.len());
        let mut rtin = Self {
            size,
            errors: vec![0.0; heights.len()],
        };
        let last = size - 1;
        rtin.propagate(
            Point::new(0, 0),
            Point::new(last, last),
            Point::new(last, 0),
            heights,
            constrained,
        );
        rtin.propagate(
            Point::new(last, last),
            Point::new(0, 0),
            Point::new(0, last),
            heights,
            constrained,
        );
        rtin
    }

    pub fn triangles(&self, max_error: f32) -> Vec<[u32; 3]> {
        let last = self.size - 1;
        let mut triangles = Vec::new();
        self.emit(
            Point::new(0, 0),
            Point::new(last, last),
            Point::new(last, 0),
            max_error,
            &mut triangles,
        );
        self.emit(
            Point::new(last, last),
            Point::new(0, 0),
            Point::new(0, last),
            max_error,
            &mut triangles,
        );
        triangles
    }

    fn propagate(
        &mut self,
        a: Point,
        b: Point,
        c: Point,
        heights: &[f32],
        constrained: &[bool],
    ) -> f32 {
        let Some(middle) = midpoint(a, b) else {
            return 0.0;
        };
        let middle_index = middle.index(self.size);
        let interpolation = (heights[a.index(self.size)] + heights[b.index(self.size)]) * 0.5;
        let mut error = (heights[middle_index] - interpolation).abs();
        if constrained[middle_index] {
            error = f32::INFINITY;
        }
        error = error.max(self.propagate(c, a, middle, heights, constrained));
        error = error.max(self.propagate(b, c, middle, heights, constrained));
        self.errors[middle_index] = self.errors[middle_index].max(error);
        error
    }

    fn emit(&self, a: Point, b: Point, c: Point, max_error: f32, triangles: &mut Vec<[u32; 3]>) {
        if let Some(middle) = midpoint(a, b)
            && self.errors[middle.index(self.size)] > max_error
        {
            self.emit(c, a, middle, max_error, triangles);
            self.emit(b, c, middle, max_error, triangles);
        } else {
            triangles.push([
                a.index(self.size) as u32,
                b.index(self.size) as u32,
                c.index(self.size) as u32,
            ]);
        }
    }
}

#[derive(Clone, Copy)]
struct Point {
    x: usize,
    z: usize,
}

impl Point {
    const fn new(x: usize, z: usize) -> Self {
        Self { x, z }
    }

    fn index(self, size: usize) -> usize {
        self.z * size + self.x
    }
}

fn midpoint(a: Point, b: Point) -> Option<Point> {
    let x = a.x + b.x;
    let z = a.z + b.z;
    if !x.is_multiple_of(2) || !z.is_multiple_of(2) {
        return None;
    }
    let middle = Point::new(x / 2, z / 2);
    (middle.x != a.x || middle.z != a.z)
        .then_some(middle)
        .filter(|middle| middle.x != b.x || middle.z != b.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_grid_collapses_to_two_triangles() {
        let heights = vec![0.0; 25];
        let rtin = Rtin::new(5, &heights, &vec![false; 25]);
        assert_eq!(rtin.triangles(0.0).len(), 2);
    }

    #[test]
    fn center_error_refines_the_mesh() {
        let mut heights = vec![0.0; 25];
        heights[12] = 10.0;
        let rtin = Rtin::new(5, &heights, &vec![false; 25]);
        assert!(rtin.triangles(1.0).len() > 2);
    }

    #[test]
    fn constrained_boundary_remains_a_complete_upward_mesh() {
        let size = 5;
        let heights = vec![0.0; size * size];
        let mut constrained = vec![false; heights.len()];
        for i in 0..size {
            constrained[i] = true;
            constrained[(size - 1) * size + i] = true;
            constrained[i * size] = true;
            constrained[i * size + size - 1] = true;
        }
        let triangles = Rtin::new(size, &heights, &constrained).triangles(0.0);
        let mut twice_area = 0i32;
        for [a, b, c] in triangles {
            let point = |index: u32| {
                (
                    (index as usize % size) as i32,
                    (index as usize / size) as i32,
                )
            };
            let (ax, az) = point(a);
            let (bx, bz) = point(b);
            let (cx, cz) = point(c);
            let winding = (bz - az) * (cx - ax) - (bx - ax) * (cz - az);
            assert!(winding > 0);
            twice_area += winding;
        }
        assert_eq!(twice_area, 2 * (size as i32 - 1).pow(2));
    }
}
