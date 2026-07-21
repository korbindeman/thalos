#[derive(Clone, Debug)]
pub struct Grid {
    pub size: usize,
    pub values: Vec<f32>,
}

impl Grid {
    pub fn zeros(size: usize) -> Self {
        Self {
            size,
            values: vec![0.0; size * size],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.values[y * self.size + x]
    }

    pub fn add(&mut self, x: usize, y: usize, value: f32) {
        self.values[y * self.size + x] += value;
    }

    pub fn subtract_mean(&mut self) {
        let mean = self.values.iter().sum::<f32>() / self.values.len() as f32;
        for value in &mut self.values {
            *value -= mean;
        }
    }

    pub fn rms(&self) -> f32 {
        (self.values.iter().map(|value| value * value).sum::<f32>() / self.values.len() as f32)
            .sqrt()
    }

    pub fn max_abs(&self) -> f32 {
        self.values
            .iter()
            .fold(0.0f32, |maximum, value| maximum.max(value.abs()))
    }

    pub fn slope_rms(&self, metres_per_pixel: f32) -> f32 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for y in 1..self.size - 1 {
            for x in 1..self.size - 1 {
                let dx = (self.get(x + 1, y) - self.get(x - 1, y)) / (2.0 * metres_per_pixel);
                let dy = (self.get(x, y + 1) - self.get(x, y - 1)) / (2.0 * metres_per_pixel);
                sum += f64::from(dx * dx + dy * dy);
                count += 1;
            }
        }
        (sum / count as f64).sqrt() as f32
    }
}
