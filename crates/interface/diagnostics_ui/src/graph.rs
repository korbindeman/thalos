use bevy::{
    prelude::*, render::render_resource::AsBindGroup, shader::ShaderRef,
    ui_render::prelude::UiMaterial,
};

use crate::FRAME_HISTORY_LEN;

const SERIES_VEC4S: usize = FRAME_HISTORY_LEN / 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagnosticsGraphMode {
    FrameTime,
    Memory,
}

impl DiagnosticsGraphMode {
    const fn shader_value(self) -> f32 {
        match self {
            Self::FrameTime => 0.0,
            Self::Memory => 1.0,
        }
    }
}

/// Two-series graph rendered as one UI material quad with no UI entity churn.
#[derive(Asset, AsBindGroup, TypePath, Clone)]
pub struct DiagnosticsGraphMaterial {
    #[uniform(0)]
    params: Vec4,
    #[uniform(1)]
    marks: Vec4,
    #[uniform(2)]
    series_a: [Vec4; SERIES_VEC4S],
    #[uniform(3)]
    series_b: [Vec4; SERIES_VEC4S],
}

impl DiagnosticsGraphMaterial {
    pub fn frame_time() -> Self {
        Self {
            params: Vec4::new(
                0.0,
                33.4,
                DiagnosticsGraphMode::FrameTime.shader_value(),
                0.0,
            ),
            marks: Vec4::new(1000.0 / 60.0, 1000.0 / 30.0, 0.0, 0.0),
            series_a: [Vec4::ZERO; SERIES_VEC4S],
            series_b: [Vec4::ZERO; SERIES_VEC4S],
        }
    }

    pub fn memory() -> Self {
        Self {
            params: Vec4::new(0.0, 1.0, DiagnosticsGraphMode::Memory.shader_value(), 0.0),
            marks: Vec4::ZERO,
            series_a: [Vec4::ZERO; SERIES_VEC4S],
            series_b: [Vec4::ZERO; SERIES_VEC4S],
        }
    }

    pub fn set_series(
        &mut self,
        first: impl Iterator<Item = f32>,
        second: impl Iterator<Item = f32>,
        minimum_scale: f32,
        mode: DiagnosticsGraphMode,
    ) {
        self.series_a.fill(Vec4::ZERO);
        self.series_b.fill(Vec4::ZERO);
        let (count, max_first) = pack_series(&mut self.series_a, first);
        let (_, max_second) = pack_series(&mut self.series_b, second);
        self.params = Vec4::new(
            count as f32,
            minimum_scale.max(1.15 * max_first.max(max_second)),
            mode.shader_value(),
            0.0,
        );
    }
}

impl Default for DiagnosticsGraphMaterial {
    fn default() -> Self {
        Self::frame_time()
    }
}

impl UiMaterial for DiagnosticsGraphMaterial {
    fn fragment_shader() -> ShaderRef {
        "shaders/perf_graph.wgsl".into()
    }
}

fn pack_series(
    destination: &mut [Vec4; SERIES_VEC4S],
    values: impl Iterator<Item = f32>,
) -> (usize, f32) {
    let mut count = 0;
    let mut max = 0.0_f32;
    for (index, value) in values.enumerate().take(FRAME_HISTORY_LEN) {
        destination[index / 4][index % 4] = value;
        count = index + 1;
        max = max.max(value);
    }
    (count, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_series_are_packed_four_values_per_uniform_vector() {
        let mut material = DiagnosticsGraphMaterial::memory();
        material.set_series(
            [1.0, 2.0, 3.0, 4.0, 5.0].into_iter(),
            [6.0, 7.0].into_iter(),
            1.0,
            DiagnosticsGraphMode::Memory,
        );
        assert_eq!(material.series_a[0], Vec4::new(1.0, 2.0, 3.0, 4.0));
        assert_eq!(material.series_a[1].x, 5.0);
        assert_eq!(material.series_b[0], Vec4::new(6.0, 7.0, 0.0, 0.0));
        assert_eq!(material.params.x, 5.0);
        assert_eq!(material.params.y, 8.05);
        assert_eq!(material.params.z, 1.0);
    }
}
