//! Renderer-independent terrain material classification.

const SLOPE_REF_STEP_M: f32 = 30.0;
const SLOPE_SPECTRUM_EXP: f32 = 0.35;

pub(crate) fn material_masks_from_heights(
    height_m: f32,
    h_l: f32,
    h_r: f32,
    h_d: f32,
    h_u: f32,
    step_m: f32,
) -> [u8; 4] {
    let grad_x = (h_r - h_l) / (2.0 * step_m);
    let grad_y = (h_u - h_d) / (2.0 * step_m);
    let slope = (grad_x * grad_x + grad_y * grad_y).sqrt();
    let slope_gain = (step_m / SLOPE_REF_STEP_M)
        .max(1.0)
        .powf(SLOPE_SPECTRUM_EXP);
    let laplacian = ((h_l + h_r + h_d + h_u) * 0.25 - height_m) / step_m.max(1.0);

    let slope_rock = smoothstep(0.20, 0.75, slope * slope_gain);
    let high_rock = smoothstep(2_200.0, 6_000.0, height_m);
    let convex_rock = smoothstep(0.04, 0.20, -laplacian);
    let rock = (slope_rock * 0.82 + high_rock * 0.18 + convex_rock * 0.16).clamp(0.0, 0.95);
    let hollow = smoothstep(0.035, 0.18, laplacian);
    let wetness = (hollow * (1.0 - smoothstep(1_500.0, 4_500.0, height_m)) * (1.0 - rock * 0.7))
        .clamp(0.0, 1.0);
    let soil = (smoothstep(0.035, 0.28, slope) * (1.0 - smoothstep(0.45, 0.9, slope))
        + hollow * 0.45
        + wetness * 0.25)
        .clamp(0.0, 1.0)
        * (1.0 - rock * 0.65);
    let grass = ((1.0 - rock) * (1.0 - soil * 0.45) * (1.0 - wetness * 0.25)).clamp(0.0, 1.0);
    let sum = (grass + soil + rock).max(1.0e-4);
    [
        quantize_unit_to_u8(grass / sum),
        quantize_unit_to_u8(soil / sum),
        quantize_unit_to_u8(rock / sum),
        quantize_unit_to_u8(wetness),
    ]
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let denominator = edge1 - edge0;
    if denominator.abs() < f32::EPSILON {
        return if value >= edge0 { 1.0 } else { 0.0 };
    }
    let amount = ((value - edge0) / denominator).clamp(0.0, 1.0);
    amount * amount * (3.0 - 2.0 * amount)
}

fn quantize_unit_to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * u8::MAX as f32).round() as u8
}
