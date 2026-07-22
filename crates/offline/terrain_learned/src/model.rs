use burn::module::Param;
use burn::nn::{
    Linear, LinearConfig, PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig, ConvTranspose2d, ConvTranspose2dConfig},
};
use burn::prelude::*;
use burn::tensor::activation::silu;
use serde::{Deserialize, Serialize};

/// Noisy residual, coarse height, four process controls, mare mask, physical
/// scale, stable x/y coordinates, and normalized diffusion time.
pub const CONDITION_CHANNELS: usize = 11;

/// Input-channel index of the broadcast normalized diffusion time plane.
const TIME_CHANNEL: usize = CONDITION_CHANNELS - 1;
/// Number of Fourier frequency pairs in the timestep embedding.
const FOURIER_FREQUENCIES: usize = 8;
/// Hidden width of the timestep-embedding MLP.
const TIME_HIDDEN: usize = 64;

/// Decoder upsampling operator. `Transposed` is the historical stride-2
/// `ConvTranspose2d`; `Resize` is nearest-neighbour ×2 followed by an ordinary
/// 3×3 convolution, which cannot produce checkerboard/overlap periodicity.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Upsampling {
    #[default]
    Transposed,
    Resize,
}

impl Upsampling {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Transposed => "transposed",
            Self::Resize => "resize",
        }
    }
}

/// How diffusion time conditions the denoiser. `Broadcast` is the historical
/// single normalized-time input plane; `Fourier` zeroes that plane and instead
/// injects a Fourier-feature MLP embedding as per-channel biases into every
/// encoder/decoder level.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeConditioning {
    #[default]
    Broadcast,
    Fourier,
}

impl TimeConditioning {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Broadcast => "broadcast",
            Self::Fourier => "fourier",
        }
    }
}

#[derive(Config, Debug)]
pub struct AirlessDenoiserConfig {
    #[config(default = 32)]
    pub hidden_channels: usize,
    #[config(default = "Upsampling::Transposed")]
    pub upsampling: Upsampling,
    #[config(default = "TimeConditioning::Broadcast")]
    pub time_conditioning: TimeConditioning,
}

/// Two-level residual U-Net for conditioned airless-terrain diffusion.
///
/// The encoder gives a 64 px patch a roughly 40 px receptive field while skip
/// connections preserve crater rims and other sharp high-frequency structure.
#[derive(Module, Debug)]
pub struct AirlessDenoiser<B: Backend> {
    input: Conv2d<B>,
    encoder_1a: Conv2d<B>,
    encoder_1b: Conv2d<B>,
    down_1: Conv2d<B>,
    encoder_2a: Conv2d<B>,
    encoder_2b: Conv2d<B>,
    down_2: Conv2d<B>,
    middle_a: Conv2d<B>,
    middle_b: Conv2d<B>,
    up_2: Upsample<B>,
    decoder_2a: Conv2d<B>,
    decoder_2b: Conv2d<B>,
    up_1: Upsample<B>,
    decoder_1a: Conv2d<B>,
    decoder_1b: Conv2d<B>,
    output: Conv2d<B>,
    time: Option<TimeEmbedding<B>>,
}

/// Fourier-feature timestep embedding with one bias projection per U-Net level.
#[derive(Module, Debug)]
pub struct TimeEmbedding<B: Backend> {
    hidden: Linear<B>,
    level_1: Linear<B>,
    level_2: Linear<B>,
    middle: Linear<B>,
    decode_2: Linear<B>,
    decode_1: Linear<B>,
}

impl<B: Backend> TimeEmbedding<B> {
    /// Extract the (spatially constant) normalized time from the broadcast
    /// plane and lift it through Fourier features and the shared MLP.
    fn hidden_features(&self, conditioned: &Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, _, _, _] = conditioned.dims();
        let time = conditioned
            .clone()
            .narrow(1, TIME_CHANNEL, 1)
            .mean_dim(3)
            .mean_dim(2)
            .reshape([batch, 1]);
        let mut features = Vec::with_capacity(2 * FOURIER_FREQUENCIES);
        for frequency in 0..FOURIER_FREQUENCIES {
            let angle = time
                .clone()
                .mul_scalar((1u32 << frequency) as f32 * core::f32::consts::TAU);
            features.push(angle.clone().sin());
            features.push(angle.cos());
        }
        silu(self.hidden.forward(Tensor::cat(features, 1)))
    }
}

/// Add a per-channel time bias to a feature map.
fn inject_time<B: Backend>(
    features: Tensor<B, 4>,
    projection: &Linear<B>,
    hidden: &Tensor<B, 2>,
) -> Tensor<B, 4> {
    let bias = projection.forward(hidden.clone());
    let [batch, channels] = bias.dims();
    features + bias.reshape([batch, channels, 1, 1])
}

/// Zero the broadcast time plane so Fourier conditioning is the sole time path.
fn mask_time_channel<B: Backend>(conditioned: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, _, height, width] = conditioned.dims();
    let spatial = conditioned.clone().narrow(1, 0, TIME_CHANNEL);
    let zeros = Tensor::zeros([batch, 1, height, width], &conditioned.device());
    Tensor::cat(vec![spatial, zeros], 1)
}

impl AirlessDenoiserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AirlessDenoiser<B> {
        let base = self.hidden_channels;
        let conv = |input, output| {
            Conv2dConfig::new([input, output], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        let down = |input, output| {
            Conv2dConfig::new([input, output], [3, 3])
                .with_stride([2, 2])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        let up = |input, output| match self.upsampling {
            Upsampling::Transposed => Upsample::Transposed(
                ConvTranspose2dConfig::new([input, output], [4, 4])
                    .with_stride([2, 2])
                    .with_padding([1, 1])
                    .init(device),
            ),
            Upsampling::Resize => Upsample::Resize(
                Conv2dConfig::new([input, output], [3, 3])
                    .with_padding(PaddingConfig2d::Same)
                    .init(device),
            ),
        };
        AirlessDenoiser {
            input: conv(CONDITION_CHANNELS, base),
            encoder_1a: conv(base, base),
            encoder_1b: conv(base, base),
            down_1: down(base, base * 2),
            encoder_2a: conv(base * 2, base * 2),
            encoder_2b: conv(base * 2, base * 2),
            down_2: down(base * 2, base * 4),
            middle_a: conv(base * 4, base * 4),
            middle_b: conv(base * 4, base * 4),
            up_2: up(base * 4, base * 2),
            decoder_2a: conv(base * 4, base * 2),
            decoder_2b: conv(base * 2, base * 2),
            up_1: up(base * 2, base),
            decoder_1a: conv(base * 2, base),
            decoder_1b: conv(base, base),
            output: Conv2dConfig::new([base, 1], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device),
            time: match self.time_conditioning {
                TimeConditioning::Broadcast => None,
                TimeConditioning::Fourier => Some(TimeEmbedding {
                    hidden: LinearConfig::new(2 * FOURIER_FREQUENCIES, TIME_HIDDEN).init(device),
                    level_1: LinearConfig::new(TIME_HIDDEN, base).init(device),
                    level_2: LinearConfig::new(TIME_HIDDEN, base * 2).init(device),
                    middle: LinearConfig::new(TIME_HIDDEN, base * 4).init(device),
                    decode_2: LinearConfig::new(TIME_HIDDEN, base * 2).init(device),
                    decode_1: LinearConfig::new(TIME_HIDDEN, base).init(device),
                }),
            },
        }
    }
}

/// One decoder upsampling stage; see [`Upsampling`].
#[derive(Module, Debug)]
pub enum Upsample<B: Backend> {
    Transposed(ConvTranspose2d<B>),
    Resize(Conv2d<B>),
}

impl<B: Backend> Upsample<B> {
    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Transposed(conv) => conv.forward(input),
            Self::Resize(conv) => conv.forward(nearest_double(input)),
        }
    }
}

/// Nearest-neighbour ×2 upsampling from primitive reshape/repeat ops, so it is
/// deterministic and differentiable on every Burn backend.
fn nearest_double<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    input
        .reshape([batch, channels, height, 1, width, 1])
        .repeat_dim(3, 2)
        .repeat_dim(5, 2)
        .reshape([batch, channels, height * 2, width * 2])
}

impl<B: Backend> AirlessDenoiser<B> {
    pub fn forward(&self, conditioned: Tensor<B, 4>) -> Tensor<B, 4> {
        let hidden = self
            .time
            .as_ref()
            .map(|embedding| embedding.hidden_features(&conditioned));
        let conditioned = if self.time.is_some() {
            mask_time_channel(conditioned)
        } else {
            conditioned
        };

        let level_1 = silu(self.input.forward(conditioned));
        let level_1 = match (&self.time, &hidden) {
            (Some(embedding), Some(hidden)) => inject_time(level_1, &embedding.level_1, hidden),
            _ => level_1,
        };
        let level_1 = silu(
            self.encoder_1b
                .forward(silu(self.encoder_1a.forward(level_1.clone()))),
        ) + level_1;

        let level_2 = silu(self.down_1.forward(level_1.clone()));
        let level_2 = match (&self.time, &hidden) {
            (Some(embedding), Some(hidden)) => inject_time(level_2, &embedding.level_2, hidden),
            _ => level_2,
        };
        let level_2 = silu(
            self.encoder_2b
                .forward(silu(self.encoder_2a.forward(level_2.clone()))),
        ) + level_2;

        let middle = silu(self.down_2.forward(level_2.clone()));
        let middle = match (&self.time, &hidden) {
            (Some(embedding), Some(hidden)) => inject_time(middle, &embedding.middle, hidden),
            _ => middle,
        };
        let middle = silu(
            self.middle_b
                .forward(silu(self.middle_a.forward(middle.clone()))),
        ) + middle;

        let decoded_2 = silu(self.up_2.forward(middle));
        let decoded_2 = match (&self.time, &hidden) {
            (Some(embedding), Some(hidden)) => inject_time(decoded_2, &embedding.decode_2, hidden),
            _ => decoded_2,
        };
        let decoded_2 = Tensor::cat(vec![decoded_2, level_2], 1);
        let decoded_2 = silu(self.decoder_2a.forward(decoded_2));
        let decoded_2 = silu(self.decoder_2b.forward(decoded_2.clone())) + decoded_2;

        let decoded_1 = silu(self.up_1.forward(decoded_2));
        let decoded_1 = match (&self.time, &hidden) {
            (Some(embedding), Some(hidden)) => inject_time(decoded_1, &embedding.decode_1, hidden),
            _ => decoded_1,
        };
        let decoded_1 = Tensor::cat(vec![decoded_1, level_1], 1);
        let decoded_1 = silu(self.decoder_1a.forward(decoded_1));
        let decoded_1 = silu(self.decoder_1b.forward(decoded_1.clone())) + decoded_1;
        self.output.forward(decoded_1)
    }

    pub fn ema_update(self, current: &Self, decay: f32) -> Self {
        assert!((0.0..1.0).contains(&decay));
        Self {
            input: ema_conv(self.input, &current.input, decay),
            encoder_1a: ema_conv(self.encoder_1a, &current.encoder_1a, decay),
            encoder_1b: ema_conv(self.encoder_1b, &current.encoder_1b, decay),
            down_1: ema_conv(self.down_1, &current.down_1, decay),
            encoder_2a: ema_conv(self.encoder_2a, &current.encoder_2a, decay),
            encoder_2b: ema_conv(self.encoder_2b, &current.encoder_2b, decay),
            down_2: ema_conv(self.down_2, &current.down_2, decay),
            middle_a: ema_conv(self.middle_a, &current.middle_a, decay),
            middle_b: ema_conv(self.middle_b, &current.middle_b, decay),
            up_2: ema_upsample(self.up_2, &current.up_2, decay),
            decoder_2a: ema_conv(self.decoder_2a, &current.decoder_2a, decay),
            decoder_2b: ema_conv(self.decoder_2b, &current.decoder_2b, decay),
            up_1: ema_upsample(self.up_1, &current.up_1, decay),
            decoder_1a: ema_conv(self.decoder_1a, &current.decoder_1a, decay),
            decoder_1b: ema_conv(self.decoder_1b, &current.decoder_1b, decay),
            output: ema_conv(self.output, &current.output, decay),
            time: ema_time(self.time, &current.time, decay),
        }
    }
}

fn ema_time<B: Backend>(
    ema: Option<TimeEmbedding<B>>,
    current: &Option<TimeEmbedding<B>>,
    decay: f32,
) -> Option<TimeEmbedding<B>> {
    match (ema, current) {
        (Some(ema), Some(current)) => Some(TimeEmbedding {
            hidden: ema_linear(ema.hidden, &current.hidden, decay),
            level_1: ema_linear(ema.level_1, &current.level_1, decay),
            level_2: ema_linear(ema.level_2, &current.level_2, decay),
            middle: ema_linear(ema.middle, &current.middle, decay),
            decode_2: ema_linear(ema.decode_2, &current.decode_2, decay),
            decode_1: ema_linear(ema.decode_1, &current.decode_1, decay),
        }),
        (None, None) => None,
        _ => panic!("EMA time-conditioning structure differs from current model"),
    }
}

fn ema_linear<B: Backend>(mut ema: Linear<B>, current: &Linear<B>, decay: f32) -> Linear<B> {
    ema.weight = ema
        .weight
        .map(|value| value * decay + current.weight.val() * (1.0 - decay));
    ema.bias = ema_bias(ema.bias.take(), current.bias.as_ref(), decay);
    ema
}

fn ema_conv<B: Backend>(mut ema: Conv2d<B>, current: &Conv2d<B>, decay: f32) -> Conv2d<B> {
    ema.weight = ema
        .weight
        .map(|value| value * decay + current.weight.val() * (1.0 - decay));
    ema.bias = ema_bias(ema.bias.take(), current.bias.as_ref(), decay);
    ema
}

fn ema_upsample<B: Backend>(ema: Upsample<B>, current: &Upsample<B>, decay: f32) -> Upsample<B> {
    match (ema, current) {
        (Upsample::Transposed(ema), Upsample::Transposed(current)) => {
            Upsample::Transposed(ema_conv_transpose(ema, current, decay))
        }
        (Upsample::Resize(ema), Upsample::Resize(current)) => {
            Upsample::Resize(ema_conv(ema, current, decay))
        }
        _ => panic!("EMA upsampling variant differs from current model"),
    }
}

fn ema_conv_transpose<B: Backend>(
    mut ema: ConvTranspose2d<B>,
    current: &ConvTranspose2d<B>,
    decay: f32,
) -> ConvTranspose2d<B> {
    ema.weight = ema
        .weight
        .map(|value| value * decay + current.weight.val() * (1.0 - decay));
    ema.bias = ema_bias(ema.bias.take(), current.bias.as_ref(), decay);
    ema
}

fn ema_bias<B: Backend>(
    ema: Option<Param<Tensor<B, 1>>>,
    current: Option<&Param<Tensor<B, 1>>>,
    decay: f32,
) -> Option<Param<Tensor<B, 1>>> {
    match (ema, current) {
        (Some(ema), Some(current)) => {
            Some(ema.map(|value| value * decay + current.val() * (1.0 - decay)))
        }
        (None, None) => None,
        _ => panic!("EMA convolution bias structure differs from current model"),
    }
}
