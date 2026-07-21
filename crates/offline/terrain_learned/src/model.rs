use burn::nn::{
    PaddingConfig2d,
    conv::{Conv2d, Conv2dConfig},
};
use burn::prelude::*;
use burn::tensor::activation::silu;

/// Noisy residual, coarse height, crater density, gardening, mare mask,
/// stable x/y coordinates, and normalized diffusion time.
pub const CONDITION_CHANNELS: usize = 8;

#[derive(Config, Debug)]
pub struct AirlessDenoiserConfig {
    #[config(default = 32)]
    pub hidden_channels: usize,
}

/// Compact conditional residual denoiser used to prove the MIRA tensor path.
#[derive(Module, Debug)]
pub struct AirlessDenoiser<B: Backend> {
    input: Conv2d<B>,
    hidden_a: Conv2d<B>,
    hidden_b: Conv2d<B>,
    output: Conv2d<B>,
}

impl AirlessDenoiserConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AirlessDenoiser<B> {
        let padding = PaddingConfig2d::Same;
        AirlessDenoiser {
            input: Conv2dConfig::new([CONDITION_CHANNELS, self.hidden_channels], [3, 3])
                .with_padding(padding.clone())
                .init(device),
            hidden_a: Conv2dConfig::new([self.hidden_channels, self.hidden_channels], [3, 3])
                .with_padding(padding.clone())
                .init(device),
            hidden_b: Conv2dConfig::new([self.hidden_channels, self.hidden_channels], [3, 3])
                .with_padding(padding.clone())
                .init(device),
            output: Conv2dConfig::new([self.hidden_channels, 1], [3, 3])
                .with_padding(padding)
                .init(device),
        }
    }
}

impl<B: Backend> AirlessDenoiser<B> {
    pub fn forward(&self, conditioned: Tensor<B, 4>) -> Tensor<B, 4> {
        let features = silu(self.input.forward(conditioned));
        let residual = features.clone();
        let features = silu(self.hidden_a.forward(features));
        let features = silu(self.hidden_b.forward(features)) + residual;
        self.output.forward(features)
    }

    /// Blend this inference model toward the latest trained weights.
    pub fn ema_update(self, current: &Self, decay: f32) -> Self {
        assert!((0.0..1.0).contains(&decay));
        Self {
            input: ema_conv(self.input, &current.input, decay),
            hidden_a: ema_conv(self.hidden_a, &current.hidden_a, decay),
            hidden_b: ema_conv(self.hidden_b, &current.hidden_b, decay),
            output: ema_conv(self.output, &current.output, decay),
        }
    }
}

fn ema_conv<B: Backend>(mut ema: Conv2d<B>, current: &Conv2d<B>, decay: f32) -> Conv2d<B> {
    let current_weight = current.weight.val();
    ema.weight = ema
        .weight
        .map(|value| value * decay + current_weight * (1.0 - decay));
    ema.bias = match (ema.bias.take(), current.bias.as_ref()) {
        (Some(ema_bias), Some(current_bias)) => {
            let current_bias = current_bias.val();
            Some(ema_bias.map(|value| value * decay + current_bias * (1.0 - decay)))
        }
        (None, None) => None,
        _ => panic!("EMA convolution bias structure differs from current model"),
    };
    ema
}
