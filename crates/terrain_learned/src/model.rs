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
}
