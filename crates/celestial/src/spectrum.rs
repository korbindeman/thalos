//! Spectral energy distributions and passband integration.
//!
//! The renderer never sees RGB directly. Sources emit flux as a
//! function of wavelength (a [`Spectrum`]), and a [`Passband`] filters
//! that into a scalar flux in a given band. Visible-light skybox
//! rendering uses three approximate CIE passbands; telescope sensors
//! will pick arbitrary bands.

/// Planck's constant × speed of light in SI units. Cached for the
/// blackbody evaluation path.
const HC: f64 = 1.986_445_857e-25; // J·m
/// Boltzmann constant, SI.
const K_B: f64 = 1.380_649e-23; // J/K

/// Spectral energy distribution.
///
/// Flux units are arbitrary and self-consistent: the same units come
/// out of [`Spectrum::evaluate`] regardless of variant, and consumers
/// (the baker, the telescope renderer) interpret them as
/// relative-flux-per-nanometre. This avoids dragging absolute
/// radiometric units through the whole pipeline while still being
/// physically meaningful.
#[derive(Debug, Clone)]
pub enum Spectrum {
    /// Planck blackbody at `temperature_k`, scaled by `scale`.
    Blackbody { temperature_k: f32, scale: f32 },
    /// F(λ) = scale · (λ / λ₀)^(-α)
    PowerLaw {
        alpha: f32,
        scale: f32,
        reference_wavelength_nm: f32,
    },
    /// Piecewise-linear tabulated curve. `wavelengths_nm` must be
    /// strictly monotonically increasing.
    Tabulated {
        wavelengths_nm: Vec<f32>,
        flux: Vec<f32>,
    },
}

impl Spectrum {
    /// Evaluate the spectrum at a given wavelength, in nm.
    pub fn evaluate(&self, wavelength_nm: f32) -> f32 {
        match self {
            Spectrum::Blackbody { temperature_k, scale } => {
                let lambda_m = (wavelength_nm as f64) * 1e-9;
                let t = *temperature_k as f64;
                // B(λ, T) = (2hc² / λ⁵) · 1 / (exp(hc / (λkT)) - 1)
                // We drop the 2hc² prefactor — it's a constant scale
                // across all stars and would just cancel out against
                // the `scale` field. What matters is the shape.
                let exponent = HC / (lambda_m * K_B * t);
                let denom = (exponent.exp() - 1.0).max(f64::MIN_POSITIVE);
                let value = 1.0 / (lambda_m.powi(5) * denom);
                (*scale as f64 * value) as f32
            }
            Spectrum::PowerLaw {
                alpha,
                scale,
                reference_wavelength_nm,
            } => {
                let ratio = wavelength_nm / reference_wavelength_nm.max(1e-6);
                scale * ratio.powf(-alpha)
            }
            Spectrum::Tabulated { wavelengths_nm, flux } => {
                if wavelengths_nm.is_empty() {
                    return 0.0;
                }
                if wavelength_nm <= wavelengths_nm[0] {
                    return flux[0];
                }
                if wavelength_nm >= *wavelengths_nm.last().unwrap() {
                    return *flux.last().unwrap();
                }
                // Binary search for the interval.
                let idx = wavelengths_nm
                    .binary_search_by(|p| {
                        p.partial_cmp(&wavelength_nm).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or_else(|i| i);
                let i1 = idx.min(wavelengths_nm.len() - 1).max(1);
                let i0 = i1 - 1;
                let x0 = wavelengths_nm[i0];
                let x1 = wavelengths_nm[i1];
                let t = (wavelength_nm - x0) / (x1 - x0).max(1e-6);
                flux[i0] * (1.0 - t) + flux[i1] * t
            }
        }
    }
}

/// A passband is a tabulated response curve. `integrate` convolves it
/// with a spectrum via the trapezoidal rule.
#[derive(Debug, Clone)]
pub struct Passband {
    pub name: &'static str,
    pub wavelengths_nm: Vec<f32>,
    pub response: Vec<f32>,
}

impl Passband {
    pub fn integrate(&self, spectrum: &Spectrum) -> f32 {
        if self.wavelengths_nm.len() < 2 {
            return 0.0;
        }
        let mut total = 0.0f64;
        for i in 0..self.wavelengths_nm.len() - 1 {
            let x0 = self.wavelengths_nm[i];
            let x1 = self.wavelengths_nm[i + 1];
            let w = (x1 - x0) as f64;
            let f0 = (spectrum.evaluate(x0) * self.response[i]) as f64;
            let f1 = (spectrum.evaluate(x1) * self.response[i + 1]) as f64;
            total += 0.5 * (f0 + f1) * w;
        }
        total as f32
    }

    /// Effective wavelength — the response-weighted centroid. Used by
    /// the baker to drive chromatic effects without re-integrating.
    pub fn effective_wavelength_nm(&self) -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (w, r) in self.wavelengths_nm.iter().zip(self.response.iter()) {
            num += (*w as f64) * (*r as f64);
            den += *r as f64;
        }
        if den > 0.0 { (num / den) as f32 } else { 0.0 }
    }
}

/// Canned passbands. Visible-light approximations of CIE RGB response —
/// good enough for a skybox, and replaceable with photometric standards
/// (Johnson UBV, SDSS ugriz) for telescope work without touching
/// consumers.
pub mod passbands {
    use super::Passband;

    fn gaussian_band(name: &'static str, center: f32, fwhm: f32) -> Passband {
        let sigma = fwhm / 2.3548;
        let samples = 16;
        let start = center - 2.5 * fwhm;
        let end = center + 2.5 * fwhm;
        let step = (end - start) / (samples - 1) as f32;
        let mut wavelengths_nm = Vec::with_capacity(samples);
        let mut response = Vec::with_capacity(samples);
        for i in 0..samples {
            let w = start + step * i as f32;
            let x = (w - center) / sigma;
            let r = (-0.5 * x * x).exp();
            wavelengths_nm.push(w);
            response.push(r);
        }
        Passband { name, wavelengths_nm, response }
    }

    pub fn visible_red() -> Passband {
        gaussian_band("visible_red", 610.0, 80.0)
    }
    pub fn visible_green() -> Passband {
        gaussian_band("visible_green", 545.0, 90.0)
    }
    pub fn visible_blue() -> Passband {
        gaussian_band("visible_blue", 465.0, 80.0)
    }

    pub fn visible_rgb() -> [Passband; 3] {
        [visible_red(), visible_green(), visible_blue()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackbody_hotter_is_bluer() {
        let hot = Spectrum::Blackbody { temperature_k: 30_000.0, scale: 1.0 };
        let cool = Spectrum::Blackbody { temperature_k: 3_000.0, scale: 1.0 };
        let red = passbands::visible_red();
        let blue = passbands::visible_blue();
        let hot_ratio = blue.integrate(&hot) / red.integrate(&hot);
        let cool_ratio = blue.integrate(&cool) / red.integrate(&cool);
        assert!(
            hot_ratio > cool_ratio,
            "hot star should be bluer: hot {hot_ratio}, cool {cool_ratio}"
        );
    }

    #[test]
    fn sun_like_has_finite_flux() {
        let sun = Spectrum::Blackbody { temperature_k: 5_778.0, scale: 1.0 };
        for band in passbands::visible_rgb() {
            let f = band.integrate(&sun);
            assert!(f.is_finite() && f > 0.0, "band {} = {f}", band.name);
        }
    }
}
