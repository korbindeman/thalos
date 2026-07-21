//! Design tokens — the single source of truth for the game UI's look.
//!
//! Everything visual (colours, spacing, radii, type scale, font roles) is
//! defined here once; widgets and screens compose these tokens and never
//! hard-code their own values. The direction is *modern, light, compact*:
//! frosted dark-glass surfaces, hairline light strokes, a near-white text
//! ramp, and one warm accent — game UI, not a debug overlay.
//!
//! Fonts: **Inter** (OFL — see `assets/fonts/Inter-OFL.txt`; the refined
//! neutral grotesque of the visionOS-style references) carries all interface
//! text in three weights; **Fira Code** stays for numeric/mono readouts where
//! tabular digits matter (altitude, Δv, coordinates). Barlow ships alongside
//! as the characterful alternative — swap the three handles in
//! `init_ui_theme` to try it.

use bevy::prelude::*;

// ---------------------------------------------------------------------------
// Colour palette
// ---------------------------------------------------------------------------

/// Near-white primary text (slightly cool, like the reference frostwork).
pub const TEXT_PRIMARY: Color = Color::srgba(0.97, 0.975, 0.98, 1.0);
/// Secondary text: descriptions, sublabels, inactive values. Translucent
/// white, not opaque grey — it stays legible over glass of any brightness.
pub const TEXT_DIM: Color = Color::srgba(1.0, 1.0, 1.0, 0.72);
/// Faint text: placeholders, disabled labels, fine print, headings.
pub const TEXT_FAINT: Color = Color::srgba(1.0, 1.0, 1.0, 0.48);
/// The one warm accent — Thalos amber, lifted and aired out.
pub const ACCENT: Color = Color::srgba(1.0, 0.78, 0.40, 1.0);
/// Dimmed accent for secondary accent uses (latched toggles at rest).
pub const ACCENT_DIM: Color = Color::srgba(0.80, 0.64, 0.36, 1.0);
/// Text drawn on top of an accent-filled control.
pub const ON_ACCENT: Color = Color::srgba(0.10, 0.08, 0.04, 1.0);
/// Destructive / warning.
pub const DANGER: Color = Color::srgba(1.0, 0.45, 0.35, 1.0);
/// Positive / confirmed.
pub const OK: Color = Color::srgba(0.55, 0.88, 0.60, 1.0);

/// Opaque backdrop for world-less screens (loading, deferred-boot menu) —
/// the same cool near-black the glass tint resolves to over an empty scene.
pub const SCREEN_BG: Color = Color::srgb(0.016, 0.020, 0.028);

/// Hairline stroke on glass edges — whisper-quiet; the surface reads from
/// its blur and fill, not its outline.
pub const STROKE: Color = Color::srgba(1.0, 1.0, 1.0, 0.09);
/// Brighter stroke: focus outlines and latched accents.
pub const STROKE_BRIGHT: Color = Color::srgba(1.0, 1.0, 1.0, 0.30);

/// Transparent-white interaction fills, layered *on top of* glass.
/// Controls are **fill-based** (visionOS-style): a soft resting fill, no
/// outline; hierarchy comes from fill brightness.
pub const FILL_REST: Color = Color::srgba(1.0, 1.0, 1.0, 0.07);
pub const FILL_HOVER: Color = Color::srgba(1.0, 1.0, 1.0, 0.13);
pub const FILL_ACTIVE: Color = Color::srgba(1.0, 1.0, 1.0, 0.20);
/// Selected row/item fill (accent-tinted).
pub const FILL_SELECTED: Color = Color::srgba(1.0, 0.78, 0.40, 0.16);

/// Glass tint (the colour mixed over the blurred backdrop). Luminous, not
/// black — the panel takes on the scene's light through the blur; a cool
/// mid-dark tint keeps white text readable. `w` is the tint opacity over the
/// blur, not the panel's final alpha.
pub const GLASS_TINT: Vec4 = Vec4::new(0.075, 0.085, 0.105, 0.58);
/// Stronger variant for overlay dialogs that must dominate the scene.
pub const GLASS_TINT_STRONG: Vec4 = Vec4::new(0.05, 0.06, 0.08, 0.72);

/// The floating-sheet drop shadow under glass panels.
pub const PANEL_SHADOW: Color = Color::srgba(0.0, 0.0, 0.0, 0.30);

// ---------------------------------------------------------------------------
// Spacing / radii / sizes
// ---------------------------------------------------------------------------

pub const SPACE_XS: f32 = 4.0;
pub const SPACE_SM: f32 = 8.0;
pub const SPACE_MD: f32 = 12.0;
pub const SPACE_LG: f32 = 16.0;
pub const SPACE_XL: f32 = 24.0;

/// Panel corner radius — generous, sheet-like.
pub const RADIUS_PANEL: f32 = 16.0;
/// Control (button/field/row) corner radius.
pub const RADIUS_CTRL: f32 = 9.0;

/// Standard control heights: compact rows everywhere.
pub const CTRL_H: f32 = 26.0;
pub const CTRL_H_SM: f32 = 22.0;
pub const CTRL_H_LG: f32 = 34.0;

// ---------------------------------------------------------------------------
// Type scale
// ---------------------------------------------------------------------------

/// Logotype / hero display.
pub const FS_DISPLAY: f32 = 52.0;
/// Screen titles ("SPACE CENTER").
pub const FS_TITLE: f32 = 20.0;
/// Section headings (caps, semibold, faint).
pub const FS_HEADING: f32 = 11.0;
/// Standard interface text.
pub const FS_BODY: f32 = 13.0;
/// Secondary text / descriptions.
pub const FS_SMALL: f32 = 11.0;
/// Mono readouts.
pub const FS_MONO: f32 = 12.0;

// ---------------------------------------------------------------------------
// Theme resource
// ---------------------------------------------------------------------------

/// Font handles + shared glass material handles, loaded once at startup by
/// [`init_ui_theme`](crate::init_ui_theme). Colours and metrics are consts
/// above — only asset-backed values live on the resource.
#[derive(Resource, Clone)]
pub struct UiTheme {
    /// Titillium Light — hero/display text.
    pub font_display: FontSource,
    /// Titillium Regular — all standard interface text.
    pub font_ui: FontSource,
    /// Titillium SemiBold — titles, headings, button labels.
    pub font_strong: FontSource,
    /// Fira Code — numeric/mono readouts.
    pub font_mono: FontSource,
    /// Shared frosted-glass panel material (regular tint).
    pub glass_regular: Handle<crate::glass::GlassMaterial>,
    /// Shared frosted-glass material for dominant overlays (dialogs).
    pub glass_strong: Handle<crate::glass::GlassMaterial>,
}

impl UiTheme {
    // -- text bundle helpers ------------------------------------------------

    fn text(
        &self,
        content: impl Into<String>,
        font: FontSource,
        size: f32,
        color: Color,
    ) -> (Text, TextFont, TextColor) {
        (
            Text::new(content),
            TextFont {
                font,
                font_size: FontSize::Px(size),
                ..Default::default()
            },
            TextColor(color),
        )
    }

    /// Hero display text (logotype).
    pub fn display(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_display.clone(), FS_DISPLAY, TEXT_PRIMARY)
    }

    /// Screen title ("SPACE CENTER").
    pub fn title(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_strong.clone(), FS_TITLE, TEXT_PRIMARY)
    }

    /// Section heading — caps by convention, faint, semibold.
    pub fn heading(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_strong.clone(), FS_HEADING, TEXT_FAINT)
    }

    /// Standard body text.
    pub fn body(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_ui.clone(), FS_BODY, TEXT_PRIMARY)
    }

    /// Emphasised body text (semibold).
    pub fn body_strong(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_strong.clone(), FS_BODY, TEXT_PRIMARY)
    }

    /// Secondary text.
    pub fn small(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_ui.clone(), FS_SMALL, TEXT_DIM)
    }

    /// Faint fine print.
    pub fn faint(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_ui.clone(), FS_SMALL, TEXT_FAINT)
    }

    /// Mono/numeric readout.
    pub fn mono(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_mono.clone(), FS_MONO, TEXT_PRIMARY)
    }

    /// Mono/numeric readout, dim.
    pub fn mono_dim(&self, content: impl Into<String>) -> (Text, TextFont, TextColor) {
        self.text(content, self.font_mono.clone(), FS_MONO, TEXT_DIM)
    }
}
