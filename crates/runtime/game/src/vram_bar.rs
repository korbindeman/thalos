//! Segmented whole-card VRAM bar — one widget, two screens.
//!
//! Answers "how full is the card, and what put it there" in a glance rather
//! than a table. The loading screen and the F3 debug view both spawn it, so the
//! decomposition and its honesty caveats live in exactly one place.
//!
//! # The decomposition
//!
//! The bar's full width is the card's **total** VRAM. The filled part is what
//! the driver reports as used — by every process, not just this one, which is
//! the whole reason it is worth showing (Thalos is routinely run two instances
//! at a time, and per-process accounting cannot see the peer eating the
//! headroom — INC-20260725T012104Z). It is split into four:
//!
//! - **terrain** — `TileTerrainRoot::resident_bytes`, the tile meshes.
//! - **meshes** — the rest of Bevy's mesh-allocator slabs. Slabs *contain* the
//!   tile meshes, so this is `slabs − terrain`, never the raw slab figure;
//!   showing both un-subtracted would double-count the ground.
//! - **textures** — estimated bytes of all `GpuImage` asset textures (terrain
//!   package charts, vegetation atlases, LUTs); mesh slabs hold only buffers,
//!   so this overlaps neither segment above.
//! - **other** — everything left in the driver's number: render targets,
//!   shadow maps, the swapchain, pipelines, plus every other process on the
//!   card. Deliberately labelled as unattributed rather than dressed up as a
//!   measurement.
//!
//! Only the first three are measured. If `other` dominates beyond the desktop's
//! usual share, that is the finding — it is what a render-target leak or a
//! second Thalos instance looks like.
//!
//! Without a whole-card reading (non-Windows, no NVIDIA driver, before the
//! first poll) there is no denominator, so the bar hides itself entirely and
//! the headline says so. A bar with an invented total would be worse than none.

use bevy::prelude::*;

use crate::perf::{PerfSamples, fmt_bytes, fmt_mib};

/// Refresh cadence. The inputs move at 2 Hz (`gpu_memory`'s poller) and 2 Hz
/// (`PerfSamples`' gauge), so a faster bar would animate noise.
const REFRESH_S: f32 = 0.25;

/// One contributor, in draw order left to right.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
pub enum VramPart {
    /// Tile-terrain meshes.
    Terrain,
    /// Mesh-allocator slabs beyond the terrain's share.
    Meshes,
    /// `GpuImage` asset textures (estimated from their descriptors).
    Textures,
    /// Everything the driver counts that we cannot attribute.
    Other,
}

impl VramPart {
    const ALL: [VramPart; 4] = [
        VramPart::Terrain,
        VramPart::Meshes,
        VramPart::Textures,
        VramPart::Other,
    ];

    fn label(self) -> &'static str {
        match self {
            VramPart::Terrain => "terrain",
            VramPart::Meshes => "meshes",
            VramPart::Textures => "textures",
            VramPart::Other => "other",
        }
    }

    /// A gold ramp for what we own and can act on, neutral grey for what we
    /// cannot attribute — so a bar dominated by grey reads as "not us"
    /// without needing the legend.
    fn color(self) -> Color {
        match self {
            VramPart::Terrain => thalos_ui::tokens::ACCENT,
            VramPart::Meshes => thalos_ui::tokens::ACCENT_DIM,
            VramPart::Textures => thalos_ui::tokens::ACCENT_FAINT,
            VramPart::Other => Color::srgba(1.0, 1.0, 1.0, 0.22),
        }
    }
}

/// Unused VRAM: the track showing through behind the segments.
const FREE_COLOR: Color = Color::srgba(1.0, 1.0, 1.0, 0.07);

/// The `4.8 / 12.0 GiB (40 %)` headline.
#[derive(Component)]
struct VramHeadline;

/// A bar segment; its `Node.width` is a percentage of the card total.
#[derive(Component)]
struct VramSegment(VramPart);

/// A legend entry's value text (`terrain 512 MiB`).
#[derive(Component)]
struct VramLegendValue(VramPart);

/// Root of the whole widget — hidden wholesale when there is no card reading.
#[derive(Component)]
struct VramBarRoot;

/// How the host screen wants the widget drawn. The loading screen and the F3
/// panel have different type scales and different palettes (the loading screen
/// may not touch `HudTheme` — it renders on frame 1, before the theme resource
/// exists), so the widget takes both rather than reaching for a global.
pub struct VramBarStyle {
    pub width: Val,
    pub bar_height: f32,
    pub font: FontSource,
    pub font_size: f32,
    /// The `VRAM` caption and the legend labels.
    pub label_color: Color,
    /// The used/total headline.
    pub value_color: Color,
    /// Caption text to the left of the headline. Empty hides the caption row's
    /// label, keeping the headline (F3 puts the word in its section header).
    pub caption: &'static str,
}

/// Spawn the widget as a child of `parent`.
pub fn spawn_vram_bar(parent: &mut ChildSpawnerCommands<'_>, style: &VramBarStyle) {
    parent
        .spawn((
            VramBarRoot,
            Node {
                width: style.width,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            Name::new("VramBar"),
        ))
        .with_children(|root| {
            // Caption + headline, pushed to opposite ends.
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    flex_direction: FlexDirection::Row,
                    justify_content: JustifyContent::SpaceBetween,
                    ..default()
                },
                Name::new("VramBarHeader"),
            ))
            .with_children(|header| {
                header.spawn((
                    Text::new(style.caption),
                    TextFont {
                        font: style.font.clone(),
                        font_size: FontSize::Px(style.font_size),
                        ..default()
                    },
                    TextColor(style.label_color),
                ));
                header.spawn((
                    VramHeadline,
                    Text::new("—"),
                    TextFont {
                        font: style.font.clone(),
                        font_size: FontSize::Px(style.font_size),
                        ..default()
                    },
                    TextColor(style.value_color),
                ));
            });

            // The bar: a track with three left-aligned segments over it.
            root.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(style.bar_height),
                    border_radius: BorderRadius::all(Val::Px(style.bar_height * 0.5)),
                    flex_direction: FlexDirection::Row,
                    align_items: AlignItems::Stretch,
                    overflow: Overflow::clip(),
                    ..default()
                },
                BackgroundColor(FREE_COLOR),
                Name::new("VramBarTrack"),
            ))
            .with_children(|track| {
                for part in VramPart::ALL {
                    track.spawn((
                        VramSegment(part),
                        Node {
                            width: Val::Percent(0.0),
                            height: Val::Percent(100.0),
                            ..default()
                        },
                        BackgroundColor(part.color()),
                    ));
                }
            });

            // Legend: swatch + value per contributor.
            root.spawn((
                Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(12.0),
                    align_items: AlignItems::Center,
                    ..default()
                },
                Name::new("VramBarLegend"),
            ))
            .with_children(|legend| {
                for part in VramPart::ALL {
                    legend
                        .spawn((
                            Node {
                                flex_direction: FlexDirection::Row,
                                align_items: AlignItems::Center,
                                column_gap: Val::Px(4.0),
                                ..default()
                            },
                            Name::new("VramBarLegendItem"),
                        ))
                        .with_children(|item| {
                            let swatch = (style.font_size * 0.55).max(5.0);
                            item.spawn((
                                Node {
                                    width: Val::Px(swatch),
                                    height: Val::Px(swatch),
                                    border_radius: BorderRadius::all(Val::Px(1.5)),
                                    ..default()
                                },
                                BackgroundColor(part.color()),
                            ));
                            item.spawn((
                                VramLegendValue(part),
                                Text::new(part.label()),
                                TextFont {
                                    font: style.font.clone(),
                                    font_size: FontSize::Px(style.font_size),
                                    ..default()
                                },
                                TextColor(style.label_color),
                            ));
                        });
                }
            });
        });
}

pub struct VramBarPlugin;

impl Plugin for VramBarPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, update_vram_bars);
    }
}

/// The measured slice of the card, in MiB.
struct Contributions {
    terrain: f32,
    meshes: f32,
    textures: f32,
    other: f32,
}

/// Split `card_used_mib` into the four parts. Each measured figure is clamped
/// into the driver's total: they are accounted independently, and a gauge
/// sampled a beat apart from the card reading must never produce a segment
/// wider than the bar.
fn contributions(card_used_mib: f32, samples: Option<&PerfSamples>) -> Contributions {
    let (terrain, slabs, textures) = match samples {
        Some(samples) => (samples.tile_mib, samples.slab_mib(), samples.texture_mib),
        None => (0.0, 0.0, 0.0),
    };
    let terrain = terrain.clamp(0.0, card_used_mib);
    // Slabs contain the tile meshes; subtracting keeps the ground from being
    // drawn twice. Textures are a disjoint pool (slabs hold only buffers).
    let meshes = (slabs - terrain).clamp(0.0, card_used_mib - terrain);
    let textures = textures.clamp(0.0, card_used_mib - terrain - meshes);
    Contributions {
        terrain,
        meshes,
        textures,
        other: (card_used_mib - terrain - meshes - textures).max(0.0),
    }
}

fn update_vram_bars(
    time: Res<Time<Real>>,
    mut until_refresh_s: Local<f32>,
    samples: Option<Res<PerfSamples>>,
    mut roots: Query<&mut Visibility, With<VramBarRoot>>,
    mut headlines: Query<&mut Text, (With<VramHeadline>, Without<VramLegendValue>)>,
    mut segments: Query<(&VramSegment, &mut Node)>,
    mut legends: Query<(&VramLegendValue, &mut Text), Without<VramHeadline>>,
) {
    if roots.is_empty() {
        return;
    }
    *until_refresh_s -= time.delta_secs();
    if *until_refresh_s > 0.0 {
        return;
    }
    *until_refresh_s = REFRESH_S;

    let card = thalos_diagnostics::gpu_memory();

    // No denominator, no bar. Everything else on the host screen still stands.
    let visibility = if card.is_some() {
        Visibility::Inherited
    } else {
        Visibility::Hidden
    };
    for mut vis in &mut roots {
        if *vis != visibility {
            *vis = visibility;
        }
    }
    let Some(card) = card else {
        for mut text in &mut headlines {
            let unavailable = "no card reading".to_string();
            if text.0 != unavailable {
                **text = unavailable;
            }
        }
        return;
    };

    let total_mib = card.total_bytes as f32 / (1024.0 * 1024.0);
    let used_mib = card.used_bytes as f32 / (1024.0 * 1024.0);
    let parts = contributions(used_mib, samples.as_deref());

    for mut text in &mut headlines {
        let headline = format!(
            "{} / {} ({:.0} %)",
            fmt_bytes(card.used_bytes),
            fmt_bytes(card.total_bytes),
            card.used_frac() * 100.0
        );
        if text.0 != headline {
            **text = headline;
        }
    }

    let percent = |mib: f32| {
        if total_mib <= 0.0 {
            0.0
        } else {
            (mib / total_mib * 100.0).clamp(0.0, 100.0)
        }
    };
    for (segment, mut node) in &mut segments {
        let width = Val::Percent(percent(match segment.0 {
            VramPart::Terrain => parts.terrain,
            VramPart::Meshes => parts.meshes,
            VramPart::Textures => parts.textures,
            VramPart::Other => parts.other,
        }));
        if node.width != width {
            node.width = width;
        }
    }

    for (legend, mut text) in &mut legends {
        let mib = match legend.0 {
            VramPart::Terrain => parts.terrain,
            VramPart::Meshes => parts.meshes,
            VramPart::Textures => parts.textures,
            VramPart::Other => parts.other,
        };
        let line = format!("{} {}", legend.0.label(), fmt_mib(mib));
        if text.0 != line {
            **text = line;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slabs_are_not_double_counted_against_terrain() {
        // 2 GiB of slabs of which 1.5 GiB is terrain: the mesh segment carries
        // the 0.5 GiB remainder, not the full slab figure. Textures are a
        // disjoint pool and subtract from "other" alone.
        let mut samples = PerfSamples::default();
        samples.seed_gauges(0, 0, 0, 1536.0, 2048.0, 1024.0, 0.0, 0.0, 0.0);
        let parts = contributions(6144.0, Some(&samples));
        assert_eq!(parts.terrain, 1536.0);
        assert_eq!(parts.meshes, 512.0);
        assert_eq!(parts.textures, 1024.0);
        assert_eq!(parts.other, 6144.0 - 2048.0 - 1024.0);
    }

    #[test]
    fn segments_never_exceed_the_card_reading() {
        // The gauges and the card reading are sampled independently, so a
        // gauge that momentarily exceeds the driver's used figure must clamp
        // rather than produce a segment wider than the bar.
        let mut samples = PerfSamples::default();
        samples.seed_gauges(0, 0, 0, 8192.0, 9000.0, 2048.0, 0.0, 0.0, 0.0);
        let parts = contributions(4096.0, Some(&samples));
        let total = parts.terrain + parts.meshes + parts.textures + parts.other;
        assert!(
            total <= 4096.0 + f32::EPSILON,
            "segments summed to {total}, past the 4096 MiB reading"
        );
    }

    #[test]
    fn without_gauges_the_whole_card_is_unattributed() {
        let parts = contributions(4096.0, None);
        assert_eq!(parts.terrain, 0.0);
        assert_eq!(parts.meshes, 0.0);
        assert_eq!(parts.textures, 0.0);
        assert_eq!(parts.other, 4096.0);
    }
}
