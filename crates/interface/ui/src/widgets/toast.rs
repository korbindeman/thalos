//! Transient toast notifications — top-centre pills that fade out.
//!
//! Replaces persistent status-bar lines: fire-and-forget feedback
//! ("Saved 'Meridian'", "Nothing to launch") that doesn't cost screen space.

use bevy::prelude::*;

use crate::UiTheme;
use crate::tokens::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ToastKind {
    #[default]
    Info,
    Success,
    Warn,
}

/// A live toast; despawned by [`update_toasts`] when its time is up.
#[derive(Component)]
pub struct Toast {
    age: f32,
    lifetime: f32,
}

/// The container toasts stack into (spawned by the plugin).
#[derive(Component)]
#[require(thalos_photo_mode::HideInPhotoMode)]
pub struct ToastArea;

pub(crate) fn setup_toast_area(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(64.0),
            left: Val::Px(0.0),
            right: Val::Px(0.0),
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: Val::Px(SPACE_SM),
            ..Default::default()
        },
        Pickable::IGNORE,
        GlobalZIndex(950),
        ToastArea,
        Name::new("UiToastArea"),
    ));
}

/// Fire a toast. Call from any system with `Commands` + the area query.
pub fn spawn_toast(
    commands: &mut Commands,
    area: Entity,
    theme: &UiTheme,
    message: impl Into<String>,
    kind: ToastKind,
) {
    let color = match kind {
        ToastKind::Info => TEXT_PRIMARY,
        ToastKind::Success => OK,
        ToastKind::Warn => DANGER,
    };
    let mut text = theme.body(message);
    text.2 = TextColor(color);
    commands.entity(area).with_children(|area| {
        area.spawn((
            Node {
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(999.0)),
                padding: UiRect::axes(Val::Px(SPACE_LG), Val::Px(6.0)),
                align_items: AlignItems::Center,
                ..Default::default()
            },
            BackgroundColor(Color::srgba(0.02, 0.025, 0.032, 0.92)),
            BorderColor::all(STROKE_BRIGHT),
            Pickable::IGNORE,
            Toast {
                age: 0.0,
                lifetime: 2.6,
            },
        ))
        .with_children(|pill| {
            pill.spawn(text);
        });
    });
}

/// Age, fade, and despawn toasts. Uses real time — toasts outlive sim pause.
pub fn update_toasts(
    mut commands: Commands,
    time: Res<Time<Real>>,
    mut toasts: Query<(
        Entity,
        &mut Toast,
        &mut BackgroundColor,
        &mut BorderColor,
        &Children,
    )>,
    mut texts: Query<&mut TextColor>,
) {
    for (entity, mut toast, mut bg, mut border, children) in &mut toasts {
        toast.age += time.delta_secs();
        if toast.age >= toast.lifetime {
            commands.entity(entity).despawn();
            continue;
        }
        // Hold, then fade over the last 0.6 s.
        let fade_start = toast.lifetime - 0.6;
        let fade = if toast.age > fade_start {
            1.0 - (toast.age - fade_start) / 0.6
        } else {
            1.0
        };
        bg.0 = bg.0.with_alpha(0.92 * fade);
        *border = BorderColor::all(STROKE_BRIGHT.with_alpha(0.38 * fade));
        for child in children.iter() {
            if let Ok(mut tc) = texts.get_mut(child) {
                tc.0 = tc.0.with_alpha(fade);
            }
        }
    }
}
