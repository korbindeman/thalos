//! Shared clean-view state and visibility arbitration for Thalos applications.
//!
//! Applications own their input adapters. This crate owns the one photo-mode
//! state and the rule that marked scene overlays stay hidden until the mode is
//! exited, even when another UI system refreshes its ordinary visibility.

use bevy::{camera::visibility::VisibilitySystems, prelude::*};

/// Global clean-view state.
#[derive(Resource, Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhotoMode {
    pub active: bool,
}

impl PhotoMode {
    pub fn toggle(&mut self) {
        self.active = !self.active;
    }
}

/// Marks an entity that must be hidden while photo mode is active.
///
/// Put this on an overlay root so visibility inheritance hides its children.
#[derive(Component, Debug, Default, Clone, Copy)]
#[require(Visibility)]
pub struct HideInPhotoMode;

/// The visibility that photo mode must restore when it releases an entity.
#[derive(Component, Debug, Clone, Copy)]
struct VisibilityBeforePhotoMode(Visibility);

/// True while photo mode is inactive.
pub fn not_in_photo_mode(photo_mode: Res<PhotoMode>) -> bool {
    !photo_mode.active
}

/// Installs the canonical photo-mode state and visibility arbiter.
///
/// The application still decides which input toggles [`PhotoMode`] and which
/// modal states are allowed to consume that input.
pub struct PhotoModePlugin;

impl Plugin for PhotoModePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<PhotoMode>().add_systems(
            PostUpdate,
            apply_photo_mode_visibility.before(VisibilitySystems::VisibilityPropagate),
        );
    }
}

fn apply_photo_mode_visibility(
    mut commands: Commands,
    photo_mode: Res<PhotoMode>,
    mut overlays: Query<
        (Entity, &mut Visibility, Option<&VisibilityBeforePhotoMode>),
        With<HideInPhotoMode>,
    >,
) {
    for (entity, mut visibility, saved) in &mut overlays {
        if photo_mode.active {
            if saved.is_none() {
                commands
                    .entity(entity)
                    .insert(VisibilityBeforePhotoMode(*visibility));
            }
            if *visibility != Visibility::Hidden {
                *visibility = Visibility::Hidden;
            }
        } else if let Some(saved) = saved {
            *visibility = saved.0;
            commands
                .entity(entity)
                .remove::<VisibilityBeforePhotoMode>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restores_each_overlays_previous_visibility() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, PhotoModePlugin));
        let inherited = app.world_mut().spawn(HideInPhotoMode).id();
        let hidden = app
            .world_mut()
            .spawn((HideInPhotoMode, Visibility::Hidden))
            .id();

        app.world_mut().resource_mut::<PhotoMode>().active = true;
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(inherited),
            Some(&Visibility::Hidden)
        );
        assert_eq!(
            app.world().get::<Visibility>(hidden),
            Some(&Visibility::Hidden)
        );

        app.world_mut().resource_mut::<PhotoMode>().active = false;
        app.update();
        assert_eq!(
            app.world().get::<Visibility>(inherited),
            Some(&Visibility::Inherited)
        );
        assert_eq!(
            app.world().get::<Visibility>(hidden),
            Some(&Visibility::Hidden)
        );
    }

    #[test]
    fn photo_mode_wins_over_an_ordinary_visibility_refresh() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, PhotoModePlugin));
        let overlay = app.world_mut().spawn(HideInPhotoMode).id();
        app.world_mut().resource_mut::<PhotoMode>().active = true;
        app.update();

        *app.world_mut()
            .get_mut::<Visibility>(overlay)
            .expect("required visibility") = Visibility::Inherited;
        app.update();

        assert_eq!(
            app.world().get::<Visibility>(overlay),
            Some(&Visibility::Hidden)
        );
    }
}
