use bevy_enhanced_input::prelude::{ActionSources, ContextActivity};

pub fn set_mouse_sources(action_sources: &mut ActionSources, enabled: bool) {
    action_sources.mouse_buttons = enabled;
    action_sources.mouse_motion = enabled;
    action_sources.mouse_wheel = enabled;
}

pub fn set_keyboard_source(action_sources: &mut ActionSources, enabled: bool) {
    action_sources.keyboard = enabled;
}

pub fn context_activity<C>(active: bool) -> ContextActivity<C> {
    ContextActivity::new(active)
}
