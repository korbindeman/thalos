//! Raw HOTAS / joystick axis reader.
//!
//! Bevy's `bevy_gilrs` converter (`convert_axis`) maps only the standard named
//! gamepad axes (`LeftStickX`, `RightZ`, …) and discards every axis gilrs
//! labels `gilrs::Axis::Unknown`. A flight stick with no SDL gamepad profile —
//! e.g. the Thrustmaster T.16000M — reports its twist (rudder) and throttle
//! slider as `Unknown`, so those axes never reach Bevy's gamepad state and
//! cannot be bound through `GamepadAxis`.
//!
//! To recover them we run our **own** private `gilrs::Gilrs` instance alongside
//! Bevy's and read every axis by its raw platform `Code` (`Code::into_u32`).
//! Each frame [`poll_joysticks`] drains the private instance's event queue
//! (which updates its internal state) and snapshots `code -> value` for every
//! connected device into [`RawJoystickState`]. HOTAS bindings in
//! `assets/input.ron` reference these raw `u32` codes directly.
//!
//! Note: raw codes are platform-specific (a Windows code does not match a Linux
//! one), which is fine — bindings are authored per machine. The standalone
//! `cargo run -p thalos_input --example gamepad_axes` probe prints the codes.

use bevy::prelude::*;
use gilrs::Gilrs;

use crate::settings::HotasDeviceSelector;

/// Private gilrs instance used purely to read raw axes Bevy drops. Held as a
/// non-send resource because `gilrs::Gilrs` is `!Sync`.
pub struct Joysticks {
    gilrs: Gilrs,
}

/// Per-frame snapshot of raw axis values for every connected device, keyed by
/// the raw platform axis code (`gilrs::Code::into_u32`).
///
/// **Sole writer:** [`poll_joysticks`].
#[derive(Resource, Default)]
pub struct RawJoystickState {
    pub devices: Vec<RawJoystickDevice>,
}

pub struct RawJoystickDevice {
    pub name: String,
    pub vendor_id: Option<u16>,
    pub product_id: Option<u16>,
    /// `(raw code, current value in [-1, 1])` for each axis that has reported.
    pub axes: Vec<(u32, f32)>,
}

impl RawJoystickState {
    /// Raw value of `code` on the first device matching `selector`, if present.
    pub fn axis(&self, selector: &HotasDeviceSelector, code: u32) -> Option<f32> {
        self.devices
            .iter()
            .find(|device| device_matches(selector, device))
            .and_then(|device| {
                device
                    .axes
                    .iter()
                    .find(|(c, _)| *c == code)
                    .map(|(_, value)| *value)
            })
    }
}

fn device_matches(selector: &HotasDeviceSelector, device: &RawJoystickDevice) -> bool {
    match selector {
        HotasDeviceSelector::Any => true,
        HotasDeviceSelector::NameContains(needle) => device
            .name
            .to_ascii_lowercase()
            .contains(&needle.to_ascii_lowercase()),
        HotasDeviceSelector::Usb {
            vendor_id,
            product_id,
        } => {
            device.vendor_id == Some(*vendor_id)
                && product_id.is_none_or(|id| device.product_id == Some(id))
        }
    }
}

/// Create the private gilrs instance. Returns `None` (with a logged warning) if
/// the platform backend fails to start; HOTAS axes then simply do nothing.
pub fn init_joysticks() -> Option<Joysticks> {
    match Gilrs::new() {
        Ok(gilrs) => Some(Joysticks { gilrs }),
        Err(err) => {
            warn!(
                "HOTAS: could not start private gilrs instance: {err}; raw joystick axes disabled"
            );
            None
        }
    }
}

/// Drain the private gilrs event queue (updating its internal state) and
/// snapshot every connected device's raw axes into [`RawJoystickState`].
pub fn poll_joysticks(
    joysticks: Option<NonSendMut<Joysticks>>,
    mut state: ResMut<RawJoystickState>,
) {
    let Some(mut joysticks) = joysticks else {
        return;
    };
    // Pumping events is what advances gilrs's internal axis state.
    while joysticks.gilrs.next_event().is_some() {}

    state.devices.clear();
    for (_id, gamepad) in joysticks.gilrs.gamepads() {
        let axes = gamepad
            .state()
            .axes()
            .map(|(code, data)| (code.into_u32(), data.value()))
            .collect();
        state.devices.push(RawJoystickDevice {
            name: gamepad.name().to_string(),
            vendor_id: gamepad.vendor_id(),
            product_id: gamepad.product_id(),
            axes,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> RawJoystickState {
        RawJoystickState {
            devices: vec![RawJoystickDevice {
                name: "T16000M".to_string(),
                vendor_id: Some(0x044f),
                product_id: Some(0xb10a),
                axes: vec![(65536, 0.1), (65538, -0.7)],
            }],
        }
    }

    #[test]
    fn axis_matches_by_name_substring_case_insensitive() {
        let state = state();
        let sel = HotasDeviceSelector::NameContains("t16000".to_string());
        assert_eq!(state.axis(&sel, 65538), Some(-0.7));
        assert_eq!(state.axis(&sel, 65536), Some(0.1));
    }

    #[test]
    fn axis_returns_none_for_unknown_code_or_unmatched_device() {
        let state = state();
        assert_eq!(state.axis(&HotasDeviceSelector::Any, 99999), None);
        let other = HotasDeviceSelector::NameContains("warthog".to_string());
        assert_eq!(state.axis(&other, 65538), None);
    }

    #[test]
    fn axis_matches_by_usb_ids() {
        let state = state();
        let sel = HotasDeviceSelector::Usb {
            vendor_id: 0x044f,
            product_id: Some(0xb10a),
        };
        assert_eq!(state.axis(&sel, 65538), Some(-0.7));
        // Vendor-only match (product unspecified) also resolves.
        let vendor_only = HotasDeviceSelector::Usb {
            vendor_id: 0x044f,
            product_id: None,
        };
        assert_eq!(state.axis(&vendor_only, 65536), Some(0.1));
    }
}
