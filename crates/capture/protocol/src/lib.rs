//! Portable control-plane types shared by capture clients and the renderer.
//!
//! This crate intentionally contains no ECS, GPU, camera, or application
//! types. It describes requests and results; `thalos_capture_runtime` turns
//! them into work performed by the complete Bevy game renderer.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const CAPTURE_PROTOCOL_SCHEMA: u32 = 1;

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureAction {
    #[default]
    Capture,
    Shutdown,
    #[serde(other)]
    Unsupported,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureRequest {
    pub schema_version: u32,
    pub id: String,
    #[serde(default)]
    pub action: CaptureAction,
    #[serde(default)]
    pub preset: String,
    #[serde(default)]
    pub overrides: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureResponse {
    pub schema_version: u32,
    pub id: String,
    pub ok: bool,
    pub message: String,
    pub output: Option<String>,
    pub completed_unix_ms: u128,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CaptureServerState {
    pub schema_version: u32,
    pub pid: u32,
    pub preset: String,
    pub width: u32,
    pub height: u32,
    pub ready: bool,
    pub busy: bool,
    pub completed_captures: u64,
    pub code_reload_unix_ms: u128,
    pub shader_reload_unix_ms: u128,
    pub heartbeat_unix_ms: u128,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_capture_request_deserializes() {
        let request: CaptureRequest = serde_json::from_str(
            r#"{"schema_version":1,"id":"abc","action":"capture","preset":"hub","overrides":{}}"#,
        )
        .expect("deserialize request");
        assert_eq!(request.action, CaptureAction::Capture);
        assert_eq!(request.preset, "hub");
    }
}
