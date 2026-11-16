// OHOS backend. Formerly a single ~3200-line mod.rs; split by concern
// (tao issue #84, item 6), with this module kept as a re-export facade so all
// `crate::platform_impl::ohos::*` / `crate::platform_impl::*` paths are
// unchanged:
//
// - `event_loop`: EventLoop/Proxy/WindowTarget, input-event translation,
//   run_loop dispatch, theme/cursor/monitor shared helpers.
// - `window`: Window (builder attrs → setters → drop), WindowId, the
//   windowStatusChange state mirror registry, scancode stubs.
// - `monitor`: MonitorHandle/VideoMode — per-display handles over the
//   openharmony-ability display enumeration (multi-display, issue #106).
// - `keycodes`: OHOS keycode → tao Key/KeyLocation mapping.
//
// NOTE: the former `decor_watch` module (per-window set_inner_size decor
// self-correction) was removed by issue Eulogizethesun/tauri#97 — set_inner_size
// now dispatches through the precise ArkTS-side `resize-inner` bridge call
// (getWindowProperties snapshot) and needs no post-hoc correction.

mod event_loop;
mod keycodes;
mod monitor;
mod window;

pub use event_loop::{DeviceId, EventLoop, EventLoopProxy, EventLoopWindowTarget, KeyEventExtra};
pub use monitor::{MonitorHandle, VideoMode};
pub use window::{
  drain_destroy_failures, keycode_from_scancode, keycode_to_scancode, OHOSWindowKind, OsError,
  PlatformSpecificWindowBuilderAttributes,
};

pub(crate) use event_loop::PlatformSpecificEventLoopAttributes;
pub(crate) use window::{Window, WindowId};

pub(crate) use crate::icon::NoIcon as PlatformIcon;
