//! # OpenHarmony
//!
//! The OpenHarmony backend builds on (and exposes types from) the [`ohos-rs`](https://docs.rs/ohos-rs/) crate.
//!
//! Native OpenHarmony applications need some form of "glue" crate that is responsible
//! for defining the main entry point for your Rust application as well as tracking
//! various life-cycle events and synchronizing with the main thread.
//!
//! tao uses the [openharmony-ability](https://docs.rs/openharmony-ability/) as a
//! glue crate.
//!
//! ## Event loop semantics
//!
//! OHOS is callback-driven: the main thread must be handed back to ArkTS, so
//! [`EventLoop::run`] registers the event handler and **returns immediately**
//! — it does not block until the app exits like on desktop platforms. Events
//! are dispatched from ArkTS lifecycle callbacks afterwards.
//!
//! Consequences for [`ControlFlow`](crate::event_loop::ControlFlow):
//!
//! - Setting `ControlFlow::Exit` from the handler terminates the UIAbility at
//!   the end of the current event dispatch (it does not make `run` return —
//!   `run` has already returned).
//! - `EventLoopExtRunReturn` is not implemented on OHOS (the same as on iOS):
//!   a "run until exit, then give the caller back control" loop cannot exist
//!   when the main thread belongs to ArkTS. Runtimes alias `run_return` to
//!   `run` on OHOS.
//!

#![cfg(target_env = "ohos")]

use crate::event_loop::{EventLoop, EventLoopBuilder};
use crate::window::{Window, WindowAttributes, WindowBuilder};
use openharmony_ability::{Configuration, OpenHarmonyApp, Rect};

/// Additional methods on [`EventLoop`] that are specific to OpenHarmony.
pub trait EventLoopExtOpenHarmony {}

impl<T> EventLoopExtOpenHarmony for EventLoop<T> {}

/// Additional methods on [`Window`] that are specific to OpenHarmony.
pub trait WindowExtOpenHarmony {
  fn content_rect(&self) -> Rect;

  fn config(&self) -> Configuration;

  /// Returns the OS-level window ID, used to distinguish main (0) vs sub-windows.
  fn window_id(&self) -> Option<i64>;

  /// Returns the resolved OHOS window kind (issue Eulogizethesun/tauri#238):
  /// `UIAbility` for UIAbility-container windows (the bootstrap main window
  /// and spawned UIAbility instances), `SubWindow` for sub-windows created
  /// via `windowStage.createSubWindow` (issue Eulogizethesun/tauri#240).
  /// Consumed by the embedding runtime (tauri-runtime-wry) to capture
  /// `is_ui_ability_main` and gate UIAbility-kind-window operations (the
  /// bootstrap main window and spawned instances) at the dispatcher layer.
  fn window_kind(&self) -> OHOSWindowKind;

  /// Returns the `BridgeRuntime` for constructing bridge-based facade clients
  /// (e.g. wry's `WebviewClient::from_bridge`).
  fn bridge_runtime(&self) -> openharmony_ability::BridgeRuntime;

  /// Destroys the OS-level window. Only non-id-0 windows have an executable
  /// path here — SubWindow sub-windows and spawned UIAbility-kind instances
  /// call the bridge's `destroyWindow`; the ArkTS bridge REJECTS
  /// `destroy-window` for window id 0 (openharmony-ability
  /// WindowPlugin.ets), so the main window is system-managed and a
  /// main-window `close()` can only resolve Err. The embedding runtime's
  /// main-window close arm never dispatches `inner.close()` at all (its
  /// main-window gate runs the exit gate instead; the only executable
  /// `inner.close()` call is the non-main-window branch) — the main window is
  /// terminated by the system through that exit chain, not by this call.
  ///
  /// Fire-and-forget — the bridge call is spawned on the window's bridge
  /// executor. On a successful resolve the window id is enqueued as a
  /// destruction confirmation and the event loop synthesizes
  /// `Event::WindowEvent::Destroyed`: that confirmation channel serves the
  /// NON-main windows (sub-windows / spawned UIAbility instances — for the
  /// main window an Ok resolve is unreachable, so there is no re-emitted
  /// `Destroyed` on that path). Failure has three legs: a window whose
  /// `window_id()` is `None` (the creation handshake never settled) fails
  /// before anything can be keyed — nothing is dispatched and that leg is
  /// log-only, with no enqueue and no safe sentinel (0 is the main window's
  /// REAL id) — a different situation from the id-0 rejection above; the
  /// embedding runtime observes the None id directly at its initiate site,
  /// which is where its in-flight close guard lives. A window whose bridge
  /// client is unavailable is accounted per attempt, same as a failed resolve.
  /// On a failed resolve tao warns and accounts the id in the destroy-failure
  /// twin queue (`drain_destroy_failures`, re-export below) so the runtime can
  /// clear its in-flight close guard; no Event is emitted for a failure.
  /// Recursion-safe against the confirmation-driven close flow: the loopback
  fn close(&self);
}

impl WindowExtOpenHarmony for Window {
  fn content_rect(&self) -> Rect {
    self.window.content_rect()
  }

  fn config(&self) -> Configuration {
    self.window.config()
  }

  fn window_id(&self) -> Option<i64> {
    self.window.window_id()
  }

  fn window_kind(&self) -> OHOSWindowKind {
    self.window.kind()
  }

  fn bridge_runtime(&self) -> openharmony_ability::BridgeRuntime {
    self
      .window
      .bridge_runtime()
      .expect("BridgeRuntime not available — EventLoop not initialized")
  }

  fn close(&self) {
    self.window.close();
  }
}

/// Additional methods on [`WindowAttributes`] that are specific to OpenHarmony.
pub trait WindowAttributesExtOpenHarmony {}

impl WindowAttributesExtOpenHarmony for WindowAttributes {}

/// Additional methods on [`WindowBuilder`] that are specific to OpenHarmony.
pub trait WindowBuilderExtOpenHarmony {
  /// Sets the window label, used as the OS-level window name for sub-windows.
  fn with_label(self, label: &str) -> Self;

  /// Sets the OHOS window kind.
  ///
  /// - `UIAbility`: a window backed by its own UIAbility instance. The FIRST
  ///   window of the process must be this kind (it bootstraps the
  ///   EntryAbility container; a SubWindow has nothing to attach to);
  ///   subsequent windows may opt in explicitly and take the
  ///   spawned-instance path.
  /// - `SubWindow`: Sub-window created via `windowStage.createSubWindow`, living
  ///   inside the existing UIAbility container -- not the OS-wide floating window
  ///   type (`WindowType.TYPE_FLOAT`), a conflation the former `Float` variant
  ///   name wrongly suggested (issue Eulogizethesun/tauri#240).
  ///
  /// Default when not specified (PR#28 review T28-8): the first window is
  /// `UIAbility` (bootstrap requirement), any later window is `SubWindow`.
  fn with_window_kind(self, kind: OHOSWindowKind) -> Self;

  /// Returns the current OHOS window kind, if set.
  fn ohos_window_kind(&self) -> Option<OHOSWindowKind>;
}

impl WindowBuilderExtOpenHarmony for WindowBuilder {
  fn with_label(mut self, label: &str) -> Self {
    self.platform_specific.label = Some(label.to_string());
    self
  }

  fn with_window_kind(mut self, kind: OHOSWindowKind) -> Self {
    self.platform_specific.window_kind = Some(kind);
    self
  }

  fn ohos_window_kind(&self) -> Option<OHOSWindowKind> {
    self.platform_specific.window_kind
  }
}

pub use crate::platform_impl::OHOSWindowKind;

/// Re-export of the destroy-failure drain (issue Eulogizethesun/tauri#255):
/// the embedding runtime (tauri-runtime-wry) drains the OHOS window ids whose
/// `close()`-dispatched destroy failed, to clear its in-flight close guard.
/// Same export path as the destroy-confirmation twin's platform surface
/// ([`crate::platform_impl::OHOSWindowKind`] et al.).
pub use crate::platform_impl::drain_destroy_failures;

pub trait EventLoopBuilderExtOpenHarmony {
  /// Associates the [`OpenHarmonyApp`] that was passed to `openharmony-ability::ability` with the event loop
  ///
  /// This must be called on OpenHarmony since the [`OpenHarmonyApp`] is not global state.
  fn with_openharmony_app(&mut self, app: OpenHarmonyApp) -> &mut Self;
}

impl<T> EventLoopBuilderExtOpenHarmony for EventLoopBuilder<T> {
  fn with_openharmony_app(&mut self, app: OpenHarmonyApp) -> &mut Self {
    self.platform_specific.openharmony_app = Some(app);
    self
  }
}

/// Re-export of the `openharmony-ability` API
///
/// tao re-exports the `openharmony-ability` API for convenience so that most
/// applications can rely on the tao crate to resolve the required version of
/// `openharmony-ability` and avoid any chance of a conflict between tao and the
/// application crate.
///
///
/// For compatibility applications should then import the [`OpenHarmonyApp`] type for
/// their `init(app: OpenHarmonyApp)` function and use `openharmony-ability-derive` to
/// implement entry like:
/// ```rust
/// #[cfg(target_env = "ohos")]
/// use tao::platform::ohos::ability::OpenHarmonyApp;
/// use openharmony_ability_derive::ability;
///
/// #[ability]
/// fn init(app: OpenHarmonyApp) {
///     // ...
/// }
/// ```
pub mod ability {
  #[doc(no_inline)]
  pub use openharmony_ability::{
    drain_pending_window_closes, drain_pending_window_status, OpenHarmonyApp,
  };

  #[doc(no_inline)]
  pub use openharmony_ability_derive::*;
}
