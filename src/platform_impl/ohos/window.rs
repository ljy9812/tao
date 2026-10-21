use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering};
use std::sync::Arc;

use openharmony_ability::window::{
  create_os_window, set_cursor_grab, set_window_privacy_mode, WindowCreateParams,
};
use openharmony_ability::{Configuration, OpenHarmonyApp, Rect};
use openharmony_ability_plugin_window::WindowExt;

use crate::dpi::{PhysicalPosition, PhysicalSize, PixelUnit, Position, Size};
use crate::error::{self};
use crate::keyboard::{KeyCode, NativeKeyCode};
use crate::monitor;
use crate::window::{self, Fullscreen, ResizeDirection, Theme, WindowSizeConstraints};

use super::event_loop::{
  cursor_position_from_app, effective_theme, monitor_from_point_for_app, set_app_theme,
  BridgeExecutor, EventLoopWindowTarget, HAS_FOCUS, LAST_DISPATCHED_RECTS,
};
use super::monitor::MonitorHandle;

// Phase 3 (design.md D6): WindowId was a ZST — every OHOS window hashed to the
// same key (0), so tauri-runtime-wry's window_id_map.get(&ZST) always returned the
// last-inserted window (typically the main window). Carrying the OHOS windowId
// (0 = main, >0 = Float sub-window) as the inner value makes per-window event
// routing work: distinct windows hash to distinct keys. This type lives entirely
// inside `#[cfg(target_env = "ohos")]` (platform_impl/mod.rs:29), so other
// platforms are unaffected (rule 2).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(crate) struct WindowId(pub(crate) i64);

impl WindowId {
  pub const fn dummy() -> Self {
    WindowId(0)
  }
}

impl From<WindowId> for u64 {
  fn from(id: WindowId) -> Self {
    id.0 as u64
  }
}

impl From<u64> for WindowId {
  fn from(id: u64) -> Self {
    WindowId(id as i64)
  }
}

/// OHOS window kind: determines whether this window lives in a UIAbility
/// container (UIAbility) or creates a new OS-level floating window (Float).
///
/// Default is UIAbility for the first window, Float for any later window.
/// The first window must be a UIAbility (Float sub-windows attach to an
/// existing UIAbility container); subsequent UIAbility windows spawn a new
/// EntryAbility instance via `start_ui_ability` with a pre-allocated window
/// id (openspec multi-uiability-windows, design D1/D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OHOSWindowKind {
  UIAbility,
  Float,
}

/// Latches once the first (UIAbility) window has been created. Only
/// constraint it enforces: the first window must be a UIAbility — a Float
/// window has no container to attach to. Replaces the old `UIABILITY_CREATED`
/// singleton guard (design D2).
///
/// Process-lifetime bootstrap marker (committer Eulogizethesun/tauri#169):
/// it is deliberately NOT reset on window `Drop` — `Drop` only unregisters
/// the per-window state mirror, so the latch keeps recording "a UIAbility
/// container exists in this process". Later windows therefore default to
/// `OHOSWindowKind::Float`, while an explicit UIAbility request stays legal
/// (it spawns a fresh EntryAbility via the spawn chain).
static FIRST_WINDOW_CREATED: AtomicBool = AtomicBool::new(false);

/// Decoration button bitfield constants (aligned with openharmony-ability ArkTS).
const FLAG_CLOSABLE: u8 = 1;
const FLAG_MAXIMIZABLE: u8 = 2;
const FLAG_MINIMIZABLE: u8 = 4;
const FLAG_RESIZABLE: u8 = 8;
const FLAG_ALL_DECORATIONS: u8 =
  FLAG_CLOSABLE | FLAG_MAXIMIZABLE | FLAG_MINIMIZABLE | FLAG_RESIZABLE;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PlatformSpecificWindowBuilderAttributes {
  pub label: Option<String>,
  pub window_kind: Option<OHOSWindowKind>,
}

/// Window state bits that are backfilled from system events
/// (`windowStatusChange`): kept in a separate `Arc` so the event loop can
/// apply system truth through the global [`WINDOW_MIRRORS`] registry without
/// owning the `Window` itself.
pub(crate) struct WindowStateMirror {
  /// Window state mirror. visible/fullscreen are maintained by windowStatusChange backfill.
  /// Defaults: visible=true, fullscreen=false.
  pub(crate) visible: AtomicBool,
  pub(crate) fullscreen: AtomicBool,
  /// State mirror for is_maximized() — written by setter intent AND backfilled
  /// by apply_window_status (windowStatusChange events), so it reflects the last
  /// known system truth (the bridge facade has no synchronous system query).
  pub(crate) maximized: AtomicBool,
  /// State mirror for is_minimized() — same backfill contract as `maximized`.
  pub(crate) minimized: AtomicBool,
  /// Per-window focused bit (#173): written by the event-loop focus arms
  /// (GainedFocus / LostFocus) through [`set_mirror_focused`] so the mirror
  /// records per-window system truth. Intentionally NOT read by
  /// `Window::is_focused()` — that stays on the app-level HAS_FOCUS static
  /// (changing the read source was explicitly out of scope for #173).
  pub(crate) focused: AtomicBool,
}

impl WindowStateMirror {
  fn new(focused: bool) -> Self {
    Self {
      visible: AtomicBool::new(true),
      fullscreen: AtomicBool::new(false),
      maximized: AtomicBool::new(false),
      minimized: AtomicBool::new(false),
      focused: AtomicBool::new(focused),
    }
  }

  /// Backfills system window status (issue 5, 5.3): applies a raw OHOS
  /// `WindowStatusType` to the mirror bits. Owned by tao — the event loop
  /// drain applies `drain_pending_window_status` entries through the
  /// [`WINDOW_MIRRORS`] registry.
  pub(crate) fn apply_window_status(&self, status: i32) {
    match WindowStatus::from(status) {
      WindowStatus::FullScreen => {
        // System fullscreen: visible + fullscreen + maximized (tauri's fullscreen entry path mirrors this synchronously).
        self.visible.store(true, Ordering::Release);
        self.fullscreen.store(true, Ordering::Release);
        self.maximized.store(true, Ordering::Release);
        self.minimized.store(false, Ordering::Release);
      }
      WindowStatus::Maximize => {
        // Maximize: visible, not fullscreen, not minimized.
        self.visible.store(true, Ordering::Release);
        self.fullscreen.store(false, Ordering::Release);
        self.maximized.store(true, Ordering::Release);
        self.minimized.store(false, Ordering::Release);
      }
      WindowStatus::Minimize => {
        // Minimize: not visible, not fullscreen, not maximized.
        self.visible.store(false, Ordering::Release);
        self.fullscreen.store(false, Ordering::Release);
        self.maximized.store(false, Ordering::Release);
        self.minimized.store(true, Ordering::Release);
      }
      WindowStatus::Floating => {
        // Free floating (normal): visible, not fullscreen, not maximized, not minimized.
        self.visible.store(true, Ordering::Release);
        self.fullscreen.store(false, Ordering::Release);
        self.maximized.store(false, Ordering::Release);
        self.minimized.store(false, Ordering::Release);
      }
      WindowStatus::SplitScreen => {
        // Split screen: visible, not fullscreen (tao has no split-screen concept,
        // treat as visible); maximized left untouched — a split half is neither
        // maximized nor floating, cannot be reliably inferred.
        self.visible.store(true, Ordering::Release);
        self.fullscreen.store(false, Ordering::Release);
      }
      WindowStatus::Other => {
        // UNDEFINED/unknown value: don't change anything, avoid accidental clearing.
      }
    }
  }
}

/// Live window state mirrors keyed by OHOS window id (0 = main window).
/// Populated by `Window::new`, removed on `Drop` (with lazy cleanup of dead
/// entries in the event loop drain). The drain at the start of each MainEvent
/// dispatch in `run_loop` applies `drain_pending_window_status` entries and
/// synthesizes `CloseRequested` events for `drain_pending_window_closes`
/// through this registry / window ids.
pub(crate) static WINDOW_MIRRORS: std::sync::LazyLock<
  std::sync::Mutex<std::collections::HashMap<i64, std::sync::Weak<WindowStateMirror>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Serializes the tests that populate [`WINDOW_MIRRORS`] with live mirrors
/// (window.rs's `ut_set_mirror_focused_writes_and_tolerates_miss` and
/// event_loop.rs's `live_window_ids_filters_dead_mirrors_...`): they run in
/// parallel in one test binary and each asserts exact registry contents.
#[cfg(test)]
pub(crate) static WINDOW_MIRRORS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Writes the per-window focused bit (#173) through [`WINDOW_MIRRORS`].
/// A miss (unknown id, or the mirror already unregistered because the
/// `Window` was dropped) is silently skipped — same tolerance as the
/// unregister-vs-backfill timing of the other mirror bits. Called by the
/// event-loop focus arms right after the app-level HAS_FOCUS store.
pub(crate) fn set_mirror_focused(window_id: i64, focused: bool) {
  let mirror = WINDOW_MIRRORS
    .lock()
    .expect("WINDOW_MIRRORS poisoned")
    .get(&window_id)
    .and_then(|weak| weak.upgrade());
  if let Some(mirror) = mirror {
    mirror.focused.store(focused, Ordering::Release);
  }
}

/// #184 lazy bridge-client cache policy (shared seam for `bridge_client` and
/// its unit tests). On a cache hit the cached client is cloned; on a miss the
/// client is re-fetched once through `refetch` (the app handle's
/// `WindowExt::window()`, a non-blocking RwLock read) and the cache is
/// overwritten on success so later calls are served from it; a persistent
/// failure returns `None` with a warn emitted exactly once — gated on the
/// false→true transition of `warned` (swap-based debounce, so hot paths like
/// the decoration seed cannot storm the log when the bridge is down).
///
/// Generic over the client type: a real `WindowClient` needs a live bridge
/// session (NAPI) that a test binary does not have, and the policy is what
/// design D8 specifies; the production call site instantiates
/// `C = openharmony_ability_plugin_window::WindowClient`.
fn bridge_client_cached_or_refetch<C: Clone>(
  cache: &std::sync::Mutex<Option<C>>,
  refetch: impl FnOnce() -> Option<C>,
  op: &str,
  warned: &AtomicBool,
) -> Option<C> {
  let cached = cache
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .clone();
  if cached.is_some() {
    return cached;
  }
  match refetch() {
    Some(client) => {
      *cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(client.clone());
      Some(client)
    }
    None => {
      if !warned.swap(true, Ordering::Relaxed) {
        log::warn!(
          "[tao-ohos] {op}: bridge client re-fetch failed — window operations degrade to no-ops (reported once)"
        );
      }
      None
    }
  }
}

/// A deferred window operation waiting for its target's UIAbility stage
/// handshake (issue 7, doc/OHOS窗口遗留问题.md in the tauri repo): boxed so
/// plain futures ([`Window::spawn_or_queue`]) and non-future actions share
/// one queue. Mirrors wry's per-webview `pending_ops` (wry/src/ohos/mod.rs).
type PendingWindowOp = Box<dyn FnOnce() + Send + 'static>;

/// Window ops dispatched to a window before its creation handshake completes
/// — the "Unknown OS sub-window" race window. Two gated populations: spawned
/// UIAbility windows awaiting stage registration (issue 7), and Float
/// sub-windows awaiting the ArkTS creation chain to settle (issue-7 addendum; gated
/// only when the ProcessInitializer capability handshake armed the Rust-side
/// registry — a stale HAR keeps Float ops on the old fire-and-forget path).
/// Keyed by OHOS window id, FIFO per window. Drained by the event loop
/// ([`drain_ready_pending_window_ops`]) whenever the D7 registration waker or
/// the Float settle waker fires, or inline by [`Window::dispatch_or_queue`]
/// when registration lands between its ready-check and push (push-then-check,
/// the same TOCTOU discipline as wry's pending_ops).
static PENDING_WINDOW_OPS: std::sync::LazyLock<
  std::sync::Mutex<std::collections::HashMap<i64, Vec<PendingWindowOp>>>,
> = std::sync::LazyLock::new(|| {
  std::sync::Mutex::new(std::collections::HashMap::new())
});

/// Event-loop drain (called on every MainEvent dispatch after a D7
/// registration waker or Float settle waker wakes the loop): replay queued
/// ops for every window whose creation handshake completed. Ready ids are
/// collected under the queue lock and replayed outside it — each op
/// self-spawns on the bridge executor and must not run with the lock held.
///
/// Note on ordering: replayed ops and fast-path ops dispatched after the
/// handshake race on the bridge executor with no FIFO guarantee — same
/// pre-existing limitation as the UIAbility queue (wry pending_ops parity).
pub(crate) fn drain_ready_pending_window_ops() {
  let ready: Vec<(i64, Vec<PendingWindowOp>)> = {
    // Poison-tolerant like the ability-side registries this queue
    // coordinates with (window/mod.rs recovers with into_inner): the
    // critical section is Map ops + is_window_ready only, so a poisoned
    // guard has nothing half-mutated worth aborting the event loop for —
    // and a panic here would cross the NAPI/event-loop boundary.
    let mut map = PENDING_WINDOW_OPS
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ids: Vec<i64> = map
      .keys()
      .filter(|id| openharmony_ability::is_window_ready(**id))
      .copied()
      .collect();
    ids
      .into_iter()
      .filter_map(|id| map.remove(&id).map(|ops| (id, ops)))
      .collect()
  };
  for (id, ops) in ready {
    // info (not debug): this is the third leg of the creation-handshake
    // evidence trio (register_pending_* → notify_*_registered → replay) —
    // hilog's default INFO level must show it on device.
    log::info!(
      "[tao-ohos] replaying {} queued window op(s) for window {} (creation handshake registered)",
      ops.len(),
      id
    );
    for op in ops {
      op();
    }
  }
}

/// Discard queued ops for a window whose tao handle is gone ([`Window::drop`])
/// or whose spawn was rolled back (the start_ui_ability Err leg — the handle
/// still lives there, but the id must never settle). Queued attribute ops die
/// with the window either way — the OS window outlives the handle on OHOS,
/// but nobody is left to observe those attributes; this matches the
/// pre-queue behavior where the dispatches simply failed.
pub(crate) fn drop_pending_window_ops(window_id: i64) {
  let ops = PENDING_WINDOW_OPS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .remove(&window_id);
  if let Some(ops) = ops {
    // info: mirrors the replay log — makes the drop race (V5) observable
    // instead of silent. The reason is deliberately generic: this runs on
    // both the handle-drop leg and the spawn-rollback Err leg.
    log::info!(
      "[tao-ohos] dropped {} queued window op(s) for window {} (handle dropped or spawn rolled back)",
      ops.len(),
      window_id
    );
  }
}

pub(crate) struct Window {
  app: OpenHarmonyApp,
  window_id: Option<i64>,
  /// Window kind resolution happens at creation (local `kind` in `new()`).
  /// Historically a field driving the set_inner_size title-bar compensation
  /// (Float skipped it — G7); since issue Eulogizethesun/tauri#97 the
  /// compensation is computed per-window on the ArkTS side from the window's
  /// own `getWindowProperties()` snapshot (Float windows get chrome 0
  /// automatically), so no per-window kind state is needed anymore.
  /// Bridge facade for async window operations. Mutex'd (#184): a `None`
  /// captured at construction (bridge not yet ready during early init) is
  /// lazily re-fetched on the next `bridge_client` call and overwritten on
  /// success, instead of pinning the no-op degradation for the window's
  /// whole lifetime. Direct reads outside `bridge_client` go through
  /// `cached_client` (lock, clone, drop the guard).
  window_client: std::sync::Mutex<Option<openharmony_ability_plugin_window::WindowClient>>,
  /// Warn-once debounce for bridge-client re-fetch failures (#184): the lazy
  /// re-fetch runs on hot paths (e.g. the decoration seed), so the failure
  /// warn is emitted only on the false→true transition of this flag.
  bridge_refetch_warned: AtomicBool,
  /// Background runtime handle for spawning async bridge calls.
  runtime: BridgeExecutor,
  /// Shared state mirror (see [`WindowStateMirror`]): the event loop drain
  /// applies system truth to it through [`WINDOW_MIRRORS`] without owning
  /// this window.
  mirror: Arc<WindowStateMirror>,
  /// Phase 2: window decoration state (title bar visibility).
  /// AtomicBool supports runtime toggle from arbitrary threads.
  decorations: AtomicBool,
  /// Phase 3: whether window was created with transparent=true.
  /// Immutable after construction — set_background_color is a no-op when true.
  transparent: bool,
  // Mirror-bit sync state (issue 5):
  //   - maximized/minimized: local mirror, set_* writes intent + apply_window_status
  //     backfills system truth (the facade has no sync query; mirror + backfill is
  //     the only viable read source for sync is_*).
  //   - visible/fullscreen: local mirror, backfilled by windowStatusChange events
  //     (apply_window_status, tao run_loop drain route). set_* writes intent, events write truth.
  //   - decorations/decoration_flags: app-owned local mirror, no system state to
  //     backfill (records app intent only; related to issue 4 semantic mismatch,
  //     out of scope for this fix).
  //   - always_on_top: pure intent flag, OHOS has no z-order API (does not reflect
  //     the real system z-order).
  //   - theme: per-window field removed, now reads global APP_THEME_OVERRIDE +
  //     app.config() colorMode (continuously refreshed by ConfigChanged).
  //   See doc/OHOS-window-residual-issues.md (issue 5, 5.1/5.2/5.3).
  /// always_on_top intent flag (OHOS has no direct API; records intent only, see set_always_on_top).
  always_on_top: AtomicBool,
  /// Decoration button availability bitfield. bit0 closable, bit1 maximizable,
  /// bit2 minimizable, bit3 resizable. Defaults to 0b1111=15 (all enabled).
  decoration_flags: AtomicU8,
  /// Window size constraint cache (min/max w/h, px). OHOS `setWindowLimits`
  /// writes all four values at once (0 = unlimited), non-incrementally; so
  /// `set_min_inner_size`/`set_max_inner_size` must send both constraint sets
  /// together, otherwise the later call resets the other dimension to 0 (losing
  /// the constraint). Each setter updates its own cache slots, then reads the
  /// other's cache and sends all four. AtomicU32 because setters may be called
  /// from any thread.
  min_inner_width: AtomicU32,
  min_inner_height: AtomicU32,
  max_inner_width: AtomicU32,
  max_inner_height: AtomicU32,
}

// Upstream PR#20 window-type constants (ArkTS WindowType). UIAbility windows
// (both the primary EntryAbility window and multi-UIAbility spawned windows —
// that path is ported, see `Window::new`) never construct a variant of this
// enum: their creation path passes the literal 0 (which coincides with
// `TypeApp`'s value — the enum has no "none" sentinel), matching how the OS
// creates the bootstrap window, so `TypeMain` itself stays unconstructed.
// Float sub-windows are the only constructors of an explicit variant
// (`TypeFloat`). The remaining variants are kept for parity with
// upstream.
#[allow(dead_code)]
enum OHOSWindowType {
  TypeApp = 0,
  TypeSystemAlert = 1,
  TypeFloat = 8,
  TypeDialog = 16,
  TypeMain = 32,
}

/// OHOS `WindowStatusType` (API 11+) — the system's window mode, reported via
/// `window.on('windowStatusChange')`. Values match the ArkTS enum order:
/// 1=FULL_SCREEN, 2=MAXIMIZE, 3=MINIMIZE, 4=FLOATING, 5=SPLIT_SCREEN.
///
/// Used by [`WindowStateMirror::apply_window_status`] to backfill system truth
/// into tao mirror bits. See doc/OHOS-window-residual-issues.md (issue 5, 5.3).
enum WindowStatus {
  FullScreen,
  Maximize,
  Minimize,
  Floating,
  SplitScreen,
  /// UNDEFINED(0) or any unrecognized value.
  Other,
}

impl From<i32> for WindowStatus {
  fn from(value: i32) -> Self {
    match value {
      1 => WindowStatus::FullScreen,
      2 => WindowStatus::Maximize,
      3 => WindowStatus::Minimize,
      4 => WindowStatus::Floating,
      5 => WindowStatus::SplitScreen,
      _ => WindowStatus::Other,
    }
  }
}

/// Maps tao `CursorIcon` to OHOS `pointer.PointerStyle` enum value.
///
/// OHOS PointerStyle declaration order (see `@ohos.multimodalInput.pointer`):
/// DEFAULT=0, EAST=1, WEST=2, SOUTH=3, NORTH=4, WEST_EAST=5, NORTH_SOUTH=6,
/// NORTH_EAST=7, NORTH_WEST=8, SOUTH_EAST=9, SOUTH_WEST=10,
/// NORTH_EAST_SOUTH_WEST=11, NORTH_WEST_SOUTH_EAST=12, CROSS=13, CURSOR_COPY=14,
/// CURSOR_FORBID=15, ..., HAND_GRABBING=17, HAND_OPEN=18, HAND_POINTING=19,
/// HELP=20, MOVE=21, ..., TEXT_CURSOR=26, ZOOM_IN=27, ZOOM_OUT=28,
/// HORIZONTAL_TEXT_CURSOR=39, LOADING=42.
fn ohos_pointer_style(icon: window::CursorIcon) -> i32 {
  match icon {
    window::CursorIcon::Default
    | window::CursorIcon::Arrow
    | window::CursorIcon::ContextMenu
    | window::CursorIcon::Cell => 0,
    window::CursorIcon::Crosshair => 13,
    window::CursorIcon::Hand => 19,
    window::CursorIcon::Move | window::CursorIcon::AllScroll => 21,
    window::CursorIcon::Text => 26,
    window::CursorIcon::VerticalText => 39,
    window::CursorIcon::Wait | window::CursorIcon::Progress => 42,
    window::CursorIcon::Help => 20,
    window::CursorIcon::NotAllowed | window::CursorIcon::NoDrop => 15,
    window::CursorIcon::Alias | window::CursorIcon::Copy => 14,
    window::CursorIcon::Grab => 18,
    window::CursorIcon::Grabbing => 17,
    window::CursorIcon::ZoomIn => 27,
    window::CursorIcon::ZoomOut => 28,
    window::CursorIcon::EResize => 1,
    window::CursorIcon::WResize => 2,
    window::CursorIcon::SResize => 3,
    window::CursorIcon::NResize => 4,
    window::CursorIcon::EwResize | window::CursorIcon::ColResize => 5,
    window::CursorIcon::NsResize | window::CursorIcon::RowResize => 6,
    window::CursorIcon::NeResize => 7,
    window::CursorIcon::NwResize => 8,
    window::CursorIcon::SeResize => 9,
    window::CursorIcon::SwResize => 10,
    window::CursorIcon::NeswResize => 11,
    window::CursorIcon::NwseResize => 12,
  }
}

/// Builder-requested logical geometry → the startAbility `StartOptions`
/// tuples (`windowWidth`/`windowHeight`, `windowLeft`/`windowTop`, physical
/// px) used by the UIAbility spawn path. An axis the builder left
/// unspecified stays `None` so the OS default sizing/positioning applies;
/// zero/negative values pass through unchanged — the ArkTS
/// start-ui-ability handler rejects them asynchronously (surfaced via
/// `note_ui_ability_spawn_rejection`, see the spawn closure in
/// `Window::new`), which is why the builder cannot report them as an `Err`.
fn start_options_geometry(
  inner_size: Option<Size>,
  position: Option<Position>,
  scale: f64,
) -> (Option<(i32, i32)>, Option<(i32, i32)>) {
  let inner_size = inner_size
    .map(|s| s.to_physical::<i32>(scale))
    .map(|p| (p.width, p.height));
  let position = position
    .map(|p| p.to_physical::<i32>(scale))
    .map(|p| (p.x, p.y));
  (inner_size, position)
}

/// Converts tao's RGBA tuple to OHOS `0xAARRGGBB` u32 format.
///
/// When `transparent` is true, returns `Some(0x00000000)` regardless of `bg`
/// (transparent takes priority over background_color, consistent with
/// Windows/macOS behavior).
///
/// Used by both `Window::new()` (creation path) and `set_background_color()`
/// (runtime path) to avoid duplicated conversion logic.
fn rgba_to_ohos_color(transparent: bool, bg: Option<window::RGBA>) -> Option<u32> {
  if transparent {
    Some(0x00000000)
  } else {
    bg.map(|(r, g, b, a)| ((a as u32) << 24) | ((r as u32) << 16) | ((g as u32) << 8) | (b as u32))
  }
}

impl Window {
  pub(crate) fn new<T: 'static>(
    el: &EventLoopWindowTarget<T>,
    window_attrs: window::WindowAttributes,
    pl_attrs: PlatformSpecificWindowBuilderAttributes,
  ) -> Result<Self, error::OsError> {
    // Resolve the window kind: explicit builder choice, else the first window
    // defaults to UIAbility and any later one to Float.
    let kind = match pl_attrs.window_kind {
      Some(kind) => kind,
      None if !FIRST_WINDOW_CREATED.load(Ordering::SeqCst) => OHOSWindowKind::UIAbility,
      None => OHOSWindowKind::Float,
    };
    let is_ui_ability_window = matches!(kind, OHOSWindowKind::UIAbility);

    // First-window latch (design D2): the first window must be a UIAbility —
    // a Float sub-window attaches to an existing UIAbility container
    // (WindowManager stage) and cannot bootstrap the process. Subsequent
    // UIAbility windows are legal and take the start_ui_ability branch below.
    // The bootstrap main window is created on the event-loop thread during
    // app setup, before any command can race this swap.
    let is_first_window = !FIRST_WINDOW_CREATED.swap(true, Ordering::SeqCst);
    if is_first_window && !is_ui_ability_window {
      // Rejection semantics: an explicit Float request for the first window
      // fails fast with Err here (this path silently continued before the
      // D2 latch introduced the check). Roll back the latch: no window was
      // created; a retry with a UIAbility window must still be able to
      // bootstrap.
      // Consumer caveat (tao#26 review follow-up, updated for the runtime-wry
      // reply channel): this Err used to be swallowed by runtime-wry — the
      // `Message::CreateWindow` dispatch had no reply channel (upstream
      // design; it logged the error and returned Ok), so `build()` resolved
      // Ok and the manager registered a label-only zombie window. On OHOS
      // runtime-wry now replies with the creation outcome and propagates the
      // failure (tauri PR of the coordinated set): `build()` rejects, no
      // zombie registration. The log line above is retained for
      // traceability. Unlike the mobile gate there is no
      // note_ui_ability_spawn_rejected surface here — rejection is conveyed
      // by the create call itself.
      FIRST_WINDOW_CREATED.store(false, Ordering::SeqCst);
      log::error!("the first OHOS window must be a UIAbility window (Float has no container to attach to)");
      return Err(os_error!(OsError));
    }

    let window_type = if is_ui_ability_window {
      // UIAbility window does not need a window_type
      0
    } else {
      // Float sub-window uses TypeFloat
      OHOSWindowType::TypeFloat as i32
    };

    let window_id = if is_ui_ability_window {
      if is_first_window {
        // First UIAbility window: the process's original EntryAbility instance
        // (launched by the OS). window_id = 0, wry takes Path 1
        // (WebViewBuilder on the existing DefaultXComponent).
        Some(0)
      } else {
        // Subsequent UIAbility window: spawn a new EntryAbility instance via
        // startAbility carrying this pre-allocated id (design D1). Window::new
        // does NOT wait for the handshake (D7): register the pending ability
        // with an event-loop waker synchronously, dispatch the startAbility
        // want fire-and-forget on the bridge executor (HC-5: no block_on),
        // and return immediately — wry queues webview ops for this id in its
        // pending_ops queue until the instance registers its stage.
        let label = pl_attrs
          .label
          .clone()
          .unwrap_or_else(|| window_attrs.title.clone());
        // OHOS form-factor gate (design.md OQ1: the mobile entry template
        // stays singleton, no onAcceptWant). On a mobile-form build the
        // spawn want would be routed back to the existing primary instance
        // (onNewWant) — AMS reports no error, the new instance's
        // onWindowStageCreate never fires, and the pending entry below
        // would never resolve: every op queued for this window held
        // silently forever, on a leg covered by neither the sync Err
        // rollback nor notify_ui_ability_start_failed. Fail fast instead,
        // before any registry state is opened. This also enforces what
        // `supports_multiple_windows()` already declares (OHOS: desktop
        // only) on the actual creation path.
        //
        // The Err is recorded under the label before returning.
        // note_ui_ability_spawn_rejected (consumable via
        // take_ui_ability_spawn_rejection) predates the runtime-wry reply
        // channel: back then the `Message::CreateWindow` dispatch had no
        // reply channel (upstream design — it logged the error and returned
        // Ok), so build() resolved Ok with a label-only zombie window and
        // the note was the only surface reporting the mobile fail-fast at
        // the tauri API layer. On OHOS the create call now propagates the
        // Err as well (tauri PR of the coordinated set); the label-keyed
        // note is kept as a durable record and for non-tauri embedders. The
        // label resolves BEFORE the gate for exactly this reason
        // (runtime-wry always sets it via with_label).
        if !openharmony_ability::is_desktop_form() {
          const MOBILE_GATE_REASON: &str = "spawning an additional UIAbility window is only supported on the desktop (PC/2in1) form — this is a mobile-form build";
          log::error!("[tao-ohos] {} (label={})", MOBILE_GATE_REASON, label);
          openharmony_ability::note_ui_ability_spawn_rejected(&label, MOBILE_GATE_REASON);
          return Err(os_error!(OsError));
        }
        // Validate BEFORE opening any handshake state: the ArkTS
        // start-ui-ability handler rejects an empty label synchronously
        // (AppControlPlugin.ets), but that rejection only surfaces on the
        // async Err leg — by then `Window::new` has already returned a live
        // handle whose every op fails at the bridge (a zombie window). Note
        // the tauri stack is NOT immune: tauri core only validates label
        // uniqueness (WindowLabelAlreadyExists, no is_empty check) and
        // runtime-wry forwards the label verbatim, so an explicitly empty
        // label reaches this point through tauri too — THIS guard is the
        // only empty-label interception in the whole stack.
        if label.is_empty() {
          log::error!(
            "[tao-ohos] spawning a UIAbility window requires a non-empty label (or window title)"
          );
          return Err(os_error!(OsError));
        }
        let window_id = openharmony_ability::next_window_id();
        // Capability handshake (G15, mirrors the Float registry's
        // create_os_window gate): only open a pending entry when the ArkTS
        // ProcessInitializer has armed UIAbility pending tracking. A stale
        // HAR (fresh .so, cached ArkTS) never arms it — and
        // `register_ui_ability_stage` ships in the same HAR generation as
        // the arming call — so the spawn stays fire-and-forget
        // (pre-registry semantics: an unknown id reads as ready) instead
        // of queueing ops against an entry no ArkTS code would ever
        // settle.
        if openharmony_ability::ui_ability_pending_tracking_enabled() {
          openharmony_ability::register_pending_ui_ability(window_id);
          openharmony_ability::set_ui_ability_waker(window_id, el.app.create_waker());
        }
        // Creation-time geometry (tao#26 review follow-up F5): the builder's
        // `inner_size`/`position` (logical px) are converted to physical px
        // and delivered through the startAbility StartOptions
        // (`windowWidth`/`windowHeight`/`windowLeft`/`windowTop`, API 11+) so
        // a spawned UIAbility window honors `.inner_size()/.position()`
        // instead of falling back to the OS default sizing. StartOptions
        // sizes the window rectangle (frame included) — same contract as the
        // Float branch's WindowCreateParams below; a decorated window's frame
        // eats into the content. Unspecified axes stay unset (OS default).
        // The bootstrap main window (id 0) remains OS/ArkTS sized — apply
        // `set_size` after ready there.
        //
        // StartOptions' official constraints (tao#26 review follow-up
        // T26-N3) — the geometry can silently NOT apply as requested:
        // - free windows only: a window the AMS opens in a non-free mode
        //   (e.g. split-screen) ignores StartOptions geometry entirely;
        // - `windowLeft`/`windowTop` are relative to the TARGET display's
        //   origin, while tao `Position` is global-screen-space and tao
        //   never passes `displayId` — on multi-display desktops a global
        //   coordinate is interpreted as if display-local;
        // - the WM clamps rects that fall outside the display bounds: a
        //   position/size partially off-screen arrives clamped, not as
        //   requested.
        let spawn_scale = el.app.scale() as f64;
        let (inner_size, position) =
          start_options_geometry(window_attrs.inner_size, window_attrs.position, spawn_scale);
        // The webview URL is delivered after the stage registers
        // (WebviewCreateRequest, OQ5). It is also mirrored into the want as
        // `tauri_window_url` by the ArkTS start-ui-ability handler, but no
        // reader consumes that key — tao always passes the empty string
        // here, so the want parameter is bookkeeping/debug payload only.
        let url = String::new();
        let transparent = window_attrs.transparent;
        let app = el.app.clone();
        // Kept for the async rejection note below: `label` is moved into
        // the spawn closure.
        let note_label = label.clone();
        el.bridge_executor.clone().spawn(async move {
          if let Err(e) = openharmony_ability_plugin_app_control::start_ui_ability(
            &app,
            window_id,
            label,
            url,
            transparent,
            inner_size,
            position,
          )
          .await
          {
            log::error!(
              "[tao-ohos] start_ui_ability failed for window {}: {:?}",
              window_id,
              e
            );
            // Durable, consumable record of the async rejection (tao#26
            // review follow-up T26-N2): build() has already returned a
            // live handle by this point — the CreateWindow reply channel
            // closed before the spawn resolved — so without this the log
            // line above is the only surface. A zero/negative
            // `inner_size` reaches exactly this leg (the ArkTS
            // start-ui-ability handler rejects non-positive width/height).
            // Embedders (and tauri) can poll
            // `take_ui_ability_spawn_rejection` with the label; the
            // registry is the same one the mobile fail-fast gate notes
            // into, so the consumption path is shared.
            openharmony_ability::note_ui_ability_spawn_rejected(
              &note_label,
              &format!("start_ui_ability failed: {:?}", e),
            );
            // Roll back the pending entry if one was opened (G15): the
            // ability-destroy callback never fires for a spawn that failed,
            // so is_window_ready would stay false forever and every op
            // queued for this window would be held silently. Unregister
            // BEFORE dropping the queued ops: an unknown id reads as ready,
            // so an op racing in between dispatches immediately (bridge
            // call fails with a warn) instead of being stranded forever.
            openharmony_ability::unregister_pending_ui_ability(window_id);
            drop_pending_window_ops(window_id);
          }
        });
        Some(window_id)
      }
    } else {
      // Float window: create a new OS-level floating window via create_os_window.
      // window_id > 0, wry takes Path 2 (load_url).
      let label = pl_attrs
        .label
        .clone()
        .unwrap_or_else(|| window_attrs.title.clone());
      // Honor the builder's inner_size/position (logical px → physical) so a
      // Float WebviewWindow sized via `.inner_size()/.position()` actually
      // applies. Without this, createOSWindow falls back to the 800×600 default
      // and ignores the requested geometry entirely.
      let scale = el.app.scale() as f64;
      let (width, height) = window_attrs
        .inner_size
        .map(|s| {
          let p = s.to_physical::<i32>(scale);
          (p.width, p.height)
        })
        .unwrap_or((800, 600));
      let (x, y) = window_attrs
        .position
        .map(|p| {
          let phys = p.to_physical::<i32>(scale);
          (phys.x, phys.y)
        })
        .unwrap_or((100, 100));
      let params = WindowCreateParams {
        name: label.clone(),
        window_type: window_type as i32,
        width,
        height,
        x,
        y,
        decorations: window_attrs.decorations,
        transparent: window_attrs.transparent,
        background_color: rgba_to_ohos_color(
          window_attrs.transparent,
          window_attrs.background_color,
        ),
      };
      match create_os_window(params) {
        Ok(id) => {
          // Float creation race (issue-7 addendum): create_os_window opened a pending
          // registry entry for this id (capability handshake permitting) before
          // dispatching the TSFN; attach the event-loop waker NOW so the settle
          // notify (notify_float_window_registered) wakes the loop and replays
          // any ops queued against the pre-registration window. If the ArkTS
          // chain already settled (synchronous pre-throw reject), the entry is
          // gone and set_float_window_waker wakes anyway — the drain pass then
          // finds this id ready and replays (review V2).
          openharmony_ability::set_float_window_waker(id, el.app.create_waker());
          Some(id)
        }
        Err(e) => {
          log::error!(
            "[tao-ohos] create_os_window failed for Float window {:?}: {:?}",
            label,
            e
          );
          return Err(os_error!(OsError));
        }
      }
    };

    // Create the WindowClient bridge facade. If the bridge runtime is not yet
    // ready (e.g. during early init), the cache starts None: that is warned
    // about once here, and later window operations lazily re-fetch through
    // the app handle (#184) — see `bridge_client`.
    let window_client = match el.app.window() {
      Ok(client) => Some(client),
      Err(e) => {
        log::warn!(
          "[tao-ohos] window bridge init failed for window {:?}: {:?} — window operations degrade to no-ops until the bridge is ready",
          window_id, e
        );
        None
      }
    };
    let runtime = el.bridge_executor.clone();

    // #173: seed the mirror's per-window focused bit from the app-level
    // HAS_FOCUS snapshot at registration time — the focus arms only write on
    // transitions, so without the seed the bit would stay false for a window
    // created while the app already had focus.
    let mirror = Arc::new(WindowStateMirror::new(HAS_FOCUS.load(Ordering::Relaxed)));
    // Seed the size-constraint cache from the builder (previously the builder
    // values were ignored — the four slots stayed 0 until a runtime
    // set_min/max_inner_size call): logical → physical via the real scale.
    let scale = el.app.scale() as f64;
    let constraints = window_attrs.inner_size_constraints;
    let constraint_px =
      |u: Option<PixelUnit>| u.map(|p| p.to_physical::<u32>(scale).0).unwrap_or(0);
    let win = Self {
      app: el.app.clone(),
      window_id,
      window_client: std::sync::Mutex::new(window_client),
      bridge_refetch_warned: AtomicBool::new(false),
      runtime,
      mirror: mirror.clone(),
      decorations: AtomicBool::new(window_attrs.decorations),
      transparent: window_attrs.transparent,
      always_on_top: AtomicBool::new(false),
      decoration_flags: AtomicU8::new(FLAG_ALL_DECORATIONS),
      min_inner_width: AtomicU32::new(constraint_px(constraints.min_width)),
      min_inner_height: AtomicU32::new(constraint_px(constraints.min_height)),
      max_inner_width: AtomicU32::new(constraint_px(constraints.max_width)),
      max_inner_height: AtomicU32::new(constraint_px(constraints.max_height)),
    };

    // Register the state mirror so the event loop drain can apply
    // windowStatusChange backfill to this window without owning it.
    if let Some(id) = window_id {
      WINDOW_MIRRORS
        .lock()
        .expect("WINDOW_MIRRORS poisoned")
        .insert(id, Arc::downgrade(&mirror));
    }

    // Push builder-specified size constraints to the system right away —
    // previously they were cached but never dispatched, so a window built
    // with `.min_inner_size(...)` had no enforced minimum until some later
    // set_min/max_inner_size call happened to fire (tao issue #84).
    if constraints.has_min() || constraints.has_max() {
      if let Some(id) = window_id {
        win.apply_window_limits(id);
      }
    }

    // Apply decorations immediately for the main window at creation time.
    // Without this, the main window retains its default OS decorations even if
    // the builder specified .decorations(false), because Window::set_decorations()
    // is only called later (if at all) by the user.
    // window_id (not hardcoded 0): a subsequent UIAbility window with
    // decorations(false) must not strip the primary window's decorations.
    if is_ui_ability_window && !window_attrs.decorations {
      // Lock-then-clone (#184): never hold the window_client lock across the
      // dispatch.
      if let Some(client) = win.cached_client() {
        let decorations_window_id = window_id.unwrap_or(0);
        win.spawn_or_queue(async move {
          if let Err(e) = client
            .set_window_decorations(decorations_window_id, false)
            .await
          {
            log::warn!(
              "[tao-ohos] set_window_decorations failed for window {}: {:?}",
              decorations_window_id,
              e
            );
          }
        });
      }
    }

    // Apply builder-specified content protection at creation time (macOS does
    // this in its create path; without it, .content_protection(true) had no
    // effect on OHOS until a later set_content_protection call — issue #115).
    // Same two-phase shape as Window::set_content_protection below.
    if window_attrs.content_protection && openharmony_ability::sdk_api_version() >= 15 {
      if let Some(id) = window_id {
        if let Some(client) = win.bridge_client("set_content_protection") {
          win.spawn_or_queue(async move {
            match client.get_real_window_id(id).await {
              Ok(real_id) => {
                if let Err(e) = set_window_privacy_mode(real_id as i32, true) {
                  log::warn!(
                    "[tao-ohos] initial content protection failed for window {} (real id {}): {}",
                    id,
                    real_id,
                    e
                  );
                }
              }
              Err(e) => {
                log::warn!(
                  "[tao-ohos] initial content protection: get_real_window_id failed for window {}: {:?}",
                  id,
                  e
                );
              }
            }
          });
        }
      }
    } else if window_attrs.content_protection
      && openharmony_ability::sdk_api_version() < 15
    {
      log::warn!(
        "[tao-ohos] builder content_protection(true) skipped: requires API 15+ (window privacy mode)"
      );
    }

    Ok(win)
  }

  /// Returns the cached bridge client without a re-fetch attempt (#184): for
  /// direct-read points that must keep their existing None semantics (degrade
  /// to NotSupported / skip) and must not hold the lock across dispatch.
  fn cached_client(&self) -> Option<openharmony_ability_plugin_window::WindowClient> {
    self
      .window_client
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
      .clone()
  }

  /// Returns a clone of the bridge client for dispatching an async window
  /// operation, lazily re-fetching it through the app handle when the cache
  /// is empty (#184): a window constructed before the bridge runtime was
  /// ready used to degrade to no-ops for its whole lifetime; now the next
  /// operation retries the construction-time `WindowExt::window()` (a
  /// non-blocking RwLock read + clone) and overwrites the cache on success.
  /// A persistent failure returns `None` with a warn emitted exactly once
  /// (see [`bridge_client_cached_or_refetch`]) — the caller then degrades to
  /// a no-op.
  fn bridge_client(&self, op: &str) -> Option<openharmony_ability_plugin_window::WindowClient> {
    bridge_client_cached_or_refetch(
      &self.window_client,
      || self.app.window().ok(),
      op,
      &self.bridge_refetch_warned,
    )
  }

  /// Dispatch a window op to the bridge executor, or queue it if the target
  /// is still awaiting its creation handshake (issue 7 + Float race): ops
  /// fired in that window reach ArkTS before `requireWindow` can find the
  /// window and are silently dropped ("Unknown OS sub-window", or a silent
  /// no-op for set-limits which does not go through requireWindow). Mirrors
  /// wry's `dispatch_or_queue`.
  ///
  /// The fast path stays synchronous for every window that cannot race: the
  /// main window (id 0) and unknown ids (UIAbility ids for a fresh tao
  /// process, zombie ids) all read as ready. Float sub-windows are gated from
  /// `create_os_window` until the ArkTS creation chain settles — but only
  /// when the ProcessInitializer capability handshake armed the registry
  /// (stale-HAR builds keep the old fire-and-forget fast path).
  fn dispatch_or_queue(&self, op: PendingWindowOp) {
    let Some(window_id) = self.window_id else {
      return;
    };
    if openharmony_ability::is_window_ready(window_id) {
      op();
      return;
    }
    // Push-then-check (wry pending_ops TOCTOU): if the handshake flipped
    // between the ready check above and this push, the settle waker already
    // fired and no further wake is coming — drain this id now. The entry may
    // already hold earlier queued ops, so drain it FIFO, not just this op.
    let mut map = PENDING_WINDOW_OPS
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    // First push for this id (empty → non-empty) is the creation-latency
    // marker (O10-3): one warn per queue lifetime is expected while a
    // spawned instance's handshake completes — a queue that is never
    // replayed nor dropped is the anomaly. AMS outright rejection is
    // covered by the ability-side rollback (notify_ui_ability_start_failed,
    // ability#53 AF3); a silent wedge here means the start neither completed
    // nor rolled back. Emitted after the lock is dropped so the logging
    // backend never runs inside the queue's critical section.
    let entry = map.entry(window_id).or_default();
    let first_queued = entry.is_empty();
    entry.push(op);
    let queued_len = entry.len();
    let replay = if openharmony_ability::is_window_ready(window_id) {
      map.remove(&window_id).unwrap_or_default()
    } else {
      Vec::new()
    };
    drop(map);
    if first_queued {
      log::warn!(
        "[tao-ohos] window {} not ready (creation handshake pending): {} window op(s) queued",
        window_id,
        queued_len
      );
    }
    for op in replay {
      op();
    }
  }

  /// Future convenience over [`Self::dispatch_or_queue`]: the future is
  /// inert until polled, so holding it in the queue is safe — it captures
  /// the op's arguments as of call time and replays in FIFO order.
  fn spawn_or_queue<F>(&self, fut: F)
  where
    F: std::future::Future<Output = ()> + Send + 'static,
  {
    let runtime = self.runtime.clone();
    self.dispatch_or_queue(Box::new(move || runtime.spawn(fut)));
  }

  /// Dispatch the cached min/max size constraints as one `setWindowLimits`
  /// call. OHOS writes all four slots atomically (0 = unlimited), so every
  /// constraints entry point (builder attrs in `new`,
  /// `set_inner_size_constraints`, `set_min_inner_size` /
  /// `set_max_inner_size`) updates the cache first and funnels through here.
  ///
  /// Inner→outer chrome compensation (review R12, closing the issue #97
  /// semantics gap): OHOS `setWindowLimits` constrains the OUTER windowRect,
  /// so the ArkTS "set-limits" action (plugins/window WindowPlugin.ets)
  /// compensates with the precise system chrome from one
  /// `getWindowProperties()` snapshot — the same algorithm as `resize-inner`
  /// (`chrome = windowRect − drawableRect` per axis). Non-zero slots only (0
  /// = no limit stays 0); Float sub-windows are decorEnabled:false so their
  /// chrome is 0 by construction; when the main-window snapshot is unreadable
  /// (e.g. the builder-time dispatch racing content load) the ArkTS side
  /// degrades to uncompensated inner-sized limits with a WARN instead of
  /// failing — limits are boundaries, not targets. A decor change after
  /// dispatch is the caller's business; re-dispatching on decor changes
  /// remains an open follow-up.
  fn apply_window_limits(&self, window_id: i64) {
    let Some(client) = self.bridge_client("set_window_limits") else {
      return;
    };
    let min_w = self.min_inner_width.load(Ordering::Acquire) as i64;
    let min_h = self.min_inner_height.load(Ordering::Acquire) as i64;
    let max_w = self.max_inner_width.load(Ordering::Acquire) as i64;
    let max_h = self.max_inner_height.load(Ordering::Acquire) as i64;
    self.spawn_or_queue(async move {
      if let Err(e) = client
        .set_window_limits(window_id, min_w, min_h, max_w, max_h)
        .await
      {
        log::warn!(
          "[tao-ohos] set_window_limits failed for window {}: {:?}",
          window_id,
          e
        );
      }
    });
  }

  pub fn request_redraw(&self) {
    // OHOS vsync auto-drives rendering; there is no app-initiated redraw API,
    // so this is a no-op (the former ArkTS bridge was removed upstream).
  }

  #[inline]
  pub fn monitor_from_point(&self, x: f64, y: f64) -> Option<monitor::MonitorHandle> {
    monitor_from_point_for_app(&self.app, x, y).map(|inner| monitor::MonitorHandle { inner })
  }

  pub fn id(&self) -> WindowId {
    // Phase 3 (design.md D6): return this window's own OHOS windowId instead of
    // the ZST constant, so tauri-runtime-wry's window_id_map routes events to the
    // correct WindowWrapper. Main window → WindowId(0), Float sub-window → WindowId(N).
    WindowId(self.window_id.unwrap_or(0))
  }

  pub fn scale_factor(&self) -> f64 {
    self.app.scale() as f64
  }

  pub fn available_monitors(&self) -> VecDeque<MonitorHandle> {
    // One handle per connected display (issue Eulogizethesun/tauri#106).
    MonitorHandle::all_for_app(&self.app).into_iter().collect()
  }

  pub fn inner_position(&self) -> Result<PhysicalPosition<i32>, error::NotSupportedError> {
    // Inner (content-area) position — computed inside openharmony-ability under
    // one lock from the system's drawableRect snapshot (issue
    // Eulogizethesun/tauri#97): window position + drawable offset. Float
    // sub-windows get offset (0,0) naturally (no system title bar).
    // (G7: the mirrored rects track the MAIN window, so this getter is only
    // meaningful for main/UIAbility windows regardless.)
    let rect = self.app.inner_rect_for(self.window_id.unwrap_or(0));
    Ok(PhysicalPosition::new(rect.left, rect.top))
  }

  pub fn inner_size(&self) -> PhysicalSize<u32> {
    // D2 hybrid (design.md): OHOS win.resize() sets the OUTER size (including
    // title bar). inner_size reads the system's drawableRect snapshot
    // (`OpenHarmonyApp::inner_rect_for`, issue Eulogizethesun/tauri#97) so
    // save→restore cycles are idempotent: save inner (drawableRect) → restore
    // resize-inner(inner) → the same drawableRect again — zero drift by
    // construction, both sides using the system's own numbers. Web content
    // sizing is unaffected: the Web component uses natural layout ("100%"),
    // so it never reads inner_size.
    let rect = self.app.inner_rect_for(self.window_id.unwrap_or(0));
    PhysicalSize::new(rect.width as _, rect.height.max(0) as u32)
  }

  pub fn set_inner_size(&self, size: Size) {
    // Guard: when FLAG_RESIZABLE is 0, disallow resize (issue 4: semantic mismatch fix)
    if (self.decoration_flags.load(Ordering::Acquire) & FLAG_RESIZABLE) == 0 {
      log::warn!("[tao-ohos] set_inner_size blocked: FLAG_RESIZABLE not set");
      return;
    }
    // Issue Eulogizethesun/tauri#97: PRECISE inner sizing. The caller expects
    // the content area to become this size; OHOS `win.resize()` sets the OUTER
    // size (including the title bar), so the inner→outer conversion must know
    // the real chrome. The conversion runs on the ArkTS side at dispatch time
    // from one atomic `getWindowProperties()` snapshot
    // (`outer = inner + (windowRect − drawableRect)` per axis) — the system's
    // own numbers, fresh on every call. This replaces the former estimate
    // chain (latched decor diff + per-window decor watcher + post-hoc
    // re-dispatch) whose errors compounded through save/restore cycles into
    // the shrinking-window bug.
    //
    // Deliberate non-goals (per issue #97): no cached decor, no post-hoc
    // correction — a decor change after dispatch (e.g. runtime menubar
    // show/hide) is the caller's business; and when the precise chrome cannot
    // be read (window not created/destroyed, content not loaded) the bridge
    // call fails and we WARN + SKIP — never resize to a guessed size.
    let s = size.to_physical::<u32>(self.scale_factor());
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_inner_size") else {
        return;
      };
      let w = s.width as i64;
      let h = s.height as i64;
      self.spawn_or_queue(async move {
        if let Err(e) = client.resize_inner_window(window_id, w, h).await {
          log::warn!(
            "[tao-ohos] set_inner_size NOT applied for window {}: resize-inner failed (precise decor unavailable): {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn set_inner_size_constraints(&self, constraints: WindowSizeConstraints) {
    // Cache all four slots (Some → physical px, None → 0 = unlimited), then
    // dispatch the merged tuple — the same all-four-at-once protocol as
    // set_min_inner_size/set_max_inner_size (see `apply_window_limits`).
    let scale = self.scale_factor();
    let px = |u: Option<PixelUnit>| u.map(|p| p.to_physical::<u32>(scale).0).unwrap_or(0);
    self
      .min_inner_width
      .store(px(constraints.min_width), Ordering::Release);
    self
      .min_inner_height
      .store(px(constraints.min_height), Ordering::Release);
    self
      .max_inner_width
      .store(px(constraints.max_width), Ordering::Release);
    self
      .max_inner_height
      .store(px(constraints.max_height), Ordering::Release);
    if let Some(window_id) = self.window_id {
      self.apply_window_limits(window_id);
    }
  }

  pub fn outer_position(&self) -> Result<PhysicalPosition<i32>, error::NotSupportedError> {
    // Raw WM rect (issue #87 major-10: sourced through outer_rect_for, which
    // documents the (0,0,0,0)-before-first-callback semantics).
    let rect = self.app.outer_rect_for(self.window_id.unwrap_or(0));
    Ok(PhysicalPosition::new(rect.left, rect.top))
  }

  pub fn set_outer_position(&self, position: Position) {
    if let Some(window_id) = self.window_id {
      let physical = position.to_physical::<i32>(self.scale_factor());
      let Some(client) = self.bridge_client("set_outer_position") else {
        return;
      };
      let x = physical.x as i64;
      let y = physical.y as i64;
      self.spawn_or_queue(async move {
        if let Err(e) = client.move_window_to(window_id, x, y).await {
          log::warn!(
            "[tao-ohos] move_window_to failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn outer_size(&self) -> PhysicalSize<u32> {
    let window = self.app.outer_rect_for(self.window_id.unwrap_or(0));
    // window_rect is set by ArkTS callback, may be (0,0,0,0) initially
    // fallback to content_rect if not yet initialized
    if window.width > 0 && window.height > 0 {
      PhysicalSize::new(window.width as _, window.height as _)
    } else {
      let content = self.app.content_rect();
      PhysicalSize::new(content.width as _, content.height as _)
    }
  }

  pub fn set_min_inner_size(&self, size: Option<Size>) {
    // OHOS setWindowLimits (API 11+) writes all four (min/max w/h) at once, where
    // 0 = no limit. It is NOT incremental — a call sets the whole tuple. So we cache
    // the min here and dispatch both constraint sets together (see
    // `apply_window_limits`); otherwise calling set_min after set_max would reset
    // max to 0 (dropping the max constraint), and vice versa.
    // ⚠️ Triggers OnSizeChange — do not call frequently (appfreeze risk).
    let (min_w, min_h) = match size {
      Some(s) => {
        let p = s.to_physical::<u32>(self.scale_factor());
        (p.width, p.height)
      }
      None => (0, 0),
    };
    self.min_inner_width.store(min_w, Ordering::Release);
    self.min_inner_height.store(min_h, Ordering::Release);
    if let Some(window_id) = self.window_id {
      self.apply_window_limits(window_id);
    }
  }

  pub fn set_max_inner_size(&self, size: Option<Size>) {
    // See set_min_inner_size note. Cache max, dispatch both sets together.
    let (max_w, max_h) = match size {
      Some(s) => {
        let p = s.to_physical::<u32>(self.scale_factor());
        (p.width, p.height)
      }
      None => (0, 0),
    };
    self.max_inner_width.store(max_w, Ordering::Release);
    self.max_inner_height.store(max_h, Ordering::Release);
    if let Some(window_id) = self.window_id {
      self.apply_window_limits(window_id);
    }
  }

  pub fn set_title(&self, title: &str) {
    // OHOS setWindowTitle (API 9+, callback form). Main window + Float sub-windows
    // both support title text. Only visible when decorations enabled (decorEnabled=true).
    // Icon is NOT changeable at runtime.
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_title") else {
        return;
      };
      let title = title.to_string();
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_window_title(window_id, title).await {
          log::warn!(
            "[tao-ohos] set_window_title failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn set_visible(&self, visibility: bool) {
    // window_id 0 (main window) is valid for minimize/restore/show/move/resize/maximize
    // (unlike set_focus/set_focusable, where the main window is OS-managed and guarded
    // with `window_id > 0`), so no guard here — programmatic minimize on the main window
    // works (verified on device).
    //
    // OHOS has no direct window-hide API, so set_visible(false) uses minimize as a
    // workaround. Since is_minimized() reads the local AtomicBool mirror (not
    // getWindowStatus()), we sync the mirror here — the same pattern as
    // set_minimized() — so is_minimized() stays consistent with the visible state.
    // set_visible(true) uses restore (API14) + show_window; on API12 restore is
    // unavailable → show_window best-effort (may not restore a minimized main
    // window). The mirror is cleared regardless, matching the restore intent.
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_visible") else {
        return;
      };
      if visibility {
        self.mirror.minimized.store(false, Ordering::Release);
        // TODO(A1): replace with AppControlExt::show_ability(env) when A1 adds the action
        self.spawn_or_queue(async move {
          if let Err(e) = client.restore_window(window_id).await {
            log::warn!(
              "[tao-ohos] restore_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
          if let Err(e) = client.show_window(window_id).await {
            log::warn!(
              "[tao-ohos] show_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
      } else {
        self.mirror.minimized.store(true, Ordering::Release);
        // TODO(A1): replace with AppControlExt::hide_ability(env) when A1 adds the action
        self.spawn_or_queue(async move {
          if let Err(e) = client.minimize_window(window_id).await {
            log::warn!(
              "[tao-ohos] minimize_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
      }
    }
  }

  pub fn set_focus(&self) {
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_focus") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.focus_window(window_id).await {
          log::warn!(
            "set_focus: focus_window failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
      // The main window (window_id = 0) is dispatched too — the ArkTS side
      // routes by kind (issue Eulogizethesun/tauri#105): main window uses
      // window.shiftAppWindowFocus (11+, works for main/sub windows), Float
      // sub-windows keep raiseToAppTop (14+, subwindow-only by contract).
    }
  }

  pub fn set_focusable(&self, focusable: bool) {
    if let Some(window_id) = self.window_id {
      // #174 (2026-10-21 device arbitration): the `window_id > 0` gate that
      // historically no-op'd the main window is removed — the main window
      // (id 0) dispatches through the same bridge path as Float sub-windows.
      // The OS arbitrates: `win.setFocusable` (WindowPlugin.ets set-focusable
      // action) succeeds on the main window (probe evidence in the phase
      // report); a rejection surfaces as the structured warn below. The
      // bridge action bypasses WindowManager.setWindowFocusable (dead code),
      // so the facade-level judging observable is the Rust-side warn, not
      // the WindowManager hilog.debug success line.
      let Some(client) = self.bridge_client("set_focusable") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_window_focusable(window_id, focusable).await {
          log::warn!(
            "set_focusable: set_window_focusable failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  /// Destroys the OS-level window: Float sub-windows call `destroyWindow`.
  /// The main (UIAbility) window is system-managed: the bridge's
  /// `destroy-window` handler rejects window id 0, so for the main window
  /// this is a rejected fire-and-forget (logged as a warning) — the ability
  /// is terminated by the system, not by an in-app close. Fire-and-forget
  /// through the bridge executor, same as the other async window operations.
  pub fn close(&self) {
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("close") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.destroy_window(window_id).await {
          log::warn!(
            "close: destroy_window failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  /// Whether the app holds focus — an app-level (per-UIAbility-stage) bit,
  /// NOT per-window (O10-5): on multi-UIAbility OHOS builds every `Window`
  /// instance reads the same `HAS_FOCUS` flag, so a raw-tao consumer cannot
  /// tell from this query which of its windows is focused and must track the
  /// focused window itself. The `Focused(bool)` window events ARE routed per
  /// window id (`GainedFocus`/`LostFocus` carry the transitioning instance's
  /// id) — that is the per-window signal. tauri does not shield consumers
  /// from this: its `get_focused_window()` returns the first window whose
  /// `is_focused()` reads true, so in a focused multi-window app it resolves
  /// to an arbitrary one of them (HashMap order) — track focus through the
  /// per-window events instead.
  pub fn is_focused(&self) -> bool {
    HAS_FOCUS.load(Ordering::Relaxed)
  }

  pub fn is_always_on_top(&self) -> bool {
    // Intent flag only — OHOS has no z-order query API (see set_always_on_top).
    self.always_on_top.load(Ordering::Acquire)
  }

  // TODO(issue 4 residual, partially addressed): set_minimizable/set_maximizable/
  // set_closable still only control decoration button visibility (Float
  // @LocalStorageProp) and do not block the programmatic APIs; the main window
  // is a no-op for them. set_resizable is the exception since issue
  // Eulogizethesun/tauri#104 — see below. is_resizable etc. still read from the
  // local mirror. See doc/OHOS-window-residual-issues.md (issue 4).
  pub fn set_resizable(&self, resizable: bool) {
    self.set_decoration_flag(FLAG_RESIZABLE, resizable);
    // Issue Eulogizethesun/tauri#104: the decoration flag only covers
    // title-bar button visibility (Float sub-windows; no-op on the main
    // window) and the programmatic-resize gate (set_inner_size checks
    // FLAG_RESIZABLE). The actual user-facing switch is edge-drag resizing —
    // dispatched through the bridge: main (UIAbility) window →
    // setResizeByDragEnabled (API 14+, effective in free-window state),
    // Float sub-window → enableDrag (API 20+). Failure is logged and ignored
    // on API levels without the call (ArkTS guards with a clear error).
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_resizable") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_resize_by_drag(window_id, resizable).await {
          log::warn!(
            "[tao-ohos] set_resize_by_drag({}) failed for window {}: {:?}",
            resizable,
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn set_minimizable(&self, minimizable: bool) {
    self.set_decoration_flag(FLAG_MINIMIZABLE, minimizable);
  }

  pub fn set_maximizable(&self, maximizable: bool) {
    self.set_decoration_flag(FLAG_MAXIMIZABLE, maximizable);
  }

  pub fn set_closable(&self, closable: bool) {
    self.set_decoration_flag(FLAG_CLOSABLE, closable);
  }

  /// Common helper: update one decoration bit and dispatch to ArkTS (FloatPage LocalStorage).
  /// Dispatched via the window bridge facade fire-and-forget (no `set-decorations`
  /// variant carrying flags exists — the ArkTS WindowManager.setDecorationFlag
  /// intercepts by reading this bitfield; here we only write the local mirror +
  /// log, dispatching via the equivalent `set_window_decoration_flags` action).
  fn set_decoration_flag(&self, flag: u8, on: bool) {
    let mut flags = self.decoration_flags.load(Ordering::Acquire);
    if on {
      flags |= flag;
    } else {
      flags &= !flag;
    }
    self.decoration_flags.store(flags, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_decoration_flag") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client
          .set_window_decoration_flags(window_id, flags as i32)
          .await
        {
          log::warn!(
            "[tao-ohos] set_window_decoration_flags failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn set_minimized(&self, minimized: bool) {
    // Guard: when FLAG_MINIMIZABLE is 0, disallow minimize (issue 4: semantic mismatch fix)
    if minimized && (self.decoration_flags.load(Ordering::Acquire) & FLAG_MINIMIZABLE) == 0 {
      log::warn!("[tao-ohos] set_minimized(true) blocked: FLAG_MINIMIZABLE not set");
      return;
    }
    // Update the mirror synchronously (setter intent); apply_window_status
    // backfills the system truth when the windowStatusChange event arrives.
    self.mirror.minimized.store(minimized, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_minimized") else {
        return;
      };
      if minimized {
        self.spawn_or_queue(async move {
          if let Err(e) = client.minimize_window(window_id).await {
            log::warn!(
              "[tao-ohos] minimize_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
      } else {
        self.spawn_or_queue(async move {
          if let Err(e) = client.restore_window(window_id).await {
            log::warn!(
              "[tao-ohos] restore_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
      }
    }
  }

  pub fn is_minimized(&self) -> bool {
    self.mirror.minimized.load(Ordering::Acquire)
  }

  pub fn set_maximized(&self, maximized: bool) {
    // Guard: when FLAG_MAXIMIZABLE is 0, disallow maximize (issue 4: semantic mismatch fix)
    if maximized && (self.decoration_flags.load(Ordering::Acquire) & FLAG_MAXIMIZABLE) == 0 {
      log::warn!("[tao-ohos] set_maximized(true) blocked: FLAG_MAXIMIZABLE not set");
      return;
    }
    // Update the mirror synchronously (setter intent); apply_window_status
    // backfills the system truth when the windowStatusChange event arrives.
    self.mirror.maximized.store(maximized, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_maximized") else {
        return;
      };
      if maximized {
        self.spawn_or_queue(async move {
          if let Err(e) = client.maximize_window(window_id).await {
            log::warn!(
              "[tao-ohos] maximize_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
      } else {
        // recover() switches MAXIMIZE/FULL_SCREEN → FLOATING (API7+, public)
        self.spawn_or_queue(async move {
          if let Err(e) = client.recover_window(window_id).await {
            log::warn!(
              "[tao-ohos] recover_window failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
      }
    }
  }

  pub fn is_maximized(&self) -> bool {
    self.mirror.maximized.load(Ordering::Acquire)
  }

  pub fn set_fullscreen(&self, monitor: Option<Fullscreen>) {
    // Delegate to the WindowClient bridge facade (plugin-window). `on=true`
    // enters an immersive fullscreen (setWindowLayoutFullScreen(true) + hide
    // system bars); `on=false` reverses it. Dispatched via `spawn_or_queue` —
    // fire-and-forget at the JS level (the ArkTS handler returns after kicking
    // off async Promises), so it does not block the main thread. Replaces the
    // legacy synchronous `set_fullscreen` NAPI call which went through the dead
    // `get_helper()` transport.
    let on = monitor.is_some();
    // Sync the fullscreen mirror (read by `fullscreen()` for sync is_fullscreen
    // queries; event backfill via apply_window_status corrects it if the
    // dispatch fails). Also sync the maximized cache: fullscreen implies
    // maximized (entering fullscreen is effectively maximize + immersive),
    // exiting fullscreen calls recover() which un-maximizes. Without this,
    // is_maximized() returns stale state after a fullscreen toggle, causing
    // the next maximize/unmaximize to be a no-op.
    self.mirror.fullscreen.store(on, Ordering::Release);
    self.mirror.maximized.store(on, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_fullscreen") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_fullscreen(window_id, on).await {
          log::warn!(
            "[tao-ohos] set_fullscreen failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn fullscreen(&self) -> Option<Fullscreen> {
    // OHOS fullscreen is an immersive layout mode, not a monitor-bound
    // Fullscreen::Exclusive/Borderless(MonitorHandle) state — report the
    // mirror bit (written by set_fullscreen and backfilled from
    // windowStatusChange events via apply_window_status) as Borderless(None),
    // matching upstream. Returning None unconditionally made is_fullscreen()
    // always false, so a fullscreen toggle could enter but never exit.
    if self.mirror.fullscreen.load(Ordering::Acquire) {
      Some(Fullscreen::Borderless(None))
    } else {
      None
    }
  }

  pub fn set_decorations(&self, decorations: bool) {
    self.decorations.store(decorations, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_decorations") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_window_decorations(window_id, decorations).await {
          log::warn!(
            "[tao-ohos] set_window_decorations failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }
  pub fn set_always_on_bottom(&self, _always_on_bottom: bool) {
    // Committer Eulogizethesun/tauri#171: surface the no-op instead of failing
    // silently (signature intentionally stays `()` — no `Err`, per the md
    // wave-0 ruling), aligned with the iOS `log::warn!` precedent.
    log::warn!("[tao-ohos] set_always_on_bottom is not supported on OHOS; call ignored");
  }

  pub fn set_always_on_top(&self, always_on_top: bool) {
    // Records intent (is_always_on_top reads this) AND dispatches to OHOS
    // setWindowTopmost (API 14+, needs ohos.permission.WINDOW_TOPMOST) via the
    // window bridge facade. Main window only per OHOS docs; Float sub-windows
    // will error (caught + warned in ArkTS, non-fatal). Only effective in
    // freeform window mode.
    self.always_on_top.store(always_on_top, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_always_on_top") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_window_topmost(window_id, always_on_top).await {
          log::warn!(
            "[tao-ohos] set_window_topmost failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }
  pub fn set_ime_position(&self, position: Position) {
    // IME position: convert to physical pixels and forward to ArkTS
    // inputMethod.getController().updateCursor(CursorInfo).
    // Prerequisite: a focused edit field inside the window (an HTML input works),
    // otherwise error 12800009 client detached is returned (expected/normal).
    // Verified OK after focusing an HTML input (2026-08-19) — works for the
    // webview scenario, not an architectural limitation.
    let p = position.to_physical::<i32>(self.scale_factor());
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_ime_position") else {
        return;
      };
      let x = p.x as i64;
      let y = p.y as i64;
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_ime_position(window_id, x, y).await {
          log::warn!(
            "[tao-ohos] set_ime_position failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn is_decorated(&self) -> bool {
    self.decorations.load(Ordering::Acquire)
  }

  pub fn is_visible(&self) -> bool {
    self.mirror.visible.load(Ordering::Acquire)
  }

  pub fn is_resizable(&self) -> bool {
    self.decoration_flags.load(Ordering::Acquire) & FLAG_RESIZABLE != 0
  }

  pub fn is_minimizable(&self) -> bool {
    self.decoration_flags.load(Ordering::Acquire) & FLAG_MINIMIZABLE != 0
  }

  pub fn is_maximizable(&self) -> bool {
    self.decoration_flags.load(Ordering::Acquire) & FLAG_MAXIMIZABLE != 0
  }

  pub fn is_closable(&self) -> bool {
    self.decoration_flags.load(Ordering::Acquire) & FLAG_CLOSABLE != 0
  }

  pub fn set_window_icon(&self, _window_icon: Option<crate::icon::Icon>) {}

  pub fn set_cursor_icon(&self, icon: window::CursorIcon) {
    // TODO(issue 6): dispatched but not yet device-tested — verify style mapping
    //   coverage, touch-mode device behavior, and whether it works on Float
    //   sub-windows. See doc/OHOS-window-residual-issues.md (issue 6).
    // Set cursor style by windowId (pointer.setPointerStyleSync), dispatched via
    // the window bridge facade fire-and-forget (ArkTS side delegates to
    // WindowManager.setPointerStyle, using the real OHOS window id).
    let style = ohos_pointer_style(icon);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_cursor_icon") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_cursor_icon(window_id, style).await {
          log::warn!(
            "[tao-ohos] set_cursor_icon failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }
  pub fn set_cursor_grab(&self, grab: bool) -> Result<(), error::ExternalError> {
    // OH_WindowManager_LockCursor/UnlockCursor (NDK C API 22+, resolved via
    // dlopen in openharmony-ability). Lock is confined-follow mode — cursor
    // keeps moving within the window area, matching Windows ClipCursor
    // semantics. Only effective while the window is focused; the system
    // releases the lock automatically on focus loss (platform difference vs
    // Windows — apps that need a persistent lock re-grab on Focused(true)).
    //
    // D3.7 two-phase: the FFI needs the REAL OHOS window id
    // (getWindowProperties().id), resolved through the window bridge facade
    // (`get-real-window-id` action) because the ability crate cannot reach the
    // plugin-window facade (dependency direction plugin-window → ability).
    // Phase 1 (sync): reject API < 22 with NotSupported — same error callers
    // saw before the feature existed. Phase 2 (async): resolve the real id and
    // invoke the FFI fire-and-forget on the bridge runtime.
    if openharmony_ability::sdk_api_version() < 22 {
      return Err(error::ExternalError::NotSupported(
        error::NotSupportedError::new(),
      ));
    }
    let window_id = self
      .window_id
      .ok_or_else(|| error::ExternalError::NotSupported(error::NotSupportedError::new()))?;
    let client = match self.cached_client() {
      Some(c) => c,
      None => {
        log::warn!(
          "[tao-ohos] set_cursor_grab: WindowClient not initialized for window {}",
          window_id
        );
        return Err(error::ExternalError::NotSupported(
          error::NotSupportedError::new(),
        ));
      }
    };
    self.spawn_or_queue(async move {
      match client.get_real_window_id(window_id).await {
        Ok(real_id) => {
          if let Err(e) = set_cursor_grab(real_id as i32, grab) {
            log::warn!(
              "[tao-ohos] set_cursor_grab({}) failed for window {} (real id {}): {}",
              grab,
              window_id,
              real_id,
              e
            );
          }
        }
        Err(e) => {
          log::warn!(
            "[tao-ohos] set_cursor_grab({}): get_real_window_id failed for window {}: {:?}",
            grab,
            window_id,
            e
          );
        }
      }
    });
    Ok(())
  }

  pub fn set_content_protection(&self, enabled: bool) {
    // OH_WindowManager_SetWindowPrivacyMode (NDK C API 15+, resolved via
    // dlopen in openharmony-ability): a privacy-mode window's content is
    // excluded from screenshots, recording, and casting. Requires
    // ohos.permission.PRIVACY_WINDOW (normal / system_grant) declared in the
    // entry module.json5.
    //
    // Same D3.7 two-phase shape as set_cursor_grab above: the FFI needs the
    // REAL OHOS window id, resolved through the window bridge facade, then
    // invoked fire-and-forget. The public tao API returns `()`, so failures
    // surface as warnings, not errors.
    if openharmony_ability::sdk_api_version() < 15 {
      log::warn!("[tao-ohos] set_content_protection: requires API 15+ (window privacy mode)");
      return;
    }
    let Some(window_id) = self.window_id else {
      log::warn!("[tao-ohos] set_content_protection: no window id");
      return;
    };
    let Some(client) = self.bridge_client("set_content_protection") else {
      return;
    };
    self.spawn_or_queue(async move {
      match client.get_real_window_id(window_id).await {
        Ok(real_id) => {
          if let Err(e) = set_window_privacy_mode(real_id as i32, enabled) {
            log::warn!(
              "[tao-ohos] set_content_protection({}) failed for window {} (real id {}): {}",
              enabled,
              window_id,
              real_id,
              e
            );
          }
        }
        Err(e) => {
          log::warn!(
            "[tao-ohos] set_content_protection({}): get_real_window_id failed for window {}: {:?}",
            enabled,
            window_id,
            e
          );
        }
      }
    });
  }

  pub fn request_user_attention(&self, _request_type: Option<window::UserAttentionType>) {
    // OHOS window layer has no requestAttention API. Emulated via
    // notificationManager on the ArkTS side (fire-and-forget; the plugin
    // handles the 1600004 enable-notification retry path).
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("request_user_attention") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.request_user_attention(window_id).await {
          log::warn!(
            "[tao-ohos] request_user_attention failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn set_cursor_position(&self, _: Position) -> Result<(), error::ExternalError> {
    Err(error::ExternalError::NotSupported(
      error::NotSupportedError::new(),
    ))
  }

  pub fn cursor_position(&self) -> Result<PhysicalPosition<f64>, error::ExternalError> {
    cursor_position_from_app(&self.app)
  }

  pub fn set_ignore_cursor_events(&self, ignore: bool) -> Result<(), error::ExternalError> {
    // window_id is None for embedded webviews with no OS-level window — cursor-event
    // ignore is genuinely unsupported there, so surface NotSupported (per design D4).
    // Main window (window_id=0) and sub-windows (window_id>0) both proceed.
    let window_id = self
      .window_id
      .ok_or_else(|| error::ExternalError::NotSupported(error::NotSupportedError::new()))?;
    // Tauri `ignore=true` (pass events through to windows below) ↔ OHOS `touchable=false`
    // (window does not consume touch/mouse events). The negation lives in this tao layer;
    // the facade client passes `touchable` through verbatim. See design D4 mapping table.
    // Lock-then-clone (#184): never hold the window_client lock across the
    // dispatch.
    if let Some(client) = self.cached_client() {
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_window_touchable(window_id, !ignore).await {
          warn!(
            "set_ignore_cursor_events: set_window_touchable failed for window {}: {:?}",
            window_id, e
          );
        }
      });
    } else {
      // WindowClient not initialized (e.g. during early init) — surface NotSupported,
      // matching the old TSFN-uninitialized error path.
      warn!(
        "set_ignore_cursor_events: WindowClient not initialized for window {}",
        window_id
      );
      return Err(error::ExternalError::NotSupported(
        error::NotSupportedError::new(),
      ));
    }
    Ok(())
  }

  pub fn set_cursor_visible(&self, visible: bool) {
    // TODO(issue 6): dispatched but not yet device-tested — verify scope (global vs
    //   window-level): pointer.setPointerVisible is a global cursor toggle, while
    //   tao's semantics are window-level; under multiple windows it would also
    //   affect other windows.
    //   See doc/OHOS-window-residual-issues.md (issue 6).
    // Global cursor visibility (pointer.setPointerVisible), dispatched via the
    // window bridge facade fire-and-forget (ArkTS side delegates to
    // WindowManager.setPointerVisible).
    // Restores the dispatch that the bridge facade migration dropped to a
    // no-op — the ArkTS implementation survived, only the Rust call was lost.
    let Some(client) = self.bridge_client("set_cursor_visible") else {
      return;
    };
    self.runtime.spawn(async move {
      if let Err(e) = client.set_cursor_visible(visible).await {
        log::warn!("[tao-ohos] set_cursor_visible failed to dispatch: {:?}", e);
      }
    });
  }
  pub fn drag_window(&self) -> Result<(), error::ExternalError> {
    // OHOS startMoving (API14+) must be called in onTouch(TouchType.Down) —
    // cannot be triggered programmatically from Rust. Float sub-windows drag
    // via FloatPage title bar onTouch→startMoving; the main UIAbility window
    // has no such path, so this is a no-op there. Returns Ok (no error) since
    // drag is handled at the UI layer (FloatPage), not via this API.
    log::debug!(
      "[tao-ohos] drag_window: no-op (startMoving must be called from onTouch in FloatPage; window_id={:?})",
      self.window_id
    );
    Ok(())
  }

  pub fn drag_resize_window(
    &self,
    _direction: ResizeDirection,
  ) -> Result<(), error::ExternalError> {
    // OHOS enableDrag (API20+) allows/disables edge drag-resize, but cannot
    // programmatically trigger a specific direction resize. System handles
    // edge drag natively. This returns Ok (no error).
    // G10: log the success path too so callers can tell "edge-drag enabled"
    // apart from an actual directional resize — the _direction is ignored and
    // no directional resize is started. Mirrors the drag_window no-op log above.
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("drag_resize_window") else {
        return Ok(());
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client.set_window_draggable(window_id, true).await {
          log::warn!(
            "[tao-ohos] set_window_draggable(true) failed for window {}: {:?}",
            window_id,
            e
          );
        } else {
          log::debug!(
            "[tao-ohos] drag_resize_window: no directional resize API (_direction ignored); enableDrag(true) set for window {}",
            window_id
          );
        }
      });
    }
    Ok(())
  }

  pub fn set_background_color(&self, color: Option<crate::window::RGBA>) {
    // Respect transparent flag: silently ignore background_color when transparent=true,
    // consistent with creation-time behavior and P3 spec.
    if self.transparent {
      log::debug!("[tao-ohos] set_background_color ignored: window is transparent");
      return;
    }
    let color_u32 = rgba_to_ohos_color(false, color).unwrap_or(0xFFFFFFFF);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_background_color") else {
        return;
      };
      self.spawn_or_queue(async move {
        if let Err(e) = client
          .set_window_background_color(window_id, color_u32)
          .await
        {
          log::warn!(
            "[tao-ohos] set_window_background_color failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
    }
  }

  pub fn theme(&self) -> Theme {
    // Issue 5, 5.2 theme backfill: read the global override; on FOLLOW fall back to app.config().
    // app.config().color_mode is continuously refreshed by ConfigChanged
    // (onConfigurationUpdated), reflecting system truth — so under FOLLOW mode it
    // stays in sync with the system without manual backfill. The same
    // effective-theme computation also drives the ThemeChanged dispatch in the
    // ConfigChanged handler (issue Eulogizethesun/tauri#108) — one source, no drift.
    effective_theme(&self.app)
  }

  pub fn set_theme(&self, theme: Option<Theme>) {
    set_app_theme(&self.app, theme);
  }

  pub fn title(&self) -> String {
    String::new()
  }

  #[cfg(feature = "rwh_04")]
  pub fn raw_window_handle_rwh_04(&self) -> rwh_04::RawWindowHandle {
    unreachable!("rwh_04 is not supported on OpenHarmony");
  }

  #[cfg(feature = "rwh_05")]
  pub fn raw_window_handle_rwh_05(&self) -> rwh_05::RawWindowHandle {
    unreachable!("rwh_05 is not supported on OpenHarmony");
  }

  #[cfg(feature = "rwh_05")]
  pub fn raw_display_handle_rwh_05(&self) -> rwh_05::RawDisplayHandle {
    unreachable!("rwh_05 is not supported on OpenHarmony");
  }

  #[cfg(feature = "rwh_06")]
  // Allow the usage of HasRawWindowHandle inside this function
  #[allow(deprecated)]
  pub fn raw_window_handle_rwh_06(&self) -> Result<rwh_06::RawWindowHandle, rwh_06::HandleError> {
    if let Some(native_window) = self.app.native_window().as_ref() {
      if let Some(win) = native_window.raw_window_handle() {
        return Ok(win);
      }
      Err(rwh_06::HandleError::Unavailable)
    } else {
      Err(rwh_06::HandleError::Unavailable)
    }
  }

  #[cfg(feature = "rwh_06")]
  pub fn raw_display_handle_rwh_06(&self) -> Result<rwh_06::RawDisplayHandle, rwh_06::HandleError> {
    Ok(rwh_06::RawDisplayHandle::Ohos(
      rwh_06::OhosDisplayHandle::new(),
    ))
  }

  pub fn config(&self) -> Configuration {
    self.app.config()
  }

  pub fn content_rect(&self) -> Rect {
    self.app.content_rect()
  }

  pub fn window_id(&self) -> Option<i64> {
    self.window_id
  }

  /// Returns the `BridgeRuntime` for this window's `OpenHarmonyApp`.
  /// Used by wry's bridge-based webview backend to construct `WebviewClient::from_bridge`.
  pub(crate) fn bridge_runtime(
    &self,
  ) -> openharmony_ability::napi_ohos::Result<openharmony_ability::BridgeRuntime> {
    self.app.bridge()
  }

  pub fn current_monitor(&self) -> Option<monitor::MonitorHandle> {
    // Display containing this window's outer-rect top-left (global coordinate
    // space, kept fresh by the windowRectChange backfill). Falls back to the
    // primary display when the rect is still unset (before the first callback)
    // or the point misses every display (issue Eulogizethesun/tauri#106).
    let rect = self.app.window_rect_for(self.window_id.unwrap_or(0));
    let handle = MonitorHandle::from_point(&self.app, rect.left as f64, rect.top as f64)
      .unwrap_or_else(|| MonitorHandle::primary_for_app(&self.app));
    Some(monitor::MonitorHandle { inner: handle })
  }

  pub fn primary_monitor(&self) -> Option<monitor::MonitorHandle> {
    Some(monitor::MonitorHandle {
      inner: MonitorHandle::primary_for_app(&self.app),
    })
  }
}

#[derive(Default, Clone, Debug)]
pub struct OsError;

use std::fmt::{self, Display, Formatter};
impl Display for OsError {
  fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), fmt::Error> {
    write!(fmt, "OpenHarmony OS Error")
  }
}

impl Drop for Window {
  /// Unregisters the state mirror from [`WINDOW_MIRRORS`]. (The former
  /// decor-watcher teardown is gone with the watcher itself — issue
  /// Eulogizethesun/tauri#97 removed the whole estimation chain.)
  fn drop(&mut self) {
    if let Some(window_id) = self.window_id {
      WINDOW_MIRRORS
        .lock()
        .expect("WINDOW_MIRRORS poisoned")
        .remove(&window_id);
      if let Ok(mut rects) = LAST_DISPATCHED_RECTS.lock() {
        rects.remove(&window_id);
      }
      // Issue 7: discard ops still queued against this window (its creation
      // handshake never completed before the handle went away), recycle the
      // UIAbility pending-registry entry (waker included — mirrors the
      // start_ui_ability Err-leg rollback in `Window::new`; without it, an
      // entry whose spawn AMS silently swallowed stays registered with a
      // live waker until process exit), and drop the Float pending-registry
      // entry so a late settle notify no-ops instead of waking a dead id.
      // Both unregisters are idempotent no-ops for unknown ids (the primary
      // window's id 0 was never registered as pending).
      openharmony_ability::unregister_pending_ui_ability(window_id);
      drop_pending_window_ops(window_id);
      openharmony_ability::unregister_pending_float(window_id);
    }
  }
}

pub fn keycode_to_scancode(_code: KeyCode) -> Option<u32> {
  None
}

pub fn keycode_from_scancode(_scancode: u32) -> KeyCode {
  KeyCode::Unidentified(NativeKeyCode::Unidentified)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::dpi::{LogicalPosition, LogicalSize};

  // Size regression coverage (set → readback exact equality, float decor=0
  // readback, save/restore zero drift — run on a DPR ≠ 1 device)
  // intentionally lives in the api demo device suite
  // (tauri repo, examples/api/src/lib/tests/window-ops.ts): a cargo test
  // binary has no UIAbility/ArkTS window, so an ohos `Window` cannot be
  // constructed outside a bridge session, and the chrome arithmetic itself
  // sits on the ArkTS side (openharmony-ability WindowPlugin.ets
  // resize-inner).
  #[test]
  fn rgba_to_ohos_color_transparent_returns_transparent_black() {
    assert_eq!(rgba_to_ohos_color(true, None), Some(0x00000000));
    assert_eq!(
      rgba_to_ohos_color(true, Some((255, 0, 0, 255))),
      Some(0x00000000)
    );
  }

  #[test]
  fn rgba_to_ohos_color_none_bg_returns_none() {
    assert_eq!(rgba_to_ohos_color(false, None), None);
  }

  #[test]
  fn rgba_to_ohos_color_packs_argb() {
    assert_eq!(
      rgba_to_ohos_color(false, Some((255, 128, 0, 200))),
      Some(0xC8FF8000)
    );
  }

  #[test]
  fn rgba_to_ohos_color_opaque_white() {
    assert_eq!(
      rgba_to_ohos_color(false, Some((255, 255, 255, 255))),
      Some(0xFFFFFFFF)
    );
  }

  #[test]
  fn rgba_to_ohos_color_zero_alpha() {
    assert_eq!(
      rgba_to_ohos_color(false, Some((0, 0, 0, 0))),
      Some(0x00000000)
    );
  }

  // T26-N4: the StartOptions geometry conversion (logical px → physical px
  // tuples for the UIAbility spawn). Unspecified axes must stay `None` (the
  // OS default sizing applies); DPR ≠ 1 rescales via dpi's round-half-away
  // i32 conversion; zero/negative values pass through unchanged for the
  // ArkTS handler to reject asynchronously.
  #[test]
  fn start_options_geometry_unspecified_axes_stay_none() {
    let (size, pos) = start_options_geometry(None, None, 2.0);
    assert_eq!(size, None);
    assert_eq!(pos, None);
  }

  #[test]
  fn start_options_geometry_identity_at_dpr_one() {
    let (size, pos) = start_options_geometry(
      Some(Size::Logical(LogicalSize::new(800.0, 600.0))),
      Some(Position::Logical(LogicalPosition::new(10.0, 20.0))),
      1.0,
    );
    assert_eq!(size, Some((800, 600)));
    assert_eq!(pos, Some((10, 20)));
  }

  #[test]
  fn start_options_geometry_rescales_at_non_integer_dpr() {
    // 1.5×: dpi's i32 Pixel conversion rounds half away from zero
    // (10.5 → 11; 800/400/20 scale exactly).
    let (size, pos) = start_options_geometry(
      Some(Size::Logical(LogicalSize::new(800.0, 400.0))),
      Some(Position::Logical(LogicalPosition::new(7.0, 20.0))),
      1.5,
    );
    assert_eq!(size, Some((1200, 600)));
    assert_eq!(pos, Some((11, 30)));
    // 2.0×: plain doubling.
    let (size, _) = start_options_geometry(
      Some(Size::Logical(LogicalSize::new(800.0, 600.0))),
      None,
      2.0,
    );
    assert_eq!(size, Some((1600, 1200)));
  }

  #[test]
  fn start_options_geometry_physical_values_ignore_scale() {
    let (size, pos) = start_options_geometry(
      Some(Size::Physical(PhysicalSize::new(1024, 768))),
      Some(Position::Physical(PhysicalPosition::new(5, 6))),
      2.5,
    );
    assert_eq!(size, Some((1024, 768)));
    assert_eq!(pos, Some((5, 6)));
  }

  #[test]
  fn start_options_geometry_zero_and_negative_pass_through() {
    // Not clamped and not dropped here: the ArkTS start-ui-ability handler
    // rejects non-positive width/height asynchronously (T26-N2's
    // note_ui_ability_spawn_rejection surface), so the conversion itself
    // must forward the request verbatim.
    let (size, pos) = start_options_geometry(
      Some(Size::Logical(LogicalSize::new(0.0, -5.0))),
      Some(Position::Logical(LogicalPosition::new(-3.0, 0.0))),
      2.0,
    );
    assert_eq!(size, Some((0, -10)));
    assert_eq!(pos, Some((-6, 0)));
  }

  // O10-6: PENDING_WINDOW_OPS lifecycle — hold while the creation handshake is
  // pending, FIFO replay once it completes, discard on handle drop. The
  // enqueue leg replicates dispatch_or_queue's one-line push (a `Window`
  // cannot be constructed outside a bridge session, per the note above); the
  // drain, the drop and the readiness registry are the real production code.
  // Sentinel ids far above any real OHOS windowId / NEXT_WINDOW_ID counter
  // value keep the static registries collision-free under parallel tests.
  #[test]
  fn pending_window_ops_hold_replay_and_drop() {
    const HELD_ID: i64 = i64::MAX - 1;
    const DROPPED_ID: i64 = i64::MAX - 2;
    let queue = || {
      PENDING_WINDOW_OPS
        .lock()
        .expect("PENDING_WINDOW_OPS poisoned")
    };
    // Idempotent reset so a previously failed run cannot leak registry
    // entries that would pin these ids as "not ready".
    openharmony_ability::unregister_pending_ui_ability(HELD_ID);
    openharmony_ability::unregister_pending_ui_ability(DROPPED_ID);
    drop_pending_window_ops(HELD_ID);
    drop_pending_window_ops(DROPPED_ID);

    // 1. Hold: a queued op stays unexecuted while the handshake is pending.
    openharmony_ability::register_pending_ui_ability(HELD_ID);
    assert!(!openharmony_ability::is_window_ready(HELD_ID));
    // Two ops with an order log so the FIFO claim is asserted, not just
    // "some op ran".
    let order: Arc<std::sync::Mutex<Vec<u32>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    for step in 1..=2u32 {
      let order = order.clone();
      queue()
        .entry(HELD_ID)
        .or_default()
        .push(Box::new(move || {
          order.lock().expect("order log poisoned").push(step);
        }));
    }
    drain_ready_pending_window_ops();
    assert!(
      order.lock().expect("order log poisoned").is_empty(),
      "op ran while the creation handshake was pending"
    );
    assert!(
      queue().contains_key(&HELD_ID),
      "queue dropped the op while the handshake was pending"
    );

    // 2. Replay: the handshake completing (registerUIAbilityStage) makes the
    //    id ready and the next drain replays in FIFO order and empties its
    //    queue.
    openharmony_ability::register_ui_ability_stage(HELD_ID);
    assert!(openharmony_ability::is_window_ready(HELD_ID));
    drain_ready_pending_window_ops();
    assert_eq!(
      *order.lock().expect("order log poisoned"),
      vec![1, 2],
      "queued ops not replayed in FIFO order"
    );
    assert!(!queue().contains_key(&HELD_ID), "drained entry not removed");
    openharmony_ability::unregister_pending_ui_ability(HELD_ID);

    // 3. Drop: ops still queued for a handle that went away are discarded
    //    without running (Window::drop leg).
    openharmony_ability::register_pending_ui_ability(DROPPED_ID);
    let dropped_ran = Arc::new(AtomicBool::new(false));
    let flag = dropped_ran.clone();
    queue()
      .entry(DROPPED_ID)
      .or_default()
      .push(Box::new(move || flag.store(true, Ordering::SeqCst)));
    drop_pending_window_ops(DROPPED_ID);
    assert!(!dropped_ran.load(Ordering::SeqCst), "dropped op ran");
    assert!(
      !queue().contains_key(&DROPPED_ID),
      "dropped entry not removed"
    );
    openharmony_ability::unregister_pending_ui_ability(DROPPED_ID);
  }

  // --- #184: lazy bridge-client re-fetch policy (design D8). The helper is
  // generic over the client type because a real WindowClient needs a live
  // bridge session (NAPI) that a test binary does not have; the policy under
  // test is hit / miss-overwrite / warn-once-debounce, and the production
  // call site instantiates C = WindowClient.

  #[derive(Clone, PartialEq, Debug)]
  struct DummyClient(u32);

  /// UT for #184: a cache hit must clone the cached client and never touch
  /// the re-fetch path (no warn either).
  #[test]
  fn ut_bridge_client_hit_skips_refetch() {
    let cache: std::sync::Mutex<Option<DummyClient>> = std::sync::Mutex::new(Some(DummyClient(7)));
    let warned = AtomicBool::new(false);
    let refetched = Arc::new(AtomicBool::new(false));
    let flag = refetched.clone();
    let got = bridge_client_cached_or_refetch(
      &cache,
      move || {
        flag.store(true, Ordering::SeqCst);
        Some(DummyClient(99))
      },
      "ut-op",
      &warned,
    );
    assert_eq!(got, Some(DummyClient(7)));
    assert!(!refetched.load(Ordering::SeqCst), "cache hit re-fetched");
    assert!(!warned.load(Ordering::SeqCst), "cache hit tripped the warn");
  }

  /// UT for #184: a cache miss re-fetches through the app handle once and
  /// overwrites the cache, so the next call is served from the cache without
  /// a second re-fetch; success must not trip the warn debounce.
  #[test]
  fn ut_bridge_client_miss_refetches_and_overwrites_cache() {
    let cache: std::sync::Mutex<Option<DummyClient>> = std::sync::Mutex::new(None);
    let warned = AtomicBool::new(false);
    let calls = Arc::new(AtomicU32::new(0));
    let counter = calls.clone();
    let got = bridge_client_cached_or_refetch(
      &cache,
      move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Some(DummyClient(5))
      },
      "ut-op",
      &warned,
    );
    assert_eq!(got, Some(DummyClient(5)));
    assert_eq!(
      calls.load(Ordering::SeqCst),
      1,
      "re-fetch ran more than once"
    );
    let counter = calls.clone();
    let got2 = bridge_client_cached_or_refetch(
      &cache,
      move || {
        counter.fetch_add(1, Ordering::SeqCst);
        Some(DummyClient(6))
      },
      "ut-op",
      &warned,
    );
    assert_eq!(got2, Some(DummyClient(5)), "overwritten cache not served");
    assert_eq!(calls.load(Ordering::SeqCst), 1, "second call re-fetched");
    assert!(
      !warned.load(Ordering::SeqCst),
      "successful re-fetch tripped the warn debounce"
    );
  }

  /// UT for #184: a persistent re-fetch failure returns None with the warn
  /// debounce tripping exactly once — the flag flips false→true on the first
  /// failure and holds on subsequent failures. Warn emission is gated 1:1 on
  /// that false→true transition (asserted by the reset leg below), so
  /// consecutive failures with a fresh flag can never emit a second warn.
  #[test]
  fn ut_bridge_client_refetch_failure_returns_none_and_warns_once() {
    let cache: std::sync::Mutex<Option<DummyClient>> = std::sync::Mutex::new(None);
    let warned = AtomicBool::new(false);
    // Bridge stays unavailable: app.window() errors → mapped to None here.
    let refetch = || None::<DummyClient>;
    assert_eq!(
      bridge_client_cached_or_refetch(&cache, refetch, "ut-op", &warned),
      None
    );
    assert!(
      warned.load(Ordering::SeqCst),
      "first failure did not trip the warn debounce"
    );
    assert_eq!(
      bridge_client_cached_or_refetch(&cache, refetch, "ut-op", &warned),
      None
    );
    assert!(
      warned.load(Ordering::SeqCst),
      "debounce flag must hold across consecutive failures"
    );
    warned.store(false, Ordering::SeqCst);
    assert_eq!(
      bridge_client_cached_or_refetch(&cache, refetch, "ut-op", &warned),
      None
    );
    assert!(
      warned.load(Ordering::SeqCst),
      "warn is not gated on the false→true transition of the debounce flag"
    );
  }

  // --- #173: per-window focused bit on the state mirror (design D7). The
  // app-level HAS_FOCUS static stays the is_focused() source; the mirror bit
  // only records per-window system truth for the event-loop drain.

  /// UT for #173: WindowStateMirror::new seeds the focused bit from the
  /// HAS_FOCUS snapshot taken at mirror registration time.
  #[test]
  fn ut_window_state_mirror_seeds_focused_bit() {
    assert!(
      WindowStateMirror::new(true).focused.load(Ordering::SeqCst),
      "new(true) must seed focused"
    );
    assert!(
      !WindowStateMirror::new(false).focused.load(Ordering::SeqCst),
      "new(false) must seed unfocused"
    );
  }

  /// UT for #173: set_mirror_focused writes the registered mirror through
  /// WINDOW_MIRRORS, and tolerates a miss (unknown id / already-dropped
  /// window Weak) without panicking — event-loop focus arms must never fail
  /// on a mirror that went away.
  #[test]
  fn ut_set_mirror_focused_writes_and_tolerates_miss() {
    // Exact-contents coexistence with event_loop.rs's live_window_ids test.
    let _guard = WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    const SENTINEL: i64 = i64::MAX - 3;
    let mirror = Arc::new(WindowStateMirror::new(false));
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .insert(SENTINEL, Arc::downgrade(&mirror));
    set_mirror_focused(SENTINEL, true);
    assert!(mirror.focused.load(Ordering::SeqCst));
    set_mirror_focused(SENTINEL, false);
    assert!(!mirror.focused.load(Ordering::SeqCst));
    // Unknown id: silently skipped.
    set_mirror_focused(i64::MIN, true);
    // Dropped window (Weak::upgrade fails): silently skipped.
    drop(mirror);
    set_mirror_focused(SENTINEL, true);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&SENTINEL);
  }
}

// --- S9 fmt batch: OsError Display (appended at file end, keeps existing line numbers) ---
#[cfg(test)]
mod fmt_tests {
  use super::OsError;

  #[test]
  fn os_error_display_writes_message() {
    assert_eq!(format!("{}", OsError), "OpenHarmony OS Error");
    assert_eq!(format!("{:?}", OsError), "OsError"); // Debug derive
  }
}
