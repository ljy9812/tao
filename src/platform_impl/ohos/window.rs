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
// (0 = main, >0 = sub-window) as the inner value makes per-window event
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
/// container (UIAbility) or is a sub-window of an existing UIAbility
/// container (SubWindow) -- created via `windowStage.createSubWindow` and
/// destroyed with its host UIAbility (issue Eulogizethesun/tauri#240: named
/// for the OS concept; "Float" conflated this with the OS-wide floating
/// window type TYPE_FLOAT, which this kind does not use).
///
/// Default is UIAbility for the first window, SubWindow for any later window.
/// The first window must be a UIAbility (SubWindow sub-windows attach to an
/// existing UIAbility container); subsequent UIAbility windows spawn a new
/// EntryAbility instance via `start_ui_ability` with a pre-allocated window
/// id (openspec multi-uiability-windows, design D1/D2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OHOSWindowKind {
  UIAbility,
  SubWindow,
}

/// Latches once the first (UIAbility) window has been created. Only
/// constraint it enforces: the first window must be a UIAbility -- a
/// SubWindow window has no container to attach to. Replaces the old
/// `UIABILITY_CREATED` singleton guard (design D2).
///
/// Process-lifetime bootstrap marker (committer Eulogizethesun/tauri#169):
/// it is deliberately NOT reset on window `Drop` -- `Drop` only unregisters
/// the per-window state mirror, so the latch keeps recording "a UIAbility
/// container exists in this process". Later windows therefore default to
/// `OHOSWindowKind::SubWindow`, while an explicit UIAbility request stays
/// legal (it spawns a fresh EntryAbility via the spawn chain).
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
  /// `Window::is_focused()` -- that stays on the app-level HAS_FOCUS static
  /// (changing the read source was explicitly out of scope for #173).
  pub(crate) focused: AtomicBool,
}

impl WindowStateMirror {
  pub(crate) fn new(focused: bool) -> Self {
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
// The lock serializes every test that touches the process-global input/focus
// statics -- WINDOW_MIRRORS + HAS_FOCUS/LAST_FOCUSED (focus_scale_tests,
// live_window_ids_tests, window.rs mirror test) and KEYBOARD_MODIFIERS (the
// axis tests' ctrl= assertions vs the gained-focus CONTROL write). One shared
// domain, because the same handler writes several of these statics per call
// (handle_gained_focus stores the focus trio AND clears the modifiers).

/// Design D2 confirmation channel (a): destroy requests this process issued
/// through [`Window::close`] whose bridge `destroy-window` resolved Ok
/// (including the idempotent already-gone ack). The `run_loop` dispatch drains
/// the queue and synthesizes `Event::WindowEvent::Destroyed`, giving the
/// embedding runtime a deterministic destruction confirmation even when the
/// ArkTS notifyWindowClose loopback (channel (b)) cannot fire -- e.g. a window
/// already destroyed on the ArkTS side has no fresh `aboutToDisappear`.
/// Belt-and-suspenders next to (b): the downstream executor is idempotent, so
/// a double confirmation is absorbed. Enqueue happens on the bridge executor
/// thread; the drain runs on the main thread -- hence the mutex.
pub(crate) static DESTROY_CONFIRMATIONS: std::sync::LazyLock<std::sync::Mutex<Vec<i64>>> =
  std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

pub(crate) fn enqueue_destroy_confirmation(window_id: i64) {
  DESTROY_CONFIRMATIONS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .push(window_id);
}

pub(crate) fn drain_destroy_confirmations() -> Vec<i64> {
  std::mem::take(
    &mut *DESTROY_CONFIRMATIONS
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner()),
  )
}

/// Failure twin of [`DESTROY_CONFIRMATIONS`] (issue Eulogizethesun/tauri#255):
/// destroy requests this process issued through [`Window::close`] that FAILED
/// to reach the bridge -- (b) the window's bridge client is unavailable, and
/// (c) the async `destroy_window` resolve returned Err. (a) A window with no
/// OHOS id (`window_id == None`) fails before anything can be keyed: there is
/// no id to enqueue, so that path stays a log-only failure -- the embedding
/// runtime can observe the None id directly at its initiate site
/// (`WindowExtOpenHarmony::window_id()`), which is where its in-flight close
/// guard lives.
///
/// The embedding runtime (tauri-runtime-wry) drains this queue through
/// [`drain_destroy_failures`] and uses it to clear the in-flight close guard
/// it set before dispatching, so a failed destroy does not make the next
/// USER-initiated `CloseRequested` be misread as the destruction loopback
/// (which would skip the app-facing dispatch and tear down accounting for a
/// window that is still alive). tao itself announces nothing: no Event variant
/// is added -- the failure is "accounted, not announced" and the runtime owns
/// the guard (design D3 keeps tao's failed resolve silent on the event lane).
/// Enqueue happens on the dispatching thread (main thread or bridge executor);
/// the drain runs on the embedding runtime's thread -- hence the mutex.
pub(crate) static DESTROY_FAILURES: std::sync::LazyLock<std::sync::Mutex<Vec<i64>>> =
  std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

pub(crate) fn enqueue_destroy_failure(window_id: i64) {
  DESTROY_FAILURES
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .push(window_id);
}

/// Drains the failure twin of the destroy-confirmation queue: the OHOS window
/// ids whose `close()`-dispatched `destroy-window` bridge call failed (bridge
/// client unavailable, or the async resolve returned Err -- see
/// [`DESTROY_FAILURES`]). Consuming read: each id is returned exactly once.
/// Exported for the embedding runtime (tauri-runtime-wry), which pairs it with
/// its own in-flight close guard; tao never drains it itself.
pub fn drain_destroy_failures() -> Vec<i64> {
  std::mem::take(
    &mut *DESTROY_FAILURES
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner()),
  )
}

/// Writes the per-window focused bit (#173) through [`WINDOW_MIRRORS`].
/// A miss (unknown id, or the mirror already unregistered because the
/// `Window` was dropped) is silently skipped -- same tolerance as the
/// unregister-vs-backfill timing of the other mirror bits. Called by the
/// event-loop focus arms right after the app-level HAS_FOCUS store.
pub(crate) fn set_mirror_focused(window_id: i64, focused: bool) {
  let mirror = WINDOW_MIRRORS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .get(&window_id)
    .and_then(|weak| weak.upgrade());
  if let Some(mirror) = mirror {
    mirror.focused.store(focused, Ordering::Release);
  }
}

/// #180(a) (design D7): reads the per-window focused bit through
/// [`WINDOW_MIRRORS`] -- the drain's paired-blur synthesis gates on it
/// ("P 仍标记 focused=true" per the focus-state spec) so a holder whose bit
/// was already cleared by a real stage blur is not double-blurred. An unknown
/// or dropped mirror reads `false` (nothing to pair-clear).
pub(crate) fn mirror_focused_bit(window_id: i64) -> bool {
  WINDOW_MIRRORS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner())
    .get(&window_id)
    .and_then(|weak| weak.upgrade())
    .is_some_and(|mirror| mirror.focused.load(Ordering::Acquire))
}

/// #184 lazy bridge-client cache policy (shared seam for `bridge_client` and
/// its unit tests). On a cache hit the cached client is cloned; on a miss the
/// client is re-fetched once through `refetch` (the app handle's
/// `WindowExt::window()`, a non-blocking RwLock read) and the cache is
/// overwritten on success so later calls are served from it; a persistent
/// failure returns `None` with a warn emitted exactly once -- gated on the
/// false->true transition of `warned` (swap-based debounce, so hot paths like
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
/// UIAbility windows awaiting stage registration (issue 7), and the
/// sub-windows awaiting the ArkTS creation chain to settle (issue-7 addendum; gated
/// only when the ProcessInitializer capability handshake armed the Rust-side
/// registry -- a stale HAR keeps sub-window ops on the old fire-and-forget path).
/// Keyed by OHOS window id, FIFO per window. Drained by the event loop
/// ([`drain_ready_pending_window_ops`]) whenever the D7 registration waker or
/// the sub-window settle waker fires, or inline by [`Window::dispatch_or_queue`]
/// when registration lands between its ready-check and push (push-then-check,
/// the same TOCTOU discipline as wry's pending_ops).
static PENDING_WINDOW_OPS: std::sync::LazyLock<
  std::sync::Mutex<std::collections::HashMap<i64, Vec<PendingWindowOp>>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Event-loop drain (called on every MainEvent dispatch after a D7
/// registration waker or sub-window settle waker wakes the loop): replay queued
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

/// #168 (design D1): the enqueue/dispatch body of [`Window::dispatch_or_queue`],
/// extracted as a free function so the creation-attr seeding path and the UTs
/// exercise the exact production mechanic: ready -> run inline (no queue
/// entry), not-ready -> hold in [`PENDING_WINDOW_OPS`] until
/// [`drain_ready_pending_window_ops`] replays it. A `Window` cannot be
/// constructed outside a bridge session, so the UT drives this entry point
/// directly with sentinel ids (same discipline as the O10-6 queue test).
pub(crate) fn dispatch_or_queue_op(window_id: i64, op: PendingWindowOp) {
  if openharmony_ability::is_window_ready(window_id) {
    op();
    return;
  }
  // Push-then-check (wry pending_ops TOCTOU): if the handshake flipped
  // between the ready check above and this push, the settle waker already
  // fired and no further wake is coming -- drain this id now. The entry may
  // already hold earlier queued ops, so drain it FIFO, not just this op.
  let mut map = PENDING_WINDOW_OPS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  // First push for this id (empty -> non-empty) is the creation-latency
  // marker (O10-3): one warn per queue lifetime is expected while a
  // spawned instance's handshake completes -- a queue that is never
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

/// PR review fix T1: enqueue a window op whose bridge client is resolved at
/// EXECUTION time instead of call time. The eager `bridge_client` resolution
/// in the setters ran BEFORE the queue check, so a seed fired while the
/// client was not yet resolvable (the #184 early-init state) returned early
/// and nothing ever entered [`PENDING_WINDOW_OPS`] -- the build()-tail
/// creation seeds (always_on_top / fullscreen / maximized / focusable /
/// decoration flags / size limits) were permanently lost, and the handshake
/// drain had nothing to replay. The op body re-resolves through the captured
/// app handle at replay (a non-blocking RwLock read) and spawns the async
/// bridge call on the executor; a still-unresolvable client degrades to a
/// structured warn (the op dies, exactly the pre-#184 no-op semantics, but
/// only after the queue had its chance).
///
/// Coverage boundary (PR#28 review T28-3): the hold covers exactly the
/// windows whose creation handshake is pending (spawned UIAbility instances
/// and sub-windows). The primary main window (id 0) is always ready, so its
/// deferred ops execute INLINE at call time -- the in-op re-resolve is a
/// same-instant second attempt, and a still-unresolvable client drops the
/// seed with the structured warn. There is no handshake to await for id 0,
/// so "creation seeds survive a not-yet-resolvable bridge client" holds for
/// the pending-handshake populations, not for the primary window.
///
/// Generic over the client type: a real `WindowClient` needs a live bridge
/// session that a test binary does not have -- the UT drives this entry point
/// with a sentinel client (same discipline as
/// [`bridge_client_cached_or_refetch`] / [`dispatch_or_queue_op`]).
fn enqueue_deferred_window_op<C, F, Fut>(
  resolve: impl Fn() -> Option<C> + Send + 'static,
  runtime: &BridgeExecutor,
  window_id: i64,
  op: &str,
  make_future: F,
) where
  F: FnOnce(C) -> Fut + Send + 'static,
  Fut: std::future::Future<Output = ()> + Send + 'static,
{
  let runtime = runtime.clone();
  let op = op.to_string();
  dispatch_or_queue_op(
    window_id,
    Box::new(move || match resolve() {
      Some(client) => runtime.spawn(make_future(client)),
      None => log::warn!(
        "[tao-ohos] {op}: bridge client still unresolvable at replay — deferred window op dropped"
      ),
    }),
  );
}

pub(crate) struct Window {
  app: OpenHarmonyApp,
  window_id: Option<i64>,
  /// Resolved window kind (issue Eulogizethesun/tauri#238): UIAbility vs
  /// SubWindow. LOCAL decision input only -- consumed by the
  /// UIAbility-kind decoration gates (the id-0 main window and spawned
  /// instances alike; three button setters error out, is_* queries
  /// report the OS-managed truth), the creation-attr seed policy
  /// (D7) and the runtime-wry `is_ui_ability_main` capture (via
  /// `WindowExtOpenHarmony::window_kind`).
  ///
  /// Distinct from what issue Eulogizethesun/tauri#97 deleted: that removed
  /// the old kind field together with the `window_type` DOWNLINK dead chain
  /// (whose only consumer was populating `WindowCreateParams.window_type`, a
  /// field the create TSFN tuple never carried). #97's zero-consumers
  /// rationale does not apply to the kind itself -- it was always resolved in
  /// `new()`, and #238 gives it real local consumers again. The
  /// `window_type` downlink (and the creation window-type enum that fed it) is
  /// deleted outright since #240.
  kind: OHOSWindowKind,
  /// Bridge facade for async window operations. Mutex'd (#184): a `None`
  /// captured at construction (bridge not yet ready during early init) is
  /// lazily re-fetched on the next `bridge_client` call and overwritten on
  /// success, instead of pinning the no-op degradation for the window's
  /// whole lifetime. Direct reads outside `bridge_client` go through
  /// `cached_client` (lock, clone, drop the guard).
  window_client: std::sync::Mutex<Option<openharmony_ability_plugin_window::WindowClient>>,
  /// Warn-once debounce for bridge-client re-fetch failures (#184): the lazy
  /// re-fetch runs on hot paths (e.g. the decoration seed), so the failure
  /// warn is emitted only on the false->true transition of this flag.
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
  //   See doc/OHOS窗口遗留问题.md in the tauri repo (issue 5, 5.1/5.2/5.3).
  /// always_on_top intent flag (OHOS has no direct API; records intent only, see set_always_on_top).
  always_on_top: AtomicBool,
  /// Decoration button availability bitfield. bit0 closable, bit1 maximizable,
  /// bit2 minimizable, bit3 resizable. Defaults to 0b1111=15 (all enabled).
  ///
  /// Sub-window contract (#238): written by the creation seed and the runtime
  /// setters, dispatched to the ArkTS LocalStorage that drives the
  /// SubWindowPage buttons.
  ///
  /// UIAbility-kind window contract (the bootstrap main window AND spawned
  /// instances -- doc corrected per PR#28 review T28-7): the bitfield IS
  /// written locally -- the creation seed caches the attr-derived
  /// `mirror_flags` (the resizable clear included) and the runtime
  /// `set_resizable` writes the resizable bit -- but it is never DISPATCHED
  /// for these windows (the ArkTS setDecorationFlags handler throws, #238;
  /// the runtime setter skips the dispatch too, T28-2), and the three button
  /// bits are OS-managed (the button setters error out). The
  /// set_minimized/set_maximized guards read FLAG_MINIMIZABLE/FLAG_MAXIMIZABLE
  /// and therefore always pass for UIAbility windows -- consistent with the
  /// OS-managed "always available" reading (disclosed, D7).
  /// is_minimizable/is_maximizable/is_closable read `kind` for UIAbility
  /// windows, not this field; is_resizable reads the cached resizable bit on
  /// every kind.
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
  /// Title cache: the `title()` read-back for the `set_title` intent.
  /// OHOS has no synchronous title query (the ArkTS LocalStorage 'title'
  /// slot is write-only from tao), so the getter reads this app-owned cache
  /// — the same "app-owned local mirror" family as `decorations`. NOT
  /// seeded from the builder title: creation never displays it on OHOS
  /// (it only serves as the sub-window label fallback), so a seeded cache
  /// would report a title the title bar never showed. None until the first
  /// `set_title` → `title()` returns the empty string (pre-fix behavior
  /// for unset windows).
  title: std::sync::Mutex<Option<String>>,
}

/// OHOS `WindowStatusType` (API 11+) — the system's window mode, reported via
/// `window.on('windowStatusChange')`. Values match the ArkTS enum order:
/// 1=FULL_SCREEN, 2=MAXIMIZE, 3=MINIMIZE, 4=FLOATING, 5=SPLIT_SCREEN.
///
/// Used by [`WindowStateMirror::apply_window_status`] to backfill system truth
/// into tao mirror bits. See doc/OHOS窗口遗留问题.md in the tauri repo (issue 5, 5.3).
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

/// #168 (design D1): the creation-time seed/replay plan derived from the
/// builder attrs. Pure decision record -- the dispatch rides the existing
/// setter channel ([`Window::dispatch_or_queue`]), while the decision itself
/// is UT-able without a bridge session (a `Window` cannot be constructed
/// outside one).
struct CreationAttrPlan {
  /// Seed `set_always_on_top(true)` -- default `false`, so only a builder
  /// `true` fires.
  always_on_top: bool,
  /// Local mirror bitfield -- stored into `Window::decoration_flags` whenever
  /// it differs from the `FLAG_ALL_DECORATIONS` default (the `set_maximized`
  /// FLAG_MAXIMIZABLE guard and the `set_inner_size` FLAG_RESIZABLE gate read
  /// that cache). Recorded for BOTH window kinds, decoupled from the
  /// `set_window_decoration_flags` dispatch below.
  mirror_flags: u8,
  /// Seed `set_window_decoration_flags` with the attr-derived bitfield;
  /// `None` = identical to the `FLAG_ALL_DECORATIONS` default -> zero
  /// dispatch. Always `None` on a UIAbility-kind window (the bootstrap main
  /// window and spawned instances alike): the ArkTS setDecorationFlags
  /// handler throws for UIAbility-kind windows (the id-0 main window and
  /// spawned UIAbility instances; issue Eulogizethesun/tauri#238), so a
  /// dispatch there is a guaranteed-fail warn -- sub-windows only.
  decoration_flags: Option<u8>,
  /// Resize seed (#238 design D7 + issue Eulogizethesun/tauri#104):
  /// `Some(false)` when `attrs.resizable` is `false` -- the resizable attr's
  /// real surface (the edge-drag switch), the same leg the runtime
  /// `set_resizable` dispatches. `None` = zero dispatch.
  ///
  /// Branch landing (PR#28 review T28-5): the branch is decided by the ArkTS
  /// `set-resize-by-drag` routing ALONE, which keys on `windowId === 0` --
  /// the primary window (id 0) lands on `setResizeByDragEnabled` while every
  /// other window (spawned UIAbility instances and sub-windows, id > 0)
  /// lands on the `enableDrag` branch. This plan never re-decides the branch
  /// locally -- the two ends agree by construction instead of by duplicated
  /// classification.
  ///
  /// PR#28 review gate F1: the seed is planned for EVERY window kind. A
  /// sub-window dispatches it IN ADDITION to the decoration-flags leg,
  /// mirroring the runtime `set_resizable` dual dispatch (before the fix a
  /// sub-window created resizable(false) never seeded `enableDrag`, so the
  /// edge-drag switch stayed at its OS default until a runtime
  /// `set_resizable` call -- creation and runtime diverged).
  resize_by_drag: Option<bool>,
  /// Replay `set_fullscreen` -- upstream windows backend form
  /// (windows/window.rs:1338-1339): fullscreen takes precedence over
  /// maximized (else-if).
  fullscreen: Option<Fullscreen>,
  /// Replay `set_maximized(true)` (default `false`, so only a builder
  /// `true` fires).
  maximized: bool,
  /// Replay the upstream "not initially focused" semantic
  /// (`attributes.focused == false`). Upstream consumes `focused` as a
  /// creation-time don't-focus marker (windows `MARKER_DONT_FOCUS` /
  /// linux `accept_focus(focusable && focused)`), never as a `set_focus()`
  /// call. The attribute default is `focused: true`, and tauri's window
  /// config `focus` (default true, settable false) is forwarded by
  /// runtime-wry into attributes -- so both "never specified" and "explicitly
  /// true" hit the differ-from-default no-op, while `false` maps to
  /// `set_focusable(false)` (#174 made setFocusable effective on every
  /// window) -- the PERSISTENT `accept_focus(false)` semantic: gtk-aligned
  /// (the window stays unfocusable), deliberately diverging from Windows'
  /// create-unfocused-but-still-focusable. A presence-based `set_focus()`
  /// would fire on every window creation and violate the spec's
  /// zero-extra-traffic scenario.
  no_initial_focus: bool,
}

/// #168 (design D1): compute the seed/replay plan. Every field is
/// "differ from the attribute default", so a default-constructed
/// `WindowAttributes` plans zero dispatches (spec: default attributes
/// produce no extra traffic).
///
/// #238 (design D7): on a UIAbility-kind window (`is_ui_ability_window` --
/// the bootstrap main window AND later spawned instances) nothing rides
/// `set_window_decoration_flags` -- the ArkTS handler throws for
/// UIAbility-kind windows (main and spawned alike). The buttons are
/// OS-managed and the platform exposes no API for these enable/disable
/// semantics; button visibility/style APIs exist, operation control does
/// not. The decoration bits are still cached locally
/// ([`CreationAttrPlan::mirror_flags`]: the set_maximized(true)
/// FLAG_MAXIMIZABLE guard and the set_inner_size FLAG_RESIZABLE gate read
/// that cache), and the one attr with a real surface is `resizable`:
/// `false` seeds `set_resize_by_drag(false)` (issue
/// Eulogizethesun/tauri#104), the same leg the runtime `set_resizable`
/// dispatches -- landing on `setResizeByDragEnabled` for window_id 0 and on
/// the `enableDrag` sub-window branch for spawned instances and sub-windows
/// (PR#28 review T28-5, unified `windowId === 0` criterion; gate F1 extended
/// the seed to sub-windows, so a sub-window created resizable(false) rides
/// BOTH the decoration-flags leg and the drag leg -- same shape as the
/// runtime setter). Only the three button setters (set_minimizable /
/// set_maximizable / set_closable) error out on UIAbility-kind windows
/// (set_resizable does not -- it rides the set_resize_by_drag leg above),
/// so the set_maximized(true) FLAG_MAXIMIZABLE guard always passes there --
/// "always maximizable" is the disclosed consistent reading (D7).
fn creation_attr_plan(
  attrs: &window::WindowAttributes,
  is_ui_ability_window: bool,
) -> CreationAttrPlan {
  let mut flags = FLAG_ALL_DECORATIONS;
  if !attrs.resizable {
    flags &= !FLAG_RESIZABLE;
  }
  if !is_ui_ability_window {
    if !attrs.minimizable {
      flags &= !FLAG_MINIMIZABLE;
    }
    if !attrs.maximizable {
      flags &= !FLAG_MAXIMIZABLE;
    }
    if !attrs.closable {
      flags &= !FLAG_CLOSABLE;
    }
  }
  CreationAttrPlan {
    always_on_top: attrs.always_on_top,
    mirror_flags: flags,
    // #238: a UIAbility-kind window never dispatches decoration flags
    // (guaranteed ArkTS throw) -- its resizable attr rides `resize_by_drag`
    // instead, and its button bits are OS-managed with no surface at all.
    decoration_flags: if !is_ui_ability_window && flags != FLAG_ALL_DECORATIONS {
      Some(flags)
    } else {
      None
    },
    // #104 + PR#28 review gate F1: EVERY window kind seeds the edge-drag
    // switch on resizable(false). For a sub-window this is the SECOND leg
    // (alongside the decoration-flags seed above), mirroring the runtime
    // `set_resizable` dual dispatch so creation and runtime stay isomorphic;
    // the ArkTS branch (setResizeByDragEnabled vs enableDrag) is still
    // decided solely by the `windowId === 0` routing (T28-5).
    resize_by_drag: if !attrs.resizable { Some(false) } else { None },
    fullscreen: attrs.fullscreen.clone(),
    maximized: attrs.maximized,
    no_initial_focus: !attrs.focused,
  }
}

impl Window {
  pub(crate) fn new<T: 'static>(
    el: &EventLoopWindowTarget<T>,
    window_attrs: window::WindowAttributes,
    pl_attrs: PlatformSpecificWindowBuilderAttributes,
  ) -> Result<Self, error::OsError> {
    // Resolve the window kind: explicit builder choice, else the first window
    // defaults to UIAbility and any later one to SubWindow.
    let kind = match pl_attrs.window_kind {
      Some(kind) => kind,
      None if !FIRST_WINDOW_CREATED.load(Ordering::SeqCst) => OHOSWindowKind::UIAbility,
      None => OHOSWindowKind::SubWindow,
    };
    let is_ui_ability_window = matches!(kind, OHOSWindowKind::UIAbility);

    // First-window latch (design D2): the first window must be a UIAbility —
    // a SubWindow attaches to an existing UIAbility container
    // (WindowManager stage) and cannot bootstrap the process. Subsequent
    // UIAbility windows are legal and take the start_ui_ability branch below.
    // The bootstrap main window is created on the event-loop thread during
    // app setup, before any command can race this swap.
    let is_first_window = !FIRST_WINDOW_CREATED.swap(true, Ordering::SeqCst);
    if is_first_window && !is_ui_ability_window {
      // Rejection semantics: an explicit SubWindow request for the first window
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
      log::error!("the first OHOS window must be a UIAbility window (a SubWindow has no container to attach to)");
      return Err(os_error!(OsError));
    }

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
        // Capability handshake (G15, mirrors the sub-window registry's
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
        // SubWindow branch's WindowCreateParams below; a decorated window's frame
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
      // SubWindow: create a new OS-level sub-window via create_os_window.
      // window_id > 0, wry takes Path 2 (load_url).
      let label = pl_attrs
        .label
        .clone()
        .unwrap_or_else(|| window_attrs.title.clone());
      // Honor the builder's inner_size/position (logical px → physical) so a
      // A SubWindow WebviewWindow sized via `.inner_size()/.position()` actually
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
          // Sub-window creation race (issue-7 addendum): create_os_window opened a pending
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
            "[tao-ohos] create_os_window failed for SubWindow window {:?}: {:?}",
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
    // the app handle (#184) -- see `bridge_client`.
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
    // HAS_FOCUS snapshot at registration time -- the focus arms only write on
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
      kind,
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
      title: std::sync::Mutex::new(None),
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

    // Apply decorations immediately for UIAbility-kind windows at creation
    // time (the id-0 main window and spawned instances alike). Without this,
    // a UIAbility-kind window retains its default OS decorations even if the
    // builder specified .decorations(false), because Window::set_decorations()
    // is only called later (if at all) by the user.
    // window_id (not hardcoded 0): a subsequent UIAbility window with
    // decorations(false) must not strip the primary window's decorations.
    if is_ui_ability_window && !window_attrs.decorations {
      // PR#28 review T28-3: this creation seed used the eager `cached_client`
      // read and was silently lost when the bridge was not yet resolvable --
      // the same T1 shape the six setter legs were patched for
      // (commit 50d04f8c). Defer with replay-time resolution: held while the
      // creation handshake is pending, executed inline (one re-resolve) for
      // the always-ready primary window (see
      // [`enqueue_deferred_window_op`] for the coverage boundary).
      // window_id (not hardcoded 0): a subsequent UIAbility window with
      // decorations(false) must not strip the primary window's decorations.
      let decorations_window_id = window_id.unwrap_or(0);
      win.spawn_or_queue_deferred("set_window_decorations", move |client| async move {
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

    // Apply builder-specified content protection at creation time (macOS does
    // this in its create path; without it, .content_protection(true) had no
    // effect on OHOS until a later set_content_protection call — issue #115).
    // Same two-phase shape as Window::set_content_protection below.
    if window_attrs.content_protection && openharmony_ability::sdk_api_version() >= 15 {
      if let Some(id) = window_id {
        // PR#28 review T28-3: same T1-shape fix as the decorations seed -- the
        // eager `bridge_client` read dropped the initial content-protection
        // dispatch when the bridge was not yet resolvable. Defer with
        // replay-time resolution (held while the handshake is pending).
        win.spawn_or_queue_deferred("set_content_protection", move |client| async move {
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
    } else if window_attrs.content_protection && openharmony_ability::sdk_api_version() < 15 {
      log::warn!(
        "[tao-ohos] builder content_protection(true) skipped: requires API 15+ (window privacy mode)"
      );
    }

    // #168 (design D1): seed/replay the remaining creation-time attrs
    // (always_on_top + decoration_flags seeding, fullscreen/maximized/focused
    // replay). Everything funnels through the pending-ops queue, so a window
    // whose creation handshake is pending holds them until the handshake
    // settles. `visible`/`window_icon`/`preferred_theme` stay unconsumed by
    // design (facade disclosure, specs/ohos-platform-limitations).
    win.seed_creation_attrs(&window_attrs);

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
  /// (see [`bridge_client_cached_or_refetch`]) -- the caller then degrades to
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
  /// is still awaiting its creation handshake (issue 7 + sub-window race): ops
  /// fired in that window reach ArkTS before `requireWindow` can find the
  /// window and are silently dropped ("Unknown OS sub-window", or a silent
  /// no-op for set-limits which does not go through requireWindow). Mirrors
  /// wry's `dispatch_or_queue`.
  ///
  /// The fast path stays synchronous for every window that cannot race: the
  /// main window (id 0) and unknown ids (UIAbility ids for a fresh tao
  /// process, zombie ids) all read as ready. Sub-windows are gated from
  /// `create_os_window` until the ArkTS creation chain settles — but only
  /// when the ProcessInitializer capability handshake armed the registry
  /// (stale-HAR builds keep the old fire-and-forget fast path).
  fn dispatch_or_queue(&self, op: PendingWindowOp) {
    let Some(window_id) = self.window_id else {
      return;
    };
    dispatch_or_queue_op(window_id, op);
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

  /// PR review fix T1: the client-unavailable fallback of
  /// [`Self::spawn_or_queue`] -- same queue, but the future is built at
  /// execution time from a freshly resolved client (see
  /// [`enqueue_deferred_window_op`], including its coverage boundary for the
  /// always-ready primary window -- PR#28 review T28-3). The client-available
  /// fast path in the setters is behavior-identical to before; only the
  /// previously-dropped `bridge_client() == None` leg now enqueues.
  fn spawn_or_queue_deferred<F, Fut>(&self, op: &str, make_future: F)
  where
    F: FnOnce(openharmony_ability_plugin_window::WindowClient) -> Fut + Send + 'static,
    Fut: std::future::Future<Output = ()> + Send + 'static,
  {
    let Some(window_id) = self.window_id else {
      return;
    };
    let app = self.app.clone();
    enqueue_deferred_window_op(
      move || app.window().ok(),
      &self.runtime,
      window_id,
      op,
      make_future,
    );
  }

  /// #168 (design D1): seed/replay the creation-time attrs at build() tail.
  ///
  /// Both halves ride the existing setter channel -- every setter below funnels
  /// through [`Self::dispatch_or_queue`], so a window whose creation handshake
  /// is still pending gets its seeds queued in [`PENDING_WINDOW_OPS`] and
  /// replayed FIFO by [`drain_ready_pending_window_ops`] (zero new
  /// machinery), while an already-ready window applies them inline. The plan
  /// is computed by [`creation_attr_plan`]: only attrs that differ from their
  /// default produce a command.
  fn seed_creation_attrs(&self, attrs: &window::WindowAttributes) {
    // Issue Eulogizethesun/tauri#238 (design D7): a UIAbility-kind window
    // created with button-seed attrs (minimizable/maximizable/closable
    // false) cannot honor
    // them -- warn + ignore (a soft preference must not fail creation), while
    // the sub-window path keeps seeding the LocalStorage that drives the
    // SubWindowPage buttons.
    let is_ui_ability_window = self.is_ui_ability_window();
    if is_ui_ability_window && (!attrs.minimizable || !attrs.maximizable || !attrs.closable) {
      log::warn!(
        "[tao-ohos] creation attrs minimizable/maximizable/closable are ignored for OpenHarmony UIAbility-kind windows (the id-0 bootstrap main window and spawned instances alike; OS-managed decoration buttons, issue #238)"
      );
    }
    let plan = creation_attr_plan(attrs, is_ui_ability_window);
    // Local mirror first: set_maximized's guard reads FLAG_MAXIMIZABLE and
    // set_inner_size's gate reads FLAG_RESIZABLE from this cache, so the
    // cache must be in place before the replay half. Decoupled from the
    // dispatch below (#238): UIAbility-kind windows cache their attr bits
    // (the resizable clear included) without any decoration dispatch.
    if plan.mirror_flags != FLAG_ALL_DECORATIONS {
      self
        .decoration_flags
        .store(plan.mirror_flags, Ordering::Release);
    }
    // Sub-window decoration seed: #238 -- the ArkTS setDecorationFlags handler
    // throws for UIAbility-kind windows (the id-0 main window and spawned
    // UIAbility instances), so only sub-windows dispatch this command (the
    // UIAbility-kind windows' resizable attr rides the set_resize_by_drag
    // seed below; their button bits are OS-managed with no surface at all).
    if let Some(flags) = plan.decoration_flags {
      if let Some(window_id) = self.window_id {
        if let Some(client) = self.bridge_client("seed_creation_attrs") {
          self.spawn_or_queue(async move {
            if let Err(e) = client
              .set_window_decoration_flags(window_id, flags as i32)
              .await
            {
              log::warn!(
                "[tao-ohos] set_window_decoration_flags (creation seed) failed for window {}: {:?}",
                window_id,
                e
              );
            }
          });
        } else {
          // PR review fix T1: the eager `if let Some(client)` dropped the
          // dispatch when the bridge client was not yet resolvable -- the
          // cache kept the flags but the ArkTS side never saw them. Defer
          // with a replay-time resolution (consistent with the setter legs
          // below).
          self.spawn_or_queue_deferred("seed_creation_attrs", move |client| async move {
            if let Err(e) = client
              .set_window_decoration_flags(window_id, flags as i32)
              .await
            {
              log::warn!(
                "[tao-ohos] set_window_decoration_flags (creation seed, deferred) failed for window {}: {:?}",
                window_id,
                e
              );
            }
          });
        }
      }
    }
    // Resize seed (#238 design D7): `resizable(false)` has a real surface
    // only in the edge-drag switch (issue Eulogizethesun/tauri#104) -- the
    // same `set_resize_by_drag` leg the runtime `set_resizable` dispatches,
    // so the seed mirrors that call shape (spawn_or_queue + the T1 deferred
    // fallback). Landing branch by the unified `windowId === 0` criterion
    // (PR#28 review T28-5): id 0 -> setResizeByDragEnabled; spawned
    // UIAbility instances and sub-windows (id > 0) -> the enableDrag
    // sub-window branch. Gate F1: every kind plans this seed when
    // resizable(false) -- a sub-window rides it ALONGSIDE the
    // decoration-flags leg (the runtime setter's dual dispatch), so the
    // edge-drag switch is seeded at creation time too.
    if let Some(drag) = plan.resize_by_drag {
      if let Some(window_id) = self.window_id {
        if let Some(client) = self.bridge_client("seed_creation_attrs") {
          self.spawn_or_queue(async move {
            if let Err(e) = client.set_resize_by_drag(window_id, drag).await {
              log::warn!(
                "[tao-ohos] set_resize_by_drag (creation seed) failed for window {}: {:?}",
                window_id,
                e
              );
            }
          });
        } else {
          // PR review fix T1: defer with a replay-time client resolution --
          // see the sub-window decoration leg above.
          self.spawn_or_queue_deferred("seed_creation_attrs", move |client| async move {
            if let Err(e) = client.set_resize_by_drag(window_id, drag).await {
              log::warn!(
                "[tao-ohos] set_resize_by_drag (creation seed, deferred) failed for window {}: {:?}",
                window_id,
                e
              );
            }
          });
        }
      }
    }
    // always_on_top seed: default false, so only a builder `true` fires.
    if plan.always_on_top {
      self.set_always_on_top(true);
    }
    // Replay half -- upstream windows/window.rs:1338-1339 form: fullscreen
    // takes precedence over maximized (else-if).
    if let Some(fullscreen) = plan.fullscreen {
      self.set_fullscreen(Some(fullscreen));
    } else if plan.maximized {
      self.set_maximized(true);
    }
    // focused: upstream never calls set_focus() from the attribute (see
    // [`CreationAttrPlan::no_initial_focus`]) -- only the explicit
    // "don't focus on creation" leg replays, through set_focusable(false).
    if plan.no_initial_focus {
      self.set_focusable(false);
    }
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
  /// = no limit stays 0); sub-windows are decorEnabled:false so their
  /// chrome is 0 by construction; when a UIAbility-kind window's snapshot
  /// (the id-0 main window's) is unreadable (e.g. the builder-time dispatch
  /// racing content load) the ArkTS side
  /// degrades to uncompensated inner-sized limits with a WARN instead of
  /// failing — limits are boundaries, not targets. A decor change after
  /// dispatch is the caller's business; re-dispatching on decor changes
  /// remains an open follow-up.
  /// A spawned UIAbility instance (windowId > 0) instead takes set-limits'
  /// silent chrome-0 arm on the drawable-degenerate path -- the same
  /// uncompensated degrade, and that arm raises no WARN (the WARN-degrade
  /// arm is main-window-only; the two rect-validation throws above it are
  /// caught by the handler's catch block, whose WARN fires for any window).
  fn apply_window_limits(&self, window_id: i64) {
    let min_w = self.min_inner_width.load(Ordering::Acquire) as i64;
    let min_h = self.min_inner_height.load(Ordering::Acquire) as i64;
    let max_w = self.max_inner_width.load(Ordering::Acquire) as i64;
    let max_h = self.max_inner_height.load(Ordering::Acquire) as i64;
    let Some(client) = self.bridge_client("set_window_limits") else {
      // PR review fix T1: the builder-time constraints dispatch (a seed leg
      // like the ones in seed_creation_attrs) was dropped when the bridge
      // client was not yet resolvable -- the cache slots kept the values but
      // the system limits were never pushed. Defer with a replay-time
      // resolution (values captured as of call time, the same queue
      // semantics as spawn_or_queue).
      self.spawn_or_queue_deferred("set_window_limits", move |client| async move {
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
      return;
    };
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
    // correct WindowWrapper. Main window -> WindowId(0), sub-window -> WindowId(N).
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
    // Eulogizethesun/tauri#97): window position + drawable offset. Sub-windows
    // get offset (0,0) naturally (no system title bar).
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
    // OHOS setWindowTitle (API 15, Promise form). Main window + sub-windows
    // both support title text. Only visible when decorations enabled (decorEnabled=true).
    // Icon is NOT changeable at runtime.
    // Cache the intent for `title()`: even when the dispatch below fails, the
    // cache reflects what the app last asked for (app-owned intent mirror,
    // same family as the decorations bit).
    *self.title.lock().expect("title cache poisoned") = Some(title.to_string());
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
    // window_id 0 (main window) is valid for restore/show/move/resize/maximize,
    // so no `window_id > 0` guard here -- and none in the sibling focus
    // methods either: set_focusable's id gate was removed in #174 (the main
    // window, id 0, dispatches through the same bridge path) and set_focus's
    // Option match dispatches id 0 too. The main window's programmatic
    // show/restore goes through that same path (verified on device).
    let Some(window_id) = self.window_id else {
      return;
    };
    // Issue Eulogizethesun/tauri#237 + PR#28 review T28-4 (committer
    // ruling): hide splits by window kind. On a UIAbility (main) window
    // OHOS has NO window-level hide API (@ohos.window d.ts carries no
    // hide(); `minimize()` on a main window is a task-bar minimize with
    // different semantics, and `hideAbility()` is UIAbility-level and
    // unavailable on PC/2in1). The former minimize downgrade lied to
    // callers: `hide()` reported success while the window merely minimized
    // AND poisoned the minimized mirror. The honest contract stays a log +
    // no-op here (tao stays void, upstream-aligned); the user-visible Err
    // is surfaced one layer up at the tauri-runtime-wry dispatcher
    // (`ohos_unsupported`), which is where tao Results reach callers. Real
    // hide lands with A1 (hide_ability action).
    //
    // On a SUB-window the d.ts dual semantics make `minimize()` THE hide
    // primitive: it hides the window and `showWindow()` restores it
    // (minimize @since 11, showWindow @since 9) -- the blanket no-op
    // regressed every sub-window hide caller (the old behavior is
    // restored). The dispatch rides the T1 deferred leg: queued while
    // the creation handshake is pending, and re-resolving the client at
    // execution time instead of being dropped when the bridge is not yet
    // up (this also makes the leg observable to the #237 UTs without a
    // live client -- PR#28 review T28-1).
    if !visibility {
      if self.is_ui_ability_window() {
        log::error!(
          "[tao-ohos] set_visible(false): hiding a UIAbility-kind window is not supported on OpenHarmony (no window-level hide API; covers the id-0 bootstrap main window and spawned instances) — window {} left untouched",
          window_id
        );
        return;
      }
      // Sub-window hide intent: an OPTIMISTIC partial mirror -- only the
      // visible/minimized pair is written (visible=false, minimized=true) so
      // is_visible()/is_minimized() read the hidden state until the system
      // truth arrives (minimized=true is what the show path's restore leg
      // reads). NOT the same shape as the Minimize backfill branch in
      // [`WindowStateMirror::apply_window_status`]: that branch writes FOUR
      // bits (it also clears fullscreen/maximized) because a system
      // windowStatusChange Minimize event IS system truth about the whole
      // window state, while this hide dispatch (the d.ts dual-semantics
      // minimize() used as a hide primitive) carries no fullscreen/maximized
      // observation -- the two bits stay at their previously recorded intent
      // until a real windowStatusChange backfills them.
      self.mirror.minimized.store(true, Ordering::Release);
      self.mirror.visible.store(false, Ordering::Release);
      self.spawn_or_queue_deferred("set_visible", move |client| async move {
        if let Err(e) = client.minimize_window(window_id).await {
          log::warn!(
            "[tao-ohos] minimize_window (sub-window hide) failed for window {}: {:?}",
            window_id,
            e
          );
        }
      });
      return;
    }
    // Issue Eulogizethesun/tauri#237 (show path): restore is dispatched ONLY
    // when the local mirror says minimized. The former unconditional
    // restore+show double-dispatch un-maximized visible maximized windows
    // (restore on a non-minimized window cancels maximization -- the
    // WindowManager.ets showWindowMethod guard documents the hazard). A plain
    // visible window gets show-only (a raise), never touching its maximize
    // state. The mirror is cleared only on the restore leg, matching the
    // restore intent (restore() is @since 14 -- a single d.ts block,
    // main-window only; below 14, e.g. API-12 deployment baselines, and on
    // sub-windows it is unavailable -> show_window best-effort, the
    // structured warn below still fires).
    let minimized = self.mirror.minimized.load(Ordering::Acquire);
    // Show intent (T28-4 mirror contract): pairs with the hide intent write
    // in the false leg -- is_visible() must not stay false after the window
    // is shown again.
    self.mirror.visible.store(true, Ordering::Release);
    // TODO(A1): replace with AppControlExt::show_ability(env) when A1 adds the action
    self.spawn_or_queue_deferred("set_visible", move |client| async move {
      if minimized {
        if let Err(e) = client.restore_window(window_id).await {
          log::warn!(
            "[tao-ohos] restore_window failed for window {}: {:?}",
            window_id,
            e
          );
        }
      }
      if let Err(e) = client.show_window(window_id).await {
        log::warn!(
          "[tao-ohos] show_window failed for window {}: {:?}",
          window_id,
          e
        );
      }
    });
    if minimized {
      self.mirror.minimized.store(false, Ordering::Release);
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
      // window.shiftAppWindowFocus (11+, works for main/sub windows), and the
      // sub-windows keep raiseToAppTop (14+, subwindow-only by contract).
    }
  }

  pub fn set_focusable(&self, focusable: bool) {
    if let Some(window_id) = self.window_id {
      // #174 (2026-10-21 device arbitration): the `window_id > 0` gate that
      // historically no-op'd the main window is removed -- the main window
      // (id 0) dispatches through the same bridge path as sub-windows.
      // The OS arbitrates: `win.setFocusable` (WindowPlugin.ets set-focusable
      // action) succeeds on the main window (probe evidence in the phase
      // report); a rejection surfaces as the structured warn below. The
      // bridge action bypasses WindowManager.setWindowFocusable (dead code),
      // so the facade-level judging observable is the Rust-side warn, not
      // the WindowManager hilog.debug success line.
      let Some(client) = self.bridge_client("set_focusable") else {
        // PR review fix T1: the client check ran before the queue check, so a
        // seed (or runtime call) in the bridge-unavailable state never
        // entered PENDING_WINDOW_OPS -- the creation-time no_initial_focus
        // seed was permanently lost. Defer with a replay-time resolution.
        self.spawn_or_queue_deferred("set_focusable", move |client| async move {
          if let Err(e) = client.set_window_focusable(window_id, focusable).await {
            log::warn!(
              "set_focusable: set_window_focusable failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
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

  /// Destroys the OS-level window: sub-windows call `destroyWindow`.
  /// The main (UIAbility) window is system-managed: the bridge's
  /// `destroy-window` handler rejects window id 0, so for the main window
  /// this is a rejected fire-and-forget (logged as a warning) — the ability
  /// is terminated by the system, not by an in-app close. Fire-and-forget
  /// through the bridge executor, same as the other async window operations.
  ///
  /// On a successful resolve the window id is enqueued as a destruction
  /// confirmation and the event loop is woken so the `run_loop` dispatch
  /// synthesizes `Event::WindowEvent::Destroyed` (design D2 channel (a)) --
  /// the deterministic confirmation for the embedding runtime's
  /// confirmation-driven accounting, covering the paths where the ArkTS
  /// notifyWindowClose loopback cannot fire. A failed resolve stays silent on
  /// the event lane (design D3: failure = no event, the caller keeps its
  /// accounting) but is ACCOUNTED: issue Eulogizethesun/tauri#255 -- the id
  /// goes into [`DESTROY_FAILURES`] (the failure twin of the confirmation
  /// queue) and the loop is woken, so the runtime can drain it and clear the
  /// in-flight close guard it set before dispatching (otherwise the next
  /// USER-initiated CloseRequested is misread as the destruction loopback).
  /// The bridge-client-unavailable leg accounts the same way; the no-OHOS-id
  /// leg is log-only (nothing to key the record on -- see the arm below).
  ///
  /// Residual boundary of that accounting (disclosed, not fixed -- issue
  /// Eulogizethesun/tauri#255 review): the bridge-client-unavailable leg (b)
  /// enqueues synchronously at call time, but the async-failure leg (c)
  /// enqueues from inside the future spawned through
  /// [`Self::spawn_or_queue`] -- its enqueue+wake happen only when the
  /// executor actually polls that queued future. If the executor never runs
  /// it (bridge never recovers AND the pending queue never drains -- an
  /// app-level wedged state), the stale in-flight close guard survives and
  /// the next USER-initiated CloseRequested is still misread as the
  /// destruction loopback. Guard-clearing is therefore "once the loop is
  /// alive again", not unconditional at call time.
  pub fn close(&self) {
    let Some(window_id) = self.window_id else {
      // Failure leg (a): no OHOS id -- the creation handshake never settled,
      // so nothing was dispatched and there is no id to key a
      // DESTROY_FAILURES record on (0 is the main window's REAL id, so no
      // sentinel is safe; the embedding runtime can observe the None id
      // directly at its initiate site via WindowExtOpenHarmony::window_id()).
      log::warn!(
        "close: window has no OHOS id (creation handshake never settled) — nothing dispatched"
      );
      return;
    };
    let Some(client) = self.bridge_client("close") else {
      // Failure leg (b): bridge client unavailable. The shared degradation
      // warn (bridge_client_cached_or_refetch) fired debounced; this leg is
      // accounted per attempt so the runtime's in-flight close guard is
      // cleared even when the bridge never comes back.
      enqueue_destroy_failure(window_id);
      self.app.create_waker().wake();
      return;
    };
    let app = self.app.clone();
    self.spawn_or_queue(async move {
      match client.destroy_window(window_id).await {
        Ok(()) => {
          enqueue_destroy_confirmation(window_id);
          app.create_waker().wake();
        }
        Err(e) => {
          // Failure leg (c): the async resolve failed. Warn AND account
          // (issue Eulogizethesun/tauri#255): the runtime drains the id and
          // clears its in-flight close guard, so the next USER close is not
          // misread as the destruction loopback. Still no event (D3).
          log::warn!(
            "close: destroy_window failed for window {}: {:?}",
            window_id,
            e
          );
          enqueue_destroy_failure(window_id);
          app.create_waker().wake();
        }
      }
    });
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
  // set_closable only control decoration button visibility (the
  // SubWindowPage @LocalStorageProp) and do not block the programmatic APIs; a
  // UIAbility-kind window errors out for them (issue Eulogizethesun/tauri#238
  // -- the buttons are OS-managed). set_resizable is the exception since issue
  // Eulogizethesun/tauri#104 -- see below (it stays ungated: it has a real
  // UIAbility-kind surface in set_resize_by_drag (main id-0 and spawned
  // instances alike)). is_resizable etc. read from the local mirror for
  // sub-windows; UIAbility-kind button queries report
  // the OS-managed truth (#238). See doc/OHOS窗口遗留问题.md in the tauri repo (issue 4).
  pub fn set_resizable(&self, resizable: bool) {
    if self.is_ui_ability_window() {
      // PR#28 review T28-2: the ArkTS setDecorationFlags handler throws for
      // EVERY UIAbility-kind window (the bootstrap main window and spawned
      // instances alike -- the same gate creation time already dropped the
      // command for, commit 18c96db), so the runtime setter must not
      // dispatch it either. Local mirror only: the set_inner_size
      // FLAG_RESIZABLE gate reads this cache.
      self.store_decoration_flag(FLAG_RESIZABLE, resizable);
    } else {
      // Sub-window: the bit rides set_window_decoration_flags (SubWindowPage
      // LocalStorage) as before.
      self.set_decoration_flag(FLAG_RESIZABLE, resizable);
    }
    // Issue Eulogizethesun/tauri#104 + PR#28 review T28-5 (unified
    // criterion): the decoration flag only covers title-bar button
    // visibility (sub-windows) and the programmatic-resize gate
    // (set_inner_size checks FLAG_RESIZABLE). The actual user-facing switch
    // is edge-drag resizing -- dispatched through the same
    // `set-resize-by-drag` action, whose ArkTS routing keys on
    // `windowId === 0`: the primary main window lands on
    // setResizeByDragEnabled (API 14+, effective in free-window state),
    // every other window (sub-windows and spawned UIAbility instances,
    // id > 0) lands on the sub-window branch, enableDrag (API 20+).
    // Failure is logged and ignored on API levels without the call (ArkTS
    // guards with a clear error).
    // Dispatched through the T1 deferred leg (same discipline as the
    // set_visible show path): the client is resolved at execution time, so a
    // call fired while the creation handshake is pending is QUEUED and
    // replayed instead of being silently dropped by the eager read -- and the
    // leg is observable to the UTs without a bridge client (PENDING_WINDOW_OPS
    // entry).
    if let Some(window_id) = self.window_id {
      self.spawn_or_queue_deferred("set_resizable", move |client| async move {
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

  /// Whether this window is of the UIAbility KIND (issue
  /// Eulogizethesun/tauri#238): the id-0 bootstrap main window AND later
  /// spawned UIAbility instances -- both get system-provided decorations, with
  /// no API mapping to the per-button enable/disable flag semantics, and
  /// both reject the UIAbility-hostile ops (decoration-flags dispatch,
  /// window-level hide). PR#28 review gate F7: renamed from
  /// `is_ui_ability_main_window` -- the predicate is KIND-based, not
  /// id-based, and "main window" understated its reach: a spawned UIAbility
  /// instance (id > 0) takes every branch gated here too. The gate must NOT
  /// be narrowed to `window_id == Some(0)`: a spawned instance falling
  /// through to the sub-window branches would dispatch the guaranteed-throw
  /// setDecorationFlags (T28-2) and route a hide onto a task-bar minimize.
  /// Matches the ArkTS `isUIAbilityMainWindow` gate, which throws for the
  /// same reason.
  fn is_ui_ability_window(&self) -> bool {
    matches!(self.kind, OHOSWindowKind::UIAbility)
  }

  pub fn set_minimizable(&self, minimizable: bool) {
    // Issue Eulogizethesun/tauri#238: UIAbility-kind windows (the id-0
    // main window and spawned instances) have OS-managed decoration
    // buttons -- OpenHarmony exposes no API mapping to the per-button
    // enable/disable flag semantics (visibility/style APIs exist; operation
    // control does not). The
    // former silent bit-write + LocalStorage dispatch faked success (and the
    // bit was never consulted for UIAbility windows anyway). Honest no-op
    // here (tao stays void, upstream-aligned); the user-visible Err surfaces
    // at the tauri-runtime-wry dispatcher. Sub-windows keep the real behavior.
    if self.is_ui_ability_window() {
      log::error!(
        "[tao-ohos] set_minimizable({}): not supported on OpenHarmony for UIAbility-kind windows (the id-0 bootstrap main window and spawned instances; OS-managed decoration buttons) — ignored for window {:?}",
        minimizable,
        self.window_id
      );
      return;
    }
    self.set_decoration_flag(FLAG_MINIMIZABLE, minimizable);
  }

  pub fn set_maximizable(&self, maximizable: bool) {
    // Issue Eulogizethesun/tauri#238 -- see set_minimizable for the rationale.
    if self.is_ui_ability_window() {
      log::error!(
        "[tao-ohos] set_maximizable({}): not supported on OpenHarmony for UIAbility-kind windows (the id-0 bootstrap main window and spawned instances; OS-managed decoration buttons) — ignored for window {:?}",
        maximizable,
        self.window_id
      );
      return;
    }
    self.set_decoration_flag(FLAG_MAXIMIZABLE, maximizable);
  }

  pub fn set_closable(&self, closable: bool) {
    // Issue Eulogizethesun/tauri#238 -- see set_minimizable for the rationale.
    if self.is_ui_ability_window() {
      log::error!(
        "[tao-ohos] set_closable({}): not supported on OpenHarmony for UIAbility-kind windows (the id-0 bootstrap main window and spawned instances; OS-managed decoration buttons) — ignored for window {:?}",
        closable,
        self.window_id
      );
      return;
    }
    self.set_decoration_flag(FLAG_CLOSABLE, closable);
  }

  /// Common helper: update one decoration bit in the local mirror only (no
  /// dispatch). The set_inner_size FLAG_RESIZABLE gate and the
  /// set_minimized/set_maximized guards read this cache; UIAbility-kind
  /// windows use this variant because the ArkTS setDecorationFlags handler
  /// throws for them (issue Eulogizethesun/tauri#238, PR#28 review T28-2).
  fn store_decoration_flag(&self, flag: u8, on: bool) {
    let mut flags = self.decoration_flags.load(Ordering::Acquire);
    if on {
      flags |= flag;
    } else {
      flags &= !flag;
    }
    self.decoration_flags.store(flags, Ordering::Release);
  }

  /// Common helper: update one decoration bit in the local mirror and
  /// dispatch the full bitfield to ArkTS (SubWindowPage LocalStorage). The
  /// dispatch rides the T1 deferred leg (client resolved at execution time):
  /// a sub-window still awaiting its creation handshake gets the dispatch
  /// QUEUED and replayed instead of silently dropped by the eager read --
  /// and the leg is observable to the UTs without a bridge client
  /// (PENDING_WINDOW_OPS entry). UIAbility-kind windows must not route here
  /// (the ArkTS setDecorationFlags handler throws for them) -- they use
  /// [`Self::store_decoration_flag`] (see set_resizable, T28-2).
  fn set_decoration_flag(&self, flag: u8, on: bool) {
    self.store_decoration_flag(flag, on);
    if let Some(window_id) = self.window_id {
      let flags = self.decoration_flags.load(Ordering::Acquire);
      self.spawn_or_queue_deferred("set_decoration_flag", move |client| async move {
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
        // PR review fix T1: defer with a replay-time client resolution (the
        // creation-time maximized seed was dropped before it could queue).
        self.spawn_or_queue_deferred("set_maximized", move |client| async move {
          if maximized {
            if let Err(e) = client.maximize_window(window_id).await {
              log::warn!(
                "[tao-ohos] maximize_window failed for window {}: {:?}",
                window_id,
                e
              );
            }
          } else {
            // recover() switches MAXIMIZE/FULL_SCREEN -> FLOATING (public).
            // Versions, kept separate: recover is @since 11 (d.ts first
            // block); the FLOATING-restore semantics is the API 12 doc
            // revision, not the introduction version.
            if let Err(e) = client.recover_window(window_id).await {
              log::warn!(
                "[tao-ohos] recover_window failed for window {}: {:?}",
                window_id,
                e
              );
            }
          }
        });
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
        // PR review fix T1: defer with a replay-time client resolution (the
        // creation-time fullscreen seed was dropped before it could queue).
        self.spawn_or_queue_deferred("set_fullscreen", move |client| async move {
          if let Err(e) = client.set_fullscreen(window_id, on).await {
            log::warn!(
              "[tao-ohos] set_fullscreen failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
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
    // silently (signature intentionally stays `()` -- no `Err`, per the md
    // wave-0 ruling), aligned with the iOS `log::warn!` precedent.
    log::warn!("[tao-ohos] set_always_on_bottom is not supported on OHOS; call ignored");
  }

  pub fn set_always_on_top(&self, always_on_top: bool) {
    // Records intent (is_always_on_top reads this) AND dispatches to OHOS
    // setWindowTopmost (API 14+, needs ohos.permission.WINDOW_TOPMOST) via the
    // window bridge facade. Main window only per OHOS docs; sub-windows
    // will error (ArkTS throws; tao catches and warns, non-fatal). Only
    // effective in freeform window mode.
    self.always_on_top.store(always_on_top, Ordering::Release);
    if let Some(window_id) = self.window_id {
      let Some(client) = self.bridge_client("set_always_on_top") else {
        // PR review fix T1: the spec scenario -- a window created with
        // with_always_on_top(true) while the bridge client is not yet ready
        // MUST queue the seed and replay it after the handshake. The eager
        // client check dropped it before the queue check; defer instead.
        self.spawn_or_queue_deferred("set_always_on_top", move |client| async move {
          if let Err(e) = client.set_window_topmost(window_id, always_on_top).await {
            log::warn!(
              "[tao-ohos] set_window_topmost failed for window {}: {:?}",
              window_id,
              e
            );
          }
        });
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
    // Issue Eulogizethesun/tauri#238: a UIAbility-kind window's buttons are
    // OS-managed
    // and always available -- report the true state instead of the local bit
    // (which no longer tracks anything for UIAbility windows since the
    // setters error out). Known edge (no upstream doc; recorded here only):
    // set_decorations(false)/setWindowDecorVisible hides the WHOLE bar; these
    // queries still report true -- same as upstream Win32, where the
    // WS_MINIMIZEBOX-style styles stay independent of the visible chrome.
    if self.is_ui_ability_window() {
      return true;
    }
    self.decoration_flags.load(Ordering::Acquire) & FLAG_MINIMIZABLE != 0
  }

  pub fn is_maximizable(&self) -> bool {
    // Issue Eulogizethesun/tauri#238 -- see is_minimizable for the rationale.
    if self.is_ui_ability_window() {
      return true;
    }
    self.decoration_flags.load(Ordering::Acquire) & FLAG_MAXIMIZABLE != 0
  }

  pub fn is_closable(&self) -> bool {
    // Issue Eulogizethesun/tauri#238 -- see is_minimizable for the rationale.
    if self.is_ui_ability_window() {
      return true;
    }
    self.decoration_flags.load(Ordering::Acquire) & FLAG_CLOSABLE != 0
  }

  pub fn set_window_icon(&self, _window_icon: Option<crate::icon::Icon>) {}

  pub fn set_cursor_icon(&self, icon: window::CursorIcon) {
    // TODO(issue 6): dispatched but not yet device-tested — verify style mapping
    //   coverage, touch-mode device behavior, and whether it works on
    //   sub-windows. See doc/OHOS窗口遗留问题.md in the tauri repo (issue 6).
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
    //   See doc/OHOS窗口遗留问题.md in the tauri repo (issue 6).
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
    // cannot be triggered programmatically from Rust. Sub-windows drag
    // via SubWindowPage title bar onTouch->startMoving; the main UIAbility window
    // has no such path, so this is a no-op there. Returns Ok (no error) since
    // drag is handled at the UI layer (SubWindowPage), not via this API.
    log::debug!(
      "[tao-ohos] drag_window: no-op (startMoving must be called from onTouch in SubWindowPage; window_id={:?})",
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
    // Cache-on-set read-back: OHOS has no synchronous title query. The cache
    // carries the last set_title intent; None until the first set returns the
    // empty string (the pre-fix return for never-titled windows).
    self
      .title
      .lock()
      .expect("title cache poisoned")
      .clone()
      .unwrap_or_default()
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

  /// Resolved window kind (issue Eulogizethesun/tauri#238), surfaced through
  /// `WindowExtOpenHarmony::window_kind` for the embedding runtime's
  /// UIAbility-kind-window dispatcher gates (the bootstrap main window and
  /// spawned instances).
  pub fn kind(&self) -> OHOSWindowKind {
    self.kind
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
      // Same poisoning policy as the ContentRectChange arm (issue
      // Eulogizethesun/tauri#135): recover the map rather than skipping the
      // remove — a skipped remove would leave a stale entry (harmless while
      // window ids are monotonically allocated and never reused, but the
      // consistency costs nothing).
      LAST_DISPATCHED_RECTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .remove(&window_id);
      // Issue 7: discard ops still queued against this window (its creation
      // handshake never completed before the handle went away), recycle the
      // UIAbility pending-registry entry (waker included — mirrors the
      // start_ui_ability Err-leg rollback in `Window::new`; without it, an
      // entry whose spawn AMS silently swallowed stays registered with a
      // live waker until process exit), and drop the sub-window pending-registry
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
  // resize-inner). The three cases by exact test name:
  // - 'window.setInnerSize actually resizes (main window)'
  // - 'window.setInnerSize save/restore zero drift (5 rounds)'
  // - 'float window setInnerSize exact readback (decor=0)'
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
      queue().entry(HELD_ID).or_default().push(Box::new(move || {
        order.lock().expect("order log poisoned").push(step);
      }));
    }
    drain_ready_pending_window_ops();
    assert!(
      order.lock().expect("order log poisoned").is_empty(),
      "op ran while the creation handshake was pending"
    );
    let held_still_queued = queue().contains_key(&HELD_ID);
    assert!(
      held_still_queued,
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
    let drained_removed = !queue().contains_key(&HELD_ID);
    assert!(drained_removed, "drained entry not removed");
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
    let dropped_removed = !queue().contains_key(&DROPPED_ID);
    assert!(dropped_removed, "dropped entry not removed");
    openharmony_ability::unregister_pending_ui_ability(DROPPED_ID);
  }

  // --- #168 (design D1): creation-attr seeding rides the existing
  // dispatch_or_queue channel. The decision is `creation_attr_plan` (pure);
  // the enqueue/inline legs run the extracted production body
  // `dispatch_or_queue_op` with sentinel ids (a `Window` cannot be
  // constructed outside a bridge session -- same discipline as the O10-6
  // test above).

  /// UT for #168 (a): a build() carrying `always_on_top(true)` with the
  /// creation handshake pending queues exactly one seed op, which replays
  /// once the handshake completes; the default attrs of the same plan stay
  /// dispatch-free.
  #[test]
  fn ut_creation_seed_always_on_top_holds_in_queue_before_ready() {
    const HELD_ID: i64 = i64::MAX - 11;
    // Idempotent reset so a previously failed run cannot leak registry
    // entries that would pin these ids as "not ready".
    openharmony_ability::unregister_pending_ui_ability(HELD_ID);
    drop_pending_window_ops(HELD_ID);

    let mut attrs = window::WindowAttributes::default();
    attrs.always_on_top = true;
    let plan = creation_attr_plan(&attrs, false);
    assert!(plan.always_on_top, "always_on_top(true) must plan a seed");
    assert_eq!(
      plan.decoration_flags, None,
      "default decoration flags must not seed"
    );
    assert!(
      plan.fullscreen.is_none(),
      "default fullscreen must not replay"
    );
    assert!(!plan.maximized, "default maximized must not replay");
    assert!(
      !plan.no_initial_focus,
      "default focused(true) must not seed a focusable flip"
    );

    openharmony_ability::register_pending_ui_ability(HELD_ID);
    assert!(!openharmony_ability::is_window_ready(HELD_ID));
    let ran = Arc::new(AtomicBool::new(false));
    let flag = ran.clone();
    dispatch_or_queue_op(
      HELD_ID,
      Box::new(move || flag.store(true, Ordering::SeqCst)),
    );
    let queued_len = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&HELD_ID)
      .map(|ops| ops.len());
    assert_eq!(
      queued_len,
      Some(1),
      "not-ready seed must hold exactly one queued op"
    );
    drain_ready_pending_window_ops();
    assert!(
      !ran.load(Ordering::SeqCst),
      "seed ran while the creation handshake was pending"
    );

    // Handshake completes: the next drain replays the held seed.
    openharmony_ability::register_ui_ability_stage(HELD_ID);
    drain_ready_pending_window_ops();
    assert!(ran.load(Ordering::SeqCst), "held seed not replayed");
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- a panic here must not poison PENDING_WINDOW_OPS.
    let still_queued = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .contains_key(&HELD_ID);
    assert!(!still_queued, "drained seed entry not removed");
    openharmony_ability::unregister_pending_ui_ability(HELD_ID);
    drop_pending_window_ops(HELD_ID);
  }

  /// UT for #168 (b): decoration-attr differences plan ONE combined
  /// set-decoration-flags seed carrying the attr-derived bitfield -- each
  /// false attr clears exactly its own bit (the maximizable clear matters
  /// downstream: set_maximized(true)'s guard reads FLAG_MAXIMIZABLE).
  /// Sub-window leg (`is_ui_ability_window = false`) -- the dispatch and the local mirror
  /// stay coupled there; the UIAbility kind (main window and spawned
  /// instances) diverges (see the #238 test below).
  /// Gate F1: a sub-window resizable(false) additionally plans the
  /// set_resize_by_drag(false) seed (the runtime setter's dual dispatch).
  #[test]
  fn ut_creation_plan_seeds_decoration_flags_when_attrs_differ() {
    let mut attrs = window::WindowAttributes::default();
    attrs.resizable = false;
    let plan = creation_attr_plan(&attrs, false);
    assert_eq!(
      plan.decoration_flags,
      Some(FLAG_ALL_DECORATIONS & !FLAG_RESIZABLE),
      "resizable(false) must seed flags with only the resizable bit cleared"
    );
    assert_eq!(
      plan.resize_by_drag,
      Some(false),
      "resizable(false) must ALSO plan the set_resize_by_drag(false) seed (gate F1: creation mirrors the runtime dual dispatch)"
    );

    let mut attrs = window::WindowAttributes::default();
    attrs.minimizable = false;
    attrs.closable = false;
    let plan = creation_attr_plan(&attrs, false);
    assert_eq!(
      plan.decoration_flags,
      Some(FLAG_ALL_DECORATIONS & !(FLAG_MINIMIZABLE | FLAG_CLOSABLE)),
      "minimizable/closable(false) must clear exactly those bits"
    );

    let mut attrs = window::WindowAttributes::default();
    attrs.maximizable = false;
    let plan = creation_attr_plan(&attrs, false);
    assert_eq!(
      plan.decoration_flags,
      Some(FLAG_ALL_DECORATIONS & !FLAG_MAXIMIZABLE),
      "maximizable(false) must clear the maximizable bit"
    );
  }

  /// UT for #168 (c): default attributes plan zero dispatches -- the spec's
  /// "default attributes produce no extra traffic" scenario, on both window
  /// kinds (the mirror bitfield stays at its default too, so the decoupled
  /// mirror store at the call site is skipped).
  #[test]
  fn ut_creation_plan_default_attrs_produce_zero_seeds() {
    for is_ui_ability_window in [false, true] {
      let attrs = window::WindowAttributes::default();
      let plan = creation_attr_plan(&attrs, is_ui_ability_window);
      assert!(!plan.always_on_top, "default always_on_top must not seed");
      assert_eq!(
        plan.decoration_flags, None,
        "default decoration flags must not seed"
      );
      assert_eq!(
        plan.mirror_flags, FLAG_ALL_DECORATIONS,
        "default attrs must not move the local mirror off its default"
      );
      assert!(
        plan.resize_by_drag.is_none(),
        "default resizable(true) must not seed a resize_by_drag flip"
      );
      assert!(
        plan.fullscreen.is_none(),
        "default fullscreen must not replay"
      );
      assert!(!plan.maximized, "default maximized must not replay");
      assert!(
        !plan.no_initial_focus,
        "default focused(true) must not seed a focusable flip"
      );
    }
  }

  /// UT for #238 (design D7): UIAbility-kind windows never plan a
  /// set_window_decoration_flags dispatch -- the ArkTS handler throws for
  /// UIAbility-kind windows (the id-0 main window and spawned UIAbility
  /// instances), so a dispatch there is a guaranteed-fail warn.
  /// The sub-window path (`is_ui_ability_window = false`, exercised by the #168 tests
  /// above) keeps the per-bit seeding.
  #[test]
  fn ut_creation_plan_main_window_no_decoration_dispatch_resize_drag_seed() {
    // Brief:
    // 1. UIAbility-kind-window attrs with all three button bits false must plan NO
    //    decoration dispatch and leave the mirror at the all-ones default
    //    (OS-managed buttons -- warn+ignore at the call site).
    // 2. resizable(false) on a UIAbility-kind window must NOT plan a decoration
    //    dispatch (guaranteed ArkTS throw) -- instead it plans the
    //    set_resize_by_drag(false) seed leg (#104) while the mirror keeps
    //    ONLY the resizable bit cleared (the programmatic-resize gate in
    //    set_inner_size reads that cache; the D7 disclosure keeps the
    //    set_maximized guard passing).
    // 3. resizable(true) (the default) on a UIAbility-kind window plans no resize
    //    seed -- default attrs produce zero traffic.
    // 4. The same button attrs on a sub window (is_ui_ability_window = false) keep
    //    today's per-bit seed and its coupled mirror (no behavior change);
    //    with the default resizable(true) no drag seed is planned either.
    // 5. Gate F1: a sub window created resizable(false) plans BOTH legs --
    //    the decoration bit (SubWindowPage LocalStorage) AND the
    //    set_resize_by_drag(false) edge-drag seed, mirroring the runtime
    //    set_resizable dual dispatch.
    let mut attrs = window::WindowAttributes::default();
    attrs.minimizable = false;
    attrs.maximizable = false;
    attrs.closable = false;
    let plan = creation_attr_plan(&attrs, true);
    assert_eq!(
      plan.decoration_flags, None,
      "UIAbility-kind windows (main id-0 and spawned): button seeds must not dispatch decoration flags (OS-managed buttons)"
    );
    assert_eq!(
      plan.mirror_flags, FLAG_ALL_DECORATIONS,
      "UIAbility-kind windows (main id-0 and spawned): button bits stay cached at the all-ones default"
    );
    assert!(
      plan.resize_by_drag.is_none(),
      "button attrs must not plan a resize_by_drag seed"
    );

    let mut attrs = window::WindowAttributes::default();
    attrs.resizable = false;
    attrs.minimizable = false;
    let plan = creation_attr_plan(&attrs, true);
    assert_eq!(
      plan.decoration_flags, None,
      "UIAbility-kind windows (main id-0 and spawned): resizable(false) must NOT dispatch decoration flags (guaranteed ArkTS throw)"
    );
    assert_eq!(
      plan.mirror_flags,
      FLAG_ALL_DECORATIONS & !FLAG_RESIZABLE,
      "UIAbility-kind windows (main id-0 and spawned): resizable(false) must still cache only the resizable bit clear (set_inner_size gate)"
    );
    assert_eq!(
      plan.resize_by_drag,
      Some(false),
      "UIAbility-kind windows (main id-0 and spawned): resizable(false) must seed the set_resize_by_drag(false) leg (#104)"
    );

    let attrs = window::WindowAttributes::default();
    let plan = creation_attr_plan(&attrs, true);
    assert!(
      plan.resize_by_drag.is_none(),
      "UIAbility-kind windows (main id-0 and spawned): resizable(true) must not seed set_resize_by_drag"
    );

    let mut attrs = window::WindowAttributes::default();
    attrs.minimizable = false;
    attrs.closable = false;
    let plan = creation_attr_plan(&attrs, false);
    assert_eq!(
      plan.decoration_flags,
      Some(FLAG_ALL_DECORATIONS & !(FLAG_MINIMIZABLE | FLAG_CLOSABLE)),
      "sub-window button seeds keep the per-bit behavior"
    );
    assert_eq!(
      plan.mirror_flags,
      FLAG_ALL_DECORATIONS & !(FLAG_MINIMIZABLE | FLAG_CLOSABLE),
      "sub-window mirror stays coupled to the dispatch payload"
    );
    assert!(
      plan.resize_by_drag.is_none(),
      "sub-windows keep the decoration-flags resizable leg, no resize_by_drag seed for button attrs"
    );

    // Gate F1 arm: a sub window created resizable(false) plans BOTH legs.
    let mut attrs = window::WindowAttributes::default();
    attrs.resizable = false;
    let plan = creation_attr_plan(&attrs, false);
    assert_eq!(
      plan.decoration_flags,
      Some(FLAG_ALL_DECORATIONS & !FLAG_RESIZABLE),
      "sub-window resizable(false) keeps the decoration-flags leg (SubWindowPage LocalStorage)"
    );
    assert_eq!(
      plan.resize_by_drag,
      Some(false),
      "sub-window resizable(false) must ALSO plan the set_resize_by_drag(false) seed (gate F1: creation mirrors the runtime dual dispatch)"
    );
  }

  /// UT for #168 (d): a seed dispatched while the window is already ready runs
  /// inline -- no queue entry, nothing to replay (same semantics as the
  /// runtime setters).
  #[test]
  fn ut_creation_seed_dispatches_inline_when_ready() {
    const READY_ID: i64 = i64::MAX - 12;
    openharmony_ability::unregister_pending_ui_ability(READY_ID);
    drop_pending_window_ops(READY_ID);
    assert!(
      openharmony_ability::is_window_ready(READY_ID),
      "unknown ids must read as ready"
    );
    let ran = Arc::new(AtomicBool::new(false));
    let flag = ran.clone();
    dispatch_or_queue_op(
      READY_ID,
      Box::new(move || flag.store(true, Ordering::SeqCst)),
    );
    assert!(
      ran.load(Ordering::SeqCst),
      "ready-window seed must dispatch inline"
    );
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- a panic here must not poison PENDING_WINDOW_OPS.
    let queued_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .contains_key(&READY_ID);
    assert!(!queued_ops, "inline dispatch must not leave a queue entry");
  }

  // --- Issue Eulogizethesun/tauri#237 + PR#28 review T28-1/T28-4: the
  // set_visible contract splits by window kind. The sub-window legs are
  // observable WITHOUT a bridge client (a real WindowClient needs a live
  // NAPI session a test binary does not have): the hide/show dispatches ride
  // the T1 deferred leg, so a pending-handshake window shows each op as a
  // PENDING_WINDOW_OPS entry -- the same dispatch-observable discipline as
  // the #168 seed UTs. This closes the T28-1 blindness (the former UT's
  // `window_client: None` early-returned on the pre-#237 code too): the
  // sub-window UTs exercise the dispatch branch itself and FAIL on both
  // prior variants (the #237 honest no-op queues nothing; the pre-#237
  // eager leg dropped everything when the client was unresolvable). A
  // `Window` literal IS constructible inside this module (private fields);
  // the sentinel id + pending registration make any accidental dispatch
  // observable as a PENDING_WINDOW_OPS entry.

  /// UT for #237 + T28-4: a SUB-window set_visible(false) queues the
  /// minimize (hide) dispatch and writes the hidden-state intent mirrors.
  #[test]
  fn ut_set_visible_false_sub_window_queues_hide_and_writes_hidden_mirrors() {
    const HIDING_ID: i64 = i64::MAX - 31;
    // Idempotent reset so a previously failed run cannot leak registry state
    // that would pin this id as "not ready" or hold stale ops.
    openharmony_ability::unregister_pending_ui_ability(HIDING_ID);
    drop_pending_window_ops(HIDING_ID);
    openharmony_ability::register_pending_ui_ability(HIDING_ID);
    assert!(!openharmony_ability::is_window_ready(HIDING_ID));

    let mirror = Arc::new(WindowStateMirror::new(true));
    let window = Window {
      app: OpenHarmonyApp::new(),
      window_id: Some(HIDING_ID),
      kind: OHOSWindowKind::SubWindow,
      window_client: std::sync::Mutex::new(None),
      bridge_refetch_warned: AtomicBool::new(false),
      runtime: BridgeExecutor::new(),
      mirror: mirror.clone(),
      decorations: AtomicBool::new(true),
      title: std::sync::Mutex::new(None),
      transparent: false,
      always_on_top: AtomicBool::new(false),
      decoration_flags: AtomicU8::new(15),
      min_inner_width: AtomicU32::new(0),
      min_inner_height: AtomicU32::new(0),
      max_inner_width: AtomicU32::new(0),
      max_inner_height: AtomicU32::new(0),
    };
    window.set_visible(false);

    assert!(
      mirror.minimized.load(Ordering::Acquire),
      "sub-window hide must write the minimized intent bit (the show path's restore leg reads it)"
    );
    assert!(
      !mirror.visible.load(Ordering::Acquire),
      "sub-window hide must flip the visible intent bit off (is_visible honesty)"
    );
    // Gate F9 (test hygiene): the queue state is bound to a local FIRST so
    // the mutex guard is released before the assert -- a panic inside the
    // assert must not poison PENDING_WINDOW_OPS and cascade into unrelated
    // UTs (the F8 fixwave mutation run hit exactly that).
    let queued_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&HIDING_ID)
      .map(|ops| ops.len());
    assert_eq!(
      queued_ops,
      Some(1),
      "sub-window set_visible(false) must queue the minimize (hide) dispatch — observable even with window_client: None (T28-1)"
    );
    openharmony_ability::unregister_pending_ui_ability(HIDING_ID);
    drop_pending_window_ops(HIDING_ID);
  }

  /// UT for #237: the UIAbility-window hide contract is an honest no-op -- no
  /// dispatch, no mirror write (the Err surfaces at the tauri-runtime-wry
  /// dispatcher). Discriminating-power note (T28-1): with
  /// `window_client: None` this leg also early-returns on the pre-#237
  /// code, so this UT alone locks little; the paired sub-window UT above
  /// drives the same constructor state through the dispatch branch (queue
  /// entry + mirror writes), proving the observables WOULD fire if the kind
  /// gate were wrong.
  #[test]
  fn ut_set_visible_false_main_window_no_dispatch_and_no_mirror_write() {
    const HIDING_ID: i64 = i64::MAX - 32;
    openharmony_ability::unregister_pending_ui_ability(HIDING_ID);
    drop_pending_window_ops(HIDING_ID);
    openharmony_ability::register_pending_ui_ability(HIDING_ID);
    assert!(!openharmony_ability::is_window_ready(HIDING_ID));

    let mirror = Arc::new(WindowStateMirror::new(true));
    let window = Window {
      app: OpenHarmonyApp::new(),
      window_id: Some(HIDING_ID),
      kind: OHOSWindowKind::UIAbility,
      window_client: std::sync::Mutex::new(None),
      bridge_refetch_warned: AtomicBool::new(false),
      runtime: BridgeExecutor::new(),
      mirror: mirror.clone(),
      decorations: AtomicBool::new(true),
      title: std::sync::Mutex::new(None),
      transparent: false,
      always_on_top: AtomicBool::new(false),
      decoration_flags: AtomicU8::new(15),
      min_inner_width: AtomicU32::new(0),
      min_inner_height: AtomicU32::new(0),
      max_inner_width: AtomicU32::new(0),
      max_inner_height: AtomicU32::new(0),
    };
    window.set_visible(false);

    assert!(
      !mirror.minimized.load(Ordering::Acquire),
      "a refused UIAbility-window hide must not write the minimized mirror"
    );
    assert!(
      mirror.visible.load(Ordering::Acquire),
      "a refused UIAbility-window hide leaves the window untouched — is_visible stays true"
    );
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- see the paired sub-window UT above.
    let queued_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .contains_key(&HIDING_ID);
    assert!(
      !queued_ops,
      "UIAbility-window set_visible(false) must not queue any window op"
    );
    openharmony_ability::unregister_pending_ui_ability(HIDING_ID);
    drop_pending_window_ops(HIDING_ID);
  }

  /// UT for #237 + T28-4: a SUB-window set_visible(true) queues the show
  /// dispatch and flips the visible intent bit back on (the mirror pair of
  /// the hide leg; also locks the T1 deferred leg of the show path -- the
  /// former eager `bridge_client` read dropped the show entirely when the
  /// client was unresolvable).
  #[test]
  fn ut_set_visible_true_sub_window_queues_show_and_restores_visible_mirror() {
    const SHOWING_ID: i64 = i64::MAX - 33;
    openharmony_ability::unregister_pending_ui_ability(SHOWING_ID);
    drop_pending_window_ops(SHOWING_ID);
    openharmony_ability::register_pending_ui_ability(SHOWING_ID);
    assert!(!openharmony_ability::is_window_ready(SHOWING_ID));

    let mirror = Arc::new(WindowStateMirror::new(true));
    mirror.visible.store(false, Ordering::Release);
    let window = Window {
      app: OpenHarmonyApp::new(),
      window_id: Some(SHOWING_ID),
      kind: OHOSWindowKind::SubWindow,
      window_client: std::sync::Mutex::new(None),
      bridge_refetch_warned: AtomicBool::new(false),
      runtime: BridgeExecutor::new(),
      mirror: mirror.clone(),
      decorations: AtomicBool::new(true),
      title: std::sync::Mutex::new(None),
      transparent: false,
      always_on_top: AtomicBool::new(false),
      decoration_flags: AtomicU8::new(15),
      min_inner_width: AtomicU32::new(0),
      min_inner_height: AtomicU32::new(0),
      max_inner_width: AtomicU32::new(0),
      max_inner_height: AtomicU32::new(0),
    };
    window.set_visible(true);

    assert!(
      mirror.visible.load(Ordering::Acquire),
      "show must flip the visible intent bit back on"
    );
    assert!(
      !mirror.minimized.load(Ordering::Acquire),
      "show on a non-minimized window must not write the minimized mirror (show-only leg)"
    );
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- see the paired hide UT above.
    let queued_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&SHOWING_ID)
      .map(|ops| ops.len());
    assert_eq!(
      queued_ops,
      Some(1),
      "sub-window set_visible(true) must queue the show dispatch even with window_client: None (T1 deferred leg)"
    );
    openharmony_ability::unregister_pending_ui_ability(SHOWING_ID);
    drop_pending_window_ops(SHOWING_ID);
  }

  /// UT for PR#28 review T28-2: the RUNTIME set_resizable must not dispatch
  /// set-decoration-flags for a UIAbility-kind window (the ArkTS handler
  /// throws for every UIAbility-kind window -- bootstrap main AND spawned
  /// instances alike, matching the kind-based `isUIAbilityMainWindow` gate in
  /// WindowManager.ets): the local FLAG_RESIZABLE mirror bit must update, and
  /// the queue must hold EXACTLY ONE op -- the edge-drag leg (whose branch
  /// landing is the ArkTS `windowId === 0` routing, T28-5). A decoration
  /// dispatch would enter a second op. The paired sub-window arm locks the
  /// dual dispatch (decoration bit + drag leg = two ops) that a UIAbility-kind window
  /// must not have. Discriminating power (T28-1 discipline): on the pre-fix
  /// code both legs were eager `bridge_client` reads that dropped with
  /// `window_client: None`, so the queue stayed EMPTY and the length
  /// assertions FAIL -- the UT cannot pass without the deferred legs.
  #[test]
  fn ut_set_resizable_ui_ability_window_no_decoration_dispatch_drag_leg_once() {
    const MAIN_ID: i64 = i64::MAX - 34;
    const SUB_ID: i64 = i64::MAX - 35;
    openharmony_ability::unregister_pending_ui_ability(MAIN_ID);
    openharmony_ability::unregister_pending_ui_ability(SUB_ID);
    drop_pending_window_ops(MAIN_ID);
    drop_pending_window_ops(SUB_ID);
    openharmony_ability::register_pending_ui_ability(MAIN_ID);
    openharmony_ability::register_pending_ui_ability(SUB_ID);
    assert!(!openharmony_ability::is_window_ready(MAIN_ID));

    let make_window = |id: i64, kind: OHOSWindowKind| Window {
      app: OpenHarmonyApp::new(),
      window_id: Some(id),
      kind,
      window_client: std::sync::Mutex::new(None),
      bridge_refetch_warned: AtomicBool::new(false),
      runtime: BridgeExecutor::new(),
      mirror: Arc::new(WindowStateMirror::new(true)),
      decorations: AtomicBool::new(true),
      title: std::sync::Mutex::new(None),
      transparent: false,
      always_on_top: AtomicBool::new(false),
      decoration_flags: AtomicU8::new(FLAG_ALL_DECORATIONS),
      min_inner_width: AtomicU32::new(0),
      min_inner_height: AtomicU32::new(0),
      max_inner_width: AtomicU32::new(0),
      max_inner_height: AtomicU32::new(0),
    };

    // Main arm: mirror-only decoration update + exactly one queued op (drag).
    let main = make_window(MAIN_ID, OHOSWindowKind::UIAbility);
    main.set_resizable(false);
    assert_eq!(
      main.decoration_flags.load(Ordering::Acquire) & FLAG_RESIZABLE,
      0,
      "the local FLAG_RESIZABLE mirror bit must clear on a UIAbility window (the set_inner_size gate reads it)"
    );
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- a panic here must not poison PENDING_WINDOW_OPS.
    let main_queued_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&MAIN_ID)
      .map(|ops| ops.len());
    assert_eq!(
      main_queued_ops,
      Some(1),
      "a UIAbility window must queue EXACTLY the drag leg — a decoration dispatch would queue a second op (T28-2)"
    );

    // Sub-window arm: the dual dispatch is intentional there (decoration bit
    // for the SubWindowPage LocalStorage + the drag leg) -- two queued ops.
    let sub = make_window(SUB_ID, OHOSWindowKind::SubWindow);
    sub.set_resizable(false);
    let sub_queued_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&SUB_ID)
      .map(|ops| ops.len());
    assert_eq!(
      sub_queued_ops,
      Some(2),
      "a sub-window keeps the dual dispatch (decoration bit + drag leg), both via the deferred legs"
    );

    openharmony_ability::unregister_pending_ui_ability(MAIN_ID);
    openharmony_ability::unregister_pending_ui_ability(SUB_ID);
    drop_pending_window_ops(MAIN_ID);
    drop_pending_window_ops(SUB_ID);
  }

  /// UT for PR#28 review gate F1: a SUB-window created with resizable(false)
  /// seeds BOTH legs -- the decoration-flags bit (SubWindowPage LocalStorage)
  /// AND the set_resize_by_drag(false) edge-drag seed -- mirroring the runtime
  /// set_resizable dual dispatch (creation/runtime isomorphism). On the
  /// pre-fix plan the drag seed was only planned for UIAbility-kind windows,
  /// so the sub-window creation leg queued ONE op and the first assert fails
  /// (a single decoration dispatch); the default-attrs arm locks the zero
  /// baseline (the 2 ops are the two legs, not stray traffic). Sentinel id +
  /// pending registration + window_client: None, same dispatch-observable
  /// discipline as the T28-1 UTs; the ArkTS branch landing stays the
  /// `windowId === 0` routing (T28-5) -- enableDrag for this id > 0 window.
  #[test]
  fn ut_creation_seed_sub_window_resizable_false_queues_both_legs() {
    const SEED_ID: i64 = i64::MAX - 36;
    const DEFAULT_ID: i64 = i64::MAX - 37;
    // Idempotent reset so a previously failed run cannot leak registry state
    // that would pin these ids as "not ready" or hold stale ops.
    openharmony_ability::unregister_pending_ui_ability(SEED_ID);
    openharmony_ability::unregister_pending_ui_ability(DEFAULT_ID);
    drop_pending_window_ops(SEED_ID);
    drop_pending_window_ops(DEFAULT_ID);
    openharmony_ability::register_pending_ui_ability(SEED_ID);
    openharmony_ability::register_pending_ui_ability(DEFAULT_ID);
    assert!(!openharmony_ability::is_window_ready(SEED_ID));

    let make_window = |id: i64| Window {
      app: OpenHarmonyApp::new(),
      window_id: Some(id),
      kind: OHOSWindowKind::SubWindow,
      window_client: std::sync::Mutex::new(None),
      bridge_refetch_warned: AtomicBool::new(false),
      runtime: BridgeExecutor::new(),
      mirror: Arc::new(WindowStateMirror::new(true)),
      decorations: AtomicBool::new(true),
      title: std::sync::Mutex::new(None),
      transparent: false,
      always_on_top: AtomicBool::new(false),
      decoration_flags: AtomicU8::new(FLAG_ALL_DECORATIONS),
      min_inner_width: AtomicU32::new(0),
      min_inner_height: AtomicU32::new(0),
      max_inner_width: AtomicU32::new(0),
      max_inner_height: AtomicU32::new(0),
    };

    // Leg 1: sub window created resizable(false) -> decoration-flags seed +
    // drag seed, two queued ops.
    let window = make_window(SEED_ID);
    let mut attrs = window::WindowAttributes::default();
    attrs.resizable = false;
    window.seed_creation_attrs(&attrs);
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- a panic here must not poison PENDING_WINDOW_OPS.
    let seeded_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&SEED_ID)
      .map(|ops| ops.len());
    assert_eq!(
      seeded_ops,
      Some(2),
      "a sub-window created resizable(false) must queue BOTH seeds: the decoration-flags leg AND the set_resize_by_drag(false) drag seed (gate F1 creation/runtime isomorphism)"
    );

    // Leg 2: default attrs (resizable(true)) -> zero seeds.
    let default_window = make_window(DEFAULT_ID);
    default_window.seed_creation_attrs(&window::WindowAttributes::default());
    let default_ops = PENDING_WINDOW_OPS
      .lock()
      .expect("PENDING_WINDOW_OPS poisoned")
      .get(&DEFAULT_ID)
      .map(|ops| ops.len());
    assert_eq!(
      default_ops, None,
      "default attrs must plan zero seeds (the two ops above are the two legs, not stray traffic)"
    );

    openharmony_ability::unregister_pending_ui_ability(SEED_ID);
    openharmony_ability::unregister_pending_ui_ability(DEFAULT_ID);
    drop_pending_window_ops(SEED_ID);
    drop_pending_window_ops(DEFAULT_ID);
  }

  // --- PR review fix T1: the deferred-client seed leg. The eager
  // `bridge_client` resolution in the setters dropped a creation seed BEFORE
  // the queue check when the client was not yet resolvable (#184 early-init
  // state) -- the build()-tail seeds never entered PENDING_WINDOW_OPS and the
  // handshake drain had nothing to replay. `enqueue_deferred_window_op` is
  // the replay-time-resolution fallback. Sentinel client (a real
  // WindowClient needs a bridge session) + sentinel ids, same discipline as
  // the #168 tests above.

  #[derive(Clone, Copy)]
  struct DeferredSeedClient;

  /// UT for T1: with the bridge unresolvable at call time and the creation
  /// handshake pending, the seed op is HELD in PENDING_WINDOW_OPS (not
  /// dropped); once the handshake completes AND the bridge resolves, the
  /// next drain replays it -- the future is then built from the freshly
  /// resolved client. A degraded leg keeps the bridge down at replay: the op
  /// still runs (the queue is emptied, nothing stranded) and degrades to the
  /// structured warn without dispatching.
  #[test]
  fn ut_deferred_seed_queues_while_bridge_down_and_replays_after_handshake() {
    const HELD_ID: i64 = i64::MAX - 21;
    const DEGRADED_ID: i64 = i64::MAX - 22;
    let queue = || {
      PENDING_WINDOW_OPS
        .lock()
        .expect("PENDING_WINDOW_OPS poisoned")
    };
    // Idempotent reset so a previously failed run cannot leak registry
    // entries that would pin these ids as "not ready".
    openharmony_ability::unregister_pending_ui_ability(HELD_ID);
    openharmony_ability::unregister_pending_ui_ability(DEGRADED_ID);
    drop_pending_window_ops(HELD_ID);
    drop_pending_window_ops(DEGRADED_ID);

    let bridge_up = Arc::new(AtomicBool::new(false));

    // Leg 1 -- hold: bridge down + handshake pending -> the op must be held,
    // exactly the spec scenario (with_always_on_top(true) while the client
    // is not yet ready SHALL queue).
    openharmony_ability::register_pending_ui_ability(HELD_ID);
    assert!(!openharmony_ability::is_window_ready(HELD_ID));
    let replayed = Arc::new(AtomicBool::new(false));
    let flag = replayed.clone();
    let resolve = bridge_up.clone();
    enqueue_deferred_window_op(
      move || {
        if resolve.load(Ordering::Acquire) {
          Some(DeferredSeedClient)
        } else {
          None
        }
      },
      &BridgeExecutor::new(),
      HELD_ID,
      "ut_deferred_seed",
      move |_| async move {
        flag.store(true, Ordering::SeqCst);
      },
    );
    assert!(
      !replayed.load(Ordering::SeqCst),
      "the deferred seed must not run while the handshake is pending"
    );
    // Gate F9 (test hygiene): bind before assert, guard released on the
    // semicolon -- a panic here must not poison PENDING_WINDOW_OPS (the
    // assert-holding-guard shape cascaded a F8 mutation failure into
    // unrelated UTs).
    let held_ops = queue().get(&HELD_ID).map(|ops| ops.len());
    assert_eq!(
      held_ops,
      Some(1),
      "the deferred seed must be held while the handshake is pending"
    );

    // Leg 2 -- replay: the bridge resolves + the handshake completes -> the
    // drain runs the op, which resolves the client and dispatches.
    bridge_up.store(true, Ordering::SeqCst);
    openharmony_ability::register_ui_ability_stage(HELD_ID);
    assert!(openharmony_ability::is_window_ready(HELD_ID));
    drain_ready_pending_window_ops();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    while !replayed.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
      std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
      replayed.load(Ordering::SeqCst),
      "the deferred seed did not replay after the handshake drain"
    );
    let replay_removed = queue().contains_key(&HELD_ID);
    assert!(!replay_removed, "the replayed entry must be removed");
    openharmony_ability::unregister_pending_ui_ability(HELD_ID);

    // Leg 3 -- degraded: the bridge never resolves. The op still runs at
    // replay (the queue is emptied, no stranded entry) and the client-less
    // body degrades to the structured warn -- no dispatch, no panic.
    openharmony_ability::register_pending_ui_ability(DEGRADED_ID);
    enqueue_deferred_window_op(
      || None::<DeferredSeedClient>,
      &BridgeExecutor::new(),
      DEGRADED_ID,
      "ut_deferred_seed_degraded",
      |_| async move {
        panic!("the degraded leg must never build a future");
      },
    );
    openharmony_ability::register_ui_ability_stage(DEGRADED_ID);
    drain_ready_pending_window_ops();
    let degraded_stranded = queue().contains_key(&DEGRADED_ID);
    assert!(
      !degraded_stranded,
      "the degraded leg must not strand its queue entry"
    );
    openharmony_ability::unregister_pending_ui_ability(DEGRADED_ID);
  }

  /// UT for #168 (replay half decision record): fullscreen and maximized are
  /// both recorded when set -- the fullscreen-takes-precedence ordering
  /// (upstream windows/window.rs:1338-1339 else-if form) lives in
  /// seed_creation_attrs and is compiled against this plan shape.
  #[test]
  fn ut_creation_plan_records_fullscreen_and_maximized() {
    let mut attrs = window::WindowAttributes::default();
    attrs.fullscreen = Some(Fullscreen::Borderless(None));
    attrs.maximized = true;
    let plan = creation_attr_plan(&attrs, false);
    assert!(
      plan.fullscreen.is_some(),
      "fullscreen attr must be recorded"
    );
    assert!(plan.maximized, "maximized attr must be recorded");
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
  /// debounce tripping exactly once -- the flag flips false->true on the first
  /// failure and holds on subsequent failures. Warn emission is gated 1:1 on
  /// that false->true transition (asserted by the reset leg below), so
  /// consecutive failures with a fresh flag can never emit a second warn.
  #[test]
  fn ut_bridge_client_refetch_failure_returns_none_and_warns_once() {
    let cache: std::sync::Mutex<Option<DummyClient>> = std::sync::Mutex::new(None);
    let warned = AtomicBool::new(false);
    // Bridge stays unavailable: app.window() errors -> mapped to None here.
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
  /// window Weak) without panicking -- event-loop focus arms must never fail
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

// DESTROY_FAILURES (issue Eulogizethesun/tauri#255) — appended at file end,
// keeps existing line numbers. The queue is a process global shared with any
// close() call in the binary; these tests serialize on a dedicated lock so a
// drain assertion only ever sees its own enqueues (same discipline as
// WINDOW_MIRRORS_TEST_LOCK).
#[cfg(test)]
mod destroy_failure_tests {
  use super::{drain_destroy_failures, enqueue_destroy_failure};
  use std::sync::Mutex;

  static DESTROY_FAILURES_TEST_LOCK: Mutex<()> = Mutex::new(());

  fn lock() -> std::sync::MutexGuard<'static, ()> {
    DESTROY_FAILURES_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }

  #[test]
  fn destroy_failures_enqueue_then_drain_returns_fifo_and_consumes() {
    let _guard = lock();
    // Defensive isolation: whatever an earlier (non-serialized) close() left
    // behind is drained away before the assertion window opens.
    let _ = drain_destroy_failures();
    enqueue_destroy_failure(41);
    enqueue_destroy_failure(43);
    // FIFO: the runtime pairs drained ids against its in-flight close guards
    // in dispatch order, so order is part of the contract.
    assert_eq!(drain_destroy_failures(), vec![41, 43]);
    // Consuming read: each failure is returned exactly once.
    assert!(drain_destroy_failures().is_empty());
  }

  #[test]
  fn destroy_failures_drain_on_untouched_queue_returns_empty() {
    let _guard = lock();
    // Serialized against the enqueue test, so this observes either the
    // cold-start empty queue or the post-drain empty queue.
    assert!(drain_destroy_failures().is_empty());
  }
}
