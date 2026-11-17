use std::cell::{Cell, RefCell};
use std::collections::{HashSet, VecDeque};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

use openharmony_ability::xcomponent::{Action, MouseButton as OhosMouseButton, TouchEvent};
use openharmony_ability::{
  ime::KeyboardStatus, CursorUntracked, Event as MainEvent, ImeEvent, InputEvent, OpenHarmonyApp,
  OpenHarmonyWaker,
};
use openharmony_ability::{AxisEventData, InputSourceType, MouseAction, MouseEventData};
use openharmony_ability_plugin_app_control::{AppControlExt, ColorModeExt};

use crate::dpi::{PhysicalPosition, PhysicalSize};
use crate::error;
use crate::event::{self, ElementState, Force, StartCause};
use crate::event_loop::{self, ControlFlow, ThemeReceipt};
use crate::keyboard::{Key, KeyCode, KeyLocation, ModifiersState};
use crate::monitor;
use crate::window::{self, Theme};

use super::keycodes::{to_location, to_logical, to_physical};
use super::monitor::MonitorHandle;
use super::window::{
  drain_destroy_confirmations, drain_ready_pending_window_ops, WindowId, WINDOW_MIRRORS,
};

pub(crate) static HAS_FOCUS: AtomicBool = AtomicBool::new(true);

/// App-level theme override (issue 5, 5.2 theme backfill).
/// `set_theme(Some)` writes an explicit override; `set_theme(None)` writes FOLLOW (follow system).
/// `theme()` reads this override: on FOLLOW it falls back to `app.config().color_mode`
/// (continuously refreshed by the ConfigChanged event, reflecting system truth, no
/// manual backfill needed). Global rather than per-window, because OHOS setColorMode
/// is itself global (not window-level).
pub(crate) const THEME_OVERRIDE_LIGHT: u8 = 0;
pub(crate) const THEME_OVERRIDE_DARK: u8 = 1;
pub(crate) const THEME_OVERRIDE_FOLLOW: u8 = 2;
pub(crate) static APP_THEME_OVERRIDE: AtomicU8 = AtomicU8::new(THEME_OVERRIDE_FOLLOW);

/// FIFO queue of deferred set-theme receipts (issue Eulogizethesun/tauri#264,
/// design D2). `set_theme_with_receipt` pushes one receipt BEFORE the theme
/// change is applied, and every drained completion (see
/// [`apply_color_mode_completion`]) pops exactly one receipt — FIFO by
/// arrival — and fires it with the verdict.
///
/// Ownership semantics (the honest boundary, keep in sync with the facade
/// doc): the pairing is by ARRIVAL ORDER, not by call identity. Completion
/// producers that do NOT carry a receipt — the native Rust `set_theme` and
/// any window-level setTheme path — still emit completions into the shared
/// slot; when such a bare completion drains while a JS caller's receipt is
/// queued, the receipt is woken by the wrong verdict (off by one slot). The
/// same slot-sharing also collapses two CONCURRENT receipted JS calls: both
/// receipts queue, but the single `take_color_mode_completion` slot merges
/// their deferred ticks, so the first-queued receipt can receive the other
/// call's verdict and the second receipt starves until its command-side
/// timeout fires. Root fix for both forms = a call-id on the setColorMode
/// wire so completions can be matched to their own receipt — future item,
/// tracked in issue Eulogizethesun/tauri#264.
///
/// `Mutex::new` is const (Rust 1.63+, above this crate's 1.74 MSRV), so the
/// queue needs no LazyLock; poison recovery follows the same
/// `poisoned.into_inner()` discipline as the test lock below (a receipt
/// panicking must not wedge the queue for later callers).
static THEME_RECEIPTS: std::sync::Mutex<VecDeque<ThemeReceipt>> =
  std::sync::Mutex::new(VecDeque::new());

/// Effective theme (override if set, else system color_mode) encoded for the
/// ThemeChanged dispatch guard (issue Eulogizethesun/tauri#108). u8 pairs with
/// the THEME_OVERRIDE_* constants' Light=0/Dark=1 encoding.
const EFFECTIVE_THEME_LIGHT: u8 = 0;
const EFFECTIVE_THEME_DARK: u8 = 1;
const EFFECTIVE_THEME_UNSEEDED: u8 = 2;
static LAST_EFFECTIVE_THEME: AtomicU8 = AtomicU8::new(EFFECTIVE_THEME_UNSEEDED);

/// #180(a) (design D7): the window id currently recorded as focused -- the
/// shared change-edge snapshot read by both focus sources (SubWindowPage
/// `windowEvent` WINDOW_ACTIVE/WINDOW_INACTIVE edges drained via
/// `openharmony_ability::drain_pending_window_focus`, and the per-UIAbility
/// stage GainedFocus/LostFocus arms). 0 = none. Scope note (review R20): the
/// edge rules built on this snapshot -- a gain edge (focused=true, last!=W)
/// pairs-clears the previous holder before the gain, a loss edge only
/// applies when W is the recorded holder, and same-state duplicate edges are
/// swallowed (zero WindowEvents) -- are implemented by the DRAIN path only
/// (`EventLoop::handle_pending_window_focus`, invoked from the run_loop's
/// drained-focus loop). The stage arms only WRITE the snapshot so
/// cross-path duplicate edges dedup against the same record
/// (`EventLoop::handle_gained_focus` stores gain -> window id; the stage
/// LostFocus arm stores loss -> 0), and their semantics are asymmetric with
/// those rules: a duplicate stage GainedFocus re-dispatches `Focused(true)`
/// (no same-state swallow), a stage gain over a drain-focused holder does
/// NOT pair-clear the old holder's focused bit, and the stage LostFocus arm
/// stores 0 unconditionally (no holder guard). Read and written from the
/// single-threaded run_loop dispatch only -- Relaxed matches the HAS_FOCUS
/// discipline.
static LAST_FOCUSED_WINDOW_ID: AtomicI64 = AtomicI64::new(0);

/// #180 (design D8): the last scale factor forwarded as
/// `ScaleFactorChanged`, stored as `f64::to_bits` (NaN-safe bitwise
/// comparison). Bitwise-equal repeats are swallowed; the first event forwards
/// (0-bits is not a representable scale, so the initial value reads as
/// unseeded). Mirrors the ThemeChanged edge detection
/// ([`LAST_EFFECTIVE_THEME`]) -- the two guards are independent and must not
/// interact.
static LAST_EFFECTIVE_SCALE: AtomicU64 = AtomicU64::new(0);

/// #180 (design D8): the ThemeChanged-style edge detector for scale factors --
/// returns `true` when the event must forward (first event or a real bitwise
/// change) and updates the snapshot; `false` swallows the repeat. Extracted
/// so the UT can drive the exact production guard.
fn scale_should_forward(new_scale: f64) -> bool {
  let bits = new_scale.to_bits();
  let prev = LAST_EFFECTIVE_SCALE.swap(bits, Ordering::Relaxed);
  prev != bits
}

/// Last dispatched rect per OHOS window id — (outer left, outer top, INNER
/// width, inner height; review R11) — used to split windowRectChange events
/// into Moved/Resized (issue Eulogizethesun/tauri#107). Mutex (not
/// thread_local) because lifecycle callbacks and the run_loop may dispatch
/// from different threads.
pub(crate) static LAST_DISPATCHED_RECTS: std::sync::LazyLock<
  std::sync::Mutex<std::collections::HashMap<i64, (i32, i32, i32, i32)>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Currently active keyboard modifiers, tracked from modifier key Down/Up
/// events (the NDK KeyEventData carries no modifier state; issue
/// Eulogizethesun/tauri#109). Read by the mouse/wheel handlers to populate the
/// deprecated `modifiers` field, and diffed on every key event to dispatch
/// `WindowEvent::ModifiersChanged`, and cleared (with ModifiersChanged) on
/// focus regain (issue Eulogizethesun/tauri#137). Mutex: XComponent input
/// callbacks and the run_loop focus handlers run on different threads.
static KEYBOARD_MODIFIERS: std::sync::Mutex<ModifiersState> =
  std::sync::Mutex::new(ModifiersState::empty());

/// Effective theme for the whole app: the explicit override when set, else the
/// system color_mode (Light for NoSet — matches `Window::theme`'s fallback).
/// Shared by `Window::theme()` and the ConfigChanged ThemeChanged dispatch.
pub(crate) fn effective_theme(app: &OpenHarmonyApp) -> Theme {
  use openharmony_ability::ColorMode;
  match APP_THEME_OVERRIDE.load(Ordering::Relaxed) {
    THEME_OVERRIDE_DARK => Theme::Dark,
    THEME_OVERRIDE_LIGHT => Theme::Light,
    _ => match app.config().color_mode {
      ColorMode::Dark => Theme::Dark,
      // Light or NoSet (no ConfigChanged received before startup) → Light.
      _ => Theme::Light,
    },
  }
}

/// Last known cursor position lives in the process-level cursor statics inside
/// openharmony-ability (screen-global vp, fed by the ArkTS `onMouse` handlers
/// of MainPage AND SubWindowPage via the `update_cursor_position` NAPI function --
/// ohos-cursor-passthrough D1/D2) and is read back through
/// `OpenHarmonyApp::cursor_position()` (physical px). Never-fed reads fail
/// with the typed untracked error -> `ExternalError::NotSupported` (D3/R5);
/// the NDK XComponent mouse path never fires while the cursor is over the
/// WebView (which covers the window), so it cannot be the tracking source.

/// Background tokio runtime for spawning async bridge calls (fire-and-forget).
///
/// `WindowClient` methods are `async` and return `Result<()>`. tao's window
/// operation APIs (e.g. `set_inner_size`) are synchronous and return `()` — they
/// cannot `.await`. `BridgeExecutor` wraps a `tokio::runtime::Handle` from a
/// dedicated background thread (`ohos-bridge-rt`) that drives a current-thread
/// runtime. Calling `spawn(future)` sends the future to that background thread
/// to be polled. The TSFN NonBlocking call inside `WindowClient` returns
/// immediately; the ArkTS callback runs on the main thread → no deadlock.
///
/// `tokio::runtime::Handle` is `Clone + Send + Sync`, so `BridgeExecutor` is
/// safely cloneable and can be stored in both `EventLoop` and `Window`.
#[derive(Clone)]
pub(crate) struct BridgeExecutor {
  handle: tokio::runtime::Handle,
}

impl BridgeExecutor {
  // pub(crate) (PR review fix T1): the deferred-seed UT in window.rs builds
  // its own executor to drive the enqueue/replay mechanic without a bridge
  // session.
  pub(crate) fn new() -> Self {
    // Panics here are acceptable: this runs exactly once during EventLoop
    // construction, before the app is functional or any recovery path
    // exists. A failure to build the tokio runtime or spawn its driver
    // thread leaves the bridge (and thus all async window operations)
    // unusable, so aborting is the only sane option.
    let runtime = tokio::runtime::Builder::new_current_thread()
      .enable_all()
      .build()
      .expect("Failed to create OHOS bridge runtime");
    let handle = runtime.handle().clone();
    std::thread::Builder::new()
      .name("ohos-bridge-rt".into())
      .spawn(move || runtime.block_on(std::future::pending::<()>()))
      .expect("Failed to spawn bridge runtime thread");
    Self { handle }
  }

  /// Spawn a fire-and-forget bridge call. The result is ignored.
  pub(crate) fn spawn<F>(&self, future: F)
  where
    F: std::future::Future<Output = ()> + Send + 'static,
  {
    self.handle.spawn(future);
  }
}

// Tracks currently pressed keys for repeat detection.
// When a Down event arrives for a key already in this set, it's a repeat.
thread_local! {
    static PRESSED_KEYS: RefCell<HashSet<i32>> = RefCell::new(HashSet::new());
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct KeyEventExtra {}

/// Map an OHOS modifier keycode to its tao modifier bit (issue
/// Eulogizethesun/tauri#109). `None` for non-modifier keys. Both left/right
/// variants map to the same bit; `KeyLocation` still disambiguates them in
/// the key event itself.
fn keycode_as_modifier(
  keycode: openharmony_ability::xcomponent::KeyCode,
) -> Option<ModifiersState> {
  use openharmony_ability::xcomponent::KeyCode::*;

  let bit = match keycode {
    ShiftLeft | ShiftRight => ModifiersState::SHIFT,
    CtrlLeft | CtrlRight => ModifiersState::CONTROL,
    AltLeft | AltRight => ModifiersState::ALT,
    MetaLeft | MetaRight => ModifiersState::SUPER,
    _ => return None,
  };
  Some(bit)
}

/// Snapshot of the tracked keyboard modifiers (empty when the lock is
/// poisoned), used to populate the deprecated `modifiers` field on mouse and
/// wheel events (issue Eulogizethesun/tauri#109).
fn current_modifiers() -> ModifiersState {
  KEYBOARD_MODIFIERS
    .lock()
    .map(|modifiers| *modifiers)
    .unwrap_or_else(|poisoned| *poisoned.into_inner())
}

/// Live window ids from the mirror registry, with a window-0 fallback for
/// the pre-first-window phase. Dead mirrors (window dropped without
/// running `Drop`) are skipped — their events would only be dropped by
/// the runtime-wry id lookup anyway, and the entries themselves are
/// reaped lazily in the pending-status drain. Shared by the app-level
/// broadcast arms (ConfigChanged's ScaleFactorChanged / ThemeChanged).
fn live_window_ids() -> Vec<i64> {
  let ids: Vec<i64> = WINDOW_MIRRORS
    .lock()
    .map(|mirrors| {
      mirrors
        .iter()
        .filter(|(_, weak)| weak.strong_count() > 0)
        .map(|(id, _)| *id)
        .collect()
    })
    .unwrap_or_default();
  if ids.is_empty() {
    vec![0]
  } else {
    ids
  }
}

/// Map an OHOS NDK MouseButton to tao's MouseButton.
///
/// Returns `None` for `NoneButton` (no meaningful button to report).
fn ohos_mouse_button_to_tao(button: OhosMouseButton) -> Option<event::MouseButton> {
  match button {
    OhosMouseButton::LeftButton => Some(event::MouseButton::Left),
    OhosMouseButton::RightButton => Some(event::MouseButton::Right),
    OhosMouseButton::MiddleButton => Some(event::MouseButton::Middle),
    OhosMouseButton::BackButton => Some(event::MouseButton::Other(4)),
    OhosMouseButton::ForwardButton => Some(event::MouseButton::Other(5)),
    OhosMouseButton::NoneButton => None,
  }
}

/// Dispatch `event` to the user handler stored in `cell`, if one is installed.
/// The `if let Some(ref mut h) = *cell.borrow_mut() { h(event) }` dance
/// appeared ~27 times across the input handlers and the `run_loop` dispatch —
/// this keeps it in one place (mirrors upstream android's
/// `call_event_handler!`).
macro_rules! call_event_handler {
  ($cell:expr, $event:expr) => {
    if let Some(ref mut handler) = *$cell.borrow_mut() {
      handler($event);
    }
  };
}

/// Emit a synthesized press+release key pair on `window_id`.
///
/// The IME handlers use this to mock physical key events where OHOS only
/// reports IME-level facts: Backspace/Enter edits arrive as IME events (no
/// key events at all), and keyboard Hide needs a synthetic Enter so web
/// engines commit their composition / fire blur.
fn emit_synthetic_key<T: 'static>(
  event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
  window_id: window::WindowId,
  logical_key: Key<'static>,
  physical_key: KeyCode,
) {
  let _ = [ElementState::Pressed, ElementState::Released].map(|state| {
    call_event_handler!(
      event_loop_cell,
      event::Event::WindowEvent {
        window_id,
        event: event::WindowEvent::KeyboardInput {
          device_id: event::DeviceId(DeviceId(0)),
          event: event::KeyEvent {
            state,
            logical_key: logical_key.clone(),
            physical_key,
            platform_specific: KeyEventExtra {},
            repeat: false,
            location: KeyLocation::Standard,
            text: None,
          },
          is_synthetic: false,
        },
      }
    );
  });
}

/// Shared cursor-position read (physical px), used by both
/// `EventLoopWindowTarget::cursor_position` and `Window::cursor_position`.
/// Fed by the ArkTS `onMouse` handlers (MainPage AND SubWindowPage --
/// ohos-cursor-passthrough D2) via the `update_cursor_position` NAPI
/// function; the vp->physical conversion and the bit-unpacking of the process
/// statics live inside `OpenHarmonyApp::cursor_position` (issue #87 major-10)
/// -- see the CURSOR_POSITION note near the top of this file.
///
/// Untracked fail-safe contract (ohos-cursor-passthrough D3, R5 -- behavior
/// change): when no feed was ever recorded the read is the typed
/// `CursorUntracked` error, mapped here onto `ExternalError::NotSupported`
/// (the section 1.5 error-conversion constraint: single variant, no wrapping) --
/// NOT swallowed into `Ok((0,0))`, which was indistinguishable from a real
/// screen-origin feed. runtime-wry maps `NotSupported` to
/// `FailedToGetCursorPosition`, so the tauri command and the JS
/// `cursorPosition()` promise reject; app-side discipline (spec): catch ->
/// keep `set_ignore_cursor_events(true)`, only a hit-test match may set
/// touchable.
/// PR#28 review T28-10 (gate F11): process-global warn-once latch for the
/// untracked-cursor fail-safe below. Hoisted from a fn-local static to module
/// scope so the F11 UT can re-arm it -- device UTs share one process and
/// #[test]s run in parallel. In production the latch only "clears" by the
/// first feed arriving (after which the read succeeds and this branch never
/// fires again), so a single latch is sufficient; a UT reset merely re-opens
/// the one warn slot.
static UNTRACKED_CURSOR_WARNED: AtomicBool = AtomicBool::new(false);

pub(crate) fn cursor_position_from_app(
  app: &OpenHarmonyApp,
) -> Result<PhysicalPosition<f64>, error::ExternalError> {
  app
    .cursor_position()
    .map(|(x, y)| PhysicalPosition::new(x, y))
    .map_err(|CursorUntracked| {
      // PR#28 review T28-10: a polling caller hits this branch on every read
      // while no mouse feed has ever arrived -- warn once, then drop to debug,
      // so the hilog cannot be flooded. See UNTRACKED_CURSOR_WARNED (gate F11
      // hoisted the state to module scope for UT re-armability).
      if !UNTRACKED_CURSOR_WARNED.swap(true, Ordering::Relaxed) {
        log::warn!("cursor_position: untracked (no mouse feed ever recorded) → NotSupported");
      } else {
        log::debug!("cursor_position: untracked (no mouse feed ever recorded) → NotSupported");
      }
      error::ExternalError::NotSupported(error::NotSupportedError::new())
    })
}

/// Shared set_theme: write the global override (so `theme()` immediately
/// reflects intent) and push the color mode through the MainThreadSync bridge.
/// Used by both `EventLoopWindowTarget::set_theme` and `Window::set_theme`.
pub(crate) fn set_app_theme(app: &OpenHarmonyApp, theme: Option<Theme>) {
  use openharmony_ability::ColorMode;
  APP_THEME_OVERRIDE.store(
    match theme {
      Some(Theme::Dark) => THEME_OVERRIDE_DARK,
      Some(Theme::Light) => THEME_OVERRIDE_LIGHT,
      None => THEME_OVERRIDE_FOLLOW,
    },
    Ordering::Relaxed,
  );
  let color_mode = match theme {
    Some(Theme::Dark) => ColorMode::Dark,
    Some(Theme::Light) => ColorMode::Light,
    None => ColorMode::NoSet,
  };
  // Migrate from OpenHarmonyApp::set_color_mode (removed) to
  // ColorModeExt::set_color_mode (MainThreadSync bridge call).
  // Bridge contract: Dark=0, Light=1, NoSet=2.
  let mode_i32 = match color_mode {
    ColorMode::Dark => 0,
    ColorMode::Light => 1,
    ColorMode::NoSet => 2,
  };
  let env_cell = openharmony_ability::get_main_thread_env();
  let env_ref = env_cell.borrow();
  if let Some(env) = env_ref.as_ref() {
    if let Err(e) = app.set_color_mode(env, mode_i32) {
      log::warn!("set_theme: failed to call set_color_mode: {:?}", e);
    }
  } else {
    log::warn!("set_theme: main thread Env not available");
  }
}

/// Queues one receipt for the NEXT drained completion (issue
/// Eulogizethesun/tauri#264, design D2). Called by
/// `EventLoopWindowTarget::set_theme_with_receipt` BEFORE
/// [`set_app_theme`] applies the change, on the same thread — the registration
/// is synchronous and can never be overtaken by a completion, because
/// completions are only consumed by [`apply_color_mode_completion`] at the
/// `MainEvent` dispatch head, which this thread cannot re-enter mid-call.
pub(crate) fn register_theme_receipt(receipt: ThemeReceipt) {
  let mut queue = THEME_RECEIPTS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  queue.push_back(receipt);
}

/// Pops exactly one receipt, FIFO by arrival (`None` when no receipt is
/// queued — a bare completion, e.g. from the native Rust `set_theme`, wakes
/// nobody). Poison recovery matches [`register_theme_receipt`].
fn pop_theme_receipt() -> Option<ThemeReceipt> {
  let mut queue = THEME_RECEIPTS
    .lock()
    .unwrap_or_else(|poisoned| poisoned.into_inner());
  queue.pop_front()
}

/// Issue Eulogizethesun/tauri#222 drain leg: applies ONE deferred setColorMode
/// completion (the `openharmony_ability::take_color_mode_completion()` slot —
/// `Some(true)` executed, `Some(false)` failed, `None` nothing new). Returns
/// whether a failure completion rolled [`APP_THEME_OVERRIDE`] back to
/// FOLLOW (pure apart from that store — device-testable).
///
/// WHY a drain and not a poll at the call site: the set-color-mode chain —
/// `set_app_theme` → `ColorModeExt::set_color_mode` (the facade that decodes
/// `SetColorModeResponse`, plugin-app-control lib.rs) →
/// `BridgeMainThread::call_sync` — runs ON the ArkTS N-API main thread. The
/// run_loop callbacks execute there (so `get_main_thread_env()` is `Some`;
/// see the exit arm below), the `call_sync` Env-identity check
/// (openharmony-ability bridge/mod.rs) rejects callers outside the active
/// ArkTS environment, and worker callers are exactly what
/// `call_sync_from_worker` exists for — set-color-mode does not use it. The
/// ArkTS handler defers the actual `setColorMode` with `setTimeout(0)`, a
/// LATER macrotask that cannot run while this synchronous dispatch blocks the
/// main thread — an inline bounded poll would starve it and turn every
/// `set_theme` into a false Err. So the completion is consumed here, at the
/// start of every MainEvent dispatch (both notify legs wake the loop, so the
/// drain runs promptly after the deferred tick reports). Consequence: the JS
/// caller still sees the bridge's accepted=true → Ok for this call; a failure
/// surfaces as the error log plus the override rollback (the honest Err for
/// the JS caller would need the call chain moved off the main thread — a
/// worker-migration rework, deliberately out of scope here).
///
/// Issue Eulogizethesun/tauri#264 (design D2) extends the drain with the
/// receipt channel: `set_theme_with_receipt` queued a
/// [`ThemeReceipt`] BEFORE applying the change, and EVERY completion
/// carrying a verdict pops exactly one queued receipt — FIFO by arrival —
/// and fires it with that verdict (`true` applied, `false` failed). The
/// pairing is by arrival order, not call identity: a bare completion (from
/// the native Rust `set_theme` or a window-level setTheme) drains while a
/// JS receipt is queued wakes that receipt with the wrong verdict — the
/// disclosed off-by-one limitation; two CONCURRENT receipted JS calls share
/// the same single completion slot, so the first-queued receipt can receive
/// the other call's verdict while the second starves until its command-side
/// timeout. Root fix for both forms = call-id on the wire (future item,
/// issue #264; see THEME_RECEIPTS). `None` carries no verdict and
/// deliberately pops nothing, so a queued receipt is only ever woken by a
/// completion that actually carries one. Failure ordering: the override is
/// rolled back to FOLLOW FIRST and the receipt fires afterwards — the JS
/// caller learns of the failure only once the honest (rolled-back) state is
/// already in effect. The receipt closure itself owns its send-failure
/// swallowing (the awaiting command side times out and drops its receiver:
/// a normal outcome), so firing it here cannot panic.
pub(crate) fn apply_color_mode_completion(completion: Option<bool>) -> bool {
  match completion {
    Some(false) => {
      log::error!(
        "[tao-ohos] setColorMode failed on the deferred ArkTS tick — rolling the theme override back to follow-system"
      );
      APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
      // AFTER the rollback: the receipt is the JS caller's failure signal,
      // and it must observe the already-rolled-back override.
      if let Some(receipt) = pop_theme_receipt() {
        receipt(false);
      }
      true
    }
    // Applied: the override already reflects the intent — hand the deferred
    // caller its success.
    Some(true) => {
      if let Some(receipt) = pop_theme_receipt() {
        receipt(true);
      }
      false
    }
    // Nothing new since the last drain: no verdict, no receipt wake, no
    // override change.
    None => false,
  }
}

/// Shared monitor_from_point: hit-test the point against every display's
/// bounds in the global coordinate space (positions from ArkTS Display.x/y,
/// issue Eulogizethesun/tauri#106 — previously only the default display's
/// bounds at (0,0) were tested). Used by both
/// `EventLoopWindowTarget::monitor_from_point` and `Window::monitor_from_point`.
pub(crate) fn monitor_from_point_for_app(
  app: &OpenHarmonyApp,
  x: f64,
  y: f64,
) -> Option<MonitorHandle> {
  MonitorHandle::from_point(app, x, y)
}

pub struct EventLoop<T: 'static> {
  pub(crate) openharmony_app: OpenHarmonyApp,
  window_target: Arc<event_loop::EventLoopWindowTarget<T>>,
  user_events_sender: mpsc::Sender<T>,
  user_events_receiver: Arc<RefCell<mpsc::Receiver<T>>>,
  event_loop: Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PlatformSpecificEventLoopAttributes {
  pub(crate) openharmony_app: Option<OpenHarmonyApp>,
}

impl Default for PlatformSpecificEventLoopAttributes {
  fn default() -> Self {
    Self {
      openharmony_app: Default::default(),
    }
  }
}

impl<T: 'static> EventLoop<T> {
  pub(crate) fn new(attributes: &PlatformSpecificEventLoopAttributes) -> Self {
    let (user_events_sender, user_events_receiver) = mpsc::channel();

    let openharmony_app = attributes.openharmony_app.as_ref().expect(
      "An `OpenHarmonyApp` as passed to lib is required to create an `EventLoop` on \
             OpenHarmony or HarmonyNext",
    );

    let bridge_executor = BridgeExecutor::new();

    Self {
      openharmony_app: openharmony_app.clone(),
      window_target: Arc::new(event_loop::EventLoopWindowTarget {
        p: EventLoopWindowTarget {
          app: openharmony_app.clone(),
          bridge_executor,
          exit: Cell::new(false),
          _marker: PhantomData,
        },
        _marker: PhantomData,
      }),
      user_events_sender,
      user_events_receiver: Arc::new(RefCell::new(user_events_receiver)),
      event_loop: Arc::new(RefCell::new(None)),
    }
  }

  pub(crate) fn window_target(&self) -> &event_loop::EventLoopWindowTarget<T> {
    &*self.window_target
  }

  // TODO: For input event, we need some real examples to test it
  // Phase 4 (design.md D5): input events carry the window id of the render
  // surface that produced them (from render()'s `window_id` param). Only the
  // primary instance owns an XComponent (spawned UIAbility windows mount
  // their webview through the WindowManager multi-root path and their content
  // input is consumed by ArkWeb), so in practice this is 0 — the routing is
  // plumbed from the source instead of hardcoded.
  fn handle_input_event(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: i64,
    event: &InputEvent,
  ) {
    let window_id = window::WindowId(WindowId(window_id));
    #[allow(unreachable_patterns)]
    match event {
      InputEvent::TouchEvent(motion_event) => {
        let device_id = event::DeviceId(DeviceId(motion_event.device_id as _));
        let action = motion_event.event_type;

        let phase = match motion_event.event_type {
          TouchEvent::Down => Some(event::TouchPhase::Started),
          TouchEvent::Up => Some(event::TouchPhase::Ended),
          TouchEvent::Move => Some(event::TouchPhase::Moved),
          TouchEvent::Cancel => Some(event::TouchPhase::Cancelled),
          _ => None,
        };

        if let Some(phase) = phase {
          for pointer in motion_event.touch_points.iter() {
            let position = PhysicalPosition {
              x: pointer.x as _,
              y: pointer.y as _,
            };
            trace!(
              "Input event {device_id:?}, {action:?}, loc={position:?}, \
                                 pointer={pointer:?}"
            );

            call_event_handler!(
              event_loop_cell,
              event::Event::WindowEvent {
                window_id,
                event: event::WindowEvent::Touch(event::Touch {
                  device_id,
                  phase,
                  location: position,
                  id: pointer.id as u64,
                  force: Some(Force::Normalized(pointer.force as f64)),
                }),
              }
            );
          }
        }
      }
      InputEvent::MouseEvent(mouse_event) => {
        Self::handle_mouse_event(event_loop_cell, window_id, mouse_event);
      }
      InputEvent::AxisEvent(axis_event) => {
        Self::handle_axis_event(event_loop_cell, window_id, axis_event);
      }
      InputEvent::KeyEvent(key) => {
        let keycode = key.code;
        {
          let state = match key.action {
            Action::Down => event::ElementState::Pressed,
            Action::Up => event::ElementState::Released,
            _ => event::ElementState::Released,
          };

          // Detect key repeat: if a Down event arrives for a key already
          // in the pressed set, it's an auto-repeat from holding the key.
          let key_raw = keycode as i32;
          let repeat = PRESSED_KEYS.with(|keys| {
            let mut keys = keys.borrow_mut();
            match key.action {
              Action::Down => !keys.insert(key_raw), // false if already present → repeat
              Action::Up => {
                keys.remove(&key_raw);
                false
              }
              _ => false,
            }
          });

          // Modifier tracking (issue Eulogizethesun/tauri#109): the NDK
          // KeyEventData carries no modifier state, so derive it from the
          // modifier keycodes themselves and dispatch ModifiersChanged on
          // transitions (before the key event, matching other backends).
          // Transition-gated: pressing the pair's other side (ShiftLeft then
          // ShiftRight) does not change the modifier set — no redundant
          // ModifiersChanged is dispatched, only the key event.
          if let Some(modifier) = keycode_as_modifier(keycode) {
            if let Ok(mut modifiers) = KEYBOARD_MODIFIERS.lock() {
              // ModifiersState is bitflags: insert/remove return (), so probe
              // membership first to detect an actual state transition.
              let changed = match state {
                event::ElementState::Pressed => {
                  let newly_held = !modifiers.contains(modifier);
                  modifiers.insert(modifier);
                  newly_held
                }
                event::ElementState::Released => {
                  let was_held = modifiers.contains(modifier);
                  modifiers.remove(modifier);
                  was_held
                }
              };
              if changed {
                // Route by the same event source's window id as the
                // KeyboardInput dispatch below (Phase 4 design.md D5): the
                // modifier state itself is app-level, but the transition
                // notification belongs to the window that received the key,
                // so a spawned UIAbility's modifier press must not land on
                // the main window (O10-1).
                call_event_handler!(
                  event_loop_cell,
                  event::Event::WindowEvent {
                    window_id,
                    event: event::WindowEvent::ModifiersChanged(*modifiers),
                  }
                );
              }
            }
          }

          let physical_key = to_physical(keycode);
          let logical_key = to_logical(keycode);

          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id,
              event: event::WindowEvent::KeyboardInput {
                device_id: event::DeviceId(DeviceId(key.device_id as _)),
                event: event::KeyEvent {
                  state,
                  physical_key,
                  logical_key,
                  location: to_location(keycode),
                  repeat,
                  text: None,
                  platform_specific: KeyEventExtra {},
                },
                is_synthetic: false,
              },
            }
          );
        }
      }
      InputEvent::ImeEvent(data) => match data {
        ImeEvent::TextInputEvent(s) => {
          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id,
              event: event::WindowEvent::ReceivedImeText(s.text.clone()),
            }
          );
        }
        ImeEvent::BackspaceEvent(_) => {
          // No physical key events exist for IME edits — mock Backspace so
          // web engines / GUI toolkits can react to the deletion.
          emit_synthetic_key(
            event_loop_cell,
            window_id,
            Key::Backspace,
            KeyCode::Backspace,
          );
        }
        ImeEvent::EnterEvent(_) => {
          // Same as Backspace: mock an Enter key press.
          emit_synthetic_key(event_loop_cell, window_id, Key::Enter, KeyCode::Enter);
        }
        ImeEvent::ImeStatusEvent(s) => match s {
          KeyboardStatus::Hide => {
            // Mock an Enter key press so egui/web engines receive a key event
            // and trigger their onblur/commit behavior on keyboard hide.
            emit_synthetic_key(event_loop_cell, window_id, Key::Enter, KeyCode::Enter);
          }
          _ => {
            warn!("Unknown openharmony_ability ime status event {s:?}")
          }
        },
      },
      _ => {
        warn!("Unknown openharmony_ability input event {event:?}")
      }
    }
  }

  /// Handle mouse events from the OHOS NDK, converting them to tao WindowEvents.
  /// `window_id` is the routed window (from the originating input event,
  /// Phase 4 design.md D5).
  fn handle_mouse_event(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: window::WindowId,
    mouse_event: &MouseEventData,
  ) {
    // Use device_id 0 for mouse, consistent across events.
    let device_id = event::DeviceId(DeviceId(0));

    match mouse_event.action {
      MouseAction::Move => {
        // Cursor tracking is NOT done here: the NDK mouse callback never fires
        // while the cursor is over the WebView. See the CURSOR_POSITION note
        // near the top of this file.
        let position = PhysicalPosition {
          x: mouse_event.x as f64,
          y: mouse_event.y as f64,
        };
        call_event_handler!(
          event_loop_cell,
          event::Event::WindowEvent {
            window_id,
            event: event::WindowEvent::CursorMoved {
              device_id,
              position,
              modifiers: current_modifiers(),
            },
          }
        );
      }
      MouseAction::Press => {
        if let Some(button) = ohos_mouse_button_to_tao(mouse_event.button) {
          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id,
              event: event::WindowEvent::MouseInput {
                device_id,
                state: ElementState::Pressed,
                button,
                modifiers: current_modifiers(),
              },
            }
          );
        }
      }
      MouseAction::Release => {
        if let Some(button) = ohos_mouse_button_to_tao(mouse_event.button) {
          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id,
              event: event::WindowEvent::MouseInput {
                device_id,
                state: ElementState::Released,
                button,
                modifiers: current_modifiers(),
              },
            }
          );
        }
      }
      MouseAction::HoverEnter => {
        call_event_handler!(
          event_loop_cell,
          event::Event::WindowEvent {
            window_id,
            event: event::WindowEvent::CursorEntered { device_id },
          }
        );
      }
      MouseAction::HoverLeave => {
        call_event_handler!(
          event_loop_cell,
          event::Event::WindowEvent {
            window_id,
            event: event::WindowEvent::CursorLeft { device_id },
          }
        );
      }
      MouseAction::None => {
        // Ignore None events
      }
    }
  }

  /// Handle axis (scroll wheel) events from the OHOS ArkUI runtime.
  /// `window_id` is the routed window (from the originating input event,
  /// Phase 4 design.md D5).
  fn handle_axis_event(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: window::WindowId,
    axis_event: &AxisEventData,
  ) {
    let device_id = event::DeviceId(DeviceId(0));
    let is_touchpad = axis_event.source_type == InputSourceType::Touchpad;

    // Emit scroll wheel event.
    // Use PixelDelta for touchpad (pixel-based), LineDelta for mouse wheel (line-based).
    if axis_event.delta_x != 0.0 || axis_event.delta_y != 0.0 {
      let delta = if is_touchpad {
        event::MouseScrollDelta::PixelDelta(PhysicalPosition {
          x: axis_event.delta_x as f64,
          y: axis_event.delta_y as f64,
        })
      } else {
        event::MouseScrollDelta::LineDelta(axis_event.delta_x, axis_event.delta_y)
      };

      call_event_handler!(
        event_loop_cell,
        event::Event::WindowEvent {
          window_id,
          event: event::WindowEvent::MouseWheel {
            device_id,
            delta,
            phase: event::TouchPhase::Moved,
            modifiers: current_modifiers(),
          },
        }
      );
    }

    // Emit pinch scale as Ctrl+MouseWheel, which WebView interprets as zoom.
    // pinch_scale: 1.0 = no change, >1.0 = zoom in, <1.0 = zoom out, 0.0 = no pinch.
    if axis_event.pinch_scale != 0.0 && axis_event.pinch_scale != 1.0 {
      let zoom_delta = if axis_event.pinch_scale > 1.0 {
        // Zooming in: positive delta
        1.0
      } else {
        // Zooming out: negative delta
        -1.0
      };

      call_event_handler!(
        event_loop_cell,
        event::Event::WindowEvent {
          window_id,
          event: event::WindowEvent::MouseWheel {
            device_id,
            delta: event::MouseScrollDelta::LineDelta(0.0, zoom_delta),
            phase: event::TouchPhase::Moved,
            // Synthesized zoom gesture: Ctrl on top of the tracked modifiers
            // (issue #109 — was hardcoded CONTROL, discarding real state).
            modifiers: current_modifiers() | ModifiersState::CONTROL,
          },
        }
      );
    }
  }

  /// Handle MainEvent::GainedFocus (a per-UIAbility-instance stage event,
  /// StageEventType::ACTIVE — `window_id` is the gaining instance, 0 =
  /// primary; Phase 4 design.md D5): mark focus, clear the tracked modifier
  /// set, and dispatch Focused(true) routed to that window. Extracted from
  /// the run_loop arm for direct testing (Review R21 / issue
  /// Eulogizethesun/tauri#137).
  fn handle_gained_focus(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: i64,
  ) {
    HAS_FOCUS.store(true, Ordering::Relaxed);
    // #173: mirror the per-window focused bit through WINDOW_MIRRORS (miss
    // tolerated -- the window may have dropped between event and write).
    super::window::set_mirror_focused(window_id, true);
    // #180(a) (design D7): sync the shared change-edge snapshot so the sub-window
    // drain path and this stage path dedup duplicate edges against the same
    // record (cross-path repeats are swallowed the same way).
    LAST_FOCUSED_WINDOW_ID.store(window_id, Ordering::Relaxed);
    // Modifier state is tracked from key events (issue #109); a window
    // that regained focus may have missed modifier releases while
    // unfocused — start clean rather than trust a stale set.
    // Review R21 (issue #137): the clear must be observable — when the
    // tracked set was non-empty, dispatch ModifiersChanged(empty)
    // BEFORE Focused(true) (same ordering as the Windows backend's
    // gain_active_focus: update_modifiers precedes Focused; and as
    // this file's key path: ModifiersChanged precedes the key event).
    // Without it, event listeners keep the stale set while the
    // deprecated `modifiers` field on mouse events reads empty.
    //
    // The LOST-focus edge deliberately does NOT clear/dispatch
    // (unlike Windows lose_active_focus / macOS windowDidResignKey):
    // the reset must be observable by the regaining window to restore
    // listener/tao consistency, and this regain-side notification
    // alone achieves that. Note (tao#26 review, second round): this is
    // NOT justified by "an unfocused window receives no input" —
    // mouse/axis pointer events are routed by SOURCE window with no
    // focus filter and read the process-global current_modifiers(),
    // so an unfocused window can still observe the stale set through
    // the deprecated `modifiers` field (e.g. hover-delivered
    // CursorMoved/scroll); only key events are focus-bound.
    //
    // Cross-window sticky-modifier limitation (tao#26 review follow-up):
    // KEYBOARD_MODIFIERS is process-global, so the set tracked under the
    // previously focused window stays live after it loses focus. When a
    // (possibly different) window regains focus, that stale set is what it
    // inherits — and if non-empty, this empty ModifiersChanged is the very
    // first event on the focus path, before Focused(true), whether or not
    // it ever held those modifiers. Residual exposure until that reset:
    // pointer events landing on any window — focused or not — between the
    // stale set's last update and the next GainedFocus report the stale
    // set in their deprecated `modifiers` field. ModifiersChanged
    // LISTENERS are unaffected (they only ever saw the events that built
    // the set); the next GainedFocus's reset is the single correction
    // point.
    let had_modifiers = {
      let mut prev = ModifiersState::empty();
      if let Ok(mut modifiers) = KEYBOARD_MODIFIERS.lock() {
        prev = *modifiers;
        *modifiers = ModifiersState::empty();
      }
      !prev.is_empty()
    };
    if had_modifiers {
      call_event_handler!(
        event_loop_cell,
        event::Event::WindowEvent {
          window_id: window::WindowId(WindowId(window_id)),
          event: event::WindowEvent::ModifiersChanged(ModifiersState::empty()),
        }
      );
    }
    call_event_handler!(
      event_loop_cell,
      event::Event::WindowEvent {
        window_id: window::WindowId(WindowId(window_id)),
        event: event::WindowEvent::Focused(true),
      }
    );
  }

  /// #180(a) (design D7): the single blur write face -- per-window mirror bit
  /// (#173 write face) + the `Focused(false)` event. Shared by the stage
  /// LostFocus arm (which additionally clears the app-level HAS_FOCUS bit)
  /// and the sub-window drain's paired-blur synthesis (which must NOT touch
  /// HAS_FOCUS -- the ability may still hold focus, and the pair is completed
  /// by the immediately following gain on the new window). Extracted so the
  /// two callers cannot drift.
  fn apply_blur(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: i64,
  ) {
    // #173: mirror the per-window focused bit (miss tolerated -- same
    // unregister-vs-backfill timing as the other mirror bits).
    super::window::set_mirror_focused(window_id, false);
    call_event_handler!(
      event_loop_cell,
      event::Event::WindowEvent {
        window_id: window::WindowId(WindowId(window_id)),
        event: event::WindowEvent::Focused(false),
      }
    );
  }

  /// #180(a) (design D7): consume one drained `notify_window_focus` edge
  /// (SubWindowPage windowEvent WINDOW_ACTIVE/WINDOW_INACTIVE) with change-edge
  /// dedup against [`LAST_FOCUSED_WINDOW_ID`] plus paired-blur synthesis:
  ///
  /// - gain edge (`focused=true`, last!=W): first pair-clear the previous
  ///   holder P=last (mirror bit + `Focused(false)`) -- ONLY when P still
  ///   reports `focused=true` (a holder already blurred by a real stage event
  ///   must not be double-blurred) -- then the regular gain path
  ///   ([`Self::handle_gained_focus`], which also syncs the snapshot).
  /// - loss edge (`focused=false`, last==W): the blur write face + snapshot
  ///   -> 0. HAS_FOCUS is deliberately NOT touched (see [`Self::apply_blur`]
  ///   note): only stage events maintain the app-level bit.
  /// - same-state edges (`focused=true` && last==W; `focused=false` &&
  ///   last!=W): swallowed -- zero WindowEvents, zero mirror writes.
  fn handle_pending_window_focus(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: i64,
    focused: bool,
  ) {
    let last = LAST_FOCUSED_WINDOW_ID.load(Ordering::Relaxed);
    if focused {
      if last == window_id {
        // Duplicate gain edge (e.g. ArkTS re-reporting WINDOW_ACTIVE for the
        // window it already recorded as focused) -- swallow.
        return;
      }
      // Pair semantics: the previous holder must not keep a stale focused bit
      // when ArkTS fast-switches and only reports the new ACTIVE (no
      // INACTIVE for P). Gate on the mirror bit so an already-blurred holder
      // is not double-blurred.
      if last != 0 && super::window::mirror_focused_bit(last) {
        Self::apply_blur(event_loop_cell, last);
      }
      Self::handle_gained_focus(event_loop_cell, window_id);
    } else {
      if last != window_id {
        // Stale/unknown blur edge: the recorded holder is someone else (or
        // nothing) -- swallow so a replayed INACTIVE cannot blur a window
        // that legitimately holds focus now.
        return;
      }
      Self::apply_blur(event_loop_cell, window_id);
      LAST_FOCUSED_WINDOW_ID.store(0, Ordering::Relaxed);
    }
  }

  pub fn run<F>(self, event_handler: F) -> ()
  where
    F: 'static + FnMut(event::Event<T>, &event_loop::EventLoopWindowTarget<T>, &mut ControlFlow),
  {
    // Leak the EventLoop on purpose: OHOS is callback-driven — `run` returns
    // immediately after registering the handler (the main thread must be
    // handed back to ArkTS), but the stored handler stays reachable from the
    // ArkTS lifecycle callbacks for the entire lifetime of the ability, i.e.
    // long after the caller's stack frame is gone. This is a singleton
    // semantic: exactly one EventLoop per process, never dropped (tao issue
    // #84, item 4).
    let event_looper = Box::leak(Box::new(self));
    event_looper.run_return(event_handler);
  }

  /// Registers `event_handle` and returns immediately — OHOS is callback
  /// driven, so unlike other platforms this neither pumps events nor returns
  /// on `ControlFlow::Exit` (which instead terminates the ability; see the
  /// exit check at the end of the `run_loop` dispatch below). The shared
  /// `EventLoopExtRunReturn` trait is cfg'd out on OHOS, so this is only
  /// reached through `run` — hence the `'static` bound.
  pub fn run_return<F>(&mut self, mut event_handle: F) -> i32
  where
    F: 'static + FnMut(event::Event<T>, &event_loop::EventLoopWindowTarget<T>, &mut ControlFlow),
  {
    let mut control_flow = ControlFlow::default();
    let target = self.window_target.clone();

    // `event_handle` is `'static` (the only caller is `run`, and OHOS is
    // excluded from the shared `EventLoopExtRunReturn` trait), so the boxed
    // closure satisfies the `'static` handler slot directly. The transmute
    // this used to rely on was unsound: that trait permits non-`'static`
    // closures, whose captures could be freed while the permanently stored
    // handler is still reachable (tao issue #84, item 1).
    let handle: Box<dyn FnMut(event::Event<T>)> = Box::new(move |e| {
      event_handle(e, &*target, &mut control_flow);
      // We need to dispatch it after every event callbacks.
      event_handle(event::Event::MainEventsCleared, &*target, &mut control_flow);
      // Propagate a handler-set `ControlFlow::Exit` to the platform
      // exit check at the end of each MainEvent dispatch. Without this
      // the check is dead code: nothing else ever writes `exit`, so the
      // terminate branch below was unreachable (tao issue #84).
      if control_flow == ControlFlow::Exit {
        target.p.exit.set(true);
      }
    });
    self.event_loop.replace(Some(handle));

    // Snapshot the shared cells as `'static` clones so the dispatch closure passed
    // to `run_loop` (which requires `F: FnMut(MainEvent) + 'static`) captures no
    // borrows of `self`.
    let event_loop_cell = self.event_loop.clone();
    let user_events_rx = self.user_events_receiver.clone();
    let window_target = self.window_target.clone();
    let app = self.openharmony_app.clone();

    app.clone().run_loop(move |event| {
      // ── OHOS window lifecycle backfill (moved from tauri-runtime-wry) ──
      // ArkTS calls notifyWindowClose() synchronously (pushes the OHOS window
      // id to the ability queue) then destroyWindow() asynchronously; draining
      // at the start of every MainEvent dispatch reads the stored Rust state
      // before the async destruction completes. Closes are synthesized as
      // standard `CloseRequested` events routed by the real OHOS window id —
      // downstream runtimes see a normal tao close. Status changes are applied
      // straight to the registered window mirrors.
      for ohos_win_id in openharmony_ability::drain_pending_window_closes() {
        call_event_handler!(
          event_loop_cell,
          event::Event::WindowEvent {
            window_id: window::WindowId(WindowId(ohos_win_id as i64)),
            event: event::WindowEvent::CloseRequested,
          }
        );
      }
      // Design D2 confirmation channel (a): destroy requests issued through
      // `Window::close` that resolved Ok (including the idempotent
      // already-gone ack) come back here as synthesized Destroyed events, so
      // the embedding runtime's confirmation executor also runs for paths
      // where the notifyWindowClose loopback above cannot fire (window
      // already destroyed -> no fresh aboutToDisappear). Idempotent
      // downstream: a loopback CloseRequested for the same window is absorbed
      // by the executor's remove-or-early-exit.
      for confirmed_id in drain_destroy_confirmations() {
        call_event_handler!(
          event_loop_cell,
          event::Event::WindowEvent {
            window_id: window::WindowId(WindowId(confirmed_id)),
            event: event::WindowEvent::Destroyed,
          }
        );
      }
      // Issue Eulogizethesun/tauri#222: the deferred setColorMode completion
      // (openharmony-ability COLOR_MODE_COMPLETION slot) is consumed here --
      // the notify legs wake the loop, and an inline poll at the set_theme
      // call site would starve (that call runs ON the ArkTS main thread; see
      // apply_color_mode_completion for the full thread-model rationale).
      // A reported failure rolls APP_THEME_OVERRIDE back to FOLLOW so
      // theme() stops reporting an intent the system never applied.
      apply_color_mode_completion(openharmony_ability::take_color_mode_completion());
      for (ohos_win_id, status) in openharmony_ability::drain_pending_window_status() {
        let ohos_win_id = ohos_win_id as i64;
        let mirror = {
          let mut mirrors = WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned");
          match mirrors.get(&ohos_win_id) {
            Some(weak) => match weak.upgrade() {
              Some(mirror) => Some(mirror),
              // Dead entry: the window was dropped without running Drop
              // (e.g. leaked) — lazily remove it so the map does not grow.
              None => {
                mirrors.remove(&ohos_win_id);
                None
              }
            },
            None => None,
          }
        };
        match mirror {
          Some(mirror) => mirror.apply_window_status(status),
          None => {
            // G6 / cross-cutting: a failed sub-window has window_id=None
            // (never registered, produces no status events), so an unmatched
            // drained status means a real window was destroyed between
            // enqueue and drain (stale id). Non-zero ids are diagnosable
            // stale ids → warn; id 0 (main window before registration) stays
            // at debug to avoid noise.
            if ohos_win_id != 0 {
              log::warn!(
                "[tao-ohos] pending window status for id {} (status={}) matched no live window \
                 (stale id: window destroyed between enqueue and drain)",
                ohos_win_id,
                status
              );
            } else {
              log::debug!(
                "[tao-ohos] pending window status for id 0 (status={}) matched no live window",
                status
              );
            }
          }
        }
      }

      // #180(a) (design D7): sub-window focus edges (SubWindowPage windowEvent
      // WINDOW_ACTIVE/WINDOW_INACTIVE -> notifyWindowFocus) drain next to the
      // status backfill and run through the same change-edge dedup +
      // paired-blur synthesis the stage arms share.
      for (focus_window_id, focused) in openharmony_ability::drain_pending_window_focus() {
        Self::handle_pending_window_focus(&event_loop_cell, focus_window_id, focused);
      }

      // Issue 7: replay window ops queued against spawned UIAbility windows
      // whose stage handshake completed since the last pass (the D7
      // registration waker wakes the loop), and against sub-windows whose
      // ArkTS creation chain settled (the sub-window settle waker).
      drain_ready_pending_window_ops();

      match event {
        MainEvent::SurfaceCreate { .. } => {
          call_event_handler!(event_loop_cell, event::Event::NewEvents(StartCause::Init));
          call_event_handler!(event_loop_cell, event::Event::Resumed);
        }
        MainEvent::SurfaceDestroy { .. } => {
          call_event_handler!(event_loop_cell, event::Event::Suspended);
        }
        MainEvent::WindowResize { .. } => {
          // Consumed without dispatch (review R19 follow-up, issue
          // Eulogizethesun/tauri#135). This event has two sources with
          // different metrics, and neither is a trustworthy Resized source:
          //
          // - windowSizeChange measures the OUTER rect on decorated
          //   windows — the wrong metric for tao's client-area Resized
          //   contract — and every physical resize pairs it with a
          //   windowRectChange, so the inner-sized ContentRectChange
          //   Resized (review R11) already covers it.
          // - XComponent on_surface_changed measures the component's OWN
          //   surface, which equals the window's drawable area only when
          //   the component fills the page. Its window_id is plumbed from
          //   the ArkTS component's windowId prop (DefaultXComponent
          //   defaults it to 0), so the ability demo's SubWindowPage (a
          //   non-fullscreen second XComponent in its own window, mounted
          //   with the default id) had its surface changes dispatched as
          //   main-window Resized — the misroute this drop eliminates. On
          //   resizes it is redundant
          //   with the paired ContentRectChange Resized (same drawable
          //   metric) — dispatching it adds at best a duplicate.
          //
          // The main window's initial rect arrives via the ArkTS post-load
          // seed (NativeAbility / BridgeHost both seed windowId 0), Float
          // sub-windows' via their own post-load seed (they register no
          // windowSizeChange at all).
          //
          // RESIDUAL GAP (tao#27 review; device-probed 2026-10-12 on the
          // pre-#27 installed build — its event chain is byte-identical
          // to this arm's parent and still dispatched Resized here
          // unconditionally): a drawable-only change such as
          // setDecorations(false) on the main window
          // (setWindowDecorVisible extends the page into the title-bar
          // area; the window rect itself is unchanged) delivers NOTHING
          // on the webview stack — zero Resized, and the inner-rect
          // cache (inner_rect_for; production writers are the
          // windowRectChange closure plus the one-time post-load seed)
          // stays stale until the next rect-changing event. Probe (2in1
          // HAD-W32, CDP + hilog): the toggle executed (WMSDecor) and
          // the webview area grew by exactly the decor-chrome height
          // within ~20ms (WebviewSurface onAreaChange), while
          // innerSize() kept the stale value and no Resized fired; the
          // in-run setSize control fired exactly one inner-metric
          // Resized per round, so the pipeline was live, and a post-
          // toggle setSize healed the cache — the source of the missing
          // signal is that the XComponent surface source produced no
          // Event::WindowResize there (architecturally dormant in
          // webview apps: the Web overlays the render node,
          // MainPage.ets) and windowRectChange never fired (outer rect
          // unchanged). So for tauri/webview apps this drop removed no
          // decor-toggle signal; XComponent-rendering consumers (where
          // the surface source is live — the misroute above) may lose a
          // decor-toggle Resized there — unprobed. The stale-cache
          // defect predates and is independent of this drop: restoring
          // the surface-source dispatch would not heal it
          // (tauri-runtime-wry discards the Resized payload and re-reads
          // inner_size) and would reintroduce the misroute. Correct fix
          // (follow-up, not blocking): push a synthetic rect change from
          // the ArkTS side when the drawable actually changes — e.g. the
          // setDecorations path or the WebviewSurface onAreaChange hook
          // — carrying a fresh getWindowProperties().drawableRect
          // through the existing windowRectChange channel, which feeds
          // both the cache and this loop's ContentRectChange dedup (the
          // resize-inner bridge already reads live decor state: its
          // chrome arithmetic was 0 while decor was hidden, probe
          // round 4).
        }
        MainEvent::WindowRedraw { window_id, .. } => {
          // RedrawRequested is driven by the XComponent frame callback, keyed
          // by the window that called render() (Phase 4 design.md D5). In
          // practice that is the primary only — spawned UIAbility windows
          // and sub-windows own no XComponent.
          call_event_handler!(
            event_loop_cell,
            event::Event::RedrawRequested(window::WindowId(WindowId(window_id)))
          );
        }
        MainEvent::ContentRectChange(content_rect) => {
          // Propagate as Resized so tauri's resize handler fires and calls
          // webview.set_bounds() with the new window dimensions.
          // Phase 3 (design.md D6): route by content_rect.window_id (populated by the
          // window_rect_change lifecycle closure from the ArkTS-wrapped windowId).
          //
          // Issue Eulogizethesun/tauri#107: windowRectChange covers position AND
          // size, but this arm only ever dispatched Resized, so WindowEvent::Moved
          // never fired. Diff against the last dispatched rect per window:
          // - position change → Moved, size change → Resized (both when both);
          // - first rect for a window seeds the cache and still dispatches the
          //   legacy Resized (sub-windows have no XComponent surface, so
          //   this event is their only initial-size signal);
          // - degenerate rects (minimize/hide collapse to 0×0) only update the
          //   cache — dispatching Resized(0,0) would corrupt downstream bounds.
          //
          // Review R11 (issue #97 follow-up): Resized now carries the INNER
          // (drawable) size, matching tao's cross-platform contract ("the
          // client area's new dimensions", event.rs — Windows reads the WM_SIZE
          // client rect, macOS the NSView frame) while Moved keeps the OUTER
          // window position (macOS likewise uses the NSWindow frame for Moved).
          // The size comes from the app's inner-rect cache: the
          // window_rect_change closure stores this same event's drawableRect
          // BEFORE dispatching ContentRectChange (same closure, same thread),
          // so this always reads the current event's inner size. It falls back
          // to the outer rect only while no readable drawable has ever
          // arrived; the pair-in/pair-out cache contract plus the ArkTS
          // post-load seed keep that fallback consistent.
          let window_id = content_rect.window_id;
          let (left, top) = (content_rect.rect.left, content_rect.rect.top);
          // Degeneracy (minimize/hide) is checked on the OUTER rect — the
          // inner cache may still hold the last non-degenerate drawable.
          let outer_degenerate = content_rect.rect.width <= 0 || content_rect.rect.height <= 0;
          let inner = app.inner_rect_for(window_id);
          let (width, height) = (inner.width, inner.height);
          let degenerate = outer_degenerate || width <= 0 || height <= 0;
          // Review R19 follow-up (issue Eulogizethesun/tauri#135): recover
          // from a poisoned lock instead of skipping the insert — skipping
          // made prev=None read as "first event" and re-dispatched a
          // duplicate Resized, the exact bug this dedup cache exists to
          // prevent. into_inner is sound: the map itself is still valid
          // memory; the poisoning panic (if any) happened elsewhere.
          let prev = LAST_DISPATCHED_RECTS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(window_id, (left, top, width, height));
          match prev {
            None => {
              if !degenerate {
                let size = PhysicalSize::new(width as _, height as _);
                call_event_handler!(
                  event_loop_cell,
                  event::Event::WindowEvent {
                    window_id: window::WindowId(WindowId(window_id)),
                    event: event::WindowEvent::Resized(size),
                  }
                );
              }
            }
            Some((prev_left, prev_top, prev_width, prev_height)) if !degenerate => {
              if (prev_left, prev_top) != (left, top) {
                call_event_handler!(
                  event_loop_cell,
                  event::Event::WindowEvent {
                    window_id: window::WindowId(WindowId(window_id)),
                    event: event::WindowEvent::Moved(PhysicalPosition::new(left, top)),
                  }
                );
              }
              if (prev_width, prev_height) != (width, height) {
                let size = PhysicalSize::new(width as _, height as _);
                call_event_handler!(
                  event_loop_cell,
                  event::Event::WindowEvent {
                    window_id: window::WindowId(WindowId(window_id)),
                    event: event::WindowEvent::Resized(size),
                  }
                );
              }
            }
            _ => {}
          }
        }
        // Phase 4 (design.md D5): focus is a per-UIAbility-instance stage
        // event — route by the gaining instance's window id so a spawned
        // window's tauri Focused listeners fire. HAS_FOCUS stays a single
        // app-level bit: when a spawned instance gains focus the primary
        // receives the matching LostFocus half of the pair, keeping the
        // bit in step; read sites are unchanged. The helper also performs
        // the R21 stale-modifier reset (issue #137) before Focused(true).
        MainEvent::GainedFocus { window_id } => {
          Self::handle_gained_focus(&event_loop_cell, window_id)
        }
        MainEvent::LostFocus { window_id } => {
          // Phase 4 (design.md D5): mirror of GainedFocus — route by the
          // losing instance's window id; HAS_FOCUS stays app-level.
          //
          // The modifier set is deliberately NOT cleared or dispatched here
          // (rationale at handle_gained_focus): the process-global
          // KEYBOARD_MODIFIERS keeps the losing window's set until the next
          // GainedFocus dispatches the empty ModifiersChanged reset.
          // Cross-window sticky-modifier limitation (tao#26 review
          // follow-up, second round): until that reset, pointer events
          // routed by source window (hover/move/scroll, no focus filter)
          // still report the stale set via the deprecated `modifiers`
          // field — including on not-yet-refocused windows; the next
          // focus gainer is the correction point.
          HAS_FOCUS.store(false, Ordering::Relaxed);
          // #180(a) (design D7): the blur write face is shared with the sub-window
          // drain's paired-blur synthesis (apply_blur), and the shared
          // change-edge snapshot records the holder clearing so cross-path
          // duplicate edges dedup the same way.
          Self::apply_blur(&event_loop_cell, window_id);
          LAST_FOCUSED_WINDOW_ID.store(0, Ordering::Relaxed);
        }
        MainEvent::ConfigChanged(config) => {
          // Configuration changes are app-level (EnvironmentCallback), not tied to a
          // specific window. Broadcast to every live window like ThemeChanged
          // below (scale is app-global on OHOS; WINDOW_MIRRORS covers main +
          // spawned UIAbility + sub-windows, with a window-0 fallback
          // before any window exists). Each window gets its own
          // `new_inner_size` slot so one handler's adjustment cannot leak
          // into the next window's event. The size is the window's OWN inner
          // rect (`inner_rect_for`, the same source as the Resized dispatch
          // in the ContentRectChange arm above — issue Eulogizethesun/tauri#97),
          // not the primary's geometry broadcast to everyone: a spawned window or a
          // sub-window must be offered its own drawable size to resize to.
          // Degradation: `inner_rect_for` returns (0,0,0,0) before a window's
          // first readable windowRectChange, so fall back to the app-level
          // content rect then (the same pre-geometry degradation the Resized
          // path tolerates).
          // #180 (design D8): the broadcast is edge-guarded -- a bitwise-equal
          // repeat (ConfigChanged storms that leave the effective scale
          // untouched) forwards nothing, first event forwards (0-bits reads
          // as unseeded). Mirrors the ThemeChanged guard below; the two edge
          // detectors are independent.
          //
          // S1-061 fix wave (config-driven edge + composite payload): the
          // pre-fix guard keyed on `app.scale()` alone -- the live NDK
          // `GetDefaultDisplayScaledDensity` display value -- which no
          // user-reachable setting moves (round-4..7 device evidence: the
          // `显示大小` setting changes VPR only; the `字体大小` setting
          // changes this event's Configuration only), so the guard swallowed
          // every real change and `onScaleChanged` never fired. The edge key
          // and the payload are now the composite
          // `app.scale() * config.font_size_scale`: the density leg keeps
          // the original payload source, the font-scale leg reads the
          // ARRIVED Configuration (not re-queried), so font-size changes
          // forward. Divergence disclosed: `Window::scale_factor()` keeps
          // returning the bare display density (upstream tao contract keeps
          // font size out of scale_factor, and flipping the getter would
          // couple every LogicalSize conversion to the font setting), so
          // the event reports the composite while later reads report the
          // density. Still not covered (remains disclosed): `显示大小` /
          // VPR changes forward nothing -- neither leg moves. The debug
          // line below is the fix-wave probe: one line per dispatched
          // ConfigChanged carrying both legs plus the decision,
          // separating the "NDK source value never moves" / "NAPI
          // callback never arrived" hypotheses the round-7 zero-probe
          // could not.
          let font_scale = config.font_size_scale;
          let density = app.scale() as f64;
          let composite = density * font_scale;
          let forward = scale_should_forward(composite);
          debug!(
            "ConfigChanged: font_size_scale={}, app.scale()={}, composite={}, forward={}",
            font_scale, density, composite, forward
          );
          if forward {
            let app_rect = app.content_rect();
            for wid in live_window_ids() {
              let inner = app.inner_rect_for(wid);
              let (width, height) = if inner.width > 0 && inner.height > 0 {
                (inner.width, inner.height)
              } else {
                (app_rect.width, app_rect.height)
              };
              let mut size = PhysicalSize::new(width as _, height as _);
              call_event_handler!(
                event_loop_cell,
                event::Event::WindowEvent {
                  window_id: window::WindowId(WindowId(wid)),
                  event: event::WindowEvent::ScaleFactorChanged {
                    new_inner_size: &mut size,
                    scale_factor: composite,
                  },
                }
              );
            }
          }
          // Issue Eulogizethesun/tauri#108: onConfigurationUpdate → ConfigChanged
          // carries the new colorMode (app.config() is already updated by the
          // lifecycle closure before dispatch), but ThemeChanged was never
          // dispatched — apps had to poll theme(). Emit ThemeChanged when the
          // EFFECTIVE theme (override-aware, same source as `Window::theme`)
          // changed. First event seeds the baseline (theme is unknowable before
          // any ConfigChanged — initial color_mode is NoSet), so it also emits;
          // apps re-applying the current theme on that first event is harmless.
          // Dispatched to every live window (theme is app-global on OHOS; the
          // registry covers main + sub-windows).
          let theme = effective_theme(&app);
          let theme_bits = match theme {
            Theme::Dark => EFFECTIVE_THEME_DARK,
            Theme::Light => EFFECTIVE_THEME_LIGHT,
          };
          let prev_bits = LAST_EFFECTIVE_THEME.swap(theme_bits, Ordering::Relaxed);
          if prev_bits != theme_bits {
            // Live mirrors only — same filter as the ScaleFactorChanged
            // broadcast above (`live_window_ids`).
            for wid in live_window_ids() {
              call_event_handler!(
                event_loop_cell,
                event::Event::WindowEvent {
                  window_id: window::WindowId(WindowId(wid)),
                  event: event::WindowEvent::ThemeChanged(theme),
                }
              );
            }
          }
        }
        MainEvent::Start => {
          // WindowStageEventType::SHOWN (window visible to user). Forwarded as
          // Event::Resumed — tao's closest lifecycle signal to OHOS "window-shown".
          // Double Resumed (alongside SurfaceCreate/Resume) is acceptable; downstream
          // tauri RunEvent::Resumed handlers must be idempotent.
          // See openspec ohos-event-lifecycle-forward.
          call_event_handler!(event_loop_cell, event::Event::Resumed);
        }
        MainEvent::Resume { .. } => {
          call_event_handler!(event_loop_cell, event::Event::Resumed);
        }
        MainEvent::SaveState { .. } => {
          // onAbilitySaveState has no tao Event/StartCause equivalent (no Autosave
          // variant). Degraded: dropped with debug log. Apps must persist state via
          // tauri RunEvent::Exit/ExitRequested or custom logic.
          // See openspec ohos-event-lifecycle-forward.
          debug!(
            "SaveState has no tao Event equivalent; dropped (see ohos-event-lifecycle-forward)"
          );
        }
        MainEvent::Pause => {
          debug!("App Paused - stopped running");
          // TODO: This is incorrect - will be solved in https://github.com/rust-windowing/winit/pull/3897
          // self.running = false;
        }
        MainEvent::WindowDestroy { window_id } => {
          // This fires from the UIAbility `onWindowStageDestroy` lifecycle
          // callback, which corresponds to a UIAbility window stage being torn
          // down. Under multi-UIAbility (openspec multi-uiability-windows)
          // every UIAbility instance's destroy routes here with its own window
          // id (Phase 4 design.md D5): the primary (0) keeps the historical
          // close+destroy pair, a spawned instance's teardown reaches its own
          // window so tauri-runtime-wry removes the store entry (D13 cleanup;
          // its close path is idempotent against the already-terminating
          // ability). Sub-windows are destroyed via the separate ArkTS
          // destroyWindow() path drained by tauri-runtime-wry's
          // drain_pending_window_closes().
          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id: window::WindowId(WindowId(window_id)),
              event: event::WindowEvent::CloseRequested,
            }
          );
          // Also dispatch Destroyed so tauri-runtime-wry can clean up the window.
          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id: window::WindowId(WindowId(window_id)),
              event: event::WindowEvent::Destroyed,
            }
          );
        }
        MainEvent::Destroy => {
          call_event_handler!(event_loop_cell, event::Event::LoopDestroyed);
        }
        MainEvent::PrepareToTerminate { answer } => {
          // PC/2in1 pre-close probe (UIAbility.onPrepareToTerminateAsync, issue
          // Eulogizethesun/tauri#103): the cancellable counterpart of the
          // MainEvent::Destroy → LoopDestroyed path above. Forwarded as a
          // dedicated Event so tauri-runtime-wry runs its ExitRequested
          // dispatch BEFORE any teardown and records the decision on `answer`
          // — prevent_exit() here genuinely keeps the app alive (ArkTS cancels
          // the termination) instead of firing into an unstoppable destroy.
          // The reference lives on the ability-side NAPI closure's stack for
          // the duration of this synchronous dispatch.
          call_event_handler!(event_loop_cell, event::Event::PrepareToTerminate { answer });
        }
        MainEvent::Input { window_id, input } => {
          Self::handle_input_event(&event_loop_cell, window_id, &input);
        }
        // OHOS: intentionally diverges from Android/iOS — always emit Event::Opened
        // even when urls is empty.
        //
        // On Android/iOS, Event::Opened is a pure "open URL" signal and is skipped
        // when urls is empty. On OHOS, `onNewWant` serves as the "re-launch" signal
        // (the OS prevents creating a second instance), so we emit Event::Opened on
        // every re-launch to allow the single-instance plugin to trigger its callback.
        // The want.parameters from the global Mutex carries system-injected fields
        // even when no URI is provided.
        //
        // Impact on other consumers:
        // - deep-link plugin: has an OHOS arm — its on_event handler writes the
        //   app-level current URL here (empty urls are ignored on OHOS), which
        //   is the warm re-entry refresh path for get_current()
        // - other consumers: typically just log the urls, no functional side effects
        MainEvent::NewWant { uri } => {
          let urls = if uri.is_empty() {
            vec![]
          } else {
            match url::Url::parse(&uri) {
              Ok(url) => vec![url],
              Err(e) => {
                log::error!("failed to parse NewWant URI '{uri}': {e}");
                vec![]
              }
            }
          };
          call_event_handler!(event_loop_cell, event::Event::Opened { urls });
        }
        MainEvent::UserEvent { .. } => {
          // Drain ALL pending user events on each wake, not just one.
          //
          // Async plugin commands (window/webview/event — all `async fn`)
          // resolve on tokio worker threads and send their response
          // `EvaluateScript` ("runCallback(...)") via `proxy.send_event` →
          // waker TSFN. The TSFN NonBlocking wake can be coalesced: N queued
          // events may produce only ONE `MainEvent::UserEvent`. A single
          // `try_recv` would fetch just one and leave the rest stranded until
          // the next wake (which may never come promptly), so `runCallback`
          // never runs → the JS Promise never settles → 5000ms test timeout.
          // Custom (sync) commands don't hit this: they resolve on the main
          // thread and go through `send_user_message`'s synchronous
          // main-thread branch (direct `handle_user_message`), bypassing the
          // waker/drain path entirely.
          while let Ok(event) = user_events_rx.borrow_mut().try_recv() {
            call_event_handler!(event_loop_cell, event::Event::UserEvent(event));
          }
        }
        unknown => {
          trace!("Unknown MainEvent {unknown:?} (ignored)");
        }
      };

      if window_target.p.exit.get() {
        call_event_handler!(event_loop_cell, event::Event::LoopDestroyed);
        // Migrate from OpenHarmonyApp::exit(0) (removed) to
        // AppControlExt::terminate(env, 0) (MainThreadSync bridge call).
        // run_loop callbacks execute on the N-API main thread, so
        // get_main_thread_env() returns Some(env).
        // The 0 is hardcoded: the OHOS exit path collapses
        // `ControlFlow::Exit` (= `ExitWithCode(0)`) into a boolean
        // pending-exit flag, so any exit code requested by the app is
        // dropped here and the process always exits with 0.
        let env_cell = openharmony_ability::get_main_thread_env();
        let env_ref = env_cell.borrow();
        if let Some(env) = env_ref.as_ref() {
          if let Err(e) = app.terminate(env, 0) {
            log::warn!("[tao-ohos] terminate failed: {:?}", e);
          }
        } else {
          log::warn!("[tao-ohos] terminate failed: main thread Env not available");
        }
      }
    });
    0
  }

  pub fn create_proxy(&self) -> EventLoopProxy<T> {
    EventLoopProxy {
      user_events_sender: self.user_events_sender.clone(),
      waker: self.openharmony_app.create_waker(),
    }
  }
}

pub struct EventLoopProxy<T: 'static> {
  user_events_sender: mpsc::Sender<T>,
  waker: OpenHarmonyWaker,
}

impl<T: 'static> EventLoopProxy<T> {
  pub fn send_event(&self, event: T) -> Result<(), event_loop::EventLoopClosed<T>> {
    self
      .user_events_sender
      .send(event)
      .map_err(|err| event_loop::EventLoopClosed(err.0))?;
    self.waker.wake();
    Ok(())
  }
}

impl<T: 'static> Clone for EventLoopProxy<T> {
  fn clone(&self) -> Self {
    EventLoopProxy {
      user_events_sender: self.user_events_sender.clone(),
      waker: self.waker.clone(),
    }
  }
}

#[derive(Clone)]
pub struct EventLoopWindowTarget<T: 'static> {
  pub(crate) app: OpenHarmonyApp,
  pub(crate) bridge_executor: BridgeExecutor,
  exit: Cell<bool>,
  _marker: std::marker::PhantomData<T>,
}

impl<T: 'static> EventLoopWindowTarget<T> {
  pub fn available_monitors(&self) -> VecDeque<MonitorHandle> {
    // One handle per connected display (issue Eulogizethesun/tauri#106).
    MonitorHandle::all_for_app(&self.app).into_iter().collect()
  }

  pub fn primary_monitor(&self) -> Option<monitor::MonitorHandle> {
    Some(monitor::MonitorHandle {
      inner: MonitorHandle::primary_for_app(&self.app),
    })
  }

  #[inline]
  pub fn monitor_from_point(&self, x: f64, y: f64) -> Option<MonitorHandle> {
    monitor_from_point_for_app(&self.app, x, y)
  }

  #[cfg(feature = "rwh_05")]
  #[inline]
  pub fn raw_display_handle_rwh_05(&self) -> rwh_05::RawDisplayHandle {
    unreachable!("rwh_05 is not supported on OpenHarmony");
  }

  #[cfg(feature = "rwh_06")]
  #[inline]
  pub fn raw_display_handle_rwh_06(&self) -> Result<rwh_06::RawDisplayHandle, rwh_06::HandleError> {
    Ok(rwh_06::RawDisplayHandle::Ohos(
      rwh_06::OhosDisplayHandle::new(),
    ))
  }

  pub fn cursor_position(&self) -> Result<PhysicalPosition<f64>, error::ExternalError> {
    cursor_position_from_app(&self.app)
  }

  pub fn set_theme(&self, theme: Option<Theme>) {
    set_app_theme(&self.app, theme);
  }

  /// Issue Eulogizethesun/tauri#264 (design D2): [`Self::set_theme`] plus a
  /// receipt the runtime handed over for the deferred completion. Register
  /// FIRST, then run the full existing `set_theme` pipeline — same thread,
  /// so the receipt is queued before any completion can possibly be drained
  /// (the drain runs at the `MainEvent` dispatch head,
  /// [`apply_color_mode_completion`], which this synchronous call cannot
  /// re-enter). The pre-existing [`Self::set_theme`] keeps its exact
  /// no-receipt behavior (bare completions wake nobody).
  pub fn set_theme_with_receipt(&self, theme: Option<Theme>, receipt: ThemeReceipt) {
    register_theme_receipt(receipt);
    set_app_theme(&self.app, theme);
  }
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DeviceId(i32);

impl DeviceId {
  pub const fn dummy() -> Self {
    DeviceId(0)
  }
}

/// Direct unit tests for the input-event handlers. These handlers are pure
/// transforms (OHOS input data -> tao events via an injected callback cell);
/// the app autotest never triggers them because no user input occurs, so we
/// exercise them directly with synthetic events.
#[cfg(test)]
mod input_tests {
  use super::*;
  use openharmony_ability::xcomponent::{
    EventSource, KeyCode as OhosKeyCode, KeyEventData, TouchEventData, TouchPointData,
  };
  use openharmony_ability::TextInputEventData;
  use std::sync::Mutex;

  type LoopCell<T> = Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>;

  /// Runs `invoke` with a collector installed in a fresh event-loop cell and
  /// returns compact descriptors of every event the handler emitted.
  fn run_collected<T: 'static>(invoke: impl FnOnce(&LoopCell<T>)) -> Vec<String> {
    let out: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = out.clone();
    let cell: LoopCell<T> = Arc::new(RefCell::new(Some(Box::new(move |e: event::Event<T>| {
      let desc = match e {
        event::Event::WindowEvent { event: we, .. } => match we {
          event::WindowEvent::CursorMoved { position, .. } => {
            format!("CursorMoved({},{})", position.x, position.y)
          }
          event::WindowEvent::MouseInput { state, button, .. } => {
            format!("MouseInput({:?},{:?})", state, button)
          }
          event::WindowEvent::CursorEntered { .. } => "CursorEntered".to_string(),
          event::WindowEvent::CursorLeft { .. } => "CursorLeft".to_string(),
          // `modifiers` is deprecated in favor of ModifiersChanged, but the
          // synthetic wheel events still carry it — fine for a test matcher.
          #[allow(deprecated)]
          event::WindowEvent::MouseWheel {
            delta, modifiers, ..
          } => format!(
            "MouseWheel({:?},ctrl={})",
            delta,
            modifiers.contains(ModifiersState::CONTROL)
          ),
          event::WindowEvent::Touch(t) => format!(
            "Touch({:?},{},{},id={})",
            t.phase, t.location.x, t.location.y, t.id
          ),
          event::WindowEvent::KeyboardInput { event: ke, .. } => format!(
            "Key({:?},{:?},loc={:?},repeat={})",
            ke.state, ke.logical_key, ke.location, ke.repeat
          ),
          event::WindowEvent::ModifiersChanged(m) => format!(
            "ModifiersChanged(ctrl={},shift={},alt={},logo={})",
            m.contains(ModifiersState::CONTROL),
            m.contains(ModifiersState::SHIFT),
            m.contains(ModifiersState::ALT),
            m.contains(ModifiersState::SUPER)
          ),
          event::WindowEvent::ReceivedImeText(s) => format!("ImeText({s})"),
          event::WindowEvent::Focused(f) => format!("Focused({f})"),
          _ => "Other".to_string(),
        },
        _ => "NonWindow".to_string(),
      };
      sink.lock().unwrap().push(desc);
    }))));
    invoke(&cell);
    let x = out.lock().unwrap().clone();
    x
  }

  fn mouse(action: MouseAction, button: OhosMouseButton) -> MouseEventData {
    MouseEventData {
      x: 10.5,
      y: 20.25,
      action,
      button,
      ..Default::default()
    }
  }

  // ─── handle_mouse_event ──────────────────────────────────────────────

  #[test]
  fn mouse_move_emits_cursor_moved() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::Move, OhosMouseButton::NoneButton),
      );
    });
    assert_eq!(evs, vec!["CursorMoved(10.5,20.25)".to_string()]);
  }

  #[test]
  fn mouse_press_release_left() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::Press, OhosMouseButton::LeftButton),
      );
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::Release, OhosMouseButton::LeftButton),
      );
    });
    assert_eq!(
      evs,
      vec![
        "MouseInput(Pressed,Left)".to_string(),
        "MouseInput(Released,Left)".to_string(),
      ]
    );
  }

  #[test]
  fn mouse_press_back_button_maps_to_other4() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::Press, OhosMouseButton::BackButton),
      );
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::Release, OhosMouseButton::ForwardButton),
      );
    });
    assert_eq!(
      evs,
      vec![
        "MouseInput(Pressed,Other(4))".to_string(),
        "MouseInput(Released,Other(5))".to_string(),
      ]
    );
  }

  #[test]
  fn mouse_press_none_button_emits_nothing() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::Press, OhosMouseButton::NoneButton),
      );
    });
    assert!(evs.is_empty());
  }

  #[test]
  fn mouse_hover_enter_leave() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::HoverEnter, OhosMouseButton::NoneButton),
      );
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::HoverLeave, OhosMouseButton::NoneButton),
      );
    });
    assert_eq!(
      evs,
      vec!["CursorEntered".to_string(), "CursorLeft".to_string()]
    );
  }

  #[test]
  fn mouse_none_action_emits_nothing() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_mouse_event(
        cell,
        window::WindowId(WindowId(0)),
        &mouse(MouseAction::None, OhosMouseButton::NoneButton),
      );
    });
    assert!(evs.is_empty());
  }

  // ─── handle_axis_event ───────────────────────────────────────────────

  #[test]
  fn axis_mouse_wheel_uses_line_delta() {
    // The ctrl= assertion reads the process-level KEYBOARD_MODIFIERS static;
    // hold the shared input/focus test lock so a parallel test's CONTROL
    // write (gained_focus_clears_stale_modifiers_before_focus) cannot land
    // between the read and the assert (PR review fix: device-UT flake).
    let _guard = super::super::window::WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_modifiers(ModifiersState::empty());
    let evs = run_collected(|cell| {
      let d = AxisEventData {
        delta_x: 0.0,
        delta_y: 3.0,
        pinch_scale: 0.0,
        source_type: InputSourceType::Mouse,
        ..Default::default()
      };
      EventLoop::<()>::handle_axis_event(cell, window::WindowId(WindowId(0)), &d);
    });
    assert_eq!(
      evs,
      vec!["MouseWheel(LineDelta(0.0, 3.0),ctrl=false)".to_string()]
    );
    set_modifiers(ModifiersState::empty());
  }

  #[test]
  fn axis_touchpad_uses_pixel_delta() {
    // Same process-global discipline as axis_mouse_wheel_uses_line_delta.
    let _guard = super::super::window::WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_modifiers(ModifiersState::empty());
    let evs = run_collected(|cell| {
      let d = AxisEventData {
        delta_x: 10.0,
        delta_y: 20.0,
        pinch_scale: 0.0,
        source_type: InputSourceType::Touchpad,
        ..Default::default()
      };
      EventLoop::<()>::handle_axis_event(cell, window::WindowId(WindowId(0)), &d);
    });
    assert_eq!(
      evs,
      vec!["MouseWheel(PixelDelta(PhysicalPosition { x: 10.0, y: 20.0 }),ctrl=false)".to_string()]
    );
    set_modifiers(ModifiersState::empty());
  }

  #[test]
  fn axis_pinch_zoom_in_and_out_emit_ctrl_wheel() {
    let evs = run_collected(|cell| {
      let in_ = AxisEventData {
        pinch_scale: 1.5,
        ..Default::default()
      };
      let out_ = AxisEventData {
        pinch_scale: 0.5,
        ..Default::default()
      };
      EventLoop::<()>::handle_axis_event(cell, window::WindowId(WindowId(0)), &in_);
      EventLoop::<()>::handle_axis_event(cell, window::WindowId(WindowId(0)), &out_);
    });
    assert_eq!(
      evs,
      vec![
        "MouseWheel(LineDelta(0.0, 1.0),ctrl=true)".to_string(),
        "MouseWheel(LineDelta(0.0, -1.0),ctrl=true)".to_string(),
      ]
    );
  }

  #[test]
  fn axis_idle_event_emits_nothing() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_axis_event(
        cell,
        window::WindowId(WindowId(0)),
        &AxisEventData::default(),
      );
    });
    assert!(evs.is_empty());
  }

  // ─── handle_input_event dispatch ─────────────────────────────────────

  #[test]
  fn input_event_routes_mouse() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(
        cell,
        0,
        &InputEvent::MouseEvent(mouse(MouseAction::Move, OhosMouseButton::NoneButton)),
      );
    });
    assert_eq!(evs, vec!["CursorMoved(10.5,20.25)".to_string()]);
  }

  #[test]
  fn input_event_routes_axis() {
    // The ctrl= assertion reads the process-level KEYBOARD_MODIFIERS static;
    // hold the shared input/focus test lock and seed a known-empty modifier
    // state so a parallel test's CONTROL write cannot land between the read
    // and the assert (aligned with axis_mouse_wheel_uses_line_delta).
    let _guard = super::super::window::WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_modifiers(ModifiersState::empty());
    let evs = run_collected(|cell| {
      let d = AxisEventData {
        delta_y: 2.0,
        source_type: InputSourceType::Mouse,
        ..Default::default()
      };
      EventLoop::<()>::handle_input_event(cell, 0, &InputEvent::AxisEvent(d));
    });
    assert_eq!(
      evs,
      vec!["MouseWheel(LineDelta(0.0, 2.0),ctrl=false)".to_string()]
    );
    set_modifiers(ModifiersState::empty());
  }

  #[test]
  fn touch_down_emits_started_per_pointer() {
    let mut touch = TouchEventData {
      event_type: TouchEvent::Down,
      ..Default::default()
    };
    touch.touch_points = vec![
      TouchPointData {
        id: 7,
        x: 1.5,
        y: 2.5,
        force: 0.5,
        event_type: TouchEvent::Down,
        ..Default::default()
      },
      TouchPointData {
        id: 9,
        x: 3.5,
        y: 4.5,
        force: 0.25,
        event_type: TouchEvent::Down,
        ..Default::default()
      },
    ];
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(cell, 0, &InputEvent::TouchEvent(touch));
    });
    assert_eq!(
      evs,
      vec![
        "Touch(Started,1.5,2.5,id=7)".to_string(),
        "Touch(Started,3.5,4.5,id=9)".to_string(),
      ]
    );
  }

  #[test]
  fn touch_move_up_cancel_phases() {
    for (ty, phase) in [
      (TouchEvent::Move, "Moved"),
      (TouchEvent::Up, "Ended"),
      (TouchEvent::Cancel, "Cancelled"),
    ] {
      let mut touch = TouchEventData {
        event_type: ty,
        ..Default::default()
      };
      touch.touch_points = vec![TouchPointData {
        id: 1,
        event_type: ty,
        ..Default::default()
      }];
      let evs = run_collected(|cell| {
        EventLoop::<()>::handle_input_event(cell, 0, &InputEvent::TouchEvent(touch.clone()));
      });
      assert_eq!(
        evs,
        vec![format!("Touch({phase},0,0,id=1)")],
        "phase {phase}"
      );
    }
  }

  #[test]
  fn touch_unknown_event_type_emits_nothing() {
    let mut touch = TouchEventData {
      event_type: TouchEvent::Unknown,
      ..Default::default()
    };
    touch.touch_points = vec![TouchPointData {
      id: 1,
      event_type: TouchEvent::Unknown,
      ..Default::default()
    }];
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(cell, 0, &InputEvent::TouchEvent(touch));
    });
    assert!(evs.is_empty());
  }

  fn key(code: OhosKeyCode, action: Action) -> InputEvent {
    InputEvent::KeyEvent(KeyEventData {
      code,
      action,
      device_id: 3,
      source: EventSource::Keyboard,
      timestamp: 0,
    })
  }

  /// Set the process-level KEYBOARD_MODIFIERS static directly (a poisoned
  /// lock is ignored — no test panics while holding it).
  fn set_modifiers(modifiers: ModifiersState) {
    if let Ok(mut m) = KEYBOARD_MODIFIERS.lock() {
      *m = modifiers;
    }
  }

  #[test]
  fn key_down_up_and_autorepeat() {
    PRESSED_KEYS.with(|k| k.borrow_mut().clear());
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(cell, 0, &key(OhosKeyCode::A, Action::Down));
      EventLoop::<()>::handle_input_event(cell, 0, &key(OhosKeyCode::A, Action::Down));
      EventLoop::<()>::handle_input_event(cell, 0, &key(OhosKeyCode::A, Action::Up));
    });
    assert_eq!(evs.len(), 3);
    assert!(
      evs[0].starts_with("Key(Pressed,") && evs[0].ends_with(",repeat=false)"),
      "{}",
      evs[0]
    );
    assert!(evs[1].ends_with(",repeat=true)"), "{}", evs[1]);
    assert!(evs[2].starts_with("Key(Released,"), "{}", evs[2]);
    PRESSED_KEYS.with(|k| k.borrow_mut().clear());
  }

  #[test]
  fn key_location_for_modifier_pairs() {
    PRESSED_KEYS.with(|k| k.borrow_mut().clear());
    // KEYBOARD_MODIFIERS is a process-level static shared with the parallel
    // gained-focus test: reset before (no assumed initial state) and after
    // (this test leaves SHIFT held) so neither test flakes the other. The
    // shared WINDOW_MIRRORS_TEST_LOCK serializes those resets against the
    // other readers/writers of the static (PR review fix: device-UT flake --
    // an unlocked empty-reset landing between the gained-focus test's
    // CONTROL seed and its read suppressed the expected ModifiersChanged).
    let _guard = super::super::window::WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_modifiers(ModifiersState::empty());
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(cell, 0, &key(OhosKeyCode::ShiftLeft, Action::Down));
      EventLoop::<()>::handle_input_event(cell, 0, &key(OhosKeyCode::ShiftRight, Action::Down));
      EventLoop::<()>::handle_input_event(cell, 0, &key(OhosKeyCode::Numpad5, Action::Down));
    });
    // #109 modifier tracking: the first Shift Down transitions the modifier
    // set (none → SHIFT) and dispatches ModifiersChanged BEFORE the key
    // event; ShiftRight adds no new modifier (SHIFT already held), so no
    // redundant ModifiersChanged — just its key event; Numpad5 is not a
    // modifier. 4 events total.
    assert_eq!(evs.len(), 4, "{:?}", evs);
    assert!(evs[0].contains("ModifiersChanged"), "{}", evs[0]);
    assert!(evs[1].contains("loc=Left"), "{}", evs[1]);
    assert!(evs[2].contains("loc=Right"), "{}", evs[2]);
    assert!(evs[3].contains("loc=Numpad"), "{}", evs[3]);
    // Routing (review F8): a modifier transition at a non-primary window id
    // must dispatch both the ModifiersChanged and the key event to that id.
    // Lives here — not in input_events_dispatch_with_originating_window_id —
    // because this test already owns the KEYBOARD_MODIFIERS write discipline
    // (a third parallel writer would flake the process-level static).
    set_modifiers(ModifiersState::empty());
    let route_ids: Arc<Mutex<Vec<window::WindowId>>> = Arc::new(Mutex::new(Vec::new()));
    let route_sink = route_ids.clone();
    let route_cell: LoopCell<()> =
      Arc::new(RefCell::new(Some(Box::new(move |e: event::Event<()>| {
        if let event::Event::WindowEvent { window_id, .. } = e {
          route_sink.lock().unwrap().push(window_id);
        }
      }))));
    EventLoop::<()>::handle_input_event(&route_cell, 5, &key(OhosKeyCode::ShiftLeft, Action::Down));
    assert_eq!(
      route_ids.lock().unwrap().clone(),
      vec![window::WindowId(WindowId(5)), window::WindowId(WindowId(5)),]
    );
    PRESSED_KEYS.with(|k| k.borrow_mut().clear());
    set_modifiers(ModifiersState::empty());
  }

  // ─── handle_gained_focus ─────────────────────────────────────────────

  #[test]
  fn gained_focus_clears_stale_modifiers_before_focus() {
    // KEYBOARD_MODIFIERS is a process-level static Mutex and #[test]s run in
    // parallel — the two cases must stay serialized inside this one fn, and
    // the static is reset at both ends so neither case observes another
    // test's residue. The WINDOW_MIRRORS_TEST_LOCK additionally serializes
    // this test's GLOBAL writes (handle_gained_focus stores HAS_FOCUS /
    // LAST_FOCUSED_WINDOW_ID / WINDOW_MIRRORS and clears
    // KEYBOARD_MODIFIERS) against the focus_scale_tests family -- which hold
    // the same lock -- and against the axis tests that assert ctrl=false off
    // the process-level modifier static (PR review fix: device-UT flake,
    // ~1-2 parallel runs in 10 observed a cross-test residue).
    let _guard = super::super::window::WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    set_modifiers(ModifiersState::empty());
    // Stale modifiers (e.g. CONTROL held when focus was lost) are cleared
    // with an observable ModifiersChanged dispatched BEFORE Focused(true)
    // (Review R21 / issue Eulogizethesun/tauri#137).
    set_modifiers(ModifiersState::CONTROL);
    let evs = run_collected(|cell| EventLoop::<()>::handle_gained_focus(cell, 0));
    assert_eq!(
      evs,
      vec![
        "ModifiersChanged(ctrl=false,shift=false,alt=false,logo=false)".to_string(),
        "Focused(true)".to_string(),
      ]
    );
    // An already-empty set (first ACTIVE at startup) dispatches no redundant
    // ModifiersChanged — only Focused(true).
    let evs = run_collected(|cell| EventLoop::<()>::handle_gained_focus(cell, 0));
    assert_eq!(evs, vec!["Focused(true)".to_string()]);
    set_modifiers(ModifiersState::empty());
  }

  // ─── IME events ──────────────────────────────────────────────────────

  #[test]
  fn ime_text_input_emits_received_ime_text() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(
        cell,
        0,
        &InputEvent::ImeEvent(ImeEvent::TextInputEvent(TextInputEventData {
          text: "hello".to_string(),
        })),
      );
    });
    assert_eq!(evs, vec!["ImeText(hello)".to_string()]);
  }

  #[test]
  fn ime_backspace_and_enter_mock_press_release_pairs() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(
        cell,
        0,
        &InputEvent::ImeEvent(ImeEvent::BackspaceEvent(1)),
      );
      EventLoop::<()>::handle_input_event(cell, 0, &InputEvent::ImeEvent(ImeEvent::EnterEvent(1)));
    });
    assert_eq!(evs.len(), 4);
    assert!(evs[0].starts_with("Key(Pressed,Backspace"), "{}", evs[0]);
    assert!(evs[1].starts_with("Key(Released,Backspace"), "{}", evs[1]);
    assert!(evs[2].starts_with("Key(Pressed,Enter"), "{}", evs[2]);
    assert!(evs[3].starts_with("Key(Released,Enter"), "{}", evs[3]);
  }

  #[test]
  fn ime_status_hide_mocks_enter_show_is_ignored() {
    let evs = run_collected(|cell| {
      EventLoop::<()>::handle_input_event(
        cell,
        0,
        &InputEvent::ImeEvent(ImeEvent::ImeStatusEvent(KeyboardStatus::Hide)),
      );
      EventLoop::<()>::handle_input_event(
        cell,
        0,
        &InputEvent::ImeEvent(ImeEvent::ImeStatusEvent(KeyboardStatus::Show)),
      );
    });
    assert_eq!(evs.len(), 2);
    assert!(
      evs
        .iter()
        .all(|e| e.starts_with("Key(") && e.contains("Enter")),
      "{evs:?}"
    );
  }

  // ─── window_id routing (Phase 4, design.md D5) ─────────────────────

  #[test]
  fn input_events_dispatch_with_originating_window_id() {
    // Input dispatch must key the emitted WindowEvent by the window id the
    // input event carries (0 = primary, >0 = another render surface), not
    // the historical hardcoded primary.
    let ids: Arc<Mutex<Vec<window::WindowId>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = ids.clone();
    let cell: LoopCell<()> = Arc::new(RefCell::new(Some(Box::new(move |e: event::Event<()>| {
      if let event::Event::WindowEvent { window_id, .. } = e {
        sink.lock().unwrap().push(window_id);
      }
    }))));
    EventLoop::<()>::handle_input_event(
      &cell,
      7,
      &InputEvent::MouseEvent(mouse(MouseAction::Move, OhosMouseButton::NoneButton)),
    );
    EventLoop::<()>::handle_input_event(
      &cell,
      7,
      &InputEvent::ImeEvent(ImeEvent::TextInputEvent(TextInputEventData {
        text: "x".to_string(),
      })),
    );
    EventLoop::<()>::handle_input_event(
      &cell,
      3,
      &InputEvent::AxisEvent(AxisEventData {
        delta_y: 1.0,
        source_type: InputSourceType::Mouse,
        ..Default::default()
      }),
    );
    // Review F8: cover the touch and key paths too (pointer id 2 ≠ window id
    // 5, proving routing keys on the window id the event carries). Modifier
    // keys route in key_location_for_modifier_pairs (serialized there for the
    // KEYBOARD_MODIFIERS static); focus/close/destroy need a running loop and
    // stay with the device suite.
    let mut touch = TouchEventData {
      event_type: TouchEvent::Down,
      ..Default::default()
    };
    touch.touch_points = vec![TouchPointData {
      id: 2,
      x: 10.0,
      y: 20.0,
      force: 0.5,
      event_type: TouchEvent::Down,
      ..Default::default()
    }];
    EventLoop::<()>::handle_input_event(&cell, 5, &InputEvent::TouchEvent(touch));
    EventLoop::<()>::handle_input_event(&cell, 5, &key(OhosKeyCode::A, Action::Down));
    assert_eq!(
      ids.lock().unwrap().clone(),
      vec![
        window::WindowId(WindowId(7)),
        window::WindowId(WindowId(7)),
        window::WindowId(WindowId(3)),
        window::WindowId(WindowId(5)),
        window::WindowId(WindowId(5)),
      ]
    );
  }
}

/// T26-N4 companion coverage for `live_window_ids`, the mirror filter
/// shared by the app-level broadcast arms (ScaleFactorChanged /
/// ThemeChanged): dead mirrors are skipped and an all-dead/empty registry
/// falls back to the main window (id 0). One test function covers all
/// scenarios because they share the process-global WINDOW_MIRRORS registry
/// and #[test]s run in parallel; the sentinel ids sit far above any real
/// OHOS window id / NEXT_WINDOW_ID value (same discipline as the
/// pending_window_ops tests in window.rs).
#[cfg(test)]
mod live_window_ids_tests {
  use super::*;
  // The ohos directory is mounted as `platform_impl::platform`
  // (`#[path = "ohos/mod.rs"]` in platform_impl/mod.rs), so from this
  // module the sibling window module is two `super`s up.
  use super::super::window::WindowStateMirror;

  #[test]
  fn live_window_ids_filters_dead_mirrors_and_falls_back_to_window_zero() {
    // Exact-contents coexistence with window.rs's set_mirror_focused test.
    let _guard = super::super::window::WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner());
    const LIVE: i64 = i64::MAX - 11;
    const DEAD: i64 = i64::MAX - 12;
    // The strong ref keeps LIVE's mirror alive; DEAD is a Weak with no
    // strong ref anywhere (a window dropped without running `Drop`).
    let live_state = Arc::new(WindowStateMirror {
      visible: std::sync::atomic::AtomicBool::new(true),
      fullscreen: std::sync::atomic::AtomicBool::new(false),
      maximized: std::sync::atomic::AtomicBool::new(false),
      minimized: std::sync::atomic::AtomicBool::new(false),
      focused: std::sync::atomic::AtomicBool::new(false),
    });
    {
      // Keyed removal, not a whole-registry clear: window.rs's mirror test
      // registers its own sentinel entry in WINDOW_MIRRORS, and these tests
      // run in parallel -- only this test's own keys may be touched.
      let mut mirrors = WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned");
      mirrors.remove(&LIVE);
      mirrors.remove(&DEAD);
      mirrors.insert(LIVE, Arc::downgrade(&live_state));
      mirrors.insert(DEAD, std::sync::Weak::new());
    }
    // Only the live mirror's id is broadcast.
    assert_eq!(live_window_ids(), vec![LIVE]);
    // Drop the strong ref: LIVE's entry goes dead too, and with no live
    // mirrors left the broadcast falls back to the main window.
    drop(live_state);
    assert_eq!(live_window_ids(), vec![0]);
    // Empty registry (pre-first-window phase): same fallback. Keyed removal
    // again -- foreign sentinel entries (window.rs mirror test) stay put.
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&LIVE);
    assert_eq!(live_window_ids(), vec![0]);
  }
}

/// #180(a)/#180 coverage for the focus edge consumption (`design D7`) and the
/// scale broadcast dedup (`design D8`). The sentinel ids sit far above any
/// real OHOS window id and above the other test modules' sentinels; the
/// WINDOW_MIRRORS_TEST_LOCK serializes with the mirror-registry tests, and
/// every global the tests touch (HAS_FOCUS / LAST_FOCUSED_WINDOW_ID /
/// LAST_EFFECTIVE_SCALE / the mirrors) is saved and restored so a parallel
/// test cannot inherit our state.
#[cfg(test)]
mod focus_scale_tests {
  use super::super::window::{WindowStateMirror, WINDOW_MIRRORS_TEST_LOCK};
  use super::*;

  type LoopCell<T> = Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>;

  /// Compact descriptor of the focus-relevant events one handler invocation
  /// emitted: `"focus:<window id>:<bool>"` per Focused event.
  fn run_collected<T: 'static>(invoke: impl FnOnce(&LoopCell<T>)) -> Vec<String> {
    let out: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = out.clone();
    let cell: LoopCell<T> = Arc::new(RefCell::new(Some(Box::new(move |e: event::Event<T>| {
      if let event::Event::WindowEvent {
        window_id: window::WindowId(WindowId(id)),
        event: event::WindowEvent::Focused(focused),
      } = e
      {
        sink
          .lock()
          .expect("focus sink poisoned")
          .push(format!("focus:{id}:{focused}"));
      }
    }))));
    invoke(&cell);
    let collected = out.lock().expect("focus sink poisoned").clone();
    collected
  }

  /// Registers a live mirror for `window_id` with the given focused seed and
  /// returns the strong ref that keeps it alive.
  fn register_mirror(window_id: i64, focused: bool) -> Arc<WindowStateMirror> {
    let mirror = Arc::new(WindowStateMirror::new(focused));
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .insert(window_id, Arc::downgrade(&mirror));
    mirror
  }

  fn set_last_focused(window_id: i64) {
    LAST_FOCUSED_WINDOW_ID.store(window_id, Ordering::Relaxed);
  }

  fn last_focused() -> i64 {
    LAST_FOCUSED_WINDOW_ID.load(Ordering::Relaxed)
  }

  /// UT for #180(a) step 1 (design D7): a first sub-window gain edge forwards
  /// Focused(true); a gain edge over a still-focused previous holder
  /// pair-clears it first (mirror bit + synthesized Focused(false)) -- the
  /// paired semantics that keep ArkTS fast-switches (ACTIVE without a
  /// matching INACTIVE) from leaving a stale focused bit.
  #[test]
  fn ut_focus_drain_gain_edge_forwards_and_pairs_blur() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|p| p.into_inner());
    const PREV: i64 = i64::MAX - 21;
    const NEW: i64 = i64::MAX - 22;
    let has_focus_before = HAS_FOCUS.load(Ordering::Relaxed);
    let prev_mirror = register_mirror(PREV, true);
    let new_mirror = register_mirror(NEW, false);
    set_last_focused(PREV);
    if let Ok(mut m) = KEYBOARD_MODIFIERS.lock() {
      *m = ModifiersState::empty();
    }

    // First gain for NEW over the still-focused PREV: paired blur on PREV,
    // then the regular gain on NEW.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, NEW, true);
    });
    assert_eq!(
      events,
      vec![format!("focus:{PREV}:false"), format!("focus:{NEW}:true")],
      "gain edge must pair-clear the previous holder before the gain"
    );
    assert!(
      !prev_mirror.focused.load(Ordering::SeqCst),
      "paired blur did not clear PREV's focused bit"
    );
    assert!(
      new_mirror.focused.load(Ordering::SeqCst),
      "gain did not set NEW's focused bit"
    );
    assert_eq!(
      last_focused(),
      NEW,
      "gain edge must sync the shared snapshot"
    );

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    set_last_focused(0);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&PREV);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&NEW);
  }

  /// UT for #180(a) step 2 (design D7): a same-window same-state gain edge is
  /// swallowed -- zero WindowEvents, zero mirror writes, snapshot unchanged.
  #[test]
  fn ut_focus_drain_duplicate_gain_edge_is_swallowed() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|p| p.into_inner());
    const W: i64 = i64::MAX - 22;
    let has_focus_before = HAS_FOCUS.load(Ordering::Relaxed);
    let mirror = register_mirror(W, true);
    set_last_focused(W);
    if let Ok(mut m) = KEYBOARD_MODIFIERS.lock() {
      *m = ModifiersState::empty();
    }

    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, W, true);
    });
    assert!(events.is_empty(), "duplicate gain edge must emit nothing");
    assert!(
      mirror.focused.load(Ordering::SeqCst),
      "swallowed edge must not touch the mirror"
    );
    assert_eq!(
      last_focused(),
      W,
      "swallowed edge must not move the snapshot"
    );

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    set_last_focused(0);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&W);
  }

  /// UT for #180(a) (design D7 loss edge): a loss edge applies only when the
  /// window is the recorded holder (blur write face + snapshot -> 0); a
  /// repeat loss edge for a window that is no longer the holder is
  /// swallowed. HAS_FOCUS stays untouched by the drain path (stage events
  /// own the app-level bit).
  #[test]
  fn ut_focus_drain_loss_edge_applies_only_to_recorded_holder() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|p| p.into_inner());
    const W: i64 = i64::MAX - 23;
    const OTHER: i64 = i64::MAX - 24;
    let has_focus_before = HAS_FOCUS.load(Ordering::Relaxed);
    let mirror = register_mirror(W, true);
    let other_mirror = register_mirror(OTHER, false);
    set_last_focused(W);
    HAS_FOCUS.store(true, Ordering::Relaxed);

    // Stale blur for a non-holder: swallowed.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, OTHER, false);
    });
    assert!(events.is_empty(), "blur for a non-holder must be swallowed");
    assert!(
      mirror.focused.load(Ordering::SeqCst),
      "swallowed blur must not touch the holder"
    );
    assert_eq!(
      last_focused(),
      W,
      "swallowed blur must not clear the snapshot"
    );

    // Holder blur: the blur write face + snapshot -> 0; HAS_FOCUS untouched.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, W, false);
    });
    assert_eq!(
      events,
      vec![format!("focus:{W}:false")],
      "holder blur must dispatch Focused(false)"
    );
    assert!(
      !mirror.focused.load(Ordering::SeqCst),
      "holder blur did not clear the focused bit"
    );
    assert_eq!(last_focused(), 0, "holder blur must clear the snapshot");
    assert!(
      HAS_FOCUS.load(Ordering::Relaxed),
      "drain blur must not touch the app-level HAS_FOCUS"
    );

    // Repeat blur now finds no holder: swallowed.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, W, false);
    });
    assert!(
      events.is_empty(),
      "repeat blur with no holder must be swallowed"
    );
    assert!(!other_mirror.focused.load(Ordering::SeqCst));

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&W);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&OTHER);
  }

  /// UT for #180(a) step 3 (design D7): the stage path and the sub-window drain share
  /// LAST_FOCUSED_WINDOW_ID -- a stage gain (handle_gained_focus) syncs the
  /// snapshot the drain dedups against, so a cross-path duplicate gain edge
  /// for the same window is swallowed, while a sub-window gain over the
  /// stage-focused window still pair-clears it.
  #[test]
  fn ut_focus_stage_and_float_share_edge_snapshot() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK
      .lock()
      .unwrap_or_else(|p| p.into_inner());
    const STAGE: i64 = i64::MAX - 21;
    const FLOAT: i64 = i64::MAX - 22;
    let has_focus_before = HAS_FOCUS.load(Ordering::Relaxed);
    let stage_mirror = register_mirror(STAGE, false);
    let float_mirror = register_mirror(FLOAT, false);
    if let Ok(mut m) = KEYBOARD_MODIFIERS.lock() {
      *m = ModifiersState::empty();
    }

    // Stage gain (UIAbility ACTIVE): forwards and syncs the snapshot.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_gained_focus(cell, STAGE);
    });
    assert_eq!(
      events,
      vec![format!("focus:{STAGE}:true")],
      "stage gain must forward Focused(true)"
    );
    assert_eq!(
      last_focused(),
      STAGE,
      "stage gain must sync the shared snapshot"
    );
    assert!(stage_mirror.focused.load(Ordering::SeqCst));

    // Cross-path duplicate: a sub-window drain gain for the same window (already
    // recorded by the stage path) is swallowed.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, STAGE, true);
    });
    assert!(
      events.is_empty(),
      "cross-path duplicate gain must be swallowed"
    );

    // Sub-window gain over the stage-focused window: paired blur on STAGE first.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, FLOAT, true);
    });
    assert_eq!(
      events,
      vec![
        format!("focus:{STAGE}:false"),
        format!("focus:{FLOAT}:true")
      ],
      "sub-window gain over the stage holder must pair-clear it first"
    );
    assert!(!stage_mirror.focused.load(Ordering::SeqCst));
    assert!(float_mirror.focused.load(Ordering::SeqCst));

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    set_last_focused(0);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&STAGE);
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&FLOAT);
  }

  /// UT for #180 (design D8): the same scale value forwarded repeatedly
  /// forwards exactly once -- the first event seeds/forwards, bitwise-equal
  /// repeats are swallowed, and a real change forwards again. Also asserts
  /// the ThemeChanged guard's independence: deduping scale never touches
  /// LAST_EFFECTIVE_THEME and vice versa.
  #[test]
  fn ut_scale_dedup_forwards_change_once_and_swallows_repeats() {
    let scale_before = LAST_EFFECTIVE_SCALE.swap(0, Ordering::Relaxed);
    let theme_before = LAST_EFFECTIVE_THEME.load(Ordering::Relaxed);

    // First event (unseeded 0-bits) forwards.
    assert!(scale_should_forward(1.5), "first scale event must forward");
    // Three repeats of the same value: one repeat would be the "same value 3
    // sends" -- all three are swallowed (the single forward above is the one).
    assert!(!scale_should_forward(1.5), "repeat #1 must be swallowed");
    assert!(!scale_should_forward(1.5), "repeat #2 must be swallowed");
    assert!(!scale_should_forward(1.5), "repeat #3 must be swallowed");
    // A real change forwards again.
    assert!(scale_should_forward(2.0), "a changed scale must forward");
    assert!(
      !scale_should_forward(2.0),
      "repeat after the change must be swallowed"
    );
    // A different value whose f64 bits differ forwards. NOTE: `2.0 +
    // f64::EPSILON` is NOT a bitwise change -- EPSILON (2^-52) is half the ULP
    // at 2.0 (ULP = 2^-51 there), so the sum rounds back to exactly 2.0. Step
    // one bit instead.
    assert!(
      scale_should_forward(f64::from_bits(2.0f64.to_bits() + 1)),
      "bitwise change must forward"
    );

    // Theme independence: scale dedup left the theme snapshot untouched, and
    // a theme value still passes through its own guard.
    assert_eq!(
      LAST_EFFECTIVE_THEME.load(Ordering::Relaxed),
      theme_before,
      "scale dedup must not touch the theme snapshot"
    );

    LAST_EFFECTIVE_SCALE.store(scale_before, Ordering::Relaxed);
  }
}

/// PR#28 review gate F11: the T28-10 cursor warn-once latch had zero UT. A
/// capturing logger is required because the ONLY observable difference
/// between the first untracked read and the follow-ups is the log level --
/// the mapped error is the same NotSupported every time. The logger is
/// installed once per process (log::set_boxed_logger is one-shot) and
/// records (level, message) pairs; other #[test]s may log concurrently, so
/// the assertions filter on the exact untracked-cursor message.
///
/// PR#28 round-2 gate F6 (precondition note): this UT relies on the
/// CONVENTION that no other tao UT feeds the process cursor statics - there
/// is no mechanical lock enforcing it. The `CAPTURED_LOGS` capture buffer
/// and openharmony-ability's `EVER_FED` cursor static are shared across
/// every test in the process, so any future feed-side UT must be serialized
/// or isolated against this module, otherwise the shared state turns these
/// assertions randomly red.
#[cfg(test)]
mod cursor_untracked_tests {
  use super::*;

  // std::sync::LazyLock requires Rust 1.80. That does not conflict with
  // Cargo.toml's `rust-version = "1.74"`: 1.74 is the MSRV of consumers
  // building for standard (non-ohos) targets, and ohos-gated code - this
  // module tree is `#[cfg(target_env = "ohos")]` - is compiled out of
  // their build surface entirely. The ohos target - test and non-test code
  // alike - is built with the OHOS toolchain, and non-test ohos code already
  // relies on LazyLock (WINDOW_MIRRORS and PENDING_WINDOW_OPS in window.rs,
  // LAST_DISPATCHED_RECTS above), so that toolchain's version must already
  // be >= 1.80. That floor is a derived fact, not a mechanism: the repo has
  // no carrier that pins the OHOS toolchain (no rust-toolchain.toml, no
  // .cargo toolchain pin); the only toolchain version pin anywhere in the
  // repo is CI's dtolnay/rust-toolchain@1.74, which serves the standard
  // target matrix and neither records 1.80 nor contains an ohos target. The
  // floor is therefore inferred from the existing LazyLock use sites, and no
  // file enforces it for the ohos target. This test-only static adds no new
  // requirement beyond that derived floor.
  static CAPTURED_LOGS: std::sync::LazyLock<std::sync::Mutex<Vec<(log::Level, String)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

  struct CaptureLogger;

  impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
      metadata.level() <= log::Level::Debug
    }
    fn log(&self, record: &log::Record) {
      if let Ok(mut logs) = CAPTURED_LOGS.lock() {
        logs.push((record.level(), record.args().to_string()));
      }
    }
    fn flush(&self) {}
  }

  fn ensure_capture_logger() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
      // One-shot per process; if a logger is somehow already installed the
      // Err is ignored and its records simply do not reach CAPTURED_LOGS
      // (the asserts below would then fail loudly instead of passing
      // vacuously).
      let _ = log::set_boxed_logger(Box::new(CaptureLogger));
      // The runtime max level starts gated: open it to Debug so the latch's
      // degraded leg (log::debug!) is actually observable.
      log::set_max_level(log::LevelFilter::Debug);
    });
  }

  fn count_untracked_records(level: log::Level) -> usize {
    CAPTURED_LOGS
      .lock()
      .unwrap_or_else(|p| p.into_inner())
      .iter()
      .filter(|(l, msg)| *l == level && msg.starts_with("cursor_position: untracked"))
      .count()
  }

  /// Gate F11: the T28-10 warn-once latch. The first untracked read in the
  /// process logs at WARN, every later one degrades to DEBUG (hilog flood
  /// guard); the mapped error is NotSupported in both legs. Precondition:
  /// the process cursor statics (openharmony-ability EVER_FED) must still be
  /// unfed -- no other tao UT feeds the cursor, so the pristine state holds;
  /// a feed from a future UT would fail the first assert below instead of
  /// producing a false green.
  #[test]
  fn ut_cursor_untracked_warns_once_then_degrades_to_debug() {
    ensure_capture_logger();
    // Re-arm the latch: device UTs share one process, so the first-warn slot
    // must be owned by THIS test regardless of execution order.
    UNTRACKED_CURSOR_WARNED.store(false, Ordering::Relaxed);
    let app = OpenHarmonyApp::new();
    CAPTURED_LOGS
      .lock()
      .unwrap_or_else(|p| p.into_inner())
      .clear();

    // First read: NotSupported + exactly one WARN record.
    let first = cursor_position_from_app(&app);
    assert!(
      matches!(first, Err(error::ExternalError::NotSupported(_))),
      "an unfed cursor must map CursorUntracked onto NotSupported (got {first:?}) — if this reads Ok, another test fed the cursor statics and this UT lost its precondition"
    );
    assert_eq!(
      count_untracked_records(log::Level::Warn),
      1,
      "the first untracked read must log exactly one WARN record (T28-10)"
    );
    assert_eq!(
      count_untracked_records(log::Level::Debug),
      0,
      "the first untracked read must not log at DEBUG"
    );

    // Second read: the same NotSupported error, but the record degrades to
    // DEBUG.
    let second = cursor_position_from_app(&app);
    assert!(
      matches!(second, Err(error::ExternalError::NotSupported(_))),
      "the latch must not change the mapped error"
    );
    assert_eq!(
      count_untracked_records(log::Level::Warn),
      1,
      "follow-up untracked reads must not warn again"
    );
    assert_eq!(
      count_untracked_records(log::Level::Debug),
      1,
      "follow-up untracked reads must degrade to DEBUG (hilog flood guard)"
    );
    // Steady state: the latch stays tripped (the production steady state
    // after any untracked read).
  }
}

// Issue Eulogizethesun/tauri#222 — appended at file end, keeps existing line
// numbers. apply_color_mode_completion is pure apart from the
// APP_THEME_OVERRIDE store, so the rollback arm is testable without a bridge
// session. APP_THEME_OVERRIDE is a process global with no other test-side
// writer today; these tests serialize on a dedicated lock and restore FOLLOW
// so a future theme UT starts clean.
#[cfg(test)]
mod color_mode_completion_tests {
  use super::*;

  static APP_THEME_OVERRIDE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

  fn lock_override() -> std::sync::MutexGuard<'static, ()> {
    APP_THEME_OVERRIDE_TEST_LOCK
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
  }

  #[test]
  fn ut_color_mode_failure_completion_rolls_override_back_to_follow() {
    let _guard = lock_override();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_DARK, Ordering::Relaxed);
    assert!(apply_color_mode_completion(Some(false)));
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_FOLLOW,
      "a reported failure must stop theme() reporting an intent the system never applied"
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
  }

  #[test]
  fn ut_color_mode_success_and_empty_completions_leave_override_alone() {
    let _guard = lock_override();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_LIGHT, Ordering::Relaxed);
    // Success: the override already reflects the intent — untouched.
    assert!(!apply_color_mode_completion(Some(true)));
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_LIGHT
    );
    // Nothing new since the last drain: untouched.
    assert!(!apply_color_mode_completion(None));
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_LIGHT
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
  }

  // ---- Issue Eulogizethesun/tauri#264 (design D2) receipt-channel tests ----
  // THEME_RECEIPTS is a process global exactly like APP_THEME_OVERRIDE, so
  // every test below runs under the same APP_THEME_OVERRIDE_TEST_LOCK and
  // clears the queue on entry (idempotent reset: a previously failed run
  // must not leak receipts into the next test's FIFO assertions).

  fn clear_receipt_queue() {
    THEME_RECEIPTS
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
      .clear();
  }

  fn pending_receipt_count() -> usize {
    THEME_RECEIPTS
      .lock()
      .unwrap_or_else(|poisoned| poisoned.into_inner())
      .len()
  }

  /// UT for the D2 failure leg: one failure completion pops the queued
  /// receipt exactly once, fires it with `false`, and does so AFTER the
  /// override rollback.
  ///
  /// # Brief
  /// 1. lock + clear queue; override := DARK.
  /// 2. register one receipt recording (verdict, override-at-fire).
  /// 3. drain Some(false): return true, receipt fired exactly once with
  ///    false, override-at-fire already FOLLOW, final override FOLLOW.
  #[test]
  fn ut_theme_receipt_failure_fires_false_after_rollback() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_DARK, Ordering::Relaxed);
    let fired: std::sync::Arc<std::sync::Mutex<Vec<(bool, u8)>>> =
      std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&fired);
    register_theme_receipt(Box::new(move |applied| {
      sink
        .lock()
        .unwrap()
        .push((applied, APP_THEME_OVERRIDE.load(Ordering::Relaxed)));
    }));

    assert!(apply_color_mode_completion(Some(false)));
    let fired = fired.lock().unwrap();
    assert_eq!(
      fired.len(),
      1,
      "one failure completion must fire the receipt exactly once"
    );
    assert_eq!(
      fired[0].0, false,
      "the receipt must carry the failure verdict"
    );
    assert_eq!(
      fired[0].1, THEME_OVERRIDE_FOLLOW,
      "the rollback must land BEFORE the receipt fires (JS learns of the failure only after the honest state is restored)"
    );
    drop(fired);
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_FOLLOW
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }

  /// UT for the D2 success leg: one success completion pops the queued
  /// receipt with `true` and leaves the override untouched.
  ///
  /// # Brief
  /// 1. lock + clear queue; override := LIGHT.
  /// 2. register one receipt recording (verdict, override-at-fire).
  /// 3. drain Some(true): return false, receipt fired exactly once with
  ///    true, override still LIGHT at fire time and afterwards.
  #[test]
  fn ut_theme_receipt_success_fires_true_keeps_override() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_LIGHT, Ordering::Relaxed);
    let fired: std::sync::Arc<std::sync::Mutex<Vec<(bool, u8)>>> =
      std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&fired);
    register_theme_receipt(Box::new(move |applied| {
      sink
        .lock()
        .unwrap()
        .push((applied, APP_THEME_OVERRIDE.load(Ordering::Relaxed)));
    }));

    assert!(!apply_color_mode_completion(Some(true)));
    let fired = fired.lock().unwrap();
    assert_eq!(
      fired.len(),
      1,
      "one success completion must fire the receipt exactly once"
    );
    assert_eq!(
      fired[0].0, true,
      "the receipt must carry the success verdict"
    );
    assert_eq!(
      fired[0].1, THEME_OVERRIDE_LIGHT,
      "a success completion must not touch the override"
    );
    drop(fired);
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_LIGHT
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }

  /// UT for the bare-completion regression guard: with no receipt queued, a
  /// failure completion keeps the exact legacy behavior (rollback only, no
  /// wake) and the queue stays empty — the drain must not conjure or retain
  /// a phantom receipt.
  ///
  /// # Brief
  /// 1. lock + clear queue; override := DARK; register nothing.
  /// 2. drain Some(false): return true, override FOLLOW, queue len 0.
  #[test]
  fn ut_theme_receipt_failure_without_receipt_keeps_legacy_rollback() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_DARK, Ordering::Relaxed);

    assert!(apply_color_mode_completion(Some(false)));
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_FOLLOW
    );
    assert_eq!(
      pending_receipt_count(),
      0,
      "a bare completion must not create or retain a receipt"
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }

  /// UT for the None leg: a drain carrying no completion must neither wake
  /// nor drop the queued receipt — it stays pending for a later verdict (no
  /// false wake).
  ///
  /// # Brief
  /// 1. lock + clear queue; register one receipt.
  /// 2. drain None twice: both return false, the receipt never fires and
  ///    stays queued (len 1) after each drain.
  #[test]
  fn ut_theme_receipt_none_completion_leaves_receipt_pending() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    let fired: std::sync::Arc<std::sync::Mutex<Vec<bool>>> =
      std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&fired);
    register_theme_receipt(Box::new(move |applied| {
      sink.lock().unwrap().push(applied);
    }));

    assert!(!apply_color_mode_completion(None));
    assert!(
      fired.lock().unwrap().is_empty(),
      "a None drain must not wake the queued receipt"
    );
    assert_eq!(
      pending_receipt_count(),
      1,
      "a None drain must not drop the queued receipt"
    );
    assert!(!apply_color_mode_completion(None));
    assert!(fired.lock().unwrap().is_empty());
    assert_eq!(pending_receipt_count(), 1);

    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }

  /// UT for FIFO pairing: two receipts registered in order pair with two
  /// completions in completion order — first-queued fires first, the second
  /// stays pending until its own completion.
  ///
  /// # Brief
  /// 1. lock + clear queue; override := DARK; register receipts 0 and 1
  ///    (each records its index + verdict into one shared log).
  /// 2. drain Some(false): only receipt 0 fires with false.
  /// 3. drain Some(true): only receipt 1 fires with true.
  #[test]
  fn ut_theme_receipt_fifo_pairs_receipts_in_completion_order() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_DARK, Ordering::Relaxed);
    let fired: std::sync::Arc<std::sync::Mutex<Vec<(usize, bool)>>> =
      std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    for receipt_index in 0..2usize {
      let sink = std::sync::Arc::clone(&fired);
      register_theme_receipt(Box::new(move |applied| {
        sink.lock().unwrap().push((receipt_index, applied));
      }));
    }

    assert!(apply_color_mode_completion(Some(false)));
    assert_eq!(
      fired.lock().unwrap().as_slice(),
      &[(0usize, false)],
      "the first completion must wake the FIRST-queued receipt only"
    );

    assert!(!apply_color_mode_completion(Some(true)));
    assert_eq!(
      fired.lock().unwrap().as_slice(),
      &[(0usize, false), (1usize, true)],
      "the second completion must wake the second-queued receipt with the success verdict"
    );

    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_FOLLOW
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }

  /// UT for exactly-one-pop: each completion pops exactly one receipt — no
  /// double-fire within a completion, no leak across completions, queue
  /// drains one by one.
  ///
  /// # Brief
  /// 1. lock + clear queue; override := DARK; register receipts 0/1/2 into
  ///    one shared log.
  /// 2. drain Some(true) twice: after each, exactly one new fire and the
  ///    pending count drops 3 -> 2 -> 1.
  /// 3. drain Some(false): receipt 2 fires with false, queue empty, override
  ///    rolled back; every receipt fired exactly once in total.
  #[test]
  fn ut_theme_receipt_each_completion_pops_exactly_one() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_DARK, Ordering::Relaxed);
    let fired: std::sync::Arc<std::sync::Mutex<Vec<(usize, bool)>>> =
      std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    for receipt_index in 0..3usize {
      let sink = std::sync::Arc::clone(&fired);
      register_theme_receipt(Box::new(move |applied| {
        sink.lock().unwrap().push((receipt_index, applied));
      }));
    }

    assert!(!apply_color_mode_completion(Some(true)));
    assert_eq!(
      fired.lock().unwrap().as_slice(),
      &[(0usize, true)],
      "the first completion must pop exactly one receipt"
    );
    assert_eq!(pending_receipt_count(), 2);

    assert!(!apply_color_mode_completion(Some(true)));
    assert_eq!(
      fired.lock().unwrap().as_slice(),
      &[(0usize, true), (1usize, true)],
      "the second completion must pop exactly one more receipt"
    );
    assert_eq!(pending_receipt_count(), 1);

    assert!(apply_color_mode_completion(Some(false)));
    assert_eq!(
      fired.lock().unwrap().as_slice(),
      &[(0usize, true), (1usize, true), (2usize, false)],
      "the third completion must pop the last receipt with the failure verdict"
    );
    assert_eq!(pending_receipt_count(), 0);

    let fired = fired.lock().unwrap();
    for receipt_index in 0..3usize {
      assert_eq!(
        fired.iter().filter(|(i, _)| *i == receipt_index).count(),
        1,
        "receipt {receipt_index} must fire exactly once (no double-fire, no leak)"
      );
    }
    drop(fired);
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_FOLLOW
    );
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }

  /// UT for the register-first discipline: `set_theme_with_receipt` queues
  /// the receipt EAGERLY — by the time the call returns (and therefore
  /// before any completion can be drained) the receipt is already queued —
  /// and still applies the theme change; the receipt's own override-at-fire
  /// probe shows the applied state.
  ///
  /// # Brief
  /// 1. lock + clear queue; override := FOLLOW.
  /// 2. build a bare EventLoopWindowTarget (a UT process has no ArkTS main
  ///    thread env, so set_app_theme stores the override and skips the
  ///    bridge call — exactly the state these tests run in).
  /// 3. set_theme_with_receipt(Some(Dark), probe): queue already holds the
  ///    receipt, override already DARK, receipt NOT yet fired.
  /// 4. drain Some(true): receipt fires exactly once with true and reads
  ///    override DARK.
  ///
  /// Register-BEFORE-apply within the method itself is not externally
  /// observable (the synchronous call cannot be re-entered by a completion
  /// drain); what this test pins is the externally load-bearing half: the
  /// registration is eager, not deferred to the drain — a receipt registered
  /// lazily would read queue-len 0 here and miss its completion.
  #[test]
  fn ut_set_theme_with_receipt_registers_eagerly_before_apply() {
    let _guard = lock_override();
    clear_receipt_queue();
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    let target = EventLoopWindowTarget::<()> {
      app: OpenHarmonyApp::new(),
      bridge_executor: BridgeExecutor::new(),
      exit: Cell::new(false),
      _marker: PhantomData,
    };
    let fired: std::sync::Arc<std::sync::Mutex<Vec<(bool, u8)>>> =
      std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&fired);
    target.set_theme_with_receipt(
      Some(Theme::Dark),
      Box::new(move |applied| {
        sink
          .lock()
          .unwrap()
          .push((applied, APP_THEME_OVERRIDE.load(Ordering::Relaxed)));
      }),
    );

    assert_eq!(
      pending_receipt_count(),
      1,
      "the receipt must already be queued when set_theme_with_receipt returns (eager registration, not drain-time)"
    );
    assert_eq!(
      APP_THEME_OVERRIDE.load(Ordering::Relaxed),
      THEME_OVERRIDE_DARK,
      "set_app_theme must still run the full existing set_theme pipeline"
    );
    assert!(
      fired.lock().unwrap().is_empty(),
      "queueing must not fire the receipt"
    );

    assert!(!apply_color_mode_completion(Some(true)));
    let fired = fired.lock().unwrap();
    assert_eq!(
      fired.as_slice(),
      &[(true, THEME_OVERRIDE_DARK)],
      "the completion wakes the receipt with the success verdict and the applied override state"
    );
    drop(fired);
    assert_eq!(pending_receipt_count(), 0);
    APP_THEME_OVERRIDE.store(THEME_OVERRIDE_FOLLOW, Ordering::Relaxed);
    clear_receipt_queue();
  }
}
