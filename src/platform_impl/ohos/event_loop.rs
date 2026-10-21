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
use crate::event_loop::{self, ControlFlow};
use crate::keyboard::{Key, KeyCode, KeyLocation, ModifiersState};
use crate::monitor;
use crate::window::{self, Theme};

use super::keycodes::{to_location, to_logical, to_physical};
use super::monitor::MonitorHandle;
use super::window::{drain_ready_pending_window_ops, WindowId, WINDOW_MIRRORS};

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

/// Effective theme (override if set, else system color_mode) encoded for the
/// ThemeChanged dispatch guard (issue Eulogizethesun/tauri#108). u8 pairs with
/// the THEME_OVERRIDE_* constants' Light=0/Dark=1 encoding.
const EFFECTIVE_THEME_LIGHT: u8 = 0;
const EFFECTIVE_THEME_DARK: u8 = 1;
const EFFECTIVE_THEME_UNSEEDED: u8 = 2;
static LAST_EFFECTIVE_THEME: AtomicU8 = AtomicU8::new(EFFECTIVE_THEME_UNSEEDED);

/// #180① (design D7): the window id currently recorded as focused — the
/// shared change-edge snapshot for BOTH focus sources (FloatPage
/// `windowEvent` WINDOW_ACTIVE/WINDOW_INACTIVE edges drained via
/// `drain_pending_window_focus`, and the per-UIAbility stage
/// GainedFocus/LostFocus arms). 0 = none. The edge rules built on it:
/// a gain edge (focused=true, last≠W) pairs-clears the previous holder
/// before the gain; a loss edge only applies when W is the recorded holder;
/// same-state duplicate edges are swallowed (zero WindowEvents). Written by
/// `handle_gained_focus` (gain → window id) and the stage LostFocus arm
/// (loss → 0) so cross-path duplicate edges dedup the same way. Read and
/// written from the single-threaded run_loop dispatch only — Relaxed matches
/// the HAS_FOCUS discipline.
static LAST_FOCUSED_WINDOW_ID: AtomicI64 = AtomicI64::new(0);

/// #180 (design D8): the last scale factor forwarded as
/// `ScaleFactorChanged`, stored as `f64::to_bits` (NaN-safe bitwise
/// comparison). Bitwise-equal repeats are swallowed; the first event forwards
/// (0-bits is not a representable scale, so the initial value reads as
/// unseeded). Mirrors the ThemeChanged edge detection
/// ([`LAST_EFFECTIVE_THEME`]) — the two guards are independent and must not
/// interact.
static LAST_EFFECTIVE_SCALE: AtomicU64 = AtomicU64::new(0);

/// #180 (design D8): the ThemeChanged-style edge detector for scale factors —
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
/// of MainPage AND FloatPage via the `update_cursor_position` NAPI function —
/// ohos-cursor-passthrough D1/D2) and is read back through
/// `OpenHarmonyApp::cursor_position()` (physical px). Never-fed reads fail
/// with the typed untracked error → `ExternalError::NotSupported` (D3/R5);
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
  fn new() -> Self {
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
/// Fed by the ArkTS `onMouse` handlers (MainPage AND FloatPage —
/// ohos-cursor-passthrough D2) via the `update_cursor_position` NAPI
/// function; the vp→physical conversion and the bit-unpacking of the process
/// statics live inside `OpenHarmonyApp::cursor_position` (issue #87 major-10)
/// — see the CURSOR_POSITION note near the top of this file.
///
/// Untracked fail-safe contract (ohos-cursor-passthrough D3, R5 — behavior
/// change): when no feed was ever recorded the read is the typed
/// `CursorUntracked` error, mapped here onto `ExternalError::NotSupported`
/// (the §1.5 error-conversion constraint: single variant, no wrapping) —
/// NOT swallowed into `Ok((0,0))`, which was indistinguishable from a real
/// screen-origin feed. runtime-wry maps `NotSupported` to
/// `FailedToGetCursorPosition`, so the tauri command and the JS
/// `cursorPosition()` promise reject; app-side discipline (spec): catch →
/// keep `set_ignore_cursor_events(true)`, only a hit-test match may set
/// touchable.
pub(crate) fn cursor_position_from_app(
  app: &OpenHarmonyApp,
) -> Result<PhysicalPosition<f64>, error::ExternalError> {
  app
    .cursor_position()
    .map(|(x, y)| PhysicalPosition::new(x, y))
    .map_err(|CursorUntracked| {
      log::warn!("cursor_position: untracked (no mouse feed ever recorded) → NotSupported");
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
          emit_synthetic_key(event_loop_cell, window_id, Key::Backspace, KeyCode::Backspace);
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
  fn handle_mouse_event(event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>, window_id: window::WindowId, mouse_event: &MouseEventData) {
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
  fn handle_axis_event(event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>, window_id: window::WindowId, axis_event: &AxisEventData) {
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
    // tolerated — the window may have dropped between event and write).
    super::window::set_mirror_focused(window_id, true);
    // #180① (design D7): sync the shared change-edge snapshot so the Float
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

  /// #180① (design D7): the single blur write face — per-window mirror bit
  /// (#173 write face) + the `Focused(false)` event. Shared by the stage
  /// LostFocus arm (which additionally clears the app-level HAS_FOCUS bit)
  /// and the Float drain's paired-blur synthesis (which must NOT touch
  /// HAS_FOCUS — the ability may still hold focus, and the pair is completed
  /// by the immediately following gain on the new window). Extracted so the
  /// two callers cannot drift.
  fn apply_blur(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: i64,
  ) {
    // #173: mirror the per-window focused bit (miss tolerated — same
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

  /// #180① (design D7): consume one drained `notify_window_focus` edge
  /// (FloatPage windowEvent WINDOW_ACTIVE/WINDOW_INACTIVE) with change-edge
  /// dedup against [`LAST_FOCUSED_WINDOW_ID`] plus paired-blur synthesis:
  ///
  /// - gain edge (`focused=true`, last≠W): first pair-clear the previous
  ///   holder P=last (mirror bit + `Focused(false)`) — ONLY when P still
  ///   reports `focused=true` (a holder already blurred by a real stage event
  ///   must not be double-blurred) — then the regular gain path
  ///   ([`Self::handle_gained_focus`], which also syncs the snapshot).
  /// - loss edge (`focused=false`, last==W): the blur write face + snapshot
  ///   → 0. HAS_FOCUS is deliberately NOT touched (see [`Self::apply_blur`]
  ///   note): only stage events maintain the app-level bit.
  /// - same-state edges (`focused=true` && last==W; `focused=false` &&
  ///   last≠W): swallowed — zero WindowEvents, zero mirror writes.
  fn handle_pending_window_focus(
    event_loop_cell: &Arc<RefCell<Option<Box<dyn FnMut(event::Event<T>) + 'static>>>>,
    window_id: i64,
    focused: bool,
  ) {
    let last = LAST_FOCUSED_WINDOW_ID.load(Ordering::Relaxed);
    if focused {
      if last == window_id {
        // Duplicate gain edge (e.g. ArkTS re-reporting WINDOW_ACTIVE for the
        // window it already recorded as focused) — swallow.
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
        // nothing) — swallow so a replayed INACTIVE cannot blur a window
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
            // G6 / cross-cutting: a failed Float window has window_id=None
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

      // #180① (design D7): Float sub-window focus edges (FloatPage windowEvent
      // WINDOW_ACTIVE/WINDOW_INACTIVE → notifyWindowFocus) drain next to the
      // status backfill and run through the same change-edge dedup +
      // paired-blur synthesis the stage arms share.
      for (focus_window_id, focused) in openharmony_ability::drain_pending_window_focus() {
        Self::handle_pending_window_focus(&event_loop_cell, focus_window_id, focused);
      }

      // Issue 7: replay window ops queued against spawned UIAbility windows
      // whose stage handshake completed since the last pass (the D7
      // registration waker wakes the loop).
      drain_ready_pending_window_ops();

      match event {
        MainEvent::SurfaceCreate { .. } => {
          call_event_handler!(event_loop_cell, event::Event::NewEvents(StartCause::Init));
          call_event_handler!(event_loop_cell, event::Event::Resumed);
        }
        MainEvent::SurfaceDestroy { .. } => {
          call_event_handler!(event_loop_cell, event::Event::Suspended);
        }
        MainEvent::WindowResize { window_id, size } => {
          // Phase 3 (design.md D6): route by the originating window's id instead of
          // the ZST constant. window_id comes from the ArkTS-wrapped options
          // (lifecycle.rs window_resize closure / xcomponent.rs on_surface_changed).
          //
          // RESIDUAL GAP (review R11): this arm has MIXED sources with
          // different metrics and is deliberately left as-is. The XComponent
          // on_surface_changed path already reports the drawable (inner) area,
          // while windowSizeChange reports the outer window rect on decorated
          // windows — routing the payload through inner_rect_for would corrupt
          // the former (the surface rect may precede the cache update), so
          // neither is transformed here. tauri apps are unaffected either way
          // (tauri-runtime-wry re-reads inner_size on every Resized); raw tao
          // consumers see an outer-sized Resized from the windowSizeChange
          // source, corrected by the inner-sized ContentRectChange Resized
          // that follows the same resize.
          let size = PhysicalSize::new(size.width as _, size.height as _);
          call_event_handler!(
            event_loop_cell,
            event::Event::WindowEvent {
              window_id: window::WindowId(WindowId(window_id)),
              event: event::WindowEvent::Resized(size),
            }
          );
        }
        MainEvent::WindowRedraw { window_id, .. } => {
          // RedrawRequested is driven by the XComponent frame callback, keyed
          // by the window that called render() (Phase 4 design.md D5). In
          // practice that is the primary only — spawned UIAbility windows
          // and Float sub-windows own no XComponent.
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
          //   legacy Resized (Float sub-windows have no XComponent surface, so
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
          let prev = LAST_DISPATCHED_RECTS
            .lock()
            .ok()
            .and_then(|mut rects| rects.insert(window_id, (left, top, width, height)));
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
        MainEvent::GainedFocus { window_id } => Self::handle_gained_focus(&event_loop_cell, window_id),
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
          // #180① (design D7): the blur write face is shared with the Float
          // drain's paired-blur synthesis (apply_blur), and the shared
          // change-edge snapshot records the holder clearing so cross-path
          // duplicate edges dedup the same way.
          Self::apply_blur(&event_loop_cell, window_id);
          LAST_FOCUSED_WINDOW_ID.store(0, Ordering::Relaxed);
        }
        MainEvent::ConfigChanged { .. } => {
          // Configuration changes are app-level (EnvironmentCallback), not tied to a
          // specific window. Broadcast to every live window like ThemeChanged
          // below (scale is app-global on OHOS; WINDOW_MIRRORS covers main +
          // spawned UIAbility + Float sub-windows, with a window-0 fallback
          // before any window exists). Each window gets its own
          // `new_inner_size` slot so one handler's adjustment cannot leak
          // into the next window's event. The size is the window's OWN inner
          // rect (`inner_rect_for`, the same source as the Resized dispatch
          // in the ContentRectChange arm above — issue Eulogizethesun/tauri#97),
          // not the primary's geometry broadcast to everyone: a spawned or
          // Float window must be offered its own drawable size to resize to.
          // Degradation: `inner_rect_for` returns (0,0,0,0) before a window's
          // first readable windowRectChange, so fall back to the app-level
          // content rect then (the same pre-geometry degradation the Resized
          // path tolerates).
          // #180 (design D8): the broadcast is edge-guarded — a bitwise-equal
          // scale repeat (ConfigChanged storms that leave scale untouched)
          // forwards nothing, first event forwards (0-bits reads as
          // unseeded). Mirrors the ThemeChanged guard below; the two edge
          // detectors are independent.
          let scale = app.scale();
          if scale_should_forward(scale as f64) {
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
                    scale_factor: scale as _,
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
          // registry covers main + Float sub-windows).
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
          // ability). Float sub-windows are destroyed via the separate ArkTS
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
  }

  #[test]
  fn axis_touchpad_uses_pixel_delta() {
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
      EventLoop::<()>::handle_axis_event(cell, window::WindowId(WindowId(0)), &AxisEventData::default());
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
    // (this test leaves SHIFT held) so neither test flakes the other.
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
    let route_cell: LoopCell<()> = Arc::new(RefCell::new(Some(Box::new(
      move |e: event::Event<()>| {
        if let event::Event::WindowEvent { window_id, .. } = e {
          route_sink.lock().unwrap().push(window_id);
        }
      },
    ))));
    EventLoop::<()>::handle_input_event(
      &route_cell,
      5,
      &key(OhosKeyCode::ShiftLeft, Action::Down),
    );
    assert_eq!(
      route_ids.lock().unwrap().clone(),
      vec![
        window::WindowId(WindowId(5)),
        window::WindowId(WindowId(5)),
      ]
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
    // test's residue.
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
      EventLoop::<()>::handle_input_event(
        cell,
        0,
        &InputEvent::ImeEvent(ImeEvent::EnterEvent(1)),
      );
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
    let cell: LoopCell<()> = Arc::new(RefCell::new(Some(Box::new(
      move |e: event::Event<()>| {
        if let event::Event::WindowEvent { window_id, .. } = e {
          sink.lock().unwrap().push(window_id);
        }
      },
    ))));
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
      // run in parallel — only this test's own keys may be touched.
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
    // again — foreign sentinel entries (window.rs mirror test) stay put.
    WINDOW_MIRRORS
      .lock()
      .expect("WINDOW_MIRRORS poisoned")
      .remove(&LIVE);
    assert_eq!(live_window_ids(), vec![0]);
  }
}

/// #180①/#180 coverage for the focus edge consumption (`design D7`) and the
/// scale broadcast dedup (`design D8`). The sentinel ids sit far above any
/// real OHOS window id and above the other test modules' sentinels; the
/// WINDOW_MIRRORS_TEST_LOCK serializes with the mirror-registry tests, and
/// every global the tests touch (HAS_FOCUS / LAST_FOCUSED_WINDOW_ID /
/// LAST_EFFECTIVE_SCALE / the mirrors) is saved and restored so a parallel
/// test cannot inherit our state.
#[cfg(test)]
mod focus_scale_tests {
  use super::*;
  use super::super::window::{WindowStateMirror, WINDOW_MIRRORS_TEST_LOCK};

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

  /// UT for #180① ① (design D7): a first Float gain edge forwards
  /// Focused(true); a gain edge over a still-focused previous holder
  /// pair-clears it first (mirror bit + synthesized Focused(false)) — the
  /// paired semantics that keep ArkTS fast-switches (ACTIVE without a
  /// matching INACTIVE) from leaving a stale focused bit.
  #[test]
  fn ut_focus_drain_gain_edge_forwards_and_pairs_blur() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
      vec![
        format!("focus:{PREV}:false"),
        format!("focus:{NEW}:true")
      ],
      "gain edge must pair-clear the previous holder before the gain"
    );
    assert!(!prev_mirror.focused.load(Ordering::SeqCst), "paired blur did not clear PREV's focused bit");
    assert!(new_mirror.focused.load(Ordering::SeqCst), "gain did not set NEW's focused bit");
    assert_eq!(last_focused(), NEW, "gain edge must sync the shared snapshot");

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    set_last_focused(0);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&PREV);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&NEW);
  }

  /// UT for #180① ② (design D7): a same-window same-state gain edge is
  /// swallowed — zero WindowEvents, zero mirror writes, snapshot unchanged.
  #[test]
  fn ut_focus_drain_duplicate_gain_edge_is_swallowed() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
    assert!(mirror.focused.load(Ordering::SeqCst), "swallowed edge must not touch the mirror");
    assert_eq!(last_focused(), W, "swallowed edge must not move the snapshot");

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    set_last_focused(0);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&W);
  }

  /// UT for #180① (design D7 loss edge): a loss edge applies only when the
  /// window is the recorded holder (blur write face + snapshot → 0); a
  /// repeat loss edge for a window that is no longer the holder is
  /// swallowed. HAS_FOCUS stays untouched by the drain path (stage events
  /// own the app-level bit).
  #[test]
  fn ut_focus_drain_loss_edge_applies_only_to_recorded_holder() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
    assert!(mirror.focused.load(Ordering::SeqCst), "swallowed blur must not touch the holder");
    assert_eq!(last_focused(), W, "swallowed blur must not clear the snapshot");

    // Holder blur: the blur write face + snapshot → 0; HAS_FOCUS untouched.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, W, false);
    });
    assert_eq!(events, vec![format!("focus:{W}:false")], "holder blur must dispatch Focused(false)");
    assert!(!mirror.focused.load(Ordering::SeqCst), "holder blur did not clear the focused bit");
    assert_eq!(last_focused(), 0, "holder blur must clear the snapshot");
    assert!(HAS_FOCUS.load(Ordering::Relaxed), "drain blur must not touch the app-level HAS_FOCUS");

    // Repeat blur now finds no holder: swallowed.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, W, false);
    });
    assert!(events.is_empty(), "repeat blur with no holder must be swallowed");
    assert!(!other_mirror.focused.load(Ordering::SeqCst));

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&W);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&OTHER);
  }

  /// UT for #180① ③ (design D7): the stage path and the Float drain share
  /// LAST_FOCUSED_WINDOW_ID — a stage gain (handle_gained_focus) syncs the
  /// snapshot the drain dedups against, so a cross-path duplicate gain edge
  /// for the same window is swallowed, while a Float gain over the
  /// stage-focused window still pair-clears it.
  #[test]
  fn ut_focus_stage_and_float_share_edge_snapshot() {
    let _guard = WINDOW_MIRRORS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
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
    assert_eq!(events, vec![format!("focus:{STAGE}:true")], "stage gain must forward Focused(true)");
    assert_eq!(last_focused(), STAGE, "stage gain must sync the shared snapshot");
    assert!(stage_mirror.focused.load(Ordering::SeqCst));

    // Cross-path duplicate: a Float drain gain for the same window (already
    // recorded by the stage path) is swallowed.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, STAGE, true);
    });
    assert!(events.is_empty(), "cross-path duplicate gain must be swallowed");

    // Float gain over the stage-focused window: paired blur on STAGE first.
    let events = run_collected::<()>(|cell| {
      EventLoop::<()>::handle_pending_window_focus(cell, FLOAT, true);
    });
    assert_eq!(
      events,
      vec![
        format!("focus:{STAGE}:false"),
        format!("focus:{FLOAT}:true")
      ],
      "Float gain over the stage holder must pair-clear it first"
    );
    assert!(!stage_mirror.focused.load(Ordering::SeqCst));
    assert!(float_mirror.focused.load(Ordering::SeqCst));

    HAS_FOCUS.store(has_focus_before, Ordering::Relaxed);
    set_last_focused(0);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&STAGE);
    WINDOW_MIRRORS.lock().expect("WINDOW_MIRRORS poisoned").remove(&FLOAT);
  }

  /// UT for #180 (design D8): the same scale value forwarded repeatedly
  /// forwards exactly once — the first event seeds/forwards, bitwise-equal
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
    // sends" — all three are swallowed (the single forward above is the one).
    assert!(!scale_should_forward(1.5), "repeat #1 must be swallowed");
    assert!(!scale_should_forward(1.5), "repeat #2 must be swallowed");
    assert!(!scale_should_forward(1.5), "repeat #3 must be swallowed");
    // A real change forwards again.
    assert!(scale_should_forward(2.0), "a changed scale must forward");
    assert!(!scale_should_forward(2.0), "repeat after the change must be swallowed");
    // A different value whose f64 bits differ forwards even at a tiny delta.
    assert!(scale_should_forward(2.0 + f64::EPSILON), "bitwise change must forward");

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
