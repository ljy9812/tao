use crate::dpi::{PhysicalPosition, PhysicalSize};
use crate::monitor;
use openharmony_ability::{all_displays, default_display, DisplaySnapshot, OpenHarmonyApp};

/// Monitor handle bound to one OHOS display (issue Eulogizethesun/tauri#106 —
/// previously a single synthetic monitor: every accessor ignored which display
/// it stood for, `position()` was hardcoded (0,0) and `available_monitors`
/// always returned one entry).
///
/// The handle carries only the display id; accessors re-query the live display
/// list, so hotplug (a display disappearing) degrades gracefully instead of
/// serving stale geometry. `app` is kept for the default-display
/// content-rect fallback in `size()`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonitorHandle {
  app: OpenHarmonyApp,
  display_id: u32,
}

impl MonitorHandle {
  /// Live snapshot of this handle's display, if it is still present.
  fn snapshot(&self) -> Option<DisplaySnapshot> {
    all_displays()
      .into_iter()
      .find(|display| display.id == self.display_id)
  }

  /// One handle per connected display (multi-monitor enumeration, issue #106).
  pub(crate) fn all_for_app(app: &OpenHarmonyApp) -> Vec<MonitorHandle> {
    all_displays()
      .into_iter()
      .map(|display| Self {
        app: app.clone(),
        display_id: display.id,
      })
      .collect()
  }

  /// Handle for the default (primary) display.
  pub(crate) fn primary_for_app(app: &OpenHarmonyApp) -> MonitorHandle {
    Self {
      app: app.clone(),
      display_id: default_display(),
    }
  }

  /// Handle for the display whose bounds contain the given point (physical px
  /// in the global display coordinate space), or `None` when the point lies
  /// outside every display.
  pub(crate) fn from_point(app: &OpenHarmonyApp, x: f64, y: f64) -> Option<MonitorHandle> {
    all_displays()
      .into_iter()
      .find(|display| {
        let (left, top) = (display.x as f64, display.y as f64);
        x >= left && y >= top && x < left + display.width as f64 && y < top + display.height as f64
      })
      .map(|display| Self {
        app: app.clone(),
        display_id: display.id,
      })
  }

  pub fn name(&self) -> Option<String> {
    Some(
      self
        .snapshot()
        .map(|display| display.name)
        .unwrap_or_else(|| "OpenHarmony Device".to_owned()),
    )
  }

  pub fn size(&self) -> PhysicalSize<u32> {
    // Real physical display dimensions — NOT the window's content_rect (which is
    // the window's own content area and is smaller than the screen). Using
    // content_rect here made positioner `Center` compute to negative coords
    // (content/2 - outer/2 < 0) which OHOS clamps to (0,0), so windows snapped
    // to top-left instead of centering.
    // Prefer the display snapshot; fall back to content_rect only for the
    // default display when the snapshot is missing or zero-sized (pre-#106
    // behavior — DisplayManager query failure on old devices). See
    // ohos-monitor-real-values.
    if let Some(display) = self.snapshot() {
      if display.width > 0 && display.height > 0 {
        return PhysicalSize::new(display.width, display.height);
      }
    }
    if self.display_id == default_display() {
      let size = self.app.content_rect();
      return PhysicalSize::new(size.width as _, size.height as _);
    }
    log::warn!(
      "[tao ohos] no size for display {} (disconnected?); returning 0x0",
      self.display_id
    );
    PhysicalSize::new(0, 0)
  }

  pub fn position(&self) -> PhysicalPosition<i32> {
    // ArkTS Display.x/y (API 19+; (0,0) below — see the display module).
    let (x, y) = self
      .snapshot()
      .map(|display| (display.x, display.y))
      .unwrap_or((0, 0));
    (x, y).into()
  }

  pub fn scale_factor(&self) -> f64 {
    match self.snapshot() {
      Some(display) if display.scale > 0.0 => display.scale,
      // Snapshot missing or unreadable — fall back to the default display's
      // density (pre-#106 behavior).
      _ => self.app.scale() as f64,
    }
  }

  pub fn video_modes(&self) -> impl Iterator<Item = monitor::VideoMode> {
    let size = self.size().into();
    // Single-item disclosure (committer Eulogizethesun/tauri#183): OHOS
    // exposes exactly one "current mode" per display — the SDK has no
    // per-display video-mode list API and no color-depth API (only the
    // wide-gamut `colorSpaces` capability, which is not a mode/bit-depth
    // surface), so bit_depth is reported as the fixed 32 (RGBA8888).
    // Multi-display enumeration/sizes/refresh rates are real values from
    // DisplayManager (see ohos-monitor-real-values, issue #106).
    // refresh_rate from OHOS DisplayManager real value (see ohos-monitor-real-values).
    // bit_depth fixed at 32 (RGBA8888) — OHOS exposes no per-display color
    // depth; see ohos-monitor-degradation.
    let refresh_rate = self
      .snapshot()
      .map(|display| display.refresh_rate as u16)
      .unwrap_or_else(|| self.app.refresh_rate() as u16);
    std::iter::once(monitor::VideoMode {
      video_mode: VideoMode {
        size,
        bit_depth: 32,
        refresh_rate,
        monitor: self.clone(),
      },
    })
  }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct VideoMode {
  size: (u32, u32),
  bit_depth: u16,
  refresh_rate: u16,
  monitor: MonitorHandle,
}

impl VideoMode {
  pub fn size(&self) -> PhysicalSize<u32> {
    self.size.into()
  }

  pub fn bit_depth(&self) -> u16 {
    self.bit_depth
  }

  pub fn refresh_rate(&self) -> u16 {
    self.refresh_rate
  }

  pub fn monitor(&self) -> monitor::MonitorHandle {
    monitor::MonitorHandle {
      inner: self.monitor.clone(),
    }
  }
}
