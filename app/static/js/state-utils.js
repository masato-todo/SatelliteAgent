// state-utils.js — global state, constants, DOM helpers, formatters

export const BUDGET_MAX = 5_000_000;
export const VIEW_DEBOUNCE_MS = 500;

export const DAMAGE_STYLE = {
  "destroyed":    { color: "#ff3b3b", weight: 2, fill: false, opacity: 0.95 },
  "major-damage": { color: "#ff9933", weight: 2, fill: false, opacity: 0.9  },
  "minor-damage": { color: "#ffe14a", weight: 1, fill: false, opacity: 0.7  },
};

export const SPECTRAL_PER_SIDE_TOOLS = new Set(["fetch_band", "false_color", "compute_index"]);

export const state = {
  beforeKey: null,
  afterKey: null,
  beforeMeta: {},
  afterMeta: {},
  mapBefore: null,
  mapAfter: null,
  overlayBefore: null,
  overlayAfter: null,
  template: null,
  dm3: null,
  syncing: false,
  eventSource: null,
  // Annotation mode
  annotating: false,
  traceEvents: [],
  traceFinal: null,
  // Trace emitter callback (set by annotate-traces.js so tools.js can push
  // events without an import cycle).
  traceEmitter: null,
  // Bbox drawing state
  drawingBbox: false,
  selectedBbox: null,
  bboxRectBefore: null,
  bboxRectAfter: null,
  drawStartLatLng: null,
  // xBD damage overlay
  damageLayerBefore: null,
  damageLayerAfter: null,
  damageVisible: true,
  imgW: null,
  imgH: null,
  // Pan/zoom tracking
  viewDebounceTimer: null,
  lastViewSnapshot: null,
};

// ---- DOM helpers ----

export function $(id) { return document.getElementById(id); }

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

export function setStatus(text) { $("status").textContent = text; }

export function updateBudget(bytes) {
  const pct = Math.max(0, Math.min(100, (bytes / BUDGET_MAX) * 100));
  $("budget-fill").style.width = pct + "%";
  $("budget-text").textContent = `${bytes.toLocaleString()} bytes remaining`;
}

export function round2(x) { return Math.round(x * 100) / 100; }

// ---- Formatters (pure, no DOM) ----

export function labelFor(side, meta) {
  const m = meta || {};
  if (m.image_available === false) return `${side} — NO IMAGE`;
  const parts = [];
  if (m.datetime) parts.push(m.datetime.slice(0, 10));

  const stats = m.stats;
  if (stats && !stats.error) {
    const cp = stats.cloud_proxy;
    const nd = stats.nodata_fraction;
    const tag = cp < 0.2 ? "clear" : cp < 0.5 ? "ok" : "CLOUDY";
    parts.push(`cloud ${cp.toFixed(2)} ${tag}`);
    if (nd !== undefined && nd > 0.05) {
      // Only show when notable — keeps label terse for fully-covered images
      parts.push(`nodata ${(nd * 100).toFixed(0)}%`);
    }
    if (stats.usable === false) parts.push("NOT usable");
  } else if (m.cloud_cover !== undefined) {
    const cc = Number(m.cloud_cover);
    parts.push(`tile_cloud ${cc.toFixed(1)}%`);
  }
  return parts.length ? `${side} · ${parts.join(" · ")}` : side;
}

export function parseBbox(str) {
  if (!str) return null;
  const m = str.match(/\[?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?/);
  if (!m) return null;
  return [Number(m[1]), Number(m[2]), Number(m[3]), Number(m[4])];
}
