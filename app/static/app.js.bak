// SatelliteAgent Mission Control — frontend

const BUDGET_MAX = 5_000_000;

const state = {
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
  // Bbox drawing state
  drawingBbox: false,
  selectedBbox: null,       // [x, y, w, h] in pixel coords of the "after" image
  bboxRectBefore: null,      // L.Rectangle on before map (sync)
  bboxRectAfter: null,       // L.Rectangle on after map (interactive)
  drawStartLatLng: null,
  // xBD damage overlay
  damageLayerBefore: null,  // L.LayerGroup on before map
  damageLayerAfter: null,   // L.LayerGroup on after map
  damageVisible: true,      // user toggle for damage polygon overlay
  imgW: null,               // current image width (px)
  imgH: null,               // current image height (px)
  // Pan/zoom tracking debounce timers, set in initMaps()
  viewDebounceTimer: null,
  lastViewSnapshot: null,
};

function labelFor(side, meta) {
  const m = meta || {};
  if (m.image_available === false) return `${side} — NO IMAGE`;
  const parts = [];
  if (m.datetime) parts.push(m.datetime.slice(0, 10));

  const stats = m.stats;
  if (stats && !stats.error) {
    const cp = stats.cloud_proxy;
    const ed = stats.edge_density;
    const tag = cp < 0.2 ? "clear" : cp < 0.5 ? "ok" : "CLOUDY";
    parts.push(`cloud_proxy ${cp.toFixed(2)} ${tag}`);
    if (ed !== undefined) parts.push(`edges ${ed.toFixed(1)}`);
    if (stats.usable === false) parts.push("NOT usable");
  } else if (m.cloud_cover !== undefined) {
    // Fallback to tile cloud if stats missing (shouldn't happen for new fetches)
    const cc = Number(m.cloud_cover);
    parts.push(`tile_cloud ${cc.toFixed(1)}%`);
  }
  return parts.length ? `${side} · ${parts.join(" · ")}` : side;
}

// ---- Utilities ----

function $(id) { return document.getElementById(id); }

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => (
    {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]
  ));
}

function setStatus(text) { $("status").textContent = text; }

function updateBudget(bytes) {
  const pct = Math.max(0, Math.min(100, (bytes / BUDGET_MAX) * 100));
  $("budget-fill").style.width = pct + "%";
  $("budget-text").textContent = `${bytes.toLocaleString()} bytes remaining`;
}

// ---- Templates ----

async function loadDM3Cases() {
  try {
    const res = await fetch("/api/disasterm3/cases");
    if (!res.ok) return;
    const data = await res.json();
    const sel = $("dm3-case");
    if (!sel) return;
    sel.innerHTML = '<option value="">— none —</option>';

    const cases = data.cases || [];
    const negativeCases = cases.filter(c => c.is_negative);
    const preciseCases  = cases.filter(c => c.precise && !c.is_negative);
    const coarseCases   = cases.filter(c => !c.precise && !c.is_negative);

    const makeOption = (c, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      let label = `[${c.source}] ${c.event} · ${c.disaster_type} · ${c.capture_date}`;
      if (c.damage && c.damage.destroyed + c.damage.major > 0) {
        label += `  (${c.damage.destroyed} destroyed + ${c.damage.major} major)`;
      }
      opt.textContent = label;
      opt.dataset.case = JSON.stringify(c);
      return opt;
    };

    // Keep the case → original-index mapping stable (used by onDM3Change via dataset.case,
    // but value must still be unique). We use the array index in the full `cases` list.
    if (preciseCases.length) {
      const grpP = document.createElement("optgroup");
      grpP.label = "✅ PRECISE — per-image centroid + real pre/post + damage overlay";
      preciseCases.forEach(c => grpP.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpP);
    }
    if (coarseCases.length) {
      const grpC = document.createElement("optgroup");
      grpC.label = "⚠ COARSE — event-level lat/lon (AOI may miss damage, no overlay)";
      coarseCases.forEach(c => grpC.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpC);
    }
    if (negativeCases.length) {
      const grpN = document.createElement("optgroup");
      grpN.label = "⊘ NEGATIVE — drop expected (no_change / cloud_blocked / random)";
      negativeCases.forEach(c => grpN.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpN);
    }
  } catch (e) {
    console.error("DM3 load failed", e);
  }
}

function onDM3Change() {
  const opt = $("dm3-case").selectedOptions[0];
  if (!opt || !opt.value) { $("dm3-gt").innerHTML = ""; state.dm3 = null; clearDamageOverlay(); return; }
  const c = JSON.parse(opt.dataset.case);
  state.dm3 = c;
  clearDamageOverlay();
  // Populate form fields
  $("lat").value = c.lat;
  $("lon").value = c.lon;
  $("before_date").value = c.before_date;
  $("after_date").value  = c.after_date;
  if (c.size_km) {
    $("size_km").value = c.size_km;
    const lbl = $("size_km_val");
    if (lbl) lbl.textContent = c.size_km;
  }
  $("template").value = "";
  state.template = c.id;

  // GT panel
  const lines = [];
  const precisionTag = c.precise
    ? '<span class="gt-precise">✅ PRECISE — xBD per-image centroid + damage overlay</span>'
    : '<span class="gt-imprecise">⚠ COARSE — event-level (AOI may miss damage, no overlay)</span>';
  lines.push(precisionTag);
  lines.push(`<span class="gt-label">GT:</span> ${escapeHtml(c.disaster_type)} → ${escapeHtml(c.mapped_class)}`);
  const eventLine = c.event_name
    ? `${escapeHtml(c.event)} — ${escapeHtml(c.event_name)} (${escapeHtml(c.source)})`
    : `${escapeHtml(c.event)} (${escapeHtml(c.source)})`;
  lines.push(`<span class="gt-label">Event:</span> ${eventLine}`);
  if (c.event_start && c.event_end) {
    const periodTxt = (c.event_start === c.event_end)
      ? c.event_start
      : `${c.event_start} to ${c.event_end}`;
    lines.push(`<span class="gt-label">Disaster period:</span> ${escapeHtml(periodTxt)}`);
  }
  lines.push(`<span class="gt-label">Location:</span> ${escapeHtml(c.location)}`);
  const dateLine = c.precise
    ? `${escapeHtml(c.before_date)} → ${escapeHtml(c.after_date)} (Before=xBD pre, After=xBD post +14d)`
    : `${escapeHtml(c.capture_date)} (before = -30d)`;
  lines.push(`<span class="gt-label">Image dates:</span> ${dateLine}`);
  if (c.damage) {
    const d = c.damage;
    lines.push(
      `<span class="gt-label">Damage:</span> ${d.destroyed} destroyed · ${d.major} major · ${d.minor} minor · ${d.no_damage} ok (${d.total} bldgs)`
    );
  }
  $("dm3-gt").innerHTML = lines.join("\n");
  setStatus(`DisasterM3 case loaded: ${c.event}${c.precise ? " (precise)" : ""} — now press Fetch Images`);
}

async function loadTemplates() {
  const res = await fetch("/api/templates");
  const data = await res.json();
  $("simsat-url").textContent = data.simsat_url;
  const p = data.provider || { kind: "none", model: "?" };
  const el = $("provider-info");
  el.textContent = p.kind === "gemini" ? `Gemini · ${p.model}` : "NO PROVIDER (set GOOGLE_API_KEY)";
  el.className = p.kind === "gemini" ? "provider-gemini" : "provider-stub";
  const sel = $("template");
  sel.innerHTML = '<option value="">── custom ──</option>';
  Object.entries(data.templates).forEach(([name, t]) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    opt.dataset.payload = JSON.stringify(t);
    sel.appendChild(opt);
  });
  // Default select
  if (sel.options.length > 1) {
    sel.selectedIndex = 1;
    onTemplateChange();
  }
  sel.addEventListener("change", onTemplateChange);
}

function onTemplateChange() {
  const opt = $("template").selectedOptions[0];
  state.template = opt ? opt.value : null;
  if (!opt || !opt.dataset.payload) return;
  const t = JSON.parse(opt.dataset.payload);
  if (t.lat !== undefined)      $("lat").value = t.lat;
  if (t.lon !== undefined)      $("lon").value = t.lon;
  if (t.before)                 $("before_date").value = t.before;
  if (t.after)                  $("after_date").value = t.after;
  if (t.size_km !== undefined)  {
    $("size_km").value = t.size_km;
    $("size_km_val").textContent = t.size_km;
  }
}

// ---- Maps ----

function initMaps() {
  const common = {
    crs: L.CRS.Simple,
    minZoom: -5,
    maxZoom: 8,
    attributionControl: false,
    zoomControl: false,
    zoomSnap: 0.25,
    zoomDelta: 0.5,
    wheelPxPerZoomLevel: 80,
  };
  state.mapBefore = L.map("map-before", common);
  state.mapAfter  = L.map("map-after",  common);
  L.control.zoom({ position: "topright" }).addTo(state.mapBefore);
  L.control.zoom({ position: "topright" }).addTo(state.mapAfter);

  // Sync pan & zoom
  const sync = (src, dst) => {
    src.on("move zoom", () => {
      if (state.syncing) return;
      state.syncing = true;
      dst.setView(src.getCenter(), src.getZoom(), { animate: false });
      state.syncing = false;
    });
  };
  sync(state.mapBefore, state.mapAfter);
  sync(state.mapAfter,  state.mapBefore);

  // Pan/zoom tracking — only emits trace events while annotating, debounced so
  // that a continuous gesture collapses to a single "view" event after the
  // user stops moving. The After map is the primary; Before mirrors it via sync.
  state.mapAfter.on("moveend zoomend", () => trackViewChange("after"));
  state.mapBefore.on("moveend zoomend", () => trackViewChange("before"));
}

const VIEW_DEBOUNCE_MS = 500;

function trackViewChange(side) {
  if (!state.annotating) return;
  if (state.syncing) return;  // ignore propagated sync events; primary side wins
  if (state.viewDebounceTimer) clearTimeout(state.viewDebounceTimer);
  state.viewDebounceTimer = setTimeout(() => emitViewSnapshot(side), VIEW_DEBOUNCE_MS);
}

function emitViewSnapshot(side) {
  state.viewDebounceTimer = null;
  const map = side === "before" ? state.mapBefore : state.mapAfter;
  if (!map) return;
  const c = map.getCenter();
  const z = map.getZoom();
  const b = map.getBounds();
  // Round so noise from sub-pixel pan doesn't generate spurious distinct events.
  const snap = {
    side,
    center: [round2(c.lat), round2(c.lng)],
    zoom: Number(z.toFixed(2)),
    bounds: [
      [round2(b.getSouth()), round2(b.getWest())],
      [round2(b.getNorth()), round2(b.getEast())],
    ],
  };
  // Suppress duplicates (same snapshot as previous emission).
  const key = JSON.stringify(snap);
  if (state.lastViewSnapshot === key) return;
  state.lastViewSnapshot = key;
  pushAndRender({type: "view", ...snap});
}

function round2(x) { return Math.round(x * 100) / 100; }

function loadImageSize(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = reject;
    img.src = url;
  });
}

async function setImage(which, key) {
  const map     = which === "before" ? state.mapBefore : state.mapAfter;
  const oldOvl  = which === "before" ? state.overlayBefore : state.overlayAfter;
  if (oldOvl) map.removeLayer(oldOvl);
  if (!key) { return; }
  const url = `/api/image/${key}`;
  const { w, h } = await loadImageSize(url);
  const bounds = [[0, 0], [h, w]];
  const overlay = L.imageOverlay(url, bounds, { interactive: false }).addTo(map);
  if (which === "before") state.overlayBefore = overlay;
  else                    state.overlayAfter  = overlay;
  state.imgW = w;
  state.imgH = h;
  map.fitBounds(bounds);
}

// ---- xBD damage overlay (precise cases only) ----

function clearDamageOverlay() {
  if (state.damageLayerBefore) { state.mapBefore.removeLayer(state.damageLayerBefore); state.damageLayerBefore = null; }
  if (state.damageLayerAfter)  { state.mapAfter.removeLayer(state.damageLayerAfter);   state.damageLayerAfter  = null; }
}

// WGS84 (lat, lon) → Leaflet CRS.Simple coords relative to an image of size (W,H)
// AOI is centered at (aoiLat, aoiLon) spanning sizeKm on each side.
function wgs84ToLeaflet(lat, lon, aoiLat, aoiLon, sizeKm, W, H) {
  const dyKm = (lat - aoiLat) * 110.574;
  const dxKm = (lon - aoiLon) * 111.320 * Math.cos(aoiLat * Math.PI / 180);
  const yFrac = 0.5 + dyKm / sizeKm;    // 0 at south edge, 1 at north edge
  const xFrac = 0.5 + dxKm / sizeKm;    // 0 at west edge, 1 at east edge
  return [yFrac * H, xFrac * W];
}

const DAMAGE_STYLE = {
  "destroyed":    { color: "#ff3b3b", weight: 2, fill: false, opacity: 0.95 },
  "major-damage": { color: "#ff9933", weight: 2, fill: false, opacity: 0.9  },
  "minor-damage": { color: "#ffe14a", weight: 1, fill: false, opacity: 0.7  },
};

async function loadDamageOverlay() {
  clearDamageOverlay();
  // Always reset both labels to their canonical base FIRST — otherwise repeated
  // calls (fetch, then candidate apply) accumulate "· GT: ... · GT: ..." duplicates.
  if (state.beforeMeta) setMapLabel("before", labelFor("Before", state.beforeMeta));
  if (state.afterMeta)  setMapLabel("after",  labelFor("After",  state.afterMeta));
  if (!state.damageVisible) return;
  if (!state.dm3 || !state.dm3.precise) return;
  if (!state.imgW || !state.imgH) return;

  const aoiLat = parseFloat($("lat").value);
  const aoiLon = parseFloat($("lon").value);
  const sizeKm = parseFloat($("size_km").value);
  const params = new URLSearchParams({
    event: state.dm3.event,
    lat:   String(aoiLat),
    lon:   String(aoiLon),
    size_km: String(sizeKm),
  });

  let data;
  try {
    const res = await fetch(`/api/xbd/damage_overlay?${params}`);
    if (!res.ok) return;
    data = await res.json();
  } catch (e) {
    console.warn("damage_overlay fetch failed", e);
    return;
  }
  const polys = data.polygons || [];
  if (!polys.length) return;

  const beforeGroup = L.layerGroup();
  const afterGroup  = L.layerGroup();
  for (const p of polys) {
    const latlngs = p.points.map(([lat, lon]) =>
      wgs84ToLeaflet(lat, lon, aoiLat, aoiLon, sizeKm, state.imgW, state.imgH)
    );
    const style = DAMAGE_STYLE[p.subtype] || DAMAGE_STYLE["destroyed"];
    L.polygon(latlngs, style).addTo(beforeGroup);
    L.polygon(latlngs, style).addTo(afterGroup);
  }
  beforeGroup.addTo(state.mapBefore);
  afterGroup.addTo(state.mapAfter);
  state.damageLayerBefore = beforeGroup;
  state.damageLayerAfter  = afterGroup;

  // Append damage counts to map labels (additive — setMapLabel already ran)
  const c = data.counts || {};
  const pieces = [];
  if (c["destroyed"])    pieces.push(`${c["destroyed"]} destroyed`);
  if (c["major-damage"]) pieces.push(`${c["major-damage"]} major`);
  if (c["minor-damage"]) pieces.push(`${c["minor-damage"]} minor`);
  if (pieces.length) {
    const tag = ` · GT: ${pieces.join(", ")}`;
    for (const side of ["before", "after"]) {
      const el = document.querySelector(`#map-${side}`).parentElement.querySelector(".map-label");
      if (el) el.textContent = (el.textContent || "") + tag;
    }
  }
}

// ---- Fetch ----

function fmtMeta(side, part) {
  const m = part.meta || {};
  if (m.error) return `✗ ${side}: ${m.error}`;
  const bits = [];
  bits.push(m.cached ? "cached" : "fetched");
  if (m.datetime) bits.push(m.datetime);
  const s = m.stats;
  if (s && !s.error) {
    bits.push(`cloud_proxy=${s.cloud_proxy.toFixed(2)}`);
    bits.push(`edges=${s.edge_density.toFixed(1)}`);
    bits.push(`bright=${s.brightness_mean.toFixed(0)}`);
    bits.push(s.usable ? "usable" : "NOT_usable");
  } else if (m.cloud_cover !== undefined) {
    bits.push(`tile_cloud=${Number(m.cloud_cover).toFixed(1)}%`);
  }
  if (m.source) bits.push(m.source);
  return `✓ ${side} (${part.date}): ${bits.join(" · ")}`;
}

function setLoading(which, opts = {}) {
  const el = $(`loading-${which}`);
  if (!el) return;
  if (opts.hide) {
    el.classList.remove("active", "error");
    return;
  }
  el.classList.toggle("active", !!opts.loading);
  el.classList.toggle("error", !!opts.error);
  const msg = el.querySelector("span");
  if (msg) msg.textContent = opts.message || (opts.loading ? "Loading Sentinel-2..." : "");
}

function setFetching(isFetching) {
  $("fetch-btn").disabled = isFetching;
  $("run-btn").disabled = isFetching;
}

async function fetchImages() {
  const payload = {
    lat:  parseFloat($("lat").value),
    lon:  parseFloat($("lon").value),
    before_date: $("before_date").value,
    after_date:  $("after_date").value,
    size_km:     parseFloat($("size_km").value),
    window_days: parseInt($("window_days").value, 10),
  };
  setStatus(`Fetching...  lat=${payload.lat}, lon=${payload.lon}, size=${payload.size_km}km`);
  setLoading("before", { loading: true });
  setLoading("after",  { loading: true });
  setFetching(true);
  try {
    const res = await fetch("/api/fetch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const msg = `HTTP ${res.status}: ${await res.text()}`;
      setStatus(msg);
      setLoading("before", { error: true, message: msg });
      setLoading("after",  { error: true, message: msg });
      return;
    }
    const data = await res.json();
    setStatus(`${fmtMeta("before", data.before)}\n${fmtMeta("after", data.after)}`);

    state.beforeKey = data.before.key;
    state.afterKey  = data.after.key;
    state.beforeMeta = data.before.meta || {};
    state.afterMeta  = data.after.meta  || {};

    if (state.beforeKey) await setImage("before", state.beforeKey);
    if (state.afterKey)  await setImage("after",  state.afterKey);

    setMapLabel("before", labelFor("Before", state.beforeMeta));
    setMapLabel("after",  labelFor("After",  state.afterMeta));

    await loadDamageOverlay();

    setLoading("before", { hide: true });
    setLoading("after",  { hide: true });
    if (!state.beforeKey) setLoading("before", { error: true, message: data.before.meta.error || "no image" });
    if (!state.afterKey)  setLoading("after",  { error: true, message: data.after.meta.error  || "no image" });
  } catch (e) {
    const msg = `Fetch failed: ${e.message}`;
    setStatus(msg);
    setLoading("before", { error: true, message: msg });
    setLoading("after",  { error: true, message: msg });
  } finally {
    setFetching(false);
  }
}

// ---- Agent (SSE) ----

function clearTrace() { $("trace").innerHTML = ""; }

function renderAttentionBody(ev) {
  if (ev.action === "draw_bbox") {
    const b = ev.bbox || [];
    return `${escapeHtml(ev.side || "?")} bbox=[${b.join(",")}]`;
  }
  if (ev.action === "clear") {
    return `${escapeHtml(ev.side || "?")} bbox cleared`;
  }
  return escapeHtml(JSON.stringify(ev));
}

function renderViewBody(ev) {
  const c = ev.center || [];
  const z = ev.zoom !== undefined ? ev.zoom : "?";
  return `${escapeHtml(ev.side || "?")} center=[${c.join(",")}] zoom=${z}`;
}

function renderTraceEvent(ev) {
  const type = ev.type || "?";
  let body = "";
  if (type === "thought")          body = escapeHtml(ev.text || "");
  else if (type === "action")      body = `<code>${escapeHtml(ev.name)}(${escapeHtml(JSON.stringify(ev.arguments || {}))})</code>`;
  else if (type === "observation") body = renderObservationBody(ev);
  else if (type === "attention")   body = renderAttentionBody(ev);
  else if (type === "view")        body = renderViewBody(ev);
  else if (type === "error")       body = escapeHtml(ev.text || "");
  else if (type === "final")       body = `<b>[END]</b> final: <code>${escapeHtml(ev.name || "?")}</code>`;
  else body = escapeHtml(JSON.stringify(ev));

  const line = document.createElement("div");
  line.className = "line";
  line.innerHTML = `<span class="tag tag-${type}">${type}</span>${body}`;
  const trace = $("trace");
  trace.appendChild(line);
  trace.scrollTop = trace.scrollHeight;
}

function obsTopSignal(toolName, result) {
  // Single-line "headline" per tool — what the agent should care about most.
  if (toolName === "get_change_stats") {
    const idx = result.indices || {};
    // Pick the index with the largest |strong↓ or strong↑|.
    let best = null;
    for (const [name, s] of Object.entries(idx)) {
      const dec = s.frac_strong_decrease || 0;
      const inc = s.frac_strong_increase || 0;
      const mag = Math.max(dec, inc);
      if (!best || mag > best.mag) {
        best = {name, dec, inc, mag, dir: dec >= inc ? "↓" : "↑", val: dec >= inc ? dec : inc};
      }
    }
    if (best) {
      return `top: ${best.name} strong${best.dir}=${(best.val * 100).toFixed(1)}% (${Object.keys(idx).length} indices)`;
    }
    return "no indices";
  }
  if (toolName === "classify_change") {
    const c = (result.classes || [])[0];
    if (c) return `${c.name || "?"} ${((c.confidence || 0) * 100).toFixed(0)}%`;
    return "no class";
  }
  if (toolName === "compute_index_delta") {
    const s = result.stats || {};
    const dec = s.frac_decrease_strong;
    const inc = s.frac_increase_strong;
    return `Δ${result.index || "?"}  strong↓=${(dec * 100).toFixed(1)}%  strong↑=${(inc * 100).toFixed(1)}%`;
  }
  if (toolName === "compute_index" || toolName === "false_color" || toolName === "fetch_band") {
    return `${toolName}: map updated`;
  }
  if (toolName === "zoom_in") return `zoom ${result.zoom_ratio || ""}x applied`;
  if (toolName === "compute_area") return `area_km2 = ${result.area_km2 ?? "?"}`;
  if (toolName === "get_region_info") return `region=${result.region || "?"}`;
  // Fallback: short JSON preview
  const j = JSON.stringify(result);
  return j.length > 100 ? j.slice(0, 100) + "…" : j;
}

function renderObservationBody(ev) {
  const result = ev.result ?? null;
  if (!result || typeof result !== "object") {
    return `<code>${escapeHtml(JSON.stringify(result))}</code>`;
  }
  const toolName = ev.name || "";
  const summary = obsTopSignal(toolName, result);
  const fullJson = JSON.stringify(result, null, 2);
  const summaryHtml = `<span class="obs-summary">${escapeHtml(summary)}</span>`;
  const detailsHtml =
    `<details class="obs-raw">` +
      `<summary>details</summary>` +
      `<pre><code>${escapeHtml(fullJson)}</code></pre>` +
    `</details>`;
  let extra = "";
  if (result.zoomed_before_key && result.zoomed_after_key) {
    extra = `<div class="zoom-hint">↑ zoomed view applied to the image panels above</div>`;
  }
  return `${summaryHtml}${detailsHtml}${extra}`;
}

function setMapLabel(which, text) {
  const el = document.querySelector(`#map-${which}`).parentElement.querySelector(".map-label");
  if (el) el.textContent = text;
}

async function resetMapsToOriginal() {
  if (state.beforeKey) {
    await setImage("before", state.beforeKey);
    setMapLabel("before", labelFor("Before", state.beforeMeta));
  }
  if (state.afterKey) {
    await setImage("after", state.afterKey);
    setMapLabel("after", labelFor("After", state.afterMeta));
  }
  await loadDamageOverlay();
}

function appendInbox(event) {
  // Inbox panel removed from UI; keep function as a no-op so callers don't break.
  if (!document.querySelector("#inbox tbody")) return;
  const res = event.result || {};
  const tbody = document.querySelector("#inbox tbody");
  const tr = document.createElement("tr");
  const region = state.template
    ? (state.template.split(":")[1] || state.template).trim()
    : `${$("lat").value},${$("lon").value}`;
  const type = state.template ? state.template.split(":")[0] : "event";
  tr.innerHTML = `
    <td><code>${escapeHtml(res.report_id || "?")}</code></td>
    <td>${escapeHtml(type)}</td>
    <td>${escapeHtml(region)}</td>
    <td>-</td>
    <td>${res.attached ? "420 KB" : "2 KB"}</td>
  `;
  tbody.appendChild(tr);
}

function runAgent() {
  if (!state.beforeKey || !state.afterKey) {
    setStatus("Fetch images first.");
    return;
  }
  if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
  clearTrace();
  updateBudget(BUDGET_MAX);
  resetMapsToOriginal();

  const url = `/api/run_agent?before_key=${encodeURIComponent(state.beforeKey)}&after_key=${encodeURIComponent(state.afterKey)}`;
  const es = new EventSource(url);
  state.eventSource = es;

  es.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }
    renderTraceEvent(ev);

    if (ev.type === "observation") {
      const r = ev.result || {};

      // Budget tool: update the bar
      if (ev.name === "check_downlink_budget" && r.remaining_bytes !== undefined) {
        updateBudget(r.remaining_bytes);
      }

      // Zoom tool: swap the top map panels to the zoomed pair
      if (r.zoomed_before_key && r.zoomed_after_key) {
        setImage("before", r.zoomed_before_key);
        setImage("after",  r.zoomed_after_key);
        const ratio = r.zoom_ratio ? `${r.zoom_ratio}x` : "";
        const bbox = r.crop_pixel_bbox ? JSON.stringify(r.crop_pixel_bbox) : "";
        setMapLabel("before", `Before [ZOOMED ${ratio}]`);
        setMapLabel("after",  `After [ZOOMED ${ratio} ${bbox}]`);
      }
    }

    if (ev.type === "final" && ev.name === "submit_to_ground") appendInbox(ev);
  };
  es.addEventListener("end", () => { es.close(); state.eventSource = null; });
  es.onerror = () => { /* stream ended or server closed */ };
}

// ---- Boot ----

// ---- Annotate mode ----

function setMode(recording) {
  state.annotating = recording;
  $("annotate-controls").hidden = !recording;
  $("mode-badge").textContent = recording ? "● RECORDING" : "IDLE";
  $("mode-badge").className = recording ? "mode-annotate" : "mode-agent";
  $("annotate-btn").textContent = recording ? "■ Stop Recording" : "● Start Recording";
  $("run-btn").disabled = recording;
  if (recording) {
    setAnnotateStatus("Recording started. Tool actions will be traced.");
  } else {
    setAnnotateStatus("");
  }
}

function setAnnotateStatus(msg) {
  $("annotate-status").textContent = msg;
}

async function toggleRecording() {
  if (state.annotating) {
    // Auto-save on Stop. If there are events, persist before clearing state.
    if (state.traceEvents.length > 0) {
      await saveTrace();
    } else {
      setAnnotateStatus("No events recorded — nothing to save.");
    }
    discardTrace();
    setMode(false);
  } else {
    if (!state.beforeKey || !state.afterKey) {
      setStatus("Fetch images first before recording.");
      return;
    }
    state.traceEvents = [];
    state.traceFinal = null;
    clearTrace();
    updateBudget(BUDGET_MAX);
    // Pre-fill scenario id from current template
    $("annotate-scenario").value = state.template || "";
    setMode(true);
  }
}

function pushAndRender(ev) {
  state.traceEvents.push(ev);
  renderTraceEvent(ev);
}

function consumeThought() {
  const t = $("annotate-thought").value.trim();
  if (t) {
    pushAndRender({type: "thought", text: t});
    $("annotate-thought").value = "";
  }
}

function parseBbox(str) {
  const parts = str.split(/[,\s]+/).filter(Boolean).map(Number);
  if (parts.length !== 4 || parts.some(n => !Number.isFinite(n))) return null;
  return parts.map(n => Math.round(n));
}

async function invokeTool(toolName, args) {
  if (!state.beforeKey || !state.afterKey) {
    setAnnotateStatus("images not ready");
    return null;
  }
  const res = await fetch("/api/tool/invoke", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      before_key: state.beforeKey,
      after_key:  state.afterKey,
      tool_name:  toolName,
      arguments:  args,
    }),
  });
  if (!res.ok) {
    setAnnotateStatus(`HTTP ${res.status}: ${await res.text()}`);
    return null;
  }
  const data = await res.json();
  return data.observation;
}

function setToolsStatus(msg) {
  // Don't mirror tool-call results to annotate-status — the trace events
  // below already show them. Keep annotate-status only for recording-state
  // messages ("Recording started", "Saved: ...").
  const t = $("tools-status"); if (t) t.textContent = msg;
}

function formatToolResult(toolName, obs) {
  if (!obs) return "(no result)";
  if (obs.error) {
    let msg = `ERROR: ${obs.error}`;
    if (obs.raw_preview !== undefined) msg += `\n  raw(${obs.raw_len}ch): ${obs.raw_preview}`;
    if (obs.raw !== undefined) msg += `\n  raw: ${obs.raw}`;
    return msg;
  }

  if (toolName === "classify_change") {
    const classes = (obs.classes || []).slice(0, 3).map(c => {
      // Accept both {name, confidence} (Gemini) and {flood: 0.62} (stub) forms
      if (c && c.name !== undefined) {
        const conf = (c.confidence ?? 0) * 100;
        return `${c.name} ${conf.toFixed(0)}%`;
      }
      const k = Object.keys(c || {})[0];
      return k ? `${k} ${((c[k] || 0) * 100).toFixed(0)}%` : "?";
    }).join(", ");
    const bb = (obs.bboxes && obs.bboxes.length) ? ` | bbox ${JSON.stringify(obs.bboxes[0])}` : "";
    const src = obs.source ? ` (${obs.source})` : "";
    return `[${classes}]${bb}${src}`;
  }
  if (toolName === "get_change_stats") {
    const idx = obs.indices || {};
    const lines = [];
    for (const name of ["NBR", "NDVI", "MNDWI", "NDBI"]) {
      const s = idx[name];
      if (!s) continue;
      lines.push(
        `    ${name.padEnd(5)}  mean=${s.mean.toFixed(3)}  median=${s.median.toFixed(3)}  ` +
        `range=[${s.min.toFixed(2)},${s.max.toFixed(2)}]  ` +
        `strong↓=${(s.frac_strong_decrease*100).toFixed(1)}%  ` +
        `strong↑=${(s.frac_strong_increase*100).toFixed(1)}%`
      );
    }
    return `\n${lines.join("\n")}`;
  }
  if (toolName === "get_region_info") {
    const infra = Array.isArray(obs.infra_nearby) ? obs.infra_nearby.join(",") : "";
    return `region=${obs.region} country=${obs.country} populated=${obs.populated} infra=[${infra}]`;
  }
  if (toolName === "compute_area") {
    return `area_km2 = ${obs.area_km2 ?? "?"}`;
  }
  if (toolName === "fetch_band") {
    const cc = obs.cloud_cover !== undefined ? ` cloud ${Number(obs.cloud_cover).toFixed(0)}%` : "";
    return `band=${obs.band}${cc} → map updated`;
  }
  if (toolName === "false_color") {
    const bands = (obs.bands_rgb || []).join("/");
    return `FCC=${bands} → map updated`;
  }
  if (toolName === "compute_index") {
    const s = obs.stats || {};
    const mean = s.mean !== undefined ? `mean=${Number(s.mean).toFixed(2)}` : "";
    const minmax = (s.min !== undefined && s.max !== undefined) ? `range=[${Number(s.min).toFixed(2)},${Number(s.max).toFixed(2)}]` : "";
    return `${obs.index} → map updated  ${mean} ${minmax}`;
  }
  if (toolName === "compute_index_delta") {
    const s = obs.stats || {};
    const mean = s.mean !== undefined ? `Δmean=${Number(s.mean).toFixed(3)}` : "";
    const lo = s.min !== undefined ? Number(s.min).toFixed(2) : "?";
    const hi = s.max !== undefined ? Number(s.max).toFixed(2) : "?";
    const decFrac = s.frac_decrease_strong !== undefined ? `strong↓=${(s.frac_decrease_strong * 100).toFixed(1)}%` : "";
    const incFrac = s.frac_increase_strong !== undefined ? `strong↑=${(s.frac_increase_strong * 100).toFixed(1)}%` : "";
    const bts = obs.before_datetime ? obs.before_datetime.slice(0, 10) : "?";
    const ats = obs.after_datetime  ? obs.after_datetime.slice(0, 10)  : "?";
    let warn = "";
    if (obs.before_datetime && obs.after_datetime && obs.before_datetime === obs.after_datetime) {
      warn = " ⚠ before_ts == after_ts (same scene fetched twice)";
    }
    return `Δ${obs.index} ${bts} → ${ats}  ${mean}  range=[${lo},${hi}]  ${decFrac}  ${incFrac}${warn}`;
  }
  if (toolName === "zoom_in") {
    const r = obs.zoom_ratio ? `${obs.zoom_ratio}x` : "";
    const b = obs.crop_pixel_bbox ? JSON.stringify(obs.crop_pixel_bbox) : "";
    return `zoom ${r} ${b} → maps updated`;
  }
  return JSON.stringify(obs).slice(0, 240);
}

const SPECTRAL_PER_SIDE_TOOLS = new Set(["fetch_band", "false_color", "compute_index"]);

async function runTool_perSide(toolName, side) {
  let args;
  if (toolName === "fetch_band") {
    args = {band: $("fetch-band-sel").value, which: side};
  } else if (toolName === "false_color") {
    const bands = $("fc-preset").value.split(/[,\s]+/).filter(Boolean);
    if (bands.length !== 3) { setToolsStatus("false_color: preset malformed"); return; }
    args = {bands, which: side};
  } else if (toolName === "compute_index") {
    args = {index: $("index-sel").value, which: side};
  } else {
    return;
  }
  if (state.annotating) {
    consumeThought();
    pushAndRender({type: "action", name: toolName, arguments: args});
  }
  setToolsStatus(`${toolName} (${side}): running...`);
  const observation = await invokeTool(toolName, args);
  if (observation == null) {
    setToolsStatus(`${toolName} (${side}): invocation failed (no response)`);
    return;
  }
  if (state.annotating) {
    pushAndRender({type: "observation", name: toolName, result: observation});
  }
  if (observation.error) {
    setToolsStatus(`${toolName} (${side}): ERROR ${observation.error}`);
    return;
  }
  if (observation.image_key) {
    await setImage(side, observation.image_key);
    let label = side === "before" ? "Before" : "After";
    if (toolName === "fetch_band") label += ` · band=${observation.band}`;
    else if (toolName === "false_color") label += ` · FC=${(observation.bands_rgb || []).join("/")}`;
    else if (toolName === "compute_index") label += ` · ${observation.index}`;
    setMapLabel(side, label);
  }
  const summary = formatToolResult(toolName, observation);
  setToolsStatus(`${toolName} (${side}): ${summary}`);
}

async function runTool(toolName) {
  let args = {};
  const which = $("which-sel") ? $("which-sel").value : "after";

  // "both" on a per-side spectral tool → run twice, once per side.
  if (SPECTRAL_PER_SIDE_TOOLS.has(toolName) && which === "both") {
    await runTool_perSide(toolName, "before");
    await runTool_perSide(toolName, "after");
    return;
  }

  // Build args from UI
  if (toolName === "classify_change") {
    args = {image_before: "current_before", image_after: "current_after"};
  } else if (toolName === "get_change_stats") {
    args = {};
  } else if (toolName === "fetch_band") {
    args = {band: $("fetch-band-sel").value, which};
  } else if (toolName === "false_color") {
    const bands = $("fc-preset").value.split(/[,\s]+/).filter(Boolean);
    if (bands.length !== 3) { setToolsStatus("false_color: preset malformed"); return; }
    args = {bands, which};
  } else if (toolName === "compute_index") {
    args = {index: $("index-sel").value, which};
  } else if (toolName === "compute_index_delta") {
    args = {index: $("index-sel").value};
  } else if (toolName === "zoom_in") {
    if (!state.selectedBbox) { setToolsStatus("zoom_in: draw a bbox on the After map first"); return; }
    args = {bbox: state.selectedBbox};
  } else if (toolName === "get_region_info") {
    args = {lat: parseFloat($("lat").value), lon: parseFloat($("lon").value)};
  } else if (toolName === "compute_area") {
    if (!state.selectedBbox) { setToolsStatus("compute_area: draw a bbox on the After map first"); return; }
    args = {bbox: state.selectedBbox};
  }

  // If recording, push thought + action event first
  if (state.annotating) {
    consumeThought();
    pushAndRender({type: "action", name: toolName, arguments: args});
  }
  setToolsStatus(`${toolName}: running...`);

  const observation = await invokeTool(toolName, args);
  if (observation == null) return;

  if (state.annotating) {
    pushAndRender({type: "observation", name: toolName, result: observation});
  }

  // Side effects always apply (map updates)
  if (toolName === "zoom_in" && observation.zoomed_before_key && observation.zoomed_after_key) {
    setImage("before", observation.zoomed_before_key);
    setImage("after",  observation.zoomed_after_key);
    const ratio = observation.zoom_ratio ? `${observation.zoom_ratio}x` : "";
    const bbox = observation.crop_pixel_bbox ? JSON.stringify(observation.crop_pixel_bbox) : "";
    setMapLabel("before", `Before [ZOOMED ${ratio}]`);
    setMapLabel("after",  `After [ZOOMED ${ratio} ${bbox}]`);
  }
  if (observation.image_key) {
    // compute_index_delta always paints on the After map. Others use `which`.
    let target;
    if (toolName === "compute_index_delta") target = "after";
    else target = (which === "before") ? "before" : "after";
    setImage(target, observation.image_key);
    let label = target === "before" ? "Before" : "After";
    if (toolName === "fetch_band") label += ` · band=${observation.band}`;
    else if (toolName === "false_color") label += ` · FC=${(observation.bands_rgb || []).join("/")}`;
    else if (toolName === "compute_index") label += ` · ${observation.index}`;
    else if (toolName === "compute_index_delta") label += ` · Δ${observation.index} (red=decrease, blue=increase)`;
    setMapLabel(target, label);
  }

  const summary = formatToolResult(toolName, observation);
  let line = `${toolName}: ${summary}`;

  // DisasterM3 GT comparison for classify_change only — get_change_stats
  // returns pure stats, the agent (or human) judges the class separately.
  if (toolName === "classify_change" && state.dm3 && observation && observation.classes) {
    const expected = state.dm3.mapped_class;
    const primary = observation.classes[0] && (observation.classes[0].name || Object.keys(observation.classes[0])[0]);
    const match = primary === expected;
    line += `\n  GT: ${expected}  →  ${match ? "✓ MATCH" : "✗ MISS (primary=" + primary + ")"}`;
  }
  setToolsStatus(line);
}

function openSubmitModal() {
  $("submit-modal").hidden = false;
  // Pre-fill change_type from DM3 GT so users don't accidentally leave the
  // previous value (e.g., "flood" leaking into a santa_rosa fire trace).
  if (state.dm3 && state.dm3.mapped_class) {
    $("f-change-type").value = state.dm3.mapped_class;
  }
  $("f-description").focus();
}
function closeSubmitModal() { $("submit-modal").hidden = true; }

function onSubmitConfirm() {
  const change_type = $("f-change-type").value.trim() || "unspecified";
  const urgency = parseInt($("f-urgency").value, 10) || 0;
  const description = $("f-description").value.trim();
  const attach_image = $("f-attach").checked;

  consumeThought();
  const args = {change_type, urgency, description, attach_image};
  pushAndRender({type: "action", name: "submit_to_ground", arguments: args});
  const observation = {
    status: "ok",
    report_id: `r-${String(state.traceEvents.length).padStart(4, "0")}`,
    attached: attach_image,
  };
  pushAndRender({type: "observation", name: "submit_to_ground", result: observation});
  pushAndRender({type: "final", name: "submit_to_ground", result: observation});

  state.traceFinal = {
    action: "submit_to_ground",
    change_type,
    urgency,
    description,
    attach_image,
  };

  appendInbox({result: observation});
  closeSubmitModal();
  setAnnotateStatus("Terminal: submit_to_ground recorded. Stop Recording to save.");
}

function onDropClick() {
  consumeThought();
  pushAndRender({type: "action", name: "drop", arguments: {}});
  pushAndRender({type: "observation", name: "drop", result: {status: "dropped"}});
  pushAndRender({type: "final", name: "drop", result: {status: "dropped"}});
  state.traceFinal = {action: "drop"};
  setAnnotateStatus("Terminal: drop recorded. Ready to Save Trace.");
}

async function saveTrace() {
  if (state.traceEvents.length === 0) {
    setAnnotateStatus("Nothing to save yet.");
    return;
  }
  const metadata = {
    scenario_id: $("annotate-scenario").value.trim() || state.template || "unknown",
    profile: $("annotate-profile").value,
    annotator: "human",
    // source_scenario captures everything needed to re-fetch the same Before/After
    // pair via /api/fetch (no image bytes saved — replayable from the API).
    source_scenario: {
      lat: parseFloat($("lat").value),
      lon: parseFloat($("lon").value),
      before_date: $("before_date").value,
      after_date:  $("after_date").value,
      size_km:     parseFloat($("size_km").value),
      window_days: parseInt($("window_days").value, 10),
    },
    // Resolved Sentinel-2 scene datetimes (from the actual fetched images).
    // Useful as a sanity check when replaying — if the re-fetch returns a
    // different datetime, the catalog has changed since recording.
    resolved_scenes: {
      before_datetime: state.beforeMeta && state.beforeMeta.datetime,
      after_datetime:  state.afterMeta  && state.afterMeta.datetime,
      before_key:      state.beforeKey,
      after_key:       state.afterKey,
    },
    // DM3 ground-truth context — not needed for replay, but valuable when
    // reviewing the trace as SFT/eval material.
    dm3_case: state.dm3 ? {
      id:               state.dm3.id,
      event:            state.dm3.event,
      event_name:       state.dm3.event_name,
      event_start:      state.dm3.event_start,
      event_end:        state.dm3.event_end,
      disaster_type:    state.dm3.disaster_type,
      mapped_class:     state.dm3.mapped_class,
      precise:          state.dm3.precise,
      damage:           state.dm3.damage,
    } : null,
  };
  const final = state.traceFinal || {action: "incomplete"};

  const res = await fetch("/api/trace/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({metadata, events: state.traceEvents, final}),
  });
  if (!res.ok) {
    setAnnotateStatus(`Save failed: HTTP ${res.status}`);
    return;
  }
  const data = await res.json();
  setAnnotateStatus(`Saved: ${data.saved_path}`);
}

// ---- Saved traces browser ----

async function openTracesModal() {
  $("traces-modal").hidden = false;
  $("traces-detail").innerHTML = `<div class="traces-detail-empty">Loading list…</div>`;
  await refreshTracesList();
}

function closeTracesModal() {
  $("traces-modal").hidden = true;
}

async function refreshTracesList() {
  const listEl = $("traces-list");
  listEl.innerHTML = "Loading…";
  try {
    const res = await fetch("/api/traces");
    if (!res.ok) { listEl.innerHTML = `HTTP ${res.status}`; return; }
    const data = await res.json();
    const traces = data.traces || [];
    if (traces.length === 0) {
      listEl.innerHTML = `<div class="traces-empty">No saved traces yet.</div>`;
      $("traces-detail").innerHTML = `<div class="traces-detail-empty">(no traces)</div>`;
      return;
    }
    listEl.innerHTML = "";
    traces.forEach(t => {
      const row = document.createElement("div");
      row.className = "trace-row";
      const ct = t.final_change_type ? ` · <b>${escapeHtml(t.final_change_type)}</b>` : "";
      const fullName = t.scenario_id || t.filename;
      row.innerHTML = `
        <div class="trace-row-head" title="${escapeHtml(fullName)}">${escapeHtml(fullName)}${ct}</div>
        <div class="trace-row-meta">${escapeHtml((t.created_at || "").slice(0, 19))} · ${t.n_events || 0} events · ${escapeHtml(t.profile || "?")} · ${(t.size_bytes || 0)} B</div>
      `;
      row.addEventListener("click", async () => {
        document.querySelectorAll(".trace-row.selected").forEach(x => x.classList.remove("selected"));
        row.classList.add("selected");
        await viewTrace(t.filename);
      });
      listEl.appendChild(row);
    });
  } catch (e) {
    listEl.innerHTML = `error: ${e.message}`;
  }
}

async function viewTrace(filename) {
  const detail = $("traces-detail");
  detail.innerHTML = `<div class="traces-detail-empty">Loading ${escapeHtml(filename)}…</div>`;
  let doc;
  try {
    const res = await fetch(`/api/traces/${encodeURIComponent(filename)}`);
    if (!res.ok) { detail.innerHTML = `HTTP ${res.status}`; return; }
    const data = await res.json();
    doc = data.doc || {};
  } catch (e) {
    detail.innerHTML = `error: ${e.message}`;
    return;
  }
  const meta = doc.metadata || {};
  const events = doc.events || [];
  const final = doc.final || {};

  const eventLines = events.map(ev => {
    const tag = `<span class="tag tag-${ev.type || "?"}">${ev.type || "?"}</span>`;
    let body;
    if (ev.type === "thought") {
      body = `<span class="te-text">${escapeHtml(ev.text || "")}</span>`;
    } else if (ev.type === "action") {
      body = `<code>${escapeHtml(ev.name)}(${escapeHtml(JSON.stringify(ev.arguments || {}))})</code>`;
    } else if (ev.type === "observation") {
      const summary = obsTopSignal(ev.name || "", ev.result || {});
      const raw = JSON.stringify(ev.result || {}, null, 2);
      body = `<span class="obs-summary">${escapeHtml(summary)}</span>` +
             `<details class="obs-raw"><summary>details</summary><pre><code>${escapeHtml(raw)}</code></pre></details>`;
    } else if (ev.type === "attention") {
      body = renderAttentionBody(ev);
    } else if (ev.type === "view") {
      body = renderViewBody(ev);
    } else if (ev.type === "final") {
      body = `<b>[END]</b> <code>${escapeHtml(ev.name || "?")}</code>`;
    } else {
      body = `<code>${escapeHtml(JSON.stringify(ev))}</code>`;
    }
    return `<div class="trace-event">${tag}${body}</div>`;
  }).join("");

  detail.innerHTML = `
    <div class="trace-detail-head">
      <div class="trace-detail-title" title="${escapeHtml(filename)}">${escapeHtml(filename)}</div>
      <button id="trace-delete-btn" class="trace-delete-btn">🗑 Delete</button>
    </div>
    <div class="trace-meta">
      scenario: ${escapeHtml(meta.scenario_id || "?")}<br>
      profile: ${escapeHtml(meta.profile || "?")} · annotator: ${escapeHtml(meta.annotator || "?")}<br>
      created: ${escapeHtml((meta.created_at || "").slice(0, 19))}<br>
      lat/lon: ${escapeHtml(String(meta.source_scenario?.lat ?? "?"))} / ${escapeHtml(String(meta.source_scenario?.lon ?? "?"))} · size: ${escapeHtml(String(meta.source_scenario?.size_km ?? "?"))} km<br>
      dates: ${escapeHtml(String(meta.source_scenario?.before_date ?? "?"))} → ${escapeHtml(String(meta.source_scenario?.after_date ?? "?"))}<br>
      <b>final:</b> ${escapeHtml(final.action || "?")} · change_type=${escapeHtml(final.change_type || "?")} · urgency=${escapeHtml(String(final.urgency ?? "?"))}
    </div>
    <div class="trace-events">${eventLines || "<i>(no events)</i>"}</div>
  `;
  $("trace-delete-btn").addEventListener("click", async () => {
    if (!confirm(`Delete ${filename}? This cannot be undone.`)) return;
    const res = await fetch(`/api/traces/${encodeURIComponent(filename)}`, {method: "DELETE"});
    if (!res.ok) { alert(`Delete failed: HTTP ${res.status}`); return; }
    detail.innerHTML = `<div class="traces-detail-empty">Deleted.</div>`;
    await refreshTracesList();
  });
}

function discardTrace() {
  state.traceEvents = [];
  state.traceFinal = null;
  clearTrace();
  resetMapsToOriginal();
  clearDrawnBbox();
  setAnnotateStatus("Discarded.");
}

// ---- Bbox drawing on After map ----

function toggleDrawMode() {
  if (state.drawingBbox) {
    exitDrawMode();
  } else {
    enterDrawMode();
  }
}

function enterDrawMode() {
  if (!state.mapAfter || !state.afterKey) {
    setAnnotateStatus("fetch images first, then draw a bbox");
    return;
  }
  state.drawingBbox = true;
  // Clear any previous bbox
  clearDrawnBbox();
  $("draw-bbox-btn").classList.add("active");
  $("draw-bbox-btn").textContent = "▢ Click 1st corner on After map";
  $("map-after").parentElement.classList.add("drawing");
  // Pan / zoom は通常通り使える (クリックだけ奪う)
  state.mapAfter.on("click",     onDrawClick);
  state.mapAfter.on("mousemove", onDrawHoverMove);
  document.addEventListener("keydown", onDrawEscape);
  setAnnotateStatus("Drawing: click first corner (Esc to cancel)");
}

function exitDrawMode() {
  state.drawingBbox = false;
  $("draw-bbox-btn").classList.remove("active");
  $("draw-bbox-btn").textContent = "▢ Draw bbox";
  $("map-after").parentElement.classList.remove("drawing");
  if (state.mapAfter) {
    state.mapAfter.off("click",     onDrawClick);
    state.mapAfter.off("mousemove", onDrawHoverMove);
  }
  document.removeEventListener("keydown", onDrawEscape);
  state.drawStartLatLng = null;
}

function onDrawEscape(e) {
  if (e.key === "Escape") {
    clearDrawnBbox();
    exitDrawMode();
    setAnnotateStatus("Drawing cancelled.");
  }
}

function onDrawClick(e) {
  if (!state.drawStartLatLng) {
    // First click: anchor
    state.drawStartLatLng = e.latlng;
    $("draw-bbox-btn").textContent = "▢ Click 2nd corner";
    setAnnotateStatus("Drawing: click opposite corner");
    // Tiny marker at first corner for visibility
    const m = L.circleMarker(e.latlng, {radius: 4, color: "#6ad", fillOpacity: 1}).addTo(state.mapAfter);
    state.bboxRectAfter = m;
  } else {
    // Second click: finalize
    const start = state.drawStartLatLng;
    const end = e.latlng;
    // CRS.Simple bounds [[0,0],[h,w]]: latLng(y, x) → y=lat, x=lng
    const x1 = Math.min(start.lng, end.lng);
    const y1 = Math.min(start.lat, end.lat);
    const x2 = Math.max(start.lng, end.lng);
    const y2 = Math.max(start.lat, end.lat);
    const w = Math.max(1, Math.round(x2 - x1));
    const h = Math.max(1, Math.round(y2 - y1));
    state.selectedBbox = [Math.round(x1), Math.round(y1), w, h];

    // Replace marker + preview with final rectangle on both maps
    if (state.bboxRectAfter)  { state.mapAfter.removeLayer(state.bboxRectAfter);  state.bboxRectAfter  = null; }
    if (state.bboxRectBefore) { state.mapBefore.removeLayer(state.bboxRectBefore); state.bboxRectBefore = null; }
    const bounds = L.latLngBounds(start, end);
    const style  = { color: "#6ad", weight: 2, fillOpacity: 0.08 };
    state.bboxRectAfter  = L.rectangle(bounds, style).addTo(state.mapAfter);
    state.bboxRectBefore = L.rectangle(bounds, style).addTo(state.mapBefore);

    $("bbox-display").textContent = `bbox = [${state.selectedBbox.join(", ")}]  (${w}×${h} px)`;
    $("bbox-display").classList.add("has-bbox");
    setAnnotateStatus(`bbox captured: ${state.selectedBbox.join(", ")}`);
    if (state.annotating) {
      pushAndRender({
        type: "attention",
        action: "draw_bbox",
        side: "after",
        bbox: state.selectedBbox.slice(),
        image_size: [state.imgW, state.imgH],
      });
    }
    exitDrawMode();
  }
}

function onDrawHoverMove(e) {
  if (!state.drawStartLatLng) return;
  // Rubber-band preview between first click and current mouse pos
  const bounds = L.latLngBounds(state.drawStartLatLng, e.latlng);
  const style  = { color: "#6ad", weight: 1, dashArray: "4,4", fillOpacity: 0.05 };
  // Remove any existing preview (marker or rect), re-add as rect
  if (state.bboxRectAfter)  { state.mapAfter.removeLayer(state.bboxRectAfter);  state.bboxRectAfter  = null; }
  if (state.bboxRectBefore) { state.mapBefore.removeLayer(state.bboxRectBefore); state.bboxRectBefore = null; }
  state.bboxRectAfter  = L.rectangle(bounds, style).addTo(state.mapAfter);
  state.bboxRectBefore = L.rectangle(bounds, style).addTo(state.mapBefore);
}

function clearDrawnBbox() {
  state.selectedBbox = null;
  if (state.bboxRectAfter)  { state.mapAfter.removeLayer(state.bboxRectAfter);  state.bboxRectAfter  = null; }
  if (state.bboxRectBefore) { state.mapBefore.removeLayer(state.bboxRectBefore); state.bboxRectBefore = null; }
  const d = $("bbox-display");
  if (d) { d.textContent = "— no bbox —"; d.classList.remove("has-bbox"); }
  if (state.annotating) {
    pushAndRender({type: "attention", action: "clear", side: "after"});
  }
}

// ---- Location search (Nominatim) ----

/**
 * Parse "Tokyo 2024/3" style queries.
 * - Returns {location, dates} where dates may be null.
 * - Accepts YYYY/M, YYYY-M, YYYY/MM/DD, YYYY-MM-DD.
 * - If month only: after_date = <YYYY-MM-15>, before_date = ~30 days earlier.
 */
function parseSearchQuery(q) {
  const re = /(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?/;
  const m = q.match(re);
  if (!m) return { location: q.trim(), dates: null };
  const year = m[1];
  const month = m[2].padStart(2, "0");
  const day = (m[3] || "15").padStart(2, "0");
  const afterISO = `${year}-${month}-${day}`;
  const d = new Date(`${afterISO}T00:00:00Z`);
  d.setUTCDate(d.getUTCDate() - 30);
  const beforeISO = d.toISOString().slice(0, 10);
  const location = q.replace(m[0], "").trim();
  return { location, dates: { before: beforeISO, after: afterISO } };
}

let geoTimer = null;
let geoPendingSeq = 0;
async function geocodeSearch(raw) {
  const { location, dates } = parseSearchQuery(raw);
  if (location.length < 2) { $("geo-results").innerHTML = ""; return; }

  const seq = ++geoPendingSeq;
  $("geo-search").classList.add("pending");
  try {
    const res = await fetch(`/api/geocode?q=${encodeURIComponent(location)}`);
    if (seq !== geoPendingSeq) return;  // stale response
    const data = await res.json();
    renderGeoResults(data.results || [], dates);
    if (data.error) {
      $("geo-results").innerHTML = `<div class="geo-hit" style="color:#f88">${escapeHtml(data.error)}</div>`;
    }
  } catch (e) {
    $("geo-results").innerHTML = `<div class="geo-hit" style="color:#f88">search failed: ${escapeHtml(e.message)}</div>`;
  } finally {
    if (seq === geoPendingSeq) $("geo-search").classList.remove("pending");
  }
}

function renderGeoResults(results, parsedDates) {
  const el = $("geo-results");
  if (results.length === 0) {
    el.innerHTML = `<div class="geo-hit" style="color:#888">no matches</div>`;
    return;
  }
  el.innerHTML = "";
  results.forEach(r => {
    const div = document.createElement("div");
    div.className = "geo-hit";
    const shortName = r.display_name.split(",").slice(0, 3).join(",");
    const remaining = r.display_name.split(",").slice(3).join(",").trim();
    div.innerHTML = `
      <div class="geo-name">${escapeHtml(shortName)}</div>
      <div class="geo-meta">${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}${remaining ? ` · ${escapeHtml(remaining)}` : ""}${r.type ? ` · ${r.type}` : ""}</div>
    `;
    div.addEventListener("click", () => applyGeoHit(r, parsedDates));
    el.appendChild(div);
  });
}

// ---- Before candidates ----

async function searchBeforeCandidates() {
  const lat = parseFloat($("lat").value);
  const lon = parseFloat($("lon").value);
  const after_date = $("after_date").value;
  const size_km = parseFloat($("size_km").value);
  if (!after_date) { setStatus("Set After date first"); return; }

  const btn = $("before-cand-btn");
  btn.disabled = true;
  const prevText = btn.textContent;
  btn.textContent = "Searching (may take 15-30s)...";
  $("before-candidates").innerHTML = "";

  // For precise DM3 cases, anchor candidates on event_start so they're
  // guaranteed pre-disaster regardless of the user's current after_date.
  const anchor_date = (state.dm3 && state.dm3.event_start) || null;
  try {
    const res = await fetch("/api/before_candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, after_date, anchor_date, size_km }),
    });
    if (!res.ok) {
      setStatus(`candidates HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    renderBeforeCandidates(data.candidates || []);
  } catch (e) {
    setStatus(`candidate search failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

function renderBeforeCandidates(candidates) {
  const el = $("before-candidates");
  el.innerHTML = "";
  if (candidates.length === 0) {
    el.innerHTML = `<div class="cand-hit">(no candidates)</div>`;
    return;
  }
  // Sort: available + lowest cloud_proxy first (pixel-based), unavailable last
  const cloudProxy = (c) => {
    const s = c.meta.stats;
    if (s && !s.error && s.cloud_proxy !== undefined) return s.cloud_proxy;
    if (c.meta.cloud_cover !== undefined) return c.meta.cloud_cover / 100;  // fallback
    return 1.01;
  };
  const sorted = [...candidates].sort((a, b) => {
    const availA = a.meta.image_available !== false;
    const availB = b.meta.image_available !== false;
    if (availA !== availB) return availA ? -1 : 1;
    return cloudProxy(a) - cloudProxy(b);
  });

  const currentBefore = $("before_date").value;

  const otherDt = (state.afterMeta && state.afterMeta.datetime) || null;
  // Hide candidates that would resolve to the same S2 scene as current After —
  // they produce delta=0, so never useful.
  const visible = sorted.filter(c => {
    if (!otherDt) return true;
    const m = c.meta || {};
    if (m.image_available === false) return true;
    return m.datetime !== otherDt;
  });
  if (visible.length === 0) {
    el.innerHTML = `<div class="cand-hit">(all candidates collide with After — try wider offsets)</div>`;
    return;
  }

  visible.forEach(c => {
    const m = c.meta || {};
    const available = m.image_available !== false;
    const s = m.stats;
    const row = document.createElement("div");
    row.className = "cand-hit" + (available ? "" : " unavailable");
    if (c.target_date === currentBefore) row.classList.add("selected");

    let badge = `<span class="cand-badge cand-unavail">NO IMG</span>`;
    if (available && s && !s.error && s.cloud_proxy !== undefined) {
      const cp = s.cloud_proxy;
      const cls = cp < 0.2 ? "cand-clear" : cp < 0.5 ? "cand-ok" : "cand-cloudy";
      const usableTag = s.usable ? "" : " ✗";
      badge = `<span class="cand-badge ${cls}" title="pixel-based cloud_proxy = max(white_fraction, low_saturation_fraction)">px_cloud ${(cp * 100).toFixed(1)}%${usableTag}</span>`;
    } else if (available && m.cloud_cover !== undefined) {
      const cc = m.cloud_cover;
      const cls = cc < 30 ? "cand-clear" : cc < 70 ? "cand-ok" : "cand-cloudy";
      badge = `<span class="cand-badge ${cls}" title="STAC tile-level cloud_cover (ground metadata)">tile_cloud ${cc.toFixed(1)}%</span>`;
    }

    const actualDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
    const offsetTxt = `-${c.offset_days}d`;
    row.innerHTML = `
      <span class="cand-date">${escapeHtml(actualDate)}</span>
      ${badge}
      <span class="cand-offset">${offsetTxt}</span>
    `;
    if (available) {
      row.addEventListener("click", async () => {
        await applyCandidateAsBefore(c);
        el.querySelectorAll(".cand-hit").forEach(x => x.classList.remove("selected"));
        row.classList.add("selected");
      });
    }
    el.appendChild(row);
  });
}

async function applyCandidateAsBefore(c) {
  const m = c.meta || {};
  if (!c.key || m.image_available === false) {
    setStatus("candidate has no image");
    return;
  }
  // Use the candidate's cached key directly — no re-fetch, same exact image
  state.beforeKey = c.key;
  state.beforeMeta = m;
  const sceneDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
  $("before_date").value = sceneDate;
  await setImage("before", state.beforeKey);
  setMapLabel("before", labelFor("Before", state.beforeMeta));
  const cc = m.cloud_cover !== undefined
    ? `cloud ${Number(m.cloud_cover).toFixed(1)}%`
    : "?";
  setStatus(`Before applied from candidate:\n  scene: ${m.datetime || sceneDate}\n  ${cc} · ${m.source || "sentinel-2"} · key=${c.key}`);
}

// ---- Find clearer After candidates (symmetric to Before) ----

async function searchAfterCandidates() {
  const lat = parseFloat($("lat").value);
  const lon = parseFloat($("lon").value);
  const after_date = $("after_date").value;
  const size_km = parseFloat($("size_km").value);
  if (!after_date) { setStatus("Set After date first"); return; }

  const btn = $("after-cand-btn");
  btn.disabled = true;
  const prevText = btn.textContent;
  btn.textContent = "Searching (may take 15-30s)...";
  $("after-candidates").innerHTML = "";

  // For precise DM3 cases, anchor candidates on event_end so they span
  // the active disaster period (e.g., peak flood at +0d) through ~6 weeks post.
  const anchor_date = (state.dm3 && state.dm3.event_end) || null;
  try {
    const res = await fetch("/api/after_candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, after_date, anchor_date, size_km }),
    });
    if (!res.ok) {
      setStatus(`candidates HTTP ${res.status}`);
      return;
    }
    const data = await res.json();
    renderAfterCandidates(data.candidates || []);
  } catch (e) {
    setStatus(`candidate search failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

function renderAfterCandidates(candidates) {
  const el = $("after-candidates");
  el.innerHTML = "";
  if (candidates.length === 0) {
    el.innerHTML = `<div class="cand-hit">(no candidates)</div>`;
    return;
  }
  const cloudProxy = (c) => {
    const s = c.meta.stats;
    if (s && !s.error && s.cloud_proxy !== undefined) return s.cloud_proxy;
    if (c.meta.cloud_cover !== undefined) return c.meta.cloud_cover / 100;
    return 1.01;
  };
  const sorted = [...candidates].sort((a, b) => {
    const availA = a.meta.image_available !== false;
    const availB = b.meta.image_available !== false;
    if (availA !== availB) return availA ? -1 : 1;
    return cloudProxy(a) - cloudProxy(b);
  });

  const currentAfter = $("after_date").value;
  const otherDt = (state.beforeMeta && state.beforeMeta.datetime) || null;
  const visible = sorted.filter(c => {
    if (!otherDt) return true;
    const m = c.meta || {};
    if (m.image_available === false) return true;
    return m.datetime !== otherDt;
  });
  if (visible.length === 0) {
    el.innerHTML = `<div class="cand-hit">(all candidates collide with Before — try wider offsets)</div>`;
    return;
  }

  visible.forEach(c => {
    const m = c.meta || {};
    const available = m.image_available !== false;
    const s = m.stats;
    const row = document.createElement("div");
    row.className = "cand-hit" + (available ? "" : " unavailable");
    if (c.target_date === currentAfter) row.classList.add("selected");

    let badge = `<span class="cand-badge cand-unavail">NO IMG</span>`;
    if (available && s && !s.error && s.cloud_proxy !== undefined) {
      const cp = s.cloud_proxy;
      const cls = cp < 0.2 ? "cand-clear" : cp < 0.5 ? "cand-ok" : "cand-cloudy";
      const usableTag = s.usable ? "" : " ✗";
      badge = `<span class="cand-badge ${cls}" title="pixel-based cloud_proxy">px_cloud ${(cp * 100).toFixed(1)}%${usableTag}</span>`;
    } else if (available && m.cloud_cover !== undefined) {
      const cc = m.cloud_cover;
      const cls = cc < 30 ? "cand-clear" : cc < 70 ? "cand-ok" : "cand-cloudy";
      badge = `<span class="cand-badge ${cls}" title="STAC tile-level cloud_cover">tile_cloud ${cc.toFixed(1)}%</span>`;
    }

    const actualDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
    const offsetTxt = `+${c.offset_days}d`;
    row.innerHTML = `
      <span class="cand-date">${escapeHtml(actualDate)}</span>
      ${badge}
      <span class="cand-offset">${offsetTxt}</span>
    `;
    if (available) {
      row.addEventListener("click", async () => {
        await applyCandidateAsAfter(c);
        el.querySelectorAll(".cand-hit").forEach(x => x.classList.remove("selected"));
        row.classList.add("selected");
      });
    }
    el.appendChild(row);
  });
}

async function applyCandidateAsAfter(c) {
  const m = c.meta || {};
  if (!c.key || m.image_available === false) {
    setStatus("candidate has no image");
    return;
  }
  state.afterKey = c.key;
  state.afterMeta = m;
  const sceneDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
  $("after_date").value = sceneDate;
  await setImage("after", state.afterKey);
  setMapLabel("after", labelFor("After", state.afterMeta));
  await loadDamageOverlay();
  const cc = m.cloud_cover !== undefined
    ? `cloud ${Number(m.cloud_cover).toFixed(1)}%`
    : "?";
  setStatus(`After applied from candidate:\n  scene: ${m.datetime || sceneDate}\n  ${cc} · ${m.source || "sentinel-2"} · key=${c.key}`);
}

function applyGeoHit(hit, parsedDates) {
  $("lat").value = hit.lat.toFixed(4);
  $("lon").value = hit.lon.toFixed(4);
  if (parsedDates) {
    $("before_date").value = parsedDates.before;
    $("after_date").value  = parsedDates.after;
  }
  // Switch Template back to "custom"
  $("template").value = "";
  state.template = null;
  $("geo-results").innerHTML = "";
  setStatus(`Selected: ${hit.display_name}${parsedDates ? ` | dates ${parsedDates.before} → ${parsedDates.after}` : ""}`);
}

window.addEventListener("DOMContentLoaded", async () => {
  await loadTemplates();
  await loadDM3Cases();
  initMaps();
  $("fetch-btn").addEventListener("click", fetchImages);
  $("run-btn").addEventListener("click", runAgent);
  $("annotate-btn").addEventListener("click", toggleRecording);
  $("traces-list-btn").addEventListener("click", openTracesModal);
  $("traces-close-btn").addEventListener("click", closeTracesModal);
  document.querySelector("#traces-modal .modal-backdrop").addEventListener("click", closeTracesModal);
  $("size_km").addEventListener("input", (e) => {
    $("size_km_val").textContent = e.target.value;
  });
  $("window_days").addEventListener("input", (e) => {
    $("window_days_val").textContent = e.target.value;
  });
  document.querySelectorAll(".tool-btn").forEach(btn => {
    btn.addEventListener("click", () => runTool(btn.dataset.tool));
  });
  $("submit-btn").addEventListener("click", openSubmitModal);
  $("drop-btn").addEventListener("click", onDropClick);
  $("f-confirm").addEventListener("click", onSubmitConfirm);
  $("f-cancel").addEventListener("click", closeSubmitModal);
  // Save Trace button removed: Stop Recording now auto-saves via toggleRecording.
  $("discard-trace-btn").addEventListener("click", () => {
    if (confirm("Discard the current trace?")) discardTrace();
  });
  const drawBtn = $("draw-bbox-btn");
  if (drawBtn) drawBtn.addEventListener("click", toggleDrawMode);
  const clearBtn = $("clear-bbox-btn");
  if (clearBtn) clearBtn.addEventListener("click", clearDrawnBbox);

  $("before-cand-btn").addEventListener("click", searchBeforeCandidates);
  $("after-cand-btn").addEventListener("click", searchAfterCandidates);
  const resetBtn = $("reset-maps-btn");
  if (resetBtn) {
    resetBtn.addEventListener("click", async () => {
      if (!state.beforeKey && !state.afterKey) { setToolsStatus("nothing to reset (fetch images first)"); return; }
      setToolsStatus("resetting maps...");
      await resetMapsToOriginal();
      setToolsStatus("maps reset to original RGB");
    });
  }
  const toggleDamage = $("toggle-damage");
  if (toggleDamage) {
    toggleDamage.addEventListener("change", async (e) => {
      state.damageVisible = !!e.target.checked;
      if (state.damageVisible) {
        await loadDamageOverlay();
      } else {
        clearDamageOverlay();
      }
    });
  }
  $("dm3-case").addEventListener("change", onDM3Change);

  // Geocoding search with debounce
  $("geo-search").addEventListener("input", (e) => {
    clearTimeout(geoTimer);
    const q = e.target.value;
    if (q.trim().length < 2) { $("geo-results").innerHTML = ""; return; }
    geoTimer = setTimeout(() => geocodeSearch(q), 450);
  });
  $("geo-search").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      clearTimeout(geoTimer);
      geocodeSearch(e.target.value);
    }
  });
  // Auto-fetch on load
  fetchImages();
});
