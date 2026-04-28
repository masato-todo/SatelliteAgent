// maps.js — Leaflet maps, image overlay, sync, view tracking, damage overlay, bbox drawing

import { state, $, round2, labelFor, DAMAGE_STYLE, VIEW_DEBOUNCE_MS } from "./state-utils.js";

// ---- Init ----

export function initMaps() {
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

  // Pan/zoom tracking — only emits trace events while annotating, debounced.
  state.mapAfter.on("moveend zoomend", () => trackViewChange("after"));
  state.mapBefore.on("moveend zoomend", () => trackViewChange("before"));
}

function trackViewChange(side) {
  if (!state.annotating) return;
  if (state.syncing) return;
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
  const snap = {
    side,
    center: [round2(c.lat), round2(c.lng)],
    zoom: Number(z.toFixed(2)),
    bounds: [
      [round2(b.getSouth()), round2(b.getWest())],
      [round2(b.getNorth()), round2(b.getEast())],
    ],
  };
  const key = JSON.stringify(snap);
  if (state.lastViewSnapshot === key) return;
  state.lastViewSnapshot = key;
  if (state.traceEmitter) state.traceEmitter({type: "view", ...snap});
}

// ---- Image overlay ----

function loadImageSize(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = reject;
    img.src = url;
  });
}

export async function setImage(which, key) {
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

export function setMapLabel(which, text) {
  const el = document.querySelector(`#map-${which}`).parentElement.querySelector(".map-label");
  if (el) el.textContent = text;
}

export async function resetMapsToOriginal() {
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

// ---- xBD damage overlay ----

export function clearDamageOverlay() {
  if (state.damageLayerBefore) { state.mapBefore.removeLayer(state.damageLayerBefore); state.damageLayerBefore = null; }
  if (state.damageLayerAfter)  { state.mapAfter.removeLayer(state.damageLayerAfter);   state.damageLayerAfter  = null; }
}

function wgs84ToLeaflet(lat, lon, aoiLat, aoiLon, sizeKm, W, H) {
  const dyKm = (lat - aoiLat) * 110.574;
  const dxKm = (lon - aoiLon) * 111.320 * Math.cos(aoiLat * Math.PI / 180);
  const yFrac = 0.5 + dyKm / sizeKm;
  const xFrac = 0.5 + dxKm / sizeKm;
  return [yFrac * H, xFrac * W];
}

export async function loadDamageOverlay() {
  clearDamageOverlay();
  // Reset both labels FIRST so repeated calls don't accumulate "GT: ..." duplicates.
  if (state.beforeMeta) setMapLabel("before", labelFor("Before", state.beforeMeta));
  if (state.afterMeta)  setMapLabel("after",  labelFor("After",  state.afterMeta));
  if (!state.damageVisible) return;
  if (!state.dm3) return;
  if (!state.imgW || !state.imgH) return;
  if (state.dm3.source === "MCD64A1") {
    return loadBurnOverlay();
  }
  if (!state.dm3.precise) return;

  const aoiLat = parseFloat($("lat").value);
  const aoiLon = parseFloat($("lon").value);
  const sizeKm = parseFloat($("size_km").value);
  const params = new URLSearchParams({
    event:   state.dm3.event,
    lat:     String(aoiLat),
    lon:     String(aoiLon),
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

async function loadBurnOverlay() {
  const aoiLat = parseFloat($("lat").value);
  const aoiLon = parseFloat($("lon").value);
  const sizeKm = parseFloat($("size_km").value);

  let gj;
  try {
    const res = await fetch(`/api/scene/burn_polygon/${encodeURIComponent(state.dm3.id)}`);
    if (!res.ok) return;
    gj = await res.json();
  } catch (e) {
    console.warn("burn_polygon fetch failed", e);
    return;
  }

  const style = { color: "#ff5530", weight: 2, opacity: 0.95, fillColor: "#ff5530", fillOpacity: 0.18 };
  const beforeGroup = L.layerGroup();
  const afterGroup  = L.layerGroup();
  const rings = extractRings(gj.geometry);
  for (const ring of rings) {
    const latlngs = ring.map(([lon, lat]) =>
      wgs84ToLeaflet(lat, lon, aoiLat, aoiLon, sizeKm, state.imgW, state.imgH)
    );
    L.polygon(latlngs, style).addTo(beforeGroup);
    L.polygon(latlngs, style).addTo(afterGroup);
  }
  beforeGroup.addTo(state.mapBefore);
  afterGroup.addTo(state.mapAfter);
  state.damageLayerBefore = beforeGroup;
  state.damageLayerAfter  = afterGroup;

  const tag = ` · GT: burn area (${state.dm3.event_name || "?"})`;
  for (const side of ["before", "after"]) {
    const el = document.querySelector(`#map-${side}`).parentElement.querySelector(".map-label");
    if (el) el.textContent = (el.textContent || "") + tag;
  }
}

function extractRings(geom) {
  if (!geom) return [];
  if (geom.type === "Polygon") return geom.coordinates;
  if (geom.type === "MultiPolygon") return geom.coordinates.flat();
  return [];
}

// ---- Bbox drawing ----

export function toggleDrawMode() {
  if (state.drawingBbox) exitDrawMode();
  else                    enterDrawMode();
}

function enterDrawMode() {
  if (!state.mapAfter || !state.afterKey) {
    if (state.setAnnotateStatus) state.setAnnotateStatus("fetch images first, then draw a bbox");
    return;
  }
  state.drawingBbox = true;
  clearDrawnBbox();
  $("draw-bbox-btn").classList.add("active");
  $("draw-bbox-btn").textContent = "▢ Click 1st corner on After map";
  $("map-after").parentElement.classList.add("drawing");
  state.mapAfter.on("click",     onDrawClick);
  state.mapAfter.on("mousemove", onDrawHoverMove);
  document.addEventListener("keydown", onDrawEscape);
  if (state.setAnnotateStatus) state.setAnnotateStatus("Drawing: click first corner (Esc to cancel)");
}

function exitDrawMode() {
  state.drawingBbox = false;
  $("draw-bbox-btn").classList.remove("active");
  $("draw-bbox-btn").textContent = "▢ Draw bbox on After";
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
    if (state.setAnnotateStatus) state.setAnnotateStatus("Drawing cancelled.");
  }
}

function onDrawClick(e) {
  if (!state.drawStartLatLng) {
    state.drawStartLatLng = e.latlng;
    $("draw-bbox-btn").textContent = "▢ Click 2nd corner";
    if (state.setAnnotateStatus) state.setAnnotateStatus("Drawing: click opposite corner");
    const m = L.circleMarker(e.latlng, {radius: 4, color: "#6ad", fillOpacity: 1}).addTo(state.mapAfter);
    state.bboxRectAfter = m;
  } else {
    const start = state.drawStartLatLng;
    const end = e.latlng;
    const x1 = Math.min(start.lng, end.lng);
    const y1 = Math.min(start.lat, end.lat);
    const x2 = Math.max(start.lng, end.lng);
    const y2 = Math.max(start.lat, end.lat);
    const w = Math.max(1, Math.round(x2 - x1));
    const h = Math.max(1, Math.round(y2 - y1));
    state.selectedBbox = [Math.round(x1), Math.round(y1), w, h];

    if (state.bboxRectAfter)  { state.mapAfter.removeLayer(state.bboxRectAfter);  state.bboxRectAfter  = null; }
    if (state.bboxRectBefore) { state.mapBefore.removeLayer(state.bboxRectBefore); state.bboxRectBefore = null; }
    const bounds = L.latLngBounds(start, end);
    const style  = { color: "#6ad", weight: 2, fillOpacity: 0.08 };
    state.bboxRectAfter  = L.rectangle(bounds, style).addTo(state.mapAfter);
    state.bboxRectBefore = L.rectangle(bounds, style).addTo(state.mapBefore);

    $("bbox-display").textContent = `bbox = [${state.selectedBbox.join(", ")}]  (${w}×${h} px)`;
    $("bbox-display").classList.add("has-bbox");
    if (state.setAnnotateStatus) state.setAnnotateStatus(`bbox captured: ${state.selectedBbox.join(", ")}`);
    if (state.annotating && state.traceEmitter) {
      state.traceEmitter({
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
  const bounds = L.latLngBounds(state.drawStartLatLng, e.latlng);
  const style  = { color: "#6ad", weight: 1, dashArray: "4,4", fillOpacity: 0.05 };
  if (state.bboxRectAfter)  { state.mapAfter.removeLayer(state.bboxRectAfter);  state.bboxRectAfter  = null; }
  if (state.bboxRectBefore) { state.mapBefore.removeLayer(state.bboxRectBefore); state.bboxRectBefore = null; }
  state.bboxRectAfter  = L.rectangle(bounds, style).addTo(state.mapAfter);
  state.bboxRectBefore = L.rectangle(bounds, style).addTo(state.mapBefore);
}

export function clearDrawnBbox() {
  state.selectedBbox = null;
  if (state.bboxRectAfter)  { state.mapAfter.removeLayer(state.bboxRectAfter);  state.bboxRectAfter  = null; }
  if (state.bboxRectBefore) { state.mapBefore.removeLayer(state.bboxRectBefore); state.bboxRectBefore = null; }
  const d = $("bbox-display");
  if (d) { d.textContent = "— no bbox —"; d.classList.remove("has-bbox"); }
  if (state.annotating && state.traceEmitter) {
    state.traceEmitter({type: "attention", action: "clear", side: "after"});
  }
}
