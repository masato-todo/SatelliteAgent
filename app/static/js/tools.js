// tools.js — invokeTool, runTool, runTool_perSide, formatters

import { state, $, escapeHtml, SPECTRAL_PER_SIDE_TOOLS } from "./state-utils.js";
import { setImage, setMapLabel } from "./maps.js";

// ---- API ----

export async function invokeTool(toolName, args) {
  if (!state.beforeKey || !state.afterKey) {
    setToolsStatus("images not ready");
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
    setToolsStatus(`HTTP ${res.status}: ${await res.text()}`);
    return null;
  }
  const data = await res.json();
  return data.observation;
}

export function setToolsStatus(msg) {
  // Don't mirror tool-call results to annotate-status — the trace events show them.
  const t = $("tools-status"); if (t) t.textContent = msg;
}

// ---- Formatters (used by tools.js + annotate-traces.js) ----

export function obsTopSignal(toolName, result) {
  if (toolName === "get_change_stats") {
    const idx = result.indices || {};
    let best = null;
    for (const [name, s] of Object.entries(idx)) {
      const dec = s.frac_strong_decrease || 0;
      const inc = s.frac_strong_increase || 0;
      const mag = Math.max(dec, inc);
      if (!best || mag > best.mag) {
        best = {name, dec, inc, mag, dir: dec >= inc ? "↓" : "↑", val: dec >= inc ? dec : inc};
      }
    }
    if (best) return `top: ${best.name} strong${best.dir}=${(best.val * 100).toFixed(1)}% (${Object.keys(idx).length} indices)`;
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
  const j = JSON.stringify(result);
  return j.length > 100 ? j.slice(0, 100) + "…" : j;
}

export function formatToolResult(toolName, obs) {
  if (!obs) return "(no result)";
  if (obs.error) {
    let msg = `ERROR: ${obs.error}`;
    if (obs.raw_preview !== undefined) msg += `\n  raw(${obs.raw_len}ch): ${obs.raw_preview}`;
    if (obs.raw !== undefined) msg += `\n  raw: ${obs.raw}`;
    return msg;
  }
  if (toolName === "classify_change") {
    const classes = (obs.classes || []).slice(0, 3).map(c => {
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

// ---- runTool ----

function emit(ev) {
  if (state.annotating && state.traceEmitter) state.traceEmitter(ev);
}

function consumeThoughtIfAny() {
  if (!state.annotating) return;
  const el = $("annotate-thought");
  if (!el) return;
  const t = el.value.trim();
  if (t) {
    emit({type: "thought", text: t});
    el.value = "";
  }
}

export async function runTool_perSide(toolName, side) {
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
  consumeThoughtIfAny();
  emit({type: "action", name: toolName, arguments: args});
  setToolsStatus(`${toolName} (${side}): running...`);
  const observation = await invokeTool(toolName, args);
  if (observation == null) {
    setToolsStatus(`${toolName} (${side}): invocation failed (no response)`);
    return;
  }
  emit({type: "observation", name: toolName, result: observation});
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

export async function runTool(toolName) {
  let args = {};
  const which = $("which-sel") ? $("which-sel").value : "after";

  if (SPECTRAL_PER_SIDE_TOOLS.has(toolName) && which === "both") {
    await runTool_perSide(toolName, "before");
    await runTool_perSide(toolName, "after");
    return;
  }

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

  consumeThoughtIfAny();
  emit({type: "action", name: toolName, arguments: args});
  setToolsStatus(`${toolName}: running...`);

  const observation = await invokeTool(toolName, args);
  if (observation == null) return;

  emit({type: "observation", name: toolName, result: observation});

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

  // DM3 GT comparison for classify_change
  if (toolName === "classify_change" && state.dm3 && observation && observation.classes) {
    const expected = state.dm3.mapped_class;
    const primary = observation.classes[0] && (observation.classes[0].name || Object.keys(observation.classes[0])[0]);
    const match = primary === expected;
    line += `\n  GT: ${expected}  →  ${match ? "✓ MATCH" : "✗ MISS (primary=" + primary + ")"}`;
  }
  setToolsStatus(line);
}
