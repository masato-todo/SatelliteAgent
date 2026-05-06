// dm3-fetch.js — DM3 cases dropdown, GT panel, templates, fetchImages, candidates, geocode

import { state, $, escapeHtml, setStatus, labelFor } from "./state-utils.js";
import { setImage, setMapLabel, loadDamageOverlay, clearDamageOverlay } from "./maps.js";

// ---- DM3 cases ----

export async function loadDM3Cases() {
  try {
    const res = await fetch("/api/disasterm3/cases");
    if (!res.ok) return;
    const data = await res.json();
    const sel = $("dm3-case");
    if (!sel) return;
    sel.innerHTML = '<option value="">— none —</option>';

    const cases = data.cases || [];
    const fireedgeCases      = cases.filter(c => c.source === "FireEdge_HF");
    const hardNegCases       = cases.filter(c => c.is_hard_negative);
    const negativeCases      = cases.filter(c => c.is_negative && !c.is_hard_negative && c.source !== "FireEdge_HF");
    const volcanicCases      = cases.filter(c => c.source === "GDACS_VO");
    const deforestationCases = cases.filter(c => c.source === "PRODES");
    const habCases           = cases.filter(c => c.source === "HAB");
    const emsCases           = cases.filter(c => c.source === "EMS");
    const catalogCases       = cases.filter(c => c.source === "MCD64A1");
    const otherSrc = c => !["MCD64A1","EMS","GDACS_VO","PRODES","HAB","HARD_NEG","FireEdge_HF"].includes(c.source);
    const preciseCases  = cases.filter(c => c.precise && !c.is_negative && otherSrc(c));
    const coarseCases   = cases.filter(c => !c.precise && !c.is_negative && otherSrc(c));

    const makeOption = (c, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      const n = c.cached_count || 0;
      const saved = (c.canonical_pairs || []).length > 0;
      const mark = saved ? "★ " : n >= 2 ? "● " : n === 1 ? "◐ " : "";
      let label = `${mark}[${c.source}] ${c.event} · ${c.disaster_type} · ${c.capture_date}`;
      if (c.damage && c.damage.destroyed + c.damage.major > 0) {
        label += `  (${c.damage.destroyed} destroyed + ${c.damage.major} major)`;
      }
      opt.textContent = label;
      opt.dataset.case = JSON.stringify(c);
      return opt;
    };

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
    if (catalogCases.length) {
      const grpM = document.createElement("optgroup");
      grpM.label = "🔥 MCD64A1 — wildfire burn area (MODIS, ≥1km²)";
      catalogCases.forEach(c => grpM.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpM);
    }
    if (emsCases.length) {
      const grpE = document.createElement("optgroup");
      grpE.label = "🌊 EMS — Copernicus rapid mapping (flood / storm / quake / landslide)";
      emsCases.forEach(c => grpE.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpE);
    }
    if (volcanicCases.length) {
      const grpV = document.createElement("optgroup");
      grpV.label = "🌋 Volcanic — GDACS eruption events (lava / ash, SWIR signal)";
      volcanicCases.forEach(c => grpV.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpV);
    }
    if (deforestationCases.length) {
      const grpD = document.createElement("optgroup");
      grpD.label = "🌳 Deforestation — PRODES Amazon clearings (NDVI / NBR drop)";
      deforestationCases.forEach(c => grpD.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpD);
    }
    if (fireedgeCases.length) {
      const grpFE = document.createElement("optgroup");
      grpFE.label = "🔥 FireEdge GT — YujiYamaguchi/fireedge-sentinel2-wildfire (HF, 300 cases)";
      fireedgeCases.forEach(c => grpFE.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpFE);
    }
    if (habCases.length) {
      const grpH = document.createElement("optgroup");
      grpH.label = "🟢 Algal bloom — harmful algal blooms / red tide (NDCI, RGB color)";
      habCases.forEach(c => grpH.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpH);
    }
    if (negativeCases.length) {
      const grpN = document.createElement("optgroup");
      grpN.label = "⊘ NEGATIVE — drop expected (no_change / cloud_blocked / random)";
      negativeCases.forEach(c => grpN.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpN);
    }
    if (hardNegCases.length) {
      const grpHN = document.createElement("optgroup");
      grpHN.label = "🛑 HARD NEGATIVE — drop at positive sites in stable years (forest/volcano/pre-burn)";
      hardNegCases.forEach(c => grpHN.appendChild(makeOption(c, cases.indexOf(c))));
      sel.appendChild(grpHN);
    }
  } catch (e) {
    console.error("DM3 load failed", e);
  }
}

function updateCachePairWidget(c) {
  const sel = $("cache-pair-select");
  const btn = $("cache-use-btn");
  if (!sel || !btn) return;
  sel.innerHTML = "";
  const pairs = (c && c.canonical_pairs) || [];
  if (!pairs.length) {
    sel.style.display = "none";
    btn.style.display = "none";
    return;
  }
  pairs.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = String(i);
    opt.textContent = `${p.size_km}km · B[${p.before_date}] / A[${p.after_date}]`;
    opt.dataset.pair = JSON.stringify(p);
    sel.appendChild(opt);
  });
  sel.style.display = "";
  btn.style.display = "";
}

export function onDM3Change() {
  const opt = $("dm3-case").selectedOptions[0];
  if (!opt || !opt.value) {
    $("dm3-gt").innerHTML = "";
    state.dm3 = null;
    clearDamageOverlay();
    updateCachePairWidget(null);
    return;
  }
  const c = JSON.parse(opt.dataset.case);
  state.dm3 = c;
  clearDamageOverlay();
  $("lat").value = c.lat;
  $("lon").value = c.lon;
  $("before_date").value = c.before_date;
  $("after_date").value  = c.after_date;
  if (c.size_km) {
    $("size_km").value = c.size_km;
    const lbl = $("size_km_val");
    if (lbl) lbl.textContent = c.size_km;
  }
  if (c.window_days) {
    // Cases that pin a precise S2 frame (e.g. FireEdge_HF carries
    // sentinel_datetime + window_days=1) need their window propagated
    // to the form so Fetch Images hits the exact same item.
    const wd = $("window_days");
    if (wd) {
      wd.value = c.window_days;
      const lbl2 = $("window_days_val");
      if (lbl2) lbl2.textContent = c.window_days;
    }
  }
  state.template = c.id;
  updateCachePairWidget(c);

  // FireEdge cases pin a precise STAC item via sentinel_datetime — expose
  // a dedicated fetch button for them so users don't have to know about
  // the window=1 + ISO-timestamp trick.
  const feBtn = $("fetch-fireedge-btn");
  if (feBtn) feBtn.style.display = (c.source === "FireEdge_HF") ? "" : "none";

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
  if ((c.canonical_pairs || []).length) {
    const p = c.canonical_pairs[0];
    lines.push(`<span class="gt-precise">★ Saved canonical:</span> ${p.size_km}km · B[${escapeHtml(p.before_date)}] / A[${escapeHtml(p.after_date)}]`);
  }
  $("dm3-gt").innerHTML = lines.join("\n");
  setStatus(`DisasterM3 case loaded: ${c.event}${c.precise ? " (precise)" : ""} — now press Fetch Images`);
}

// ---- Templates ----

// "Templates" UI dropdown was retired in favour of the DM3 case dropdown.
// We still hit /api/templates for the SimSat URL + provider badge in the header.
export async function loadTemplates() {
  const res = await fetch("/api/templates");
  const data = await res.json();
  $("simsat-url").textContent = data.simsat_url;
  const p = data.provider || { kind: "none", model: "?" };
  const el = $("provider-info");
  el.textContent = p.kind === "gemini" ? `Gemini · ${p.model}` : "NO PROVIDER (set GOOGLE_API_KEY)";
  el.className = p.kind === "gemini" ? "provider-gemini" : "provider-stub";
}

// ---- Fetch images ----

function fmtMeta(side, part) {
  const m = part.meta || {};
  if (m.error) return `✗ ${side}: ${m.error}`;
  const bits = [];
  bits.push(m.cached ? "cached" : "fetched");
  if (m.datetime) bits.push(m.datetime);
  const s = m.stats;
  if (s && !s.error) {
    bits.push(`cloud=${s.cloud_proxy.toFixed(2)}`);
    if (s.nodata_fraction !== undefined && s.nodata_fraction > 0.05) {
      bits.push(`nodata=${(s.nodata_fraction * 100).toFixed(0)}%`);
    }
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
  const fe = $("fetch-fireedge-btn");
  if (fe) fe.disabled = isFetching;
}

export async function saveCurrentPair() {
  const c = state.dm3;
  if (!c) { setStatus("select a case first"); return; }
  if (!state.beforeKey || !state.afterKey) {
    setStatus("press Fetch Images first to load Before/After");
    return;
  }
  const body = {
    scene_id:    c.id,
    lat:         parseFloat($("lat").value),
    lon:         parseFloat($("lon").value),
    before_date: $("before_date").value,
    after_date:  $("after_date").value,
    size_km:     parseFloat($("size_km").value),
    before_key:  state.beforeKey,
    after_key:   state.afterKey,
    label:       c.mapped_class || c.disaster_type,
    event_type:  c.disaster_type,
    event_start: c.event_start,
    event_end:   c.event_end,
    event_name:  c.event_name,
    is_negative:     !!c.is_negative,
    negative_type:   c.negative_type || null,
    expected_action: c.expected_action || null,
  };
  const btn = $("save-pair-btn");
  btn.disabled = true;
  setStatus(`saving Before/After for ${c.id} ...`);
  try {
    const res = await fetch("/api/scene/save_pair", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      setStatus(`save failed: HTTP ${res.status} ${txt}`);
      return;
    }
    const d = await res.json();
    setStatus(`saved → ${d.saved_dir}  (canonical entries: ${d.canonical_entries})`);
  } catch (e) {
    setStatus(`save error: ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

export function useCachedPair() {
  const sel = $("cache-pair-select");
  const opt = sel && sel.selectedOptions[0];
  if (!opt) return;
  const p = JSON.parse(opt.dataset.pair);
  if (p.size_km != null) {
    $("size_km").value = p.size_km;
    const lbl = $("size_km_val");
    if (lbl) lbl.textContent = p.size_km;
  }
  if (p.before_date) $("before_date").value = p.before_date;
  if (p.after_date)  $("after_date").value  = p.after_date;
  fetchImages();
}

async function _fetchImagesWithPayload(payload, statusPrefix) {
  setStatus(`${statusPrefix}  lat=${payload.lat}, lon=${payload.lon}, size=${payload.size_km}km`);
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

export async function fetchImages() {
  const payload = {
    lat:  parseFloat($("lat").value),
    lon:  parseFloat($("lon").value),
    before_date: $("before_date").value,
    after_date:  $("after_date").value,
    size_km:     parseFloat($("size_km").value),
    window_days: parseInt($("window_days").value, 10),
  };
  await _fetchImagesWithPayload(payload, "Fetching...");
}

// FireEdge cases pin a specific S2 STAC item by sentinel_datetime (full ISO).
// Sending that timestamp + window_days=1 forces SimSat to return the exact
// frame the wildfire LoRA was trained on. Mirrors the production-conditions
// eval at scripts/eval_wildfire_hf_simsat.py --use-sentinel-datetime.
export async function fetchImagesFireEdge() {
  const c = state.dm3;
  if (!c || c.source !== "FireEdge_HF") {
    setStatus("Select a FireEdge GT case first.");
    return;
  }
  const sdt = c.sentinel_datetime;
  if (!sdt) {
    setStatus("This FireEdge case has no sentinel_datetime — cannot pin training frame.");
    return;
  }
  const payload = {
    lat:  parseFloat($("lat").value),
    lon:  parseFloat($("lon").value),
    before_date: $("before_date").value,
    after_date:  sdt,    // full ISO → SimSat returns the exact training STAC item
    size_km:     parseFloat($("size_km").value),
    window_days:        1,    // After: tight window pins the training STAC item
    before_window_days: 30,   // Before: standard wide search so sdt-180d finds a real S2 capture
  };
  await _fetchImagesWithPayload(payload, `Fetching FireEdge frame (After=${sdt} ±1d, Before=${$("before_date").value} ±30d)...`);
}

// ---- Find clearer Before / After candidates ----

export async function searchBeforeCandidates() {
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

  const anchor_date = (state.dm3 && state.dm3.event_start) || null;
  try {
    const res = await fetch("/api/before_candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, after_date, anchor_date, size_km }),
    });
    if (!res.ok) { setStatus(`candidates HTTP ${res.status}`); return; }
    const data = await res.json();
    renderCandidates(data.candidates || [], "before");
  } catch (e) {
    setStatus(`candidate search failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

export async function searchAfterCandidates() {
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

  const anchor_date = (state.dm3 && state.dm3.event_end) || null;
  try {
    const res = await fetch("/api/after_candidates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lat, lon, after_date, anchor_date, size_km }),
    });
    if (!res.ok) { setStatus(`candidates HTTP ${res.status}`); return; }
    const data = await res.json();
    renderCandidates(data.candidates || [], "after");
  } catch (e) {
    setStatus(`candidate search failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = prevText;
  }
}

function renderCandidates(candidates, side) {
  const isBefore = side === "before";
  const elId = isBefore ? "before-candidates" : "after-candidates";
  const el = $(elId);
  el.innerHTML = "";
  if (candidates.length === 0) {
    el.innerHTML = `<div class="cand-hit">(no candidates)</div>`;
    return;
  }
  // Sort priority: available > usable > low cloud + low nodata.
  // Treat nodata-heavy candidates (tile fragments) as worse than cloudy:
  // a 50%-clouded scene still has signal, a 50%-nodata scene is half-empty.
  const badnessScore = (c) => {
    const s = c.meta.stats || {};
    if (c.meta.image_available === false) return 99;
    const cp = (s.cloud_proxy !== undefined) ? s.cloud_proxy
             : (c.meta.cloud_cover !== undefined ? c.meta.cloud_cover / 100 : 1.01);
    const nd = s.nodata_fraction || 0;
    return cp + nd * 1.5;  // nodata weighted slightly heavier
  };
  const sorted = [...candidates].sort((a, b) => badnessScore(a) - badnessScore(b));

  const currentDate = isBefore ? $("before_date").value : $("after_date").value;
  const otherDt = isBefore
    ? (state.afterMeta && state.afterMeta.datetime) || null
    : (state.beforeMeta && state.beforeMeta.datetime) || null;

  // Hide candidates that resolve to the same S2 scene as the other side
  const visible = sorted.filter(c => {
    if (!otherDt) return true;
    const m = c.meta || {};
    if (m.image_available === false) return true;
    return m.datetime !== otherDt;
  });
  if (visible.length === 0) {
    const otherSide = isBefore ? "After" : "Before";
    el.innerHTML = `<div class="cand-hit">(all candidates collide with ${otherSide} — try wider offsets)</div>`;
    return;
  }

  visible.forEach(c => {
    const m = c.meta || {};
    const available = m.image_available !== false;
    const s = m.stats;
    const row = document.createElement("div");
    row.className = "cand-hit" + (available ? "" : " unavailable");
    if (c.target_date === currentDate) row.classList.add("selected");

    let cloudBadge = `<span class="cand-badge cand-unavail">NO IMG</span>`;
    let nodataBadge = "";
    if (available && s && !s.error && s.cloud_proxy !== undefined) {
      const cp = s.cloud_proxy;
      const cls = cp < 0.2 ? "cand-clear" : cp < 0.5 ? "cand-ok" : "cand-cloudy";
      const usableTag = s.usable ? "" : " ✗";
      cloudBadge = `<span class="cand-badge ${cls}" title="pixel-based cloud_proxy">cloud ${(cp * 100).toFixed(0)}%${usableTag}</span>`;
      // Show nodata badge separately when notable
      const nd = s.nodata_fraction;
      if (nd !== undefined && nd > 0.05) {
        const ndCls = nd < 0.2 ? "cand-ok" : "cand-cloudy";
        nodataBadge = `<span class="cand-badge ${ndCls}" title="fraction of pixels that are nodata (tile fragment / SimSat boundary)">nodata ${(nd * 100).toFixed(0)}%</span>`;
      }
    } else if (available && m.cloud_cover !== undefined) {
      const cc = m.cloud_cover;
      const cls = cc < 30 ? "cand-clear" : cc < 70 ? "cand-ok" : "cand-cloudy";
      cloudBadge = `<span class="cand-badge ${cls}" title="STAC tile-level cloud_cover">tile_cloud ${cc.toFixed(0)}%</span>`;
    }

    const actualDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
    const offsetTxt = isBefore ? `-${c.offset_days}d` : `+${c.offset_days}d`;
    row.innerHTML = `
      <span class="cand-date">${escapeHtml(actualDate)}</span>
      ${cloudBadge}
      ${nodataBadge}
      <span class="cand-offset">${offsetTxt}</span>
    `;
    if (available) {
      row.addEventListener("click", async () => {
        if (isBefore) await applyCandidateAsBefore(c);
        else          await applyCandidateAsAfter(c);
        el.querySelectorAll(".cand-hit").forEach(x => x.classList.remove("selected"));
        row.classList.add("selected");
      });
    }
    el.appendChild(row);
  });
}

async function applyCandidateAsBefore(c) {
  const m = c.meta || {};
  if (!c.key || m.image_available === false) { setStatus("candidate has no image"); return; }
  state.beforeKey = c.key;
  state.beforeMeta = m;
  const sceneDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
  $("before_date").value = sceneDate;
  await setImage("before", state.beforeKey);
  setMapLabel("before", labelFor("Before", state.beforeMeta));
  const cc = m.cloud_cover !== undefined ? `cloud ${Number(m.cloud_cover).toFixed(1)}%` : "?";
  setStatus(`Before applied:\n  scene: ${m.datetime || sceneDate}\n  ${cc} · ${m.source || "sentinel-2"} · key=${c.key}`);
}

async function applyCandidateAsAfter(c) {
  const m = c.meta || {};
  if (!c.key || m.image_available === false) { setStatus("candidate has no image"); return; }
  state.afterKey = c.key;
  state.afterMeta = m;
  const sceneDate = m.datetime ? m.datetime.slice(0, 10) : c.target_date;
  $("after_date").value = sceneDate;
  await setImage("after", state.afterKey);
  setMapLabel("after", labelFor("After", state.afterMeta));
  await loadDamageOverlay();
  const cc = m.cloud_cover !== undefined ? `cloud ${Number(m.cloud_cover).toFixed(1)}%` : "?";
  setStatus(`After applied:\n  scene: ${m.datetime || sceneDate}\n  ${cc} · ${m.source || "sentinel-2"} · key=${c.key}`);
}

// ---- Geocode (Nominatim) ----

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
export async function geocodeSearch(raw) {
  const { location, dates } = parseSearchQuery(raw);
  if (location.length < 2) { $("geo-results").innerHTML = ""; return; }
  const seq = ++geoPendingSeq;
  $("geo-search").classList.add("pending");
  try {
    const res = await fetch(`/api/geocode?q=${encodeURIComponent(location)}`);
    if (seq !== geoPendingSeq) return;
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

function applyGeoHit(hit, parsedDates) {
  $("lat").value = hit.lat.toFixed(4);
  $("lon").value = hit.lon.toFixed(4);
  if (parsedDates) {
    $("before_date").value = parsedDates.before;
    $("after_date").value  = parsedDates.after;
  }
  state.template = null;
  $("geo-results").innerHTML = "";
  setStatus(`Selected: ${hit.display_name}${parsedDates ? ` | dates ${parsedDates.before} → ${parsedDates.after}` : ""}`);
}

// Expose timer accessor for main.js to set up debounce binding
export function getGeoTimer() { return geoTimer; }
export function setGeoTimer(t) { geoTimer = t; }
