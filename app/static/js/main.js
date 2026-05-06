// main.js — runAgent SSE + boot wiring

import { state, $, setStatus, updateBudget, BUDGET_MAX } from "./state-utils.js";
import { initMaps, setImage, setMapLabel, resetMapsToOriginal,
         loadDamageOverlay, clearDamageOverlay, toggleDrawMode, clearDrawnBbox } from "./maps.js";
import { loadDM3Cases, onDM3Change, loadTemplates, fetchImages, fetchImagesFireEdge,
         useCachedPair, saveCurrentPair, searchBeforeCandidates, searchAfterCandidates,
         geocodeSearch } from "./dm3-fetch.js";
import { runTool, setToolsStatus } from "./tools.js";
import { initProviders, openSettings, closeSettings } from "./providers.js";
import { toggleRecording, openSubmitModal, closeSubmitModal,
         onSubmitConfirm, onDropClick, openTracesModal, closeTracesModal,
         clearTrace, renderTraceEvent, discardTrace } from "./annotate-traces.js";

// ---- Agent (SSE) ----

function setRunButton(running) {
  const btn = $("run-btn");
  if (!btn) return;
  btn.textContent = running ? "Stop Agent" : "Run Agent";
  btn.classList.toggle("danger", running);
}

function stopAgent() {
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  setRunButton(false);
}

async function runLfm2Agent() {
  const sceneId = state.dm3 ? state.dm3.id : null;
  if (!sceneId) {
    setStatus("Pick a DM3 case first (LFM2 agent reads precomputed tools by case_id)");
    return;
  }
  const btn = $("run-lfm2-btn");
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Running…";
  setStatus(`LFM2 agent: ${sceneId} (no streaming, ~30-60s)`);
  try {
    const r = await fetch("/api/run_lfm2_agent", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({scene_id: sceneId, include_images: false}),
    });
    if (!r.ok) {
      setStatus(`HTTP ${r.status}: ${(await r.text()).slice(0, 200)}`);
      return;
    }
    const data = await r.json();
    clearTrace();
    renderTraceEvent({type: "thought",
      text: `LFM2.5-VL agent · ${data.served_model || "?"} · include_images=${data.include_images}`});
    const obs = data.observations || [];
    let obsIdx = 0;
    for (const tc of data.tool_call_log || []) {
      renderTraceEvent({type: "action",  name: tc.name, arguments: tc.args || {}});
      // Terminal tools (submit_to_ground / drop) have no observation.
      // Non-terminal tool_calls get the next observation in order.
      if (tc.name !== "submit_to_ground" && tc.name !== "drop" && obsIdx < obs.length) {
        const o = obs[obsIdx++];
        let parsed = o.content;
        try { parsed = JSON.parse(o.content); } catch (_) { /* keep string */ }
        renderTraceEvent({type: "observation", name: o.name, result: parsed});
      }
    }
    renderTraceEvent({type: "final", name: data.terminal,
      result: {raw_log: data.raw_log, scene_id: data.scene_id}});
    setStatus(`LFM2 agent → terminal=${data.terminal}, ${(data.tool_call_log || []).length} tool calls`);
  } catch (e) {
    setStatus(`LFM2 agent error: ${e}`);
  } finally {
    btn.disabled = false;
    btn.textContent = orig;
  }
}


function runAgent() {
  if (state.eventSource) { stopAgent(); return; }   // toggle: button now acts as Stop
  if (!state.beforeKey || !state.afterKey) {
    setStatus("Fetch images first.");
    return;
  }
  clearTrace();
  updateBudget(BUDGET_MAX);
  resetMapsToOriginal();

  const qp = new URLSearchParams({
    before_key: state.beforeKey,
    after_key:  state.afterKey,
  });
  if (state.vlmProvider)  qp.set("provider", state.vlmProvider);
  if (state.vlmModel)     qp.set("model",    state.vlmModel);
  if (state.dm3 && state.dm3.id) qp.set("scene_id", state.dm3.id);
  const instrEl = $("agent-instructions");
  const instructions = instrEl ? (instrEl.value || "").trim() : "";
  if (instructions) qp.set("instructions", instructions);
  const url = `/api/run_agent?${qp.toString()}`;
  const es = new EventSource(url);
  state.eventSource = es;
  setRunButton(true);

  es.onmessage = (msg) => {
    let ev;
    try { ev = JSON.parse(msg.data); } catch { return; }
    renderTraceEvent(ev);

    if (ev.type === "error" || ev.type === "final") {
      // Server-side terminal event — stop reconnecting.
      stopAgent();
      return;
    }

    if (ev.type === "observation") {
      const r = ev.result || {};
      if (ev.name === "check_downlink_budget" && r.remaining_bytes !== undefined) {
        updateBudget(r.remaining_bytes);
      }
      if (r.zoomed_before_key && r.zoomed_after_key) {
        setImage("before", r.zoomed_before_key);
        setImage("after",  r.zoomed_after_key);
        const ratio = r.zoom_ratio ? `${r.zoom_ratio}x` : "";
        const bbox = r.crop_pixel_bbox ? JSON.stringify(r.crop_pixel_bbox) : "";
        setMapLabel("before", `Before [ZOOMED ${ratio}]`);
        setMapLabel("after",  `After [ZOOMED ${ratio} ${bbox}]`);
      }
    }
  };
  es.addEventListener("end", () => stopAgent());
  // Without this guard EventSource auto-reconnects on stream close → infinite loop
  es.onerror = () => stopAgent();
}

// ---- Boot wiring ----

window.addEventListener("DOMContentLoaded", async () => {
  await loadTemplates();
  await loadDM3Cases();
  await initProviders();
  initMaps();

  $("fetch-btn").addEventListener("click", fetchImages);
  $("fetch-fireedge-btn").addEventListener("click", fetchImagesFireEdge);
  $("save-pair-btn").addEventListener("click", saveCurrentPair);
  $("cache-use-btn").addEventListener("click", useCachedPair);
  $("trace-clear-btn").addEventListener("click", () => {
    clearTrace();
    setStatus("trace cleared (saved files unaffected)");
  });
  $("settings-btn").addEventListener("click", openSettings);
  $("settings-close-btn").addEventListener("click", closeSettings);
  $("settings-modal").querySelector(".modal-backdrop").addEventListener("click", closeSettings);
  $("run-btn").addEventListener("click", runAgent);
  $("run-lfm2-btn").addEventListener("click", runLfm2Agent);
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
      if (state.damageVisible) await loadDamageOverlay();
      else                      clearDamageOverlay();
    });
  }

  $("dm3-case").addEventListener("change", onDM3Change);

  // Geocode search with debounce
  let geoTimer = null;
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

});
