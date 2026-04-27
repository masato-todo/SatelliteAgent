// main.js — runAgent SSE + boot wiring

import { state, $, setStatus, updateBudget, BUDGET_MAX } from "./state-utils.js";
import { initMaps, setImage, setMapLabel, resetMapsToOriginal,
         loadDamageOverlay, clearDamageOverlay, toggleDrawMode, clearDrawnBbox } from "./maps.js";
import { loadDM3Cases, onDM3Change, loadTemplates, fetchImages,
         searchBeforeCandidates, searchAfterCandidates, geocodeSearch } from "./dm3-fetch.js";
import { runTool, setToolsStatus } from "./tools.js";
import { toggleRecording, openSubmitModal, closeSubmitModal,
         onSubmitConfirm, onDropClick, openTracesModal, closeTracesModal,
         clearTrace, renderTraceEvent, discardTrace } from "./annotate-traces.js";

// ---- Agent (SSE) ----

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
  es.addEventListener("end", () => { es.close(); state.eventSource = null; });
  es.onerror = () => { /* stream ended or server closed */ };
}

// ---- Boot wiring ----

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

  // Auto-fetch on load
  // fetchImages();
});
