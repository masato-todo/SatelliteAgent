// annotate-traces.js — recording mode, submit modal, trace events, traces browser

import { state, $, escapeHtml, setStatus, updateBudget, BUDGET_MAX } from "./state-utils.js";
import { resetMapsToOriginal, clearDrawnBbox } from "./maps.js";
import { obsTopSignal } from "./tools.js";

// ---- Mode + status ----

function setMode(recording) {
  state.annotating = recording;
  $("annotate-controls").hidden = !recording;
  $("mode-badge").textContent = recording ? "● RECORDING" : "IDLE";
  $("mode-badge").className = recording ? "mode-annotate" : "mode-agent";
  $("annotate-btn").textContent = recording ? "■ Stop Recording" : "● Start Recording";
  $("run-btn").disabled = recording;
  if (recording) setAnnotateStatus("Recording started. Tool actions will be traced.");
  else            setAnnotateStatus("");
}

export function setAnnotateStatus(msg) {
  $("annotate-status").textContent = msg;
}

// Allow other modules (maps.js bbox draw) to surface annotation messages
state.setAnnotateStatus = setAnnotateStatus;

// ---- Trace event push + render ----

export function pushAndRender(ev) {
  state.traceEvents.push(ev);
  renderTraceEvent(ev);
}

// Wire as the global emitter so tools.js / maps.js can push without import cycle.
state.traceEmitter = pushAndRender;

function consumeThought() {
  const t = $("annotate-thought").value.trim();
  if (t) {
    pushAndRender({type: "thought", text: t});
    $("annotate-thought").value = "";
  }
}

export function clearTrace() { $("trace").innerHTML = ""; }

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

export function renderTraceEvent(ev) {
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

// ---- Recording control ----

export async function toggleRecording() {
  if (state.annotating) {
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
    $("annotate-scenario").value = state.template || "";
    setMode(true);
  }
}

export function discardTrace() {
  state.traceEvents = [];
  state.traceFinal = null;
  clearTrace();
  resetMapsToOriginal();
  clearDrawnBbox();
  setAnnotateStatus("Discarded.");
}

// ---- Submit / Drop modal ----

export function openSubmitModal() {
  $("submit-modal").hidden = false;
  if (state.dm3 && state.dm3.mapped_class) {
    $("f-change-type").value = state.dm3.mapped_class;
  }
  $("f-description").focus();
}

export function closeSubmitModal() { $("submit-modal").hidden = true; }

export function onSubmitConfirm() {
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
    change_type, urgency, description, attach_image,
  };

  closeSubmitModal();
  setAnnotateStatus("Terminal: submit_to_ground recorded. Stop Recording to save.");
}

export function onDropClick() {
  consumeThought();
  pushAndRender({type: "action", name: "drop", arguments: {}});
  pushAndRender({type: "observation", name: "drop", result: {status: "dropped"}});
  pushAndRender({type: "final", name: "drop", result: {status: "dropped"}});
  state.traceFinal = {action: "drop"};
  setAnnotateStatus("Terminal: drop recorded. Stop Recording to save.");
}

// ---- Save trace ----

async function saveTrace() {
  if (state.traceEvents.length === 0) {
    setAnnotateStatus("Nothing to save yet.");
    return;
  }
  const metadata = {
    scenario_id: $("annotate-scenario").value.trim() || state.template || "unknown",
    profile: $("annotate-profile").value,
    annotator: "human",
    source_scenario: {
      lat: parseFloat($("lat").value),
      lon: parseFloat($("lon").value),
      before_date: $("before_date").value,
      after_date:  $("after_date").value,
      size_km:     parseFloat($("size_km").value),
      window_days: parseInt($("window_days").value, 10),
    },
    resolved_scenes: {
      before_datetime: state.beforeMeta && state.beforeMeta.datetime,
      after_datetime:  state.afterMeta  && state.afterMeta.datetime,
      before_key:      state.beforeKey,
      after_key:       state.afterKey,
    },
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

export async function openTracesModal() {
  $("traces-modal").hidden = false;
  $("traces-detail").innerHTML = `<div class="traces-detail-empty">Loading list…</div>`;
  await refreshTracesList();
}

export function closeTracesModal() { $("traces-modal").hidden = true; }

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
      const kindBadge = t.kind === "agent" ? `<span class="trace-kind agent">AGENT</span>`
                       : t.kind === "human" ? `<span class="trace-kind human">HUMAN</span>`
                       : "";
      const gtBadge = t.gt_match === true  ? `<span class="trace-gt ok">GT✓</span>`
                    : t.gt_match === false ? `<span class="trace-gt miss">GT✗</span>`
                    : "";
      const fullName = t.scenario_id || t.filename;
      row.innerHTML = `
        <div class="trace-row-head" title="${escapeHtml(fullName)}">${kindBadge}${gtBadge} ${escapeHtml(fullName)}${ct}</div>
        <div class="trace-row-meta">${escapeHtml((t.created_at || "").slice(0, 19))} · ${t.n_events || 0} events · ${escapeHtml(t.profile || "?")} · ${escapeHtml(t.annotator || "?")} · ${(t.size_bytes || 0)} B</div>
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
      scenario: ${escapeHtml(meta.scenario_id || meta.scene_id || "?")}<br>
      kind: ${escapeHtml(meta.scenario_type || (meta.provider ? "agent" : "human"))} ·
      profile/provider: ${escapeHtml(meta.profile || meta.provider || "?")} ·
      annotator/model: ${escapeHtml(meta.annotator || meta.model || "?")}<br>
      created: ${escapeHtml((meta.created_at || meta.collected_at || "").slice(0, 19))}<br>
      lat/lon: ${escapeHtml(String(meta.lat ?? meta.source_scenario?.lat ?? "?"))} / ${escapeHtml(String(meta.lon ?? meta.source_scenario?.lon ?? "?"))} · size: ${escapeHtml(String(meta.size_km ?? meta.source_scenario?.size_km ?? "?"))} km<br>
      dates: ${escapeHtml(String(meta.before_date ?? meta.source_scenario?.before_date ?? "?"))} → ${escapeHtml(String(meta.after_date ?? meta.source_scenario?.after_date ?? "?"))}<br>
      expected: ${escapeHtml(meta.expected_action || "?")} (${escapeHtml(meta.expected_class || "?")}) · gt_match: ${doc.gt_match === true ? "✓" : doc.gt_match === false ? "✗" : "—"}<br>
      <b>final:</b> ${escapeHtml(final.action || final.name || "?")} · change_type=${escapeHtml(final.change_type || (final.result || {}).change_type || "?")} · urgency=${escapeHtml(String(final.urgency ?? (final.result || {}).urgency ?? "?"))}
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
