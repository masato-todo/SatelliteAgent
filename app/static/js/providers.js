// providers.js — VLM provider/model selector wired to /api/providers (modal)
import { state, $ } from "./state-utils.js";

const STORAGE_KEY = "satagent.vlm";

export function openSettings()  { const m = $("settings-modal"); if (m) m.hidden = false; }
export function closeSettings() { const m = $("settings-modal"); if (m) m.hidden = true; }

// Header badge rendering. Owns the source-of-truth: whatever is currently in
// state.vlmProvider/state.vlmModel is what Run Agent / classify_change will
// use. We re-render this on every Settings change so the header is a live
// preview, not a stale snapshot of the server-resolved default.
function renderProviderBadge(cfg) {
  const el = $("provider-info");
  if (!el) return;
  const kind  = (cfg && cfg.kind)  || "none";
  const name  = (cfg && cfg.name)  || "?";
  const model = state.vlmModel || (cfg && cfg.default_model) || "?";
  if (kind === "gemini") {
    el.textContent = `Gemini · ${model}`;
    el.className   = "provider-gemini";
  } else if (kind === "openai_compat") {
    el.textContent = `${name} · ${model}`;
    el.className   = "provider-vllm";
  } else if (kind === "lfm2_multiturn") {
    el.textContent = `${name} · ${model}`;
    el.className   = "provider-vllm";
  } else {
    el.textContent = "NO PROVIDER (check config/providers.yaml or .env)";
    el.className   = "provider-stub";
  }
}

export async function initProviders() {
  const sel = $("vlm-provider"); const mod = $("vlm-model");
  if (!sel || !mod) return;

  let data;
  try {
    const res = await fetch("/api/providers");
    if (!res.ok) return;
    data = await res.json();
  } catch (e) {
    console.warn("providers fetch failed", e);
    return;
  }

  const provs = data.providers || [];
  if (!provs.length) {
    sel.innerHTML = '<option value="">(none configured)</option>';
    sel.disabled = true;
    mod.disabled = true;
    return;
  }

  let saved = {};
  try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); } catch (e) { /* */ }

  sel.innerHTML = "";
  for (const p of provs) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = `${p.name} (${p.kind})`;
    opt.dataset.cfg = JSON.stringify(p);
    sel.appendChild(opt);
  }

  const initialProv = provs.find(p => p.name === saved.provider) ? saved.provider : provs[0].name;
  sel.value = initialProv;
  populateModels(sel, mod, saved.model);

  sel.addEventListener("change", () => populateModels(sel, mod, null));
  mod.addEventListener("change", () => { persist(); renderFromSelection(sel); });

  state.vlmProvider = sel.value;
  state.vlmModel    = mod.value;
  persist();
  renderFromSelection(sel);
}

function renderFromSelection(sel) {
  const opt = sel.selectedOptions[0];
  if (!opt) return;
  let cfg = {};
  try { cfg = JSON.parse(opt.dataset.cfg || "{}"); } catch (e) { /* */ }
  renderProviderBadge(cfg);
}

function populateModels(sel, mod, preferModel) {
  const opt = sel.selectedOptions[0];
  if (!opt) return;
  const cfg = JSON.parse(opt.dataset.cfg || "{}");
  const models = cfg.models || [];
  mod.innerHTML = "";
  for (const m of models) {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    mod.appendChild(o);
  }
  const initial = models.includes(preferModel) ? preferModel : (cfg.default_model || models[0]);
  if (initial) mod.value = initial;
  state.vlmProvider = sel.value;
  state.vlmModel    = mod.value;
  persist();
  renderProviderBadge(cfg);
}

function persist() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({
      provider: state.vlmProvider,
      model:    state.vlmModel,
    }));
  } catch (e) { /* */ }
}
