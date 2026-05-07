"""Evaluate the wildfire-precursor LoRA in *production conditions*:
HF row coords → SimSat fetch (twice, T-14 + T-7) → preprocessing → LFM.

Mirrors scripts/eval_wildfire_hf_simsat.py for the precursor model.

Difference vs eval_wildfire_precursor_hf.py:
  - eval_wildfire_precursor_hf.py:        HF images   → LFM
  - eval_wildfire_precursor_hf_simsat.py: HF coords + sentinel_datetimes
                                          → SimSat → preprocess → LFM

The gap between the two recalls quantifies:
  (a) SimSat fetch differences vs the dataset's training-time SimSat
  (b) our percentile/Lanczos pipeline vs the recipe used during training
combined.

Channel encoding (matches the LoRA's system prompt):
  R = SWIR 2.2μm (B12)
  G = NIR  865nm (B8A)        ← NOTE: B8A, not B08
  B = SWIR 1.6μm (B11)

Preprocessing matches the upstream training pipeline
(yujiyamaguchi/liquid-ai-space-hackathon apps/fireguard/.../dataset_builder.py
:_arrs_to_composites): a SINGLE p2/p98 pair is computed across ALL 3
channels AND BOTH time frames, then applied identically to every
channel of every frame. This preserves the SWIR/NIR ratio across
T-14 / T-7 so the model can read NDMI-style drying as a relative
brightness shift. → uint8 RGB → Lanczos 448×448.

Endpoint: see eval_wildfire_precursor_hf.py header.

Usage:
    python scripts/eval_wildfire_precursor_hf_simsat.py
    python scripts/eval_wildfire_precursor_hf_simsat.py --split val --limit 5
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from simsat_client.sentinel import fetch_sentinel_array, SimSatError  # noqa: E402

LFM_BASE_URL = os.environ.get("LFM_PRECURSOR_BASE_URL", "http://localhost:8089/v1")
LFM_MODEL    = os.environ.get("LFM_PRECURSOR_MODEL",    "lfm2.5-vl-450m-wildfire-precursor")
SIMSAT_BASE  = os.environ.get("SIMSAT_API_URL",         "http://localhost:9005")

HF_DATASET = "YujiYamaguchi/fireguard-sentinel2-wildfire-precursor"
HF_CONFIG  = "pair14_7"

PRECURSOR_BANDS = ["swir22", "nir08", "swir16"]   # R, G, B (in that order)
TARGET_SIZE     = 448                              # LFM 2.5-VL vision encoder native
TRAINING_AOI_KM = float(os.environ.get("PRECURSOR_AOI_KM", 5.0))


def image_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _fetch_raw_rgb(lat: float, lon: float, ts: str, window_days: int) -> tuple[np.ndarray, str]:
    """Fetch [swir22, nir08, swir16] for one frame and stack into HxWx3
    (channel order matches the LoRA's expected R,G,B). Returns (rgb_array,
    actual_simsat_datetime)."""
    sa = fetch_sentinel_array(
        lat=lat, lon=lon, timestamp=ts, bands=PRECURSOR_BANDS,
        size_km=TRAINING_AOI_KM, base_url=SIMSAT_BASE,
        resolution_meters=10, window_days=window_days, timeout=120,
    )
    band_idx = {n: i for i, n in enumerate(sa.band_names)}
    rgb = np.stack([
        sa.array[band_idx["swir22"]],   # R
        sa.array[band_idx["nir08"]],    # G
        sa.array[band_idx["swir16"]],   # B
    ], axis=-1)
    actual = sa.metadata.get("date") or sa.metadata.get("datetime") or ""
    return rgb, actual


def _composite_pair_with_shared_scale(rgb_t14: np.ndarray, rgb_t7: np.ndarray) -> tuple[Image.Image, Image.Image]:
    """Apply the upstream training recipe: a single (p2, p98) pair across
    ALL 3 channels AND BOTH time frames, then normalize identically.
    Mirrors fireguard/finetune/dataset_builder.py:_arrs_to_composites.
    """
    stacked = np.concatenate([rgb_t14, rgb_t7], axis=0)
    p2  = np.percentile(stacked, 2.0)
    p98 = np.percentile(stacked, 98.0)
    out: list[Image.Image] = []
    for rgb in (rgb_t14, rgb_t7):
        n = np.clip((rgb.astype(np.float32) - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
        img = Image.fromarray((n * 255).astype(np.uint8), mode="RGB")
        if img.size != (TARGET_SIZE, TARGET_SIZE):
            img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
        out.append(img)
    return out[0], out[1]


def replace_image_placeholders(messages: list[dict], image_urls: list[str]) -> list[dict]:
    cursor = 0
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m); continue
        new_content = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image":
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_urls[cursor]},
                })
                cursor += 1
            else:
                new_content.append(c)
        out.append({"role": m["role"], "content": new_content})
    return out


def call_lfm(messages: list[dict], timeout: float = 180.0) -> dict:
    r = requests.post(
        f"{LFM_BASE_URL.rstrip('/')}/chat/completions",
        json={"model": LFM_MODEL, "max_tokens": 64, "temperature": 0.1,
              "top_p": 0.9, "messages": messages},
        timeout=timeout,
    )
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:300]}"}
    try:
        text = r.json()["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as e:
        return {"error": f"unexpected response shape: {e}"}
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        return json.loads(text)
    except Exception as e:
        return {"error": f"JSON parse failed: {e}", "raw": text[:300]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=HF_DATASET)
    ap.add_argument("--config",  default=HF_CONFIG)
    ap.add_argument("--split",   default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit",   type=int, default=0)
    ap.add_argument("--window-days", type=int, default=1,
                    help="Tight window so SimSat returns the exact training STAC item. "
                         "Increase if your SimSat catalog has gaps.")
    ap.add_argument("--out",     default=str(ROOT / "data" / "eval_precursor_hf_simsat.jsonl"))
    args = ap.parse_args()

    try:
        requests.get(f"{LFM_BASE_URL.rstrip('/')}/models", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: precursor server unreachable at {LFM_BASE_URL}: {e}", file=sys.stderr)
        return 2

    from datasets import load_dataset
    print(f"[1/2] loading {args.dataset} config={args.config} split={args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"      {len(ds)} rows  (size_km={TRAINING_AOI_KM})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fp = out_path.open("w", encoding="utf-8")

    n_tp = n_fp = n_tn = n_fn = 0
    n_err = 0
    n_dt_match = n_dt_miss = 0
    t_start = time.time()

    print(f"\n[2/2] running eval (SimSat → preprocess → LFM)...", flush=True)
    for i, row in enumerate(ds):
        gt = int(row["label"]); gt_high = gt == 1
        lat = float(row["lat"]); lon = float(row["lon"])
        sdts = row.get("sentinel_datetimes") or []
        if len(sdts) < 2:
            print(f"  [{i+1:>3}/{len(ds)}] SKIP: row has only {len(sdts)} sentinel_datetimes", flush=True)
            n_err += 1
            continue

        t0 = time.time()
        try:
            rgb_t14, actual_dt1 = _fetch_raw_rgb(lat, lon, sdts[0], args.window_days)
            rgb_t7,  actual_dt2 = _fetch_raw_rgb(lat, lon, sdts[1], args.window_days)
            img1, img2 = _composite_pair_with_shared_scale(rgb_t14, rgb_t7)
        except SimSatError as e:
            elapsed = time.time() - t0
            print(f"  [{i+1:>3}/{len(ds)}] gt={gt} SIMSAT_ERR  err={e!s:.80}  {elapsed:.1f}s", flush=True)
            n_err += 1
            out_fp.write(json.dumps({
                "idx": i, "name": row.get("name"), "label": gt,
                "lat": lat, "lon": lon, "expected_dts": sdts,
                "error": f"SimSatError: {e}",
            }) + "\n")
            continue

        # datetime sanity (compare on day-level)
        dt_match_1 = actual_dt1[:10] == sdts[0][:10]
        dt_match_2 = actual_dt2[:10] == sdts[1][:10]
        dt_match = dt_match_1 and dt_match_2
        if dt_match:
            n_dt_match += 1
        else:
            n_dt_miss += 1

        # Build messages from the row's training prompts, swap in our composites
        msgs_in = json.loads(row["messages_json"])
        msgs_in = [m for m in msgs_in if m.get("role") != "assistant"]
        urls = [image_to_data_url(img1), image_to_data_url(img2)]
        msgs = replace_image_placeholders(msgs_in, urls)

        parsed = call_lfm(msgs)
        elapsed = time.time() - t0

        if "error" in parsed:
            verdict, pred, risk = "ERR", None, None
            n_err += 1
        else:
            risk = parsed.get("risk_level")
            pred_high = (risk == "HIGH")
            pred = pred_high
            if   gt_high and pred_high:     n_tp += 1; verdict = "TP"
            elif gt_high and not pred_high: n_fn += 1; verdict = "FN"
            elif not gt_high and pred_high: n_fp += 1; verdict = "FP"
            else:                            n_tn += 1; verdict = "TN"

        match_tag = "✓" if dt_match else "✗"
        print(f"  [{i+1:>3}/{len(ds)}] src={row.get('source','?'):<12} gt={gt} "
              f"pred={str(risk):<6} {verdict}  dt={match_tag}  "
              f"sim=({actual_dt1[:10]},{actual_dt2[:10]})  {elapsed:.1f}s"
              + (("  err=" + parsed.get("error","")[:80]) if "error" in parsed else ""),
              flush=True)
        out_fp.write(json.dumps({
            "idx": i, "name": row.get("name"), "label": gt,
            "source": row.get("source"),
            "lat": lat, "lon": lon,
            "query_date": row.get("query_date"),
            "expected_sentinel_datetimes": sdts,
            "actual_sentinel_datetimes": [actual_dt1, actual_dt2],
            "datetime_day_match": dt_match,
            "frp": row.get("frp"),
            "pred_risk": risk, "verdict": verdict,
            "raw": parsed,
            "elapsed_s": round(elapsed, 2),
        }) + "\n")
        out_fp.flush()

    out_fp.close()

    n_total = n_tp + n_fp + n_tn + n_fn
    acc  = (n_tp + n_tn) / n_total if n_total else 0.0
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    rec  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print()
    print(f"[results] split={args.split}, n={n_total} (errors: {n_err})", flush=True)
    print(f"  TP={n_tp:3}  FP={n_fp:3}  TN={n_tn:3}  FN={n_fn:3}", flush=True)
    print(f"  accuracy  = {acc:.3f}", flush=True)
    print(f"  precision = {prec:.3f}", flush=True)
    print(f"  recall    = {rec:.3f}", flush=True)
    print(f"  f1        = {f1:.3f}", flush=True)
    print(f"  datetime day-level match (both frames): {n_dt_match}/{n_dt_match + n_dt_miss}", flush=True)
    print(f"  elapsed   = {time.time() - t_start:.0f}s", flush=True)
    print(f"  jsonl     = {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
