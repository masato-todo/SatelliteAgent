"""Evaluate the LFM2.5-VL wildfire-precursor LoRA on its HF dataset.

Mirrors scripts/eval_wildfire_hf.py for the precursor model:
   YujiYamaguchi/lfm2-5-vl-450m-wildfire-precursor-pair14_7
   ↑ trained on
   YujiYamaguchi/fireguard-sentinel2-wildfire-precursor (config=pair14_7)

Each row carries two images (image = T-14d, image_1 = T-7d) and a
verbatim `messages_json` (system + user) that the LoRA was fine-tuned
against. We send those exact messages with the image bytes substituted
in, then parse the `{"risk_level": "HIGH"|"LOW"}` response.

This is the cleanest possible eval — skips SimSat fetch + preprocessing
entirely. The gap between this script and
`eval_wildfire_precursor_hf_simsat.py` quantifies SimSat / preprocessing
drift vs the dataset's training-time pipeline.

Endpoint: a transformers+peft server with the precursor adapter loaded.
Spin one up alongside the existing wildfire one (different port + adapter):

    LFM_BASE_DIR=$PWD/models/wildfire-precursor-staging/base \\
    LFM_ADAPTER_DIR=$PWD/models/wildfire-precursor-staging/adapter \\
    LFM_MODEL_NAME=lfm2.5-vl-450m-wildfire-precursor \\
    uvicorn services.wildfire.server:app --host 0.0.0.0 --port 8089

(or run the same Dockerfile under services/wildfire/Dockerfile with the
precursor adapter mounted on a different host port.)

Usage:
    python scripts/eval_wildfire_precursor_hf.py
    python scripts/eval_wildfire_precursor_hf.py --split test
    python scripts/eval_wildfire_precursor_hf.py --split val --limit 5
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

import requests

ROOT = Path(__file__).resolve().parent.parent

LFM_BASE_URL = os.environ.get("LFM_PRECURSOR_BASE_URL", "http://localhost:8089/v1")
LFM_MODEL    = os.environ.get("LFM_PRECURSOR_MODEL",    "lfm2.5-vl-450m-wildfire-precursor")

HF_DATASET   = "YujiYamaguchi/fireguard-sentinel2-wildfire-precursor"
HF_CONFIG    = "pair14_7"


def image_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def replace_image_placeholders(messages: list[dict], image_urls: list[str]) -> list[dict]:
    """The dataset's messages_json marks image positions with content
    blocks of `{"type": "image"}`. We replace them in order with
    `{"type": "image_url", "image_url": {"url": <data_url>}}` so the
    OpenAI-compat server can decode them."""
    cursor = 0
    out: list[dict] = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m); continue
        new_content = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "image":
                if cursor >= len(image_urls):
                    raise RuntimeError(f"messages_json has more [image] placeholders than image fields ({len(image_urls)})")
                new_content.append({
                    "type": "image_url",
                    "image_url": {"url": image_urls[cursor]},
                })
                cursor += 1
            else:
                new_content.append(c)
        out.append({"role": m["role"], "content": new_content})
    if cursor != len(image_urls):
        raise RuntimeError(f"messages_json had only {cursor} [image] placeholders but row provided {len(image_urls)} images")
    return out


def call_lfm(messages: list[dict], timeout: float = 180.0) -> dict:
    body = {
        "model": LFM_MODEL,
        "max_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.9,
        "messages": messages,
    }
    r = requests.post(f"{LFM_BASE_URL.rstrip('/')}/chat/completions",
                      json=body, timeout=timeout)
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
    ap.add_argument("--config", default=HF_CONFIG)
    ap.add_argument("--split",  default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit",  type=int, default=0)
    ap.add_argument("--out",    default=str(ROOT / "data" / "eval_precursor_hf.jsonl"))
    args = ap.parse_args()

    # Server reachable?
    try:
        requests.get(f"{LFM_BASE_URL.rstrip('/')}/models", timeout=5).raise_for_status()
    except requests.RequestException as e:
        print(f"ERROR: precursor server unreachable at {LFM_BASE_URL}: {e}", file=sys.stderr)
        print("       see header of this script for how to spin one up.", file=sys.stderr)
        return 2

    from datasets import load_dataset
    print(f"[1/2] loading {args.dataset} config={args.config} split={args.split}...", flush=True)
    ds = load_dataset(args.dataset, args.config, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"      {len(ds)} rows", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fp = out_path.open("w", encoding="utf-8")

    n_tp = n_fp = n_tn = n_fn = 0
    n_err = 0
    pairs: list[tuple[int, float]] = []
    t_start = time.time()

    print(f"\n[2/2] running eval (HF images → LFM precursor server)...", flush=True)
    for i, row in enumerate(ds):
        gt = int(row["label"])  # 1 = HIGH-risk wildfire precursor, 0 = LOW
        gt_high = gt == 1

        # Build messages from the dataset's recorded ones (the system + user
        # prompt actually used at training time).
        msgs_in = json.loads(row["messages_json"])
        # Drop the trailing assistant turn — that's the GT we want to predict.
        msgs_in = [m for m in msgs_in if m.get("role") != "assistant"]
        urls = [image_to_data_url(row["image"]), image_to_data_url(row["image_1"])]
        try:
            msgs = replace_image_placeholders(msgs_in, urls)
        except Exception as e:
            print(f"  [{i+1:>3}/{len(ds)}] PROMPT_ERR: {e}", flush=True)
            n_err += 1
            continue

        t0 = time.time()
        parsed = call_lfm(msgs)
        elapsed = time.time() - t0

        if "error" in parsed:
            verdict, pred, risk = "ERR", None, None
            n_err += 1
        else:
            risk = parsed.get("risk_level")
            pred_high = (risk == "HIGH")
            pred = pred_high
            score = 1.0 if pred_high else 0.0
            if   gt_high and pred_high:     n_tp += 1; verdict = "TP"
            elif gt_high and not pred_high: n_fn += 1; verdict = "FN"
            elif not gt_high and pred_high: n_fp += 1; verdict = "FP"
            else:                            n_tn += 1; verdict = "TN"
            pairs.append((1 if gt_high else 0, score))

        print(f"  [{i+1:>3}/{len(ds)}] src={row.get('source','?'):<12} gt={gt} "
              f"pred={str(risk):<6} {verdict}  {elapsed:.1f}s"
              + (("  err=" + parsed.get("error","")[:80]) if "error" in parsed else ""),
              flush=True)
        out_fp.write(json.dumps({
            "idx": i, "name": row.get("name"), "label": gt,
            "source": row.get("source"),
            "lat": row.get("lat"), "lon": row.get("lon"),
            "query_date": row.get("query_date"),
            "sentinel_datetimes": row.get("sentinel_datetimes"),
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
    print(f"  elapsed   = {time.time() - t_start:.0f}s", flush=True)
    print(f"  jsonl     = {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
