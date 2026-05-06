"""Evaluate the LFM2.5-VL wildfire LoRA on the official HF dataset.

Uses YujiYamaguchi/fireedge-sentinel2-wildfire — same false-color images
the LoRA was trained on, with FT-style binary `label` (0/1) GT. This is
the cleanest possible eval: skips SimSat fetch + preprocessing entirely
and just feeds each image straight to the wildfire endpoint.

Endpoint: tools/wildfire.py docker container on http://localhost:8085/v1
Prompt:   FireEdge FIRE_DETECTION_FT_PROMPT (matches LoRA training).

Usage:
    python scripts/eval_wildfire_hf.py
    python scripts/eval_wildfire_hf.py --split test --limit 0
    python scripts/eval_wildfire_hf.py --split val
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# FT_PROMPT — exact training prompt (apps/fireedge/src/interfaces.py:FIRE_DETECTION_FT_PROMPT)
USER_PROMPT = (
    "Examine this satellite false-color composite image "
    "(R=SWIR2.2μm, G=SWIR1.6μm, B=NIR).\n\n"
    "Does this scene contain active fire or burn scar?\n"
    'Respond with JSON only: {"fire_detected": true} or {"fire_detected": false}'
)


def image_to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def call_lfm(base_url: str, model: str, data_url: str, timeout: float = 120) -> dict:
    body = {
        "model": model,
        "max_tokens": 64,
        "temperature": 0.1,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": USER_PROMPT},
        ]}],
    }
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions",
                      json=body, timeout=timeout)
    if r.status_code != 200:
        return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    text = (r.json()["choices"][0]["message"]["content"] or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except Exception as e:
        return {"error": f"json: {e}", "raw": text[:200]}
    return parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YujiYamaguchi/fireedge-sentinel2-wildfire")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default="http://localhost:8085/v1")
    ap.add_argument("--model", default="lfm2.5-vl-450m-wildfire")
    ap.add_argument("--out", default=str(ROOT / "data" / "eval_hf.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset
    print(f"[1/2] loading {args.dataset} split={args.split}...", flush=True)
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"      {len(ds)} rows", flush=True)
    print(f"      base_url={args.base_url} model={args.model}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fp = out_path.open("w", encoding="utf-8")

    n_tp = n_fp = n_tn = n_fn = 0
    n_err = 0
    pairs: list[tuple[int, float]] = []  # (gt, score)
    t_start = time.time()

    print(f"\n[2/2] running eval...", flush=True)
    for i, row in enumerate(ds):
        gt = int(row["label"])
        gt_fire = gt == 1
        src = row.get("source", "?")
        try:
            data_url = image_to_data_url(row["image"])
        except Exception as e:
            print(f"  [{i+1:>3}/{len(ds)}] image encode FAIL: {e}", flush=True)
            n_err += 1
            continue
        t0 = time.time()
        try:
            parsed = call_lfm(args.base_url, args.model, data_url)
        except Exception as e:
            parsed = {"error": f"{type(e).__name__}: {e}"}

        if "error" in parsed:
            n_err += 1
            verdict = "ERR"
            pred = None
            score = 0.0
        else:
            pred_fire = bool(parsed.get("fire_detected"))
            pred = pred_fire
            score = 1.0 if pred_fire else 0.0
            if gt_fire and pred_fire:    n_tp += 1; verdict = "TP"
            elif gt_fire and not pred_fire: n_fn += 1; verdict = "FN"
            elif not gt_fire and pred_fire: n_fp += 1; verdict = "FP"
            else: n_tn += 1; verdict = "TN"
            pairs.append((1 if gt_fire else 0, score))

        elapsed = time.time() - t0
        print(f"  [{i+1:>3}/{len(ds)}] src={src:<13} gt={gt} "
              f"pred={'fire' if pred else 'no':<4} {verdict}  {elapsed:.1f}s"
              + ("  err=" + parsed.get("error", "")[:60] if "error" in parsed else ""),
              flush=True)
        out_fp.write(json.dumps({
            "idx": i, "label": gt, "source": src,
            "pred_fire": pred, "verdict": verdict,
            "fire_detected_raw": parsed.get("fire_detected"),
            "error": parsed.get("error"),
            "elapsed_s": round(elapsed, 2),
        }) + "\n")
        out_fp.flush()

    out_fp.close()

    n_total = n_tp + n_fp + n_tn + n_fn
    acc = (n_tp + n_tn) / n_total if n_total else 0.0
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

    # AUC-ROC (binary scores 0/1 → AUC = recall when prec=1, but compute anyway)
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = sum(1 for y, _ in pairs if y == 0)
    if n_pos and n_neg:
        pairs.sort(key=lambda x: x[1])
        rank_sum = 0.0
        i_p = 0
        while i_p < len(pairs):
            j = i_p
            while j < len(pairs) and pairs[j][1] == pairs[i_p][1]:
                j += 1
            avg_rank = (i_p + j + 1) / 2
            for k in range(i_p, j):
                if pairs[k][0] == 1:
                    rank_sum += avg_rank
            i_p = j
        u = rank_sum - n_pos * (n_pos + 1) / 2
        print(f"  AUC-ROC   = {u / (n_pos * n_neg):.3f}  (pos={n_pos}, neg={n_neg})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
