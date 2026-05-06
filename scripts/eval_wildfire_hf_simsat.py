"""Eval wildfire LoRA in *production conditions*: HF row coords →
our SimSat fetch → tools.wildfire preprocessing → LFM.

Difference vs eval_wildfire_hf.py:
  - eval_wildfire_hf.py:        HF image → LFM        (skips SimSat & preprocessing)
  - eval_wildfire_hf_simsat.py: HF lat/lon/query_date → SimSat → preprocess → LFM

The gap between the two recalls quantifies:
  (a) SimSat fetch differences vs FireEdge's training-time SimSat
  (b) our percentile/Lanczos pipeline vs the recipe used during training
combined.

Usage:
    python scripts/eval_wildfire_hf_simsat.py
    python scripts/eval_wildfire_hf_simsat.py --split val --limit 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.wildfire import detect_wildfire_impl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YujiYamaguchi/fireedge-sentinel2-wildfire")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size-km", type=float, default=5.0,
                    help="FireEdge training AOI is 5 km (model_report.md §4)")
    ap.add_argument("--use-sentinel-datetime", action="store_true",
                    help="Send the row's sentinel_datetime (= the actual S2 frame "
                         "the LoRA saw at training time) as the SimSat timestamp, "
                         "instead of query_date. Combined with --window-days=1 "
                         "this reproduces the exact training image.")
    ap.add_argument("--window-days", type=int, default=12,
                    help="SimSat search window. Use 1 with --use-sentinel-datetime.")
    ap.add_argument("--out", default=str(ROOT / "data" / "eval_hf_simsat.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset
    print(f"[1/2] loading {args.dataset} split={args.split}...", flush=True)
    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"      {len(ds)} rows  (size_km={args.size_km})", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fp = out_path.open("w", encoding="utf-8")

    n_tp = n_fp = n_tn = n_fn = 0
    n_err = 0
    pairs: list[tuple[int, float]] = []
    n_dt_match = n_dt_miss = 0
    t_start = time.time()

    print(f"\n[2/2] running eval (SimSat → preprocess → LFM)...", flush=True)
    for i, row in enumerate(ds):
        gt = int(row["label"])
        gt_fire = gt == 1
        lat = float(row["lat"]); lon = float(row["lon"])
        qdate = row["query_date"]
        expected_dt = row.get("sentinel_datetime", "")
        src = row.get("source", "?")

        # When --use-sentinel-datetime is set, send the actual training-time
        # frame timestamp (and a tight 1d window) so SimSat returns the same
        # STAC item the LoRA was trained on.
        ts = expected_dt if (args.use_sentinel_datetime and expected_dt) else qdate

        t0 = time.time()
        try:
            parsed = detect_wildfire_impl(
                lat, lon, ts, size_km=args.size_km,
                window_days=args.window_days,
            )
        except Exception as e:
            parsed = {"error": f"{type(e).__name__}: {e}"}
        elapsed = time.time() - t0

        # Datetime sanity: SimSat vs HF training capture
        actual_dt = (parsed.get("sentinel") or {}).get("datetime") or ""
        dt_match = actual_dt[:10] == (expected_dt or "")[:10]
        if expected_dt:
            (n_dt_match if dt_match else n_dt_miss).__class__  # no-op
            if dt_match: n_dt_match += 1
            else: n_dt_miss += 1

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

        match_tag = "✓" if dt_match else "✗"
        print(f"  [{i+1:>3}/{len(ds)}] src={src:<13} gt={gt} "
              f"pred={'fire' if pred else 'no':<4} {verdict}  "
              f"dt={match_tag} q={qdate} sim={actual_dt[:10] if actual_dt else '?':10}  "
              f"{elapsed:.1f}s"
              + ("  err=" + parsed.get("error", "")[:80] if "error" in parsed else ""),
              flush=True)
        out_fp.write(json.dumps({
            "idx": i, "label": gt, "source": src,
            "lat": lat, "lon": lon, "query_date": qdate,
            "expected_sentinel_datetime": expected_dt,
            "actual_sentinel_datetime": actual_dt,
            "datetime_day_match": dt_match,
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
    print(f"  datetime day-level match: {n_dt_match}/{n_dt_match + n_dt_miss}", flush=True)
    print(f"  elapsed   = {time.time() - t_start:.0f}s", flush=True)
    print(f"  jsonl     = {out_path}", flush=True)

    # AUC-ROC
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
