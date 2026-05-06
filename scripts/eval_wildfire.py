"""Quick evaluation harness for detect_wildfire (LFM2.5-VL wildfire LoRA).

Iterates canonical_dataset.yaml entries with label in {fire, no_change}
and asks the wildfire tool whether the AFTER image shows an active fire.
Uses tools.wildfire.detect_wildfire_impl directly (same code path as the
UI button), so the result mirrors what users see.

Usage:
    python scripts/eval_wildfire.py
    python scripts/eval_wildfire.py --limit 20
    python scripts/eval_wildfire.py --only-fire
    python scripts/eval_wildfire.py --threshold 0.5    # default
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools.wildfire import detect_wildfire_impl


def _request_field(case: dict, key: str):
    """Read date / size_km uniformly across canonical_dataset.yaml
    (nested under 'request') and firms_fire_cases.yaml (top-level)."""
    return (case.get("request") or {}).get(key) or case.get(key)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-yaml",
                    default=str(ROOT / "data" / "canonical_dataset.yaml"),
                    help="Path to a yaml whose top-level 'cases' has fire/no_change entries.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-fire", action="store_true")
    ap.add_argument("--only-negative", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="fire_confidence cutoff for TP/FP decision (default 0.5)")
    ap.add_argument("--out", default=str(ROOT / "data" / "eval_wildfire.jsonl"))
    args = ap.parse_args()

    with open(args.cases_yaml, encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    cases = doc.get("cases") or []

    targets = []
    for c in cases:
        label = c.get("label") or c.get("event_type")
        # FIRMS yaml: event_type=fire / no_change. canonical: label=fire / no_change.
        if label not in ("fire", "no_change"):
            continue
        if args.only_fire and label != "fire":
            continue
        if args.only_negative and label != "no_change":
            continue
        c["_label"] = label
        targets.append(c)
    if args.limit:
        targets = targets[: args.limit]
    print(f"[plan] {len(targets)} cases (fire={sum(1 for c in targets if c['_label']=='fire')}, "
          f"no_change={sum(1 for c in targets if c['_label']=='no_change')}) "
          f"from {args.cases_yaml}", flush=True)

    out_path = Path(args.out)
    out_fp = out_path.open("w", encoding="utf-8")

    n_tp = n_fp = n_tn = n_fn = 0
    n_err = 0
    t_start = time.time()

    for i, case in enumerate(targets, 1):
        cid = case["id"]
        gt_label = case["_label"]
        gt_fire = gt_label == "fire"

        lat = float(case["lat"])
        lon = float(case["lon"])
        size_km = float(_request_field(case, "size_km") or 10.0)
        after_date = _request_field(case, "after_date")
        if not after_date:
            print(f"  [{i:>3}/{len(targets)}] {cid:<55} SKIP (no after_date)", flush=True)
            continue

        t0 = time.time()
        try:
            result = detect_wildfire_impl(lat, lon, after_date, size_km)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}

        if "error" in result:
            n_err += 1
            verdict = "ERR"
            pred = None
            conf = 0.0
        else:
            conf = float(result.get("fire_confidence") or 0.0)
            pred_fire = bool(result.get("fire_detected")) and conf >= args.threshold
            pred = pred_fire
            if gt_fire and pred_fire:    n_tp += 1; verdict = "TP"
            elif gt_fire and not pred_fire: n_fn += 1; verdict = "FN"
            elif not gt_fire and pred_fire: n_fp += 1; verdict = "FP"
            else: n_tn += 1; verdict = "TN"

        elapsed = time.time() - t0
        sev = result.get("severity", "?")
        err_msg = (" err=" + result.get("error", "")[:60]) if "error" in result else ""
        print(f"  [{i:>3}/{len(targets)}] {cid:<55} GT={gt_label:<9} "
              f"pred={'fire' if pred else 'no':<4} conf={conf:.2f} sev={sev:<8} "
              f"{verdict}  {elapsed:.1f}s{err_msg}", flush=True)

        out_fp.write(json.dumps({
            "id": cid, "gt_label": gt_label, "gt_fire": gt_fire,
            "predicted_fire": pred, "fire_confidence": conf,
            "severity": result.get("severity"),
            "fire_detected_raw": result.get("fire_detected"),
            "smoke_detected": result.get("smoke_detected"),
            "smoke_confidence": result.get("smoke_confidence"),
            "description": result.get("description"),
            "error": result.get("error"),
            "verdict": verdict,
            "elapsed_s": round(elapsed, 2),
        }, ensure_ascii=False) + "\n")
        out_fp.flush()

    out_fp.close()

    # Metrics
    n_total = n_tp + n_fp + n_tn + n_fn
    acc = (n_tp + n_tn) / n_total if n_total else 0.0
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    rec = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print()
    print(f"[results] {n_total} cases (errors: {n_err})", flush=True)
    print(f"  TP={n_tp:3}  FP={n_fp:3}  TN={n_tn:3}  FN={n_fn:3}", flush=True)
    print(f"  accuracy  = {acc:.3f}", flush=True)
    print(f"  precision = {prec:.3f}", flush=True)
    print(f"  recall    = {rec:.3f}", flush=True)
    print(f"  f1        = {f1:.3f}", flush=True)
    print(f"  elapsed   = {time.time() - t_start:.0f}s", flush=True)
    print(f"  per-case JSONL: {out_path}", flush=True)

    # AUC-ROC: re-read JSONL, build (gt, score) pairs, compute trapezoidal AUC.
    # No sklearn dep — pure Python implementation.
    rows: list[dict] = []
    for line in out_path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    pairs = [(int(bool(r["gt_fire"])), float(r["fire_confidence"]))
             for r in rows if r.get("error") is None]
    n_pos = sum(1 for y, _ in pairs if y == 1)
    n_neg = sum(1 for y, _ in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        print(f"  AUC-ROC   = N/A (need both pos and neg; have pos={n_pos}, neg={n_neg})", flush=True)
        return 0
    # Mann–Whitney U / Wilcoxon equivalent
    pairs.sort(key=lambda x: x[1])
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][1] == pairs[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank
        for k in range(i, j):
            if pairs[k][0] == 1:
                rank_sum += avg_rank
        i = j
    u = rank_sum - n_pos * (n_pos + 1) / 2
    auc = u / (n_pos * n_neg)
    print(f"  AUC-ROC   = {auc:.3f}  (pos={n_pos}, neg={n_neg})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
