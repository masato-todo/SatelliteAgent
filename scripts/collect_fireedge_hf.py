"""Convert YujiYamaguchi/fireedge-sentinel2-wildfire HF dataset → DM3 case yaml.

Each row exposes lat/lon/query_date/sentinel_datetime/label/source so we
can drop them straight into the SatelliteAgent dropdown. The recorded
`sentinel_datetime` is the training-time S2 frame, so apps that fetch via
SimSat with window_days=1 will reproduce the exact image the LoRA saw
(see scripts/eval_wildfire_hf_simsat.py).

Output: `data/metadata/disaster_m3/fireedge_hf_cases.yaml`.

Usage:
    python scripts/collect_fireedge_hf.py
    python scripts/collect_fireedge_hf.py --splits test val
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "fireedge_hf_cases.yaml"

# Same training period as Phase 2 NEG temporal (apps/fireedge/finetune/dataset_builder.py:NEG_OFFSET=180d)
NEG_BEFORE_OFFSET_DAYS = 180


def parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00") if s.endswith("Z") else s
        return datetime.fromisoformat(s)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YujiYamaguchi/fireedge-sentinel2-wildfire")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[1/2] loading {args.dataset} (splits: {args.splits})...", flush=True)
    cases: list[dict] = []
    for split in args.splits:
        ds = load_dataset(args.dataset, split=split)
        print(f"  {split}: {len(ds)} rows", flush=True)
        for i, r in enumerate(ds):
            try:
                lat = float(r["lat"]); lon = float(r["lon"])
            except (KeyError, ValueError, TypeError):
                continue
            qdate = r.get("query_date") or ""
            sdt   = r.get("sentinel_datetime") or ""
            label = int(r.get("label", 0))
            src   = r.get("source") or "?"

            # after_date = exact training-time S2 frame (use date part only;
            # the case carries `sentinel_datetime` separately for window=1 fetch).
            after_date = sdt[:10] if sdt else qdate
            # before_date defaults to the same temporal NEG anchor used in training.
            sdt_obj = parse_iso(sdt) or parse_iso(qdate + "T12:00:00+00:00")
            if sdt_obj is None:
                continue
            before_date = (sdt_obj - timedelta(days=NEG_BEFORE_OFFSET_DAYS)).strftime("%Y-%m-%d")

            event_type      = "fire" if label == 1 else "no_change"
            expected_action = "submit_to_ground" if label == 1 else "drop"
            case_id = f"fireedge_{split}_{src}_{i:03d}"
            cases.append({
                "id":              case_id,
                "source":          "FireEdge_HF",
                "event_type":      event_type,
                "label":           event_type,
                "expected_action": expected_action,
                "lat":             round(lat, 6),
                "lon":             round(lon, 6),
                "size_km":         5.0,
                "before_date":     before_date,
                "after_date":      after_date,
                "window_days":     1,                  # tight window → exact training frame
                "query_date":      qdate,
                "sentinel_datetime": sdt,
                "fireedge_split":  split,
                "fireedge_source": src,                # firms_pos / firms_neg / diverse_neg
                "fireedge_idx":    i,
                "nbr2":            r.get("nbr2"),
                "nbr2_min":        r.get("nbr2_min"),
                "mean_swir22":     r.get("mean_swir22"),
                "swir22_max":      r.get("swir22_max"),
            })

    print(f"\n[2/2] writing {len(cases)} cases → {args.out}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases":    len(cases),
        "source":     args.dataset,
        "splits":     args.splits,
        "cases":      cases,
    }
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    tmp.replace(out_path)

    n_pos = sum(1 for c in cases if c["label"] == "fire")
    n_neg = sum(1 for c in cases if c["label"] == "no_change")
    print(f"      pos (fire) = {n_pos}, neg (no_change) = {n_neg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
