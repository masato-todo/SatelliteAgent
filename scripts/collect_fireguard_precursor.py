"""Convert YujiYamaguchi/fireguard-sentinel2-wildfire-precursor (config=pair14_7)
into DM3 case yaml entries (one per row, no images).

Each row carries (lat, lon, query_date, sentinel_datetimes=[T-14, T-7],
label, source). We map T-14 → before_date, T-7 → after_date so the app's
"Fetch Images" path with window_days=1 retrieves the *exact* two STAC
items the precursor LoRA was trained on.

Output: data/metadata/disaster_m3/fireguard_precursor_cases.yaml

Usage:
    python scripts/collect_fireguard_precursor.py
    python scripts/collect_fireguard_precursor.py --splits test val
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "fireguard_precursor_cases.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="YujiYamaguchi/fireguard-sentinel2-wildfire-precursor")
    ap.add_argument("--config",  default="pair14_7")
    ap.add_argument("--splits",  nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out",     default=str(OUT_PATH))
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"[1/2] loading {args.dataset} config={args.config} (splits: {args.splits})...", flush=True)
    cases: list[dict] = []
    for split in args.splits:
        ds = load_dataset(args.dataset, args.config, split=split)
        print(f"  {split}: {len(ds)} rows", flush=True)
        for i, r in enumerate(ds):
            try:
                lat = float(r["lat"]); lon = float(r["lon"])
            except (KeyError, ValueError, TypeError):
                continue
            sdts = r.get("sentinel_datetimes") or []
            if len(sdts) < 2:
                continue
            label = int(r.get("label", 0))
            src   = r.get("source") or "?"
            qdate = r.get("query_date") or ""
            frp   = r.get("frp")

            # Pin the exact training-time STAC items: pass full ISO timestamps
            # to /api/fetch with window_days=1. _normalize_ts preserves the
            # time component so SimSat returns the same items at distance 0.
            before_date = sdts[0]   # T-14d
            after_date  = sdts[1]   # T-7d

            event_type      = "wildfire_precursor" if label == 1 else "no_change"
            expected_action = "submit_to_ground"   if label == 1 else "drop"
            case_id = f"precursor_{split}_{src}_{i:03d}"
            cases.append({
                "id":              case_id,
                "source":          "FireGuard_HF",
                "event_type":      event_type,
                "label":           event_type,
                "expected_action": expected_action,
                "lat":             round(lat, 6),
                "lon":             round(lon, 6),
                "size_km":         5.0,
                "before_date":     before_date,
                "after_date":      after_date,
                "window_days":     1,                # tight pin on both sides
                "query_date":      qdate,
                "sentinel_datetimes": sdts,
                "precursor_split":  split,
                "precursor_source": src,
                "precursor_idx":    i,
                "frp":             frp,
                "name":            r.get("name"),
            })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases":    len(cases),
        "source":     args.dataset,
        "config":     args.config,
        "splits":     args.splits,
        "cases":      cases,
    }
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    tmp.replace(out_path)

    n_pos = sum(1 for c in cases if c["label"] == "wildfire_precursor")
    n_neg = sum(1 for c in cases if c["label"] == "no_change")
    print(f"\n[2/2] {len(cases)} cases → {out_path}", flush=True)
    print(f"      pos (precursor) = {n_pos},  neg (no_change) = {n_neg}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
