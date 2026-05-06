"""Collect harmful algal bloom (HAB) / red tide events as Phase 1 catalog.

There is NO standardized global HAB event list (NOAA bulletin is PDF-only,
GDACS doesn't track HABs, CMEMS chlorophyll-a is raster-only). So this is
a hand-curated set of well-documented major HABs (peer-reviewed / news).

Per docs/EXPERIMENT_PLAN.md Phase 1: only catalog metadata. Phase 2 finds
imagery via SimSat probe.

S2 detectability of HABs:
  - NDCI = (B5 - B4) / (B5 + B4)  (red-edge based) — chlorophyll proxy
  - false_color RGB shows the visible color (red/green/brown) directly
  - Bloom patches scale 1 km² 〜 数千 km²; size_km=20 catches typical extents

Output: `data/metadata/disaster_m3/algal_bloom_cases.yaml`.

Usage:
  python scripts/collect_algal_bloom.py
  python scripts/collect_algal_bloom.py --target 20  # smoke
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "metadata" / "disaster_m3" / "algal_bloom_cases.yaml"


# Hand-curated list. Each entry: (id_suffix, region, country, species, color,
# lat, lon, event_yyyymm, notes).
# Date is mid-month for the event peak; Phase 2 probe finds best S2 around it.
HAB_EVENTS: list[dict] = [
    # ---------- USA ----------
    {"sfx":"florida_red_tide_2018",    "region":"Florida SW coast",      "country":"USA",       "species":"Karenia brevis",     "color":"red",
     "lat":27.00,  "lon":-82.50, "yyyymm":"2018-09", "notes":"Catastrophic red tide off Sarasota / Naples; mass marine die-off"},
    {"sfx":"florida_red_tide_2021",    "region":"Tampa Bay",             "country":"USA",       "species":"Karenia brevis",     "color":"red",
     "lat":27.70,  "lon":-82.60, "yyyymm":"2021-07", "notes":"Tampa Bay K. brevis outbreak"},
    {"sfx":"lake_erie_toledo_2014",    "region":"Lake Erie",             "country":"USA",       "species":"Microcystis",        "color":"green",
     "lat":41.65,  "lon":-83.20, "yyyymm":"2014-08", "notes":"Toledo drinking-water crisis"},
    {"sfx":"lake_erie_2017",           "region":"Lake Erie western",     "country":"USA",       "species":"Microcystis",        "color":"green",
     "lat":41.75,  "lon":-83.10, "yyyymm":"2017-09", "notes":"Annual cyanoHAB, severe year"},
    {"sfx":"lake_erie_2019",           "region":"Lake Erie western",     "country":"USA",       "species":"Microcystis",        "color":"green",
     "lat":41.60,  "lon":-82.90, "yyyymm":"2019-08", "notes":"Severe bloom"},
    {"sfx":"chesapeake_mahogany_2020", "region":"Chesapeake Bay",        "country":"USA",       "species":"Prorocentrum",       "color":"red-brown",
     "lat":38.50,  "lon":-76.40, "yyyymm":"2020-08", "notes":"Mahogany tide"},
    {"sfx":"pacific_pseudonitzschia_2015","region":"Pacific NW",         "country":"USA",       "species":"Pseudo-nitzschia",   "color":"brown",
     "lat":47.50,  "lon":-124.50,"yyyymm":"2015-06", "notes":"Largest Pseudo-nitzschia bloom on record, domoic acid"},

    # ---------- China / East Asia ----------
    {"sfx":"yellow_sea_ulva_2008",     "region":"Qingdao Yellow Sea",    "country":"China",     "species":"Ulva prolifera",     "color":"green",
     "lat":36.00,  "lon":120.50, "yyyymm":"2008-07", "notes":"Pre-Olympics green tide, mega bloom"},
    {"sfx":"yellow_sea_ulva_2021",     "region":"Yellow Sea",            "country":"China",     "species":"Ulva prolifera",     "color":"green",
     "lat":35.50,  "lon":120.50, "yyyymm":"2021-07", "notes":"Largest Ulva bloom on record (~1700 km²)"},
    {"sfx":"yellow_sea_ulva_2024",     "region":"Yellow Sea",            "country":"China",     "species":"Ulva prolifera",     "color":"green",
     "lat":35.40,  "lon":120.40, "yyyymm":"2024-07", "notes":"Annual recurrence"},
    {"sfx":"taihu_microcystis_2007",   "region":"Lake Taihu",            "country":"China",     "species":"Microcystis",        "color":"green",
     "lat":31.20,  "lon":120.20, "yyyymm":"2007-05", "notes":"Wuxi drinking-water crisis"},
    {"sfx":"taihu_microcystis_2022",   "region":"Lake Taihu",            "country":"China",     "species":"Microcystis",        "color":"green",
     "lat":31.10,  "lon":120.20, "yyyymm":"2022-08", "notes":"Annual recurrence"},
    {"sfx":"dianchi_2020",             "region":"Lake Dianchi (Kunming)","country":"China",     "species":"Microcystis",        "color":"green",
     "lat":24.85,  "lon":102.70, "yyyymm":"2020-07", "notes":"Chronic eutrophication"},
    {"sfx":"chaohu_2019",              "region":"Lake Chaohu",           "country":"China",     "species":"Microcystis",        "color":"green",
     "lat":31.55,  "lon":117.50, "yyyymm":"2019-08", "notes":"Severe bloom"},

    # ---------- Russia / Far East ----------
    {"sfx":"kamchatka_2020",           "region":"Avacha Bay",            "country":"Russia",    "species":"Karenia",            "color":"brown-red",
     "lat":52.80,  "lon":158.50, "yyyymm":"2020-09", "notes":"Catastrophic die-off, surfer deaths reported"},
    {"sfx":"hokkaido_2021",            "region":"Sea of Okhotsk",        "country":"Japan",     "species":"Karenia selliformis","color":"red",
     "lat":43.50,  "lon":145.30, "yyyymm":"2021-09", "notes":"Salmon mass mortality"},
    {"sfx":"vladivostok_2018",         "region":"Peter the Great Bay",   "country":"Russia",    "species":"Alexandrium",        "color":"red",
     "lat":43.10,  "lon":131.90, "yyyymm":"2018-08", "notes":"Annual recurrence"},

    # ---------- Europe ----------
    {"sfx":"north_sea_cyano_2022",     "region":"North Sea",             "country":"Germany",   "species":"cyanobacteria",      "color":"green",
     "lat":55.50,  "lon":7.50,   "yyyymm":"2022-07", "notes":"Summer cyanobacteria"},
    {"sfx":"baltic_cyano_2018",        "region":"Baltic Sea central",    "country":"Sweden/Finland","species":"cyanobacteria",  "color":"green",
     "lat":58.50,  "lon":19.50,  "yyyymm":"2018-07", "notes":"Recurring summer cyanobacteria"},
    {"sfx":"baltic_cyano_2021",        "region":"Gulf of Finland",       "country":"Finland",   "species":"cyanobacteria",      "color":"green",
     "lat":59.80,  "lon":23.50,  "yyyymm":"2021-07", "notes":"Severe cyanoHAB"},
    {"sfx":"adriatic_2019",            "region":"Northern Adriatic",     "country":"Italy",     "species":"Phaeocystis/Diatom", "color":"brown",
     "lat":44.80,  "lon":13.20,  "yyyymm":"2019-07", "notes":"Recurring mucilage"},
    {"sfx":"black_sea_emiliania_2017", "region":"Black Sea",             "country":"Ukraine",   "species":"Emiliania huxleyi",  "color":"turquoise",
     "lat":43.50,  "lon":33.50,  "yyyymm":"2017-06", "notes":"Coccolithophore, milky turquoise"},

    # ---------- South America ----------
    {"sfx":"chile_patagonia_2016",     "region":"Chiloé Patagonia",      "country":"Chile",     "species":"Pseudochattonella",  "color":"brown",
     "lat":-42.50, "lon":-73.20, "yyyymm":"2016-04", "notes":"Catastrophic salmon farm losses"},
    {"sfx":"pisco_peru_2017",          "region":"Pisco Bay",             "country":"Peru",      "species":"Akashiwo",           "color":"red",
     "lat":-13.70, "lon":-76.20, "yyyymm":"2017-04", "notes":"Pisco / Paracas red tide"},
    {"sfx":"argentina_coast_2021",     "region":"Buenos Aires coast",    "country":"Argentina", "species":"Alexandrium",        "color":"red",
     "lat":-38.00, "lon":-57.50, "yyyymm":"2021-04", "notes":"Annual recurrence"},

    # ---------- Africa / Middle East ----------
    {"sfx":"oman_red_tide_2008",       "region":"Gulf of Oman",          "country":"Oman",      "species":"Cochlodinium",       "color":"red",
     "lat":24.00,  "lon":58.00,  "yyyymm":"2008-09", "notes":"Massive Cochlodinium bloom"},
    {"sfx":"kuwait_red_tide_2022",     "region":"Kuwait Bay",            "country":"Kuwait",    "species":"Karenia mikimotoi",  "color":"red",
     "lat":29.40,  "lon":48.00,  "yyyymm":"2022-08", "notes":"Marine die-off"},
    {"sfx":"namibia_2019",             "region":"Namibian shelf",        "country":"Namibia",   "species":"Sulfide / Diatom",   "color":"green",
     "lat":-23.00, "lon":14.00,  "yyyymm":"2019-04", "notes":"Sulfide eruption + bloom"},

    # ---------- Oceania ----------
    {"sfx":"tasman_phytoplankton_2018","region":"Tasman Sea east coast", "country":"Australia", "species":"Phytoplankton",      "color":"green",
     "lat":-42.00, "lon":148.50, "yyyymm":"2018-12", "notes":"Anomalous bloom"},

    # ---------- Other ----------
    {"sfx":"caspian_2020",             "region":"Caspian Sea",           "country":"Iran",      "species":"cyanobacteria",      "color":"green",
     "lat":38.00,  "lon":52.00,  "yyyymm":"2020-08", "notes":"Recurring bloom"},
    {"sfx":"lake_winnipeg_2017",       "region":"Lake Winnipeg",         "country":"Canada",    "species":"cyanobacteria",      "color":"green",
     "lat":51.50,  "lon":-97.00, "yyyymm":"2017-08", "notes":"Eutrophic lake, annual"},
]


def save_yaml(path: Path, cases: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases": len(cases),
        "cases": cases,
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(OUT_PATH))
    p.add_argument("--target", type=int, default=None,
                   help="Cap N (default: include all)")
    p.add_argument("--size-km", type=float, default=10.0,
                   help="AOI size (HABs span 1〜数千 km²; 10km matches other catalogs and keeps fetch fast)")
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--before-offset-days", type=int, default=-365,
                   help="before_date offset from event (negative). Default -365 to align season.")
    p.add_argument("--after-offset-days", type=int, default=0,
                   help="after_date offset from event (peak month). Default 0.")
    args = p.parse_args()

    cases: list[dict] = []
    target = args.target if args.target is not None else len(HAB_EVENTS)
    for ev in HAB_EVENTS[:target]:
        # event_date = yyyy-mm-15 (mid-month)
        et = datetime.strptime(ev["yyyymm"] + "-15", "%Y-%m-%d").replace(tzinfo=timezone.utc)
        before_date = (et + timedelta(days=args.before_offset_days)).strftime("%Y-%m-%d")
        after_date  = (et + timedelta(days=args.after_offset_days)).strftime("%Y-%m-%d")
        case_id = f"hab_{ev['sfx']}"
        cases.append({
            "id":              case_id,
            "source":          "HAB",
            "category":        "Algal bloom",
            "event_type":      "algal_bloom",
            "expected_action": "submit_to_ground",
            "name":            f"{ev['species']} ({ev['color']}) — {ev['region']}",
            "region":          ev["region"],
            "country":         ev["country"],
            "species":         ev["species"],
            "bloom_color":     ev["color"],
            "lat":             round(ev["lat"], 4),
            "lon":             round(ev["lon"], 4),
            "size_km":         args.size_km,
            "before_date":     before_date,
            "after_date":      after_date,
            "window_days":     args.window_days,
            "event_time":      et.isoformat(),
            "notes":           ev["notes"],
        })

    out_path = Path(args.out)
    save_yaml(out_path, cases)
    print(f"[done] {len(cases)} algal-bloom cases → {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
