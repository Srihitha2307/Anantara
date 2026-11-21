"""
Build the cached demo planets for the website.

  python scripts/build_demo_data.py

Physics (orbit, habitable zone, escape velocity, shoreline) is computed from the REAL NASA Exoplanet Archive rows in
data/reference/. The two model scores are different:
  * if data/demo/runs/<slug>.json exists (made by scripts/analyse.py from a real light curve / spectrum), its Model B
    and Model A outputs are used and marked source = model_b / model_a;
  * otherwise the old placeholder numbers are kept and marked source = placeholder, and the site says so.
Writes data/demo/planets.json and web/data/planets.json (the static site reads the second one).
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anantara_pipeline as pipe

DEMO = [("TRAPPIST-1 b", dict(cf=0.97, mol={"H₂O": .12, "CO₂": .21, "CH₄": .05, "SO₂": .04}), 12.47),
        ("K2-18 b",      dict(cf=0.94, mol={"H₂O": .35, "CO₂": .88, "CH₄": .91, "SO₂": .06}), None),
        ("LHS 1140 b",   dict(cf=0.96, mol={"H₂O": .41, "CO₂": .28, "CH₄": .09, "SO₂": .05}), None)]
# Archive has no distance for TRAPPIST-1; 12.47 pc is the published Gaia value (40.7 ly).


def main():
    out = []
    for name, placeholder, dist_pc in DEMO:
        row = pipe.find_planet(name)
        star, src, _ = pipe.star_from(row)
        if not star.get("sy_dist") and dist_pc:
            star["sy_dist"] = dist_pc
        planet = {k: row[k] for k in ("pl_rade", "pl_bmasse", "pl_bmassprov", "pl_orbper", "pl_orbsmax") if row[k] == row[k]}
        rec = pipe.build_record(name, star, planet, cached_source="archive")
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        run = ROOT / "data" / "demo" / "runs" / f"{slug}.json"
        if run.exists():
            r = json.loads(run.read_text())
            for k in ("cf", "lc", "mol", "molv", "thr", "spec_mode"):
                if k in r:
                    rec[k] = r[k]
            rec["source"]["cf"] = r.get("source", {}).get("cf", rec["source"]["cf"])
            rec["source"]["mol"] = r.get("source", {}).get("mol", rec["source"]["mol"])
        if "cf" not in rec:
            rec["cf"], rec["source"]["cf"] = placeholder["cf"], "placeholder"
        if "mol" not in rec:
            rec["mol"], rec["source"]["mol"] = placeholder["mol"], "placeholder"
        out.append(rec)
        print(f"{name:14s} a={rec['a']:.4f} AU  HZ={rec['srv']['hz'][1]:.3f}-{rec['srv']['hz'][2]:.3f}  "
              f"v_esc={rec['srv']['ve']:.1f}  {rec['srv']['shoreline']:10s} cf={rec['source']['cf']}, mol={rec['source']['mol']}")
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    (ROOT / "data" / "demo").mkdir(parents=True, exist_ok=True)
    (ROOT / "web" / "data").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "demo" / "planets.json").write_text(txt)
    (ROOT / "web" / "data" / "planets.json").write_text(txt)
    print("wrote data/demo/planets.json and web/data/planets.json")


if __name__ == "__main__":
    main()
