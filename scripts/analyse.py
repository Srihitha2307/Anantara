"""
Run the full pipeline on real files and save the result for the site.

  python scripts/analyse.py --name "K2-18 b" --lc kic_lightcurve.fits --spectrum k2-18b_jwst.csv --save-demo
  python scripts/analyse.py --lc my_curve.csv --teff 3400 --st-rad 0.4 --st-mass 0.4

--save-demo writes data/demo/runs/<planet>.json; then run  python scripts/build_demo_data.py  to put the real
Model B / Model A scores into the cached demo planets.
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import anantara_pipeline as pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="")
    ap.add_argument("--lc"); ap.add_argument("--spectrum"); ap.add_argument("--period", type=float)
    ap.add_argument("--teff", type=float); ap.add_argument("--st-rad", type=float); ap.add_argument("--st-mass", type=float)
    ap.add_argument("--save-demo", action="store_true")
    a = ap.parse_args()
    lc = spec = None; hdr = {}
    if a.lc:
        df, hdr = pipe.read_table(Path(a.lc).read_bytes(), a.lc)
        lc = pipe.analyse_lightcurve(df, hdr, a.period)
    if a.spectrum:
        df, _ = pipe.read_table(Path(a.spectrum).read_bytes(), a.spectrum)
        spec = pipe.analyse_spectrum(df, a.spectrum)
    row = pipe.find_planet(a.name) or pipe.find_planet(a.spectrum) or pipe.find_planet(a.lc)
    star, src, notes = pipe.star_from(row, hdr, dict(st_teff=a.teff, st_rad=a.st_rad, st_mass=a.st_mass))
    planet = {k: row[k] for k in ("pl_rade", "pl_bmasse", "pl_bmassprov", "pl_orbper", "pl_orbsmax") if row is not None and row[k] == row[k]} if row else None
    label = row["pl_name"] if row else (a.name or "upload")
    rec = pipe.build_record(label, star, planet, lc, spec, notes, cached_source=src)
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    if a.save_demo:
        d = ROOT / "data" / "demo" / "runs"; d.mkdir(parents=True, exist_ok=True)
        f = d / (re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") + ".json")
        f.write_text(json.dumps(rec, indent=1, ensure_ascii=False)); print("saved", f)


if __name__ == "__main__":
    main()
