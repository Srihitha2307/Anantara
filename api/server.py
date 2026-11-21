"""
Anantara backend (FastAPI). One process serves BOTH the API and the website, so one URL is the whole product.

  GET  /api/health     models loaded? versions
  GET  /api/demo       the cached demo planets (same JSON the static site reads)
  POST /api/analyse    multipart: files[] (light curve CSV/FITS and/or transmission spectrum CSV),
                       optional fields: name, teff, st_rad, st_mass, period
  GET  /               web/index.html

Run:  uvicorn api.server:app --port 8000      then open http://localhost:8000
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch
import anantara_models as am
import anantara_pipeline as pipe

torch.set_num_threads(2)
MAX_BYTES = 30 * 1024 * 1024

app = FastAPI(title="Anantara API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
def health():
    return {"status": "ok", "models": am.models_ready(), "torch": torch.__version__,
            "note": "Candidate indications only; Model A is a proof of concept trained on simulated spectra."}


@app.get("/api/demo")
def demo():
    p = ROOT / "web" / "data" / "planets.json"
    if not p.exists():
        raise HTTPException(404, "planets.json not built yet: run  python scripts/build_demo_data.py")
    return json.loads(p.read_text())


def _num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


@app.post("/api/analyse")
async def analyse(files: List[UploadFile] = File(...), name: str = Form(""), teff: Optional[str] = Form(None),
                  st_rad: Optional[str] = Form(None), st_mass: Optional[str] = Form(None),
                  period: Optional[str] = Form(None)):
    lc = spec = None
    hdr, lc_name, notes = {}, "", []
    for f in files[:4]:
        data = await f.read()
        if len(data) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than {MAX_BYTES // 2**20} MB")
        try:
            df, h = pipe.read_table(data, f.filename or "")
            kind = pipe.classify(df)
            if kind == "lightcurve" and lc is None:
                hdr, lc_name = h, f.filename or ""
                lc = pipe.analyse_lightcurve(df, h, _num(period))
            elif kind == "spectrum" and spec is None:
                spec = pipe.analyse_spectrum(df, f.filename or "")
                spec["file"] = f.filename
            elif kind is None:
                notes.append(f"{f.filename}: not recognised (need time+flux columns, or wavelength+depth+error columns)")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(422, f"{f.filename}: {e}")
    if lc is None and spec is None:
        raise HTTPException(422, "No usable file. Upload a light curve (columns time, flux) or a transmission spectrum "
                                 "(columns wavelength_um, depth_ppm, depth_err_ppm). " + " ".join(notes))
    row = (pipe.find_planet(name) or pipe.find_planet(hdr.get("OBJECT"))
           or next((r for r in (pipe.find_planet(f.filename) for f in files) if r), None))
    if row is None and lc is None:
        raise HTTPException(422, "A spectrum alone needs the planet's name (e.g. 'K2-18 b') so Anantara can look up its star and orbit. "
                                 "Type it in the 'Planet name' box and upload again.")
    over = dict(st_teff=_num(teff), st_rad=_num(st_rad), st_mass=_num(st_mass))
    star, star_src, snotes = pipe.star_from(row, hdr, over)
    planet = None
    if row is not None:
        planet = {k: row.get(k) for k in ("pl_rade", "pl_bmasse", "pl_bmassprov", "pl_orbper", "pl_orbsmax")}
        planet = {k: v for k, v in planet.items() if v == v and v is not None}
    label = row["pl_name"] if row else (name.strip() or Path(lc_name or "upload").stem)
    try:
        rec = pipe.build_record(label, star, planet, lc, spec, notes + snotes, cached_source=star_src)
    except ValueError as e:
        raise HTTPException(422, str(e))
    return rec


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
