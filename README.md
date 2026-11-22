# Anantara

Explainable exoplanet triage. Upload a Kepler/TESS light curve and/or a JWST/Hubble transmission spectrum and Anantara shows the system in 3D:

1. **Transit vetting** – is the dip planet-like? (Model B, 1D CNN, Grad-CAM in the notebook)
2. **Orbit and habitable zone** – Kepler's third law and Kopparapu et al. (2013)
3. **Gravity and Cosmic Shoreline** – escape velocity against stellar XUV (Zahnle & Catling 2017)
4. **Atmosphere** – candidate indications for H₂O, CO₂, CH₄, SO₂ (Model A, 2D CNN)

> **Wording rule:** Anantara reports *candidate indications*, never "life found" and never a "detection".

Team S38 · Domain 05

---

## Repository layout

```
README.md  requirements.txt  requirements-api.txt  requirements-dev.txt
src/
  anantara_physics.py     orbit, habitable zone, escape velocity, shoreline (no ML)
  anantara_models.py      Net2D / TransitCNN1D definitions, weight loading, spectrum -> image
  anantara_pipeline.py    light curve -> BLS -> Model B -> physics ; spectrum -> Model A
api/server.py             FastAPI: serves the website AND /api/analyse
web/                      index.html (3D site), config.js, data/planets.json
models/                   modelA_v4.pt, modelA_thresholds.json, model_b_1d.pt, transit_cnn_fold_1..5.pt
data/reference/           NASA Exoplanet Archive tables, derived.csv, false positives, CV results
data/demo/                planets.json (cached demo planets); runs/ for real-run results
data/cache/               views_kepler_tess.npz (Model B training views)
notebooks/                00_data_access, model_b_1d_kepler_tess, model_a_v5_trained, model_a_real_2d, archive/
scripts/                  build_demo_data.py, analyse.py, commit_in_order.sh
tests/                    offline smoke tests (pytest)
Dockerfile  render.yaml  deploy/huggingface/  .github/workflows/
docs/PROJECT_NOTES.md     the original planning notes
```

## Run it locally (3 commands)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build is enough
pip install -r requirements-dev.txt
uvicorn api.server:app --port 8000          # then open http://localhost:8000
```

`pytest -q` runs the offline tests (physics self-test, weights load, period recovery, API routes).
Needs internet only for the three.js script on cdnjs.

### What you can upload

| File | Columns | Needed extra |
|---|---|---|
| Light curve (CSV, TSV, or Kepler/TESS `.fits`) | `time`, `flux` (aliases: `pdcsap_flux`, `sap_flux`, `bjd`, …) | Planet name, or star Teff / radius / mass. Otherwise a Sun-like star is **assumed** and the result says so |
| Transmission spectrum (CSV) | `wavelength_um`, `depth_ppm`, `depth_err_ppm` (0.3–6 µm, ≥10 rows) | Planet name (e.g. `K2-18 b`) so the star and orbit can be looked up |

Upload both together for the full result. Planet names are matched against `data/reference/planets.csv` (6,473 archive planets).

### Command line (same pipeline)

```bash
python scripts/analyse.py --name "K2-18 b" --lc kic_lightcurve.fits --spectrum k2-18b_jwst.csv --save-demo
python scripts/build_demo_data.py     # puts real Model B / Model A scores into the cached demo planets
```

## Deploy: get a link

**Render (Docker, free) — one service serves the website and the API**

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Srihitha2307/Anantara)

1. Open the button above (or <https://render.com/deploy?repo=https://github.com/Srihitha2307/Anantara>).
2. Sign in with GitHub, pick the workspace, and click **Apply**. Render reads `render.yaml`, builds the `Dockerfile`, and starts `uvicorn api.server:app`.
3. Wait for the first build (~5–8 min on free CPU), then open **<https://anantara-ai.onrender.com>**.
4. Free instances sleep after 15 minutes of inactivity; the first request afterwards takes ~30–60 s to wake.

The blueprint uses Render's free 512 MB plan. Models load lazily, so normal uploads fit; if a very large file is OOM-killed, switch the instance type in Render → Settings.

**GitHub Pages (static demo only)** — `.github/workflows/pages.yml` publishes `web/`. Set `window.ANANTARA_API` in `web/config.js` to the Render URL to make uploads work on the Pages page too; if the backend is asleep the page falls back to the cached sample planets.

**Hugging Face Space** — the `deploy/huggingface/` kit still works, but the Space must use the Docker SDK and have available `cpu-basic` quota.

## Training (Colab, free T4 GPU)

Run in this order. Every notebook mounts Drive and keeps its files in `MyDrive/anantara/` (an older folder with the previous project name is renamed automatically by the first cell).

1. `00_data_access.ipynb` – checks the four data connections.
2. `model_a_v5_trained.ipynb` – Model A on simulated spectra (≈25–30 min), writes `modelA_v4.pt` and the CV results.
3. `model_b_1d_kepler_tess.ipynb` – Model B (set both `N_PER_CLASS_*` to 30 for a 20–30 min test), writes `model_b_1d.pt`.
4. `model_a_real_2d.ipynb` – optional, real JWST spectra.

After retraining, copy the `.pt` files into `models/` and update `models/modelA_thresholds.json` from the notebook's printed thresholds.

## Honest status (put this in front of judges)

| Part | Status |
|---|---|
| Physics module | Self-test passes (Earth and Venus retained, Mars on the line). The shoreline constant K is anchored on Mars: a **modelling choice**, not a published fit. The L^0.4 XUV scaling is **unverified** against the paper |
| Model A (2D CNN) | Trained on **simulated** spectra only. Held-out simulated test: precision 0.88, recall 0.66. File-style stress test: precision **0.71** (below the 0.90 target), recall 0.63. 3-fold CV precision 0.87–0.93. **Not yet validated on real spectra** (0 real spectra were loaded). Proof of concept |
| Model B (1D CNN) | Weights load and run. On the cached Kepler views it scores the 22 known false positives lower (mean 0.22) than the rest (0.52), which is a sanity check, not a metric. Test AUC from the notebook run was not in the files provided: record it in this table |
| Demo planets | Orbit, habitable zone, escape velocity and shoreline come from real archive parameters. Their **detection and molecule scores are placeholders** (marked `placeholder` in the JSON and on the page) until you run `scripts/analyse.py` on real files |
| Not connected | Stellar XUV from MUSCLES (`xuv.csv` is empty), Hubble data in a model, a false-positive demo target |

## Data and credits

NASA Exoplanet Archive, MAST (Kepler, TESS, JWST, Hubble), MUSCLES. Public data, credited in the app. Habitable zone: Kopparapu et al. (2013, erratum 2014). Shoreline: Zahnle & Catling (2017). Mass–radius: Chen & Kipping (2017). `PS_*.csv` (the 38 MB full Archive dump) is deliberately not committed.
