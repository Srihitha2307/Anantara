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

**Option A – one link for everything (recommended): Hugging Face Space (free, 16 GB RAM, Docker)**

1. Create an account and an **empty Space** at huggingface.co/new-space → SDK **Docker** → Blank.
2. `pip install huggingface_hub && huggingface-cli login`
3. From the repo root: `bash deploy/huggingface/push.sh <your-hf-username>/anantara`
4. After a few minutes of building: `https://<your-hf-username>-anantara.hf.space`. Test it with `/api/health`.

**Option B – static site on GitHub Pages** (the cached demo planets work with no backend)

1. Repo → Settings → Pages → Source: **GitHub Actions**. The workflow in `.github/workflows/pages.yml` publishes `web/`.
2. Result: `https://<you>.github.io/anantara/`.
3. To enable real uploads on that page, put the Space URL in `web/config.js`:
   `window.ANANTARA_API = "https://<your-hf-username>-anantara.hf.space";` and commit.
   If the backend is down the page falls back to the cached sample planets and says so.

Render (`render.yaml`) also works, but its free tier has 512 MB RAM, which is tight for PyTorch.

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
