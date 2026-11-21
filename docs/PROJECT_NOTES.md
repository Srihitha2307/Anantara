# Anantara – Project Notes (chat summary)

A summary of the planning and code work done for the Anantara exoplanet project: where the data comes from, the two CNN models, the physics code, how to train in Google Colab, and website name ideas.

> **Status:** Nothing here has been run end to end. The physics module was tested offline with sample numbers. The notebooks were only syntax-checked, and the simulated-data signal was sanity-checked. Network downloads (MAST, Archive) and PyTorch training have **not** been tested. Expect to fix a few errors on the first run.

---

## 1. Project idea in plain words

- When a planet passes in front of its star, the star dims slightly for a few hours, and the dip repeats every orbit.
- **A CNN** is a pattern-finder. It is shown many labelled examples and learns what a real planet dip looks like.
- **Google Colab** is a free browser computer used to run the code. Nothing needs installing.

---

## 2. Data sources and what each is used for

| Data | Tool / API | Used for |
|---|---|---|
| Labels and star/planet parameters | NASA Exoplanet Archive (TAP API, via `requests`) | Which stars have planets; masses, radii, temperatures |
| Kepler + TESS light curves | `lightkurve` (talks to MAST) | Training Model B |
| JWST + Hubble observations | `astroquery.mast` | Model A (real data), demo planets |
| Hubble UV of stars (MUSCLES) | `astroquery.mast` (HLSP collection) | Stellar XUV for the Cosmic Shoreline |

**Which telescope gives which parameter**

| Group | Parameter | Original source |
|---|---|---|
| Planet | Period, transit time, depth | Kepler or TESS |
| Planet | Radius | Transit depth × star radius |
| Planet | Mass | Ground-based radial-velocity instruments (or transit-timing) |
| Planet | Orbit distance | Calculated (Kepler's third law) |
| Star | Temperature, radius, mass | Spectroscopy and Gaia, via catalogues |
| Star | Distance | Gaia |
| Star | Luminosity | Calculated from radius and temperature |
| Star radiation | UV / X-ray | Hubble (UV) plus X-ray telescopes, bundled in MUSCLES |
| Atmosphere | Molecules in the spectrum | JWST and Hubble |

The NASA Exoplanet Archive is a catalogue, not a telescope. It collects numbers from many telescopes and papers.

**Suggested table layout (joined on planet/host name)**

- `planets.csv`: `pl_name, hostname, pl_rade, pl_bmasse, pl_bmassprov, pl_orbper, pl_orbsmax`
- `stars.csv`: `hostname, st_teff, st_rad, st_mass, st_lum, st_age, sy_dist`
- `derived.csv` (filled by code): `pl_name, a_au, hz_inner_au, hz_outer_au, in_hz, v_esc_kms, shoreline_verdict`
- `xuv.csv` (optional, only stars with MUSCLES data)
- Keep raw downloads untouched in `data/raw/`. Record whether a mass is measured or an estimate.

---

## 3. The two models

### Model B – 1D CNN on Kepler + TESS (`model_b_1d_kepler_tess.ipynb`)
- **Input:** a star's brightness over time, folded on the planet's period, turned into a 2×201 array (global view + zoomed transit view).
- **Output:** planet / not a planet.
- **Labels:** Kepler KOI table (CONFIRMED vs FALSE POSITIVE) and TESS TOI table (CP/KP vs FP/FA).
- **Connection:** Archive gives the star list and labels → `lightkurve` downloads each star → arrays go into the CNN. Splits are by star.
- Includes Grad-CAM (shows which part of the light curve the CNN looked at).

### Model A – 2D CNN for JWST / Hubble-style spectra
Two versions:

1. **`model_a_2d_jwst_hubble.ipynb` (simulated training data).** Input is a 64×64 image: wavelength × time. Output is which of H₂O, CO₂, CH₄, SO₂ are present (multi-label). It trains on simulated images with toy molecular bands, because there are too few real JWST/Hubble spectra. It needs no downloads and trains in minutes.
2. **`model_a_real_2d.ipynb` (real JWST data only).** Downloads per-integration extracted spectra (`x1dints`) from MAST for planets you list, stacks them into wavelength × time images, and trains with heavy augmentation. Includes a leave-one-planet-out test.

**Honest limits of Model A**
- Only a handful of planets have usable JWST data in one instrument mode (the notebook uses NIRSpec G395H). Expect roughly 5–15 planets.
- Trained from scratch on so few planets, the CNN will mostly memorise. Report the leave-one-out result honestly and call it a proof of concept.
- **You must supply the labels** (which molecules each planet has) from published papers and verify every row. The notebook has one example row (WASP-39) that needs checking.
- Option: set `PRETRAINED = 'model_a_2d.pt'` to fine-tune from the simulated-data model.
- Hubble planet spectra are best taken from published tables, not trained on.

---

## 4. Notebooks and files delivered

| File | Purpose | Run order |
|---|---|---|
| `00_data_access.ipynb` | Tests that all four data sources connect | 1 |
| `model_a_2d_jwst_hubble.ipynb` | 2D CNN on simulated data (quick win) | 2 |
| `model_b_1d_kepler_tess.ipynb` | 1D CNN on Kepler + TESS | 3 |
| `model_a_real_2d.ipynb` | 2D CNN on real JWST data | 4 (last) |
| `anantara_physics.py` | Parameters + physics (no ML) | – |
| `requirements.txt` | Python packages | – |

---

## 5. How to train in Google Colab (beginner steps)

1. Open colab.research.google.com and sign in.
2. **File → Upload notebook** and pick one notebook.
3. **Runtime → Change runtime type → T4 GPU** (for the training notebooks).
4. Click the play button on each cell, top to bottom.
5. For Model B, first set both `N_PER_CLASS_*` values to `30` for a test run (about 20–30 minutes). Raise to 150+ later.
6. The notebook asks permission to use Google Drive. Click allow.

**Session safety**
- Downloads are saved to Google Drive as they go, so a disconnect loses nothing. Re-run all cells and it resumes.
- `DOWNLOAD_BUDGET` and `TRAIN_BUDGET` make the loops stop themselves.
- Downloading is the slow part; the models themselves train in minutes.
- Results are reported as **AUC**: 0.5 is a coin flip, 1.0 is perfect, and above about 0.8 is a decent start.

**Event rule to confirm:** the original rules reportedly require model training to happen during the event. Public data may be downloaded beforehand. Confirm at the briefing.

---

## 6. How the pieces connect

**Connected in code**
- Archive → Model B (star list and labels)
- `lightkurve` → Model B (Kepler/TESS brightness data)
- `astroquery.mast` → real-data Model A (JWST spectra)
- Simulated data → simulated Model A, with 2D Grad-CAM

**Not connected yet**
- Hubble data (no model uses it)
- MUSCLES stellar UV/X-ray (only described, not in a notebook)
- A single pipeline that runs Model B, then the physics, then Model A for one planet
- The website (trained models are only files on Drive)

---

## 7. Physics module (`anantara_physics.py`)

| Function | Job |
|---|---|
| `load_tables()` | Reads your `planets.csv` (+ optional `stars.csv`); accepts Archive downloads and friendly column names |
| `fetch_params()` | Live alternative: pulls parameters from the Archive |
| `orbit_distance()` | Kepler's third law |
| `habitable_zone()` | Kopparapu et al. (2013) inner and outer edges (valid 2600–7200 K) |
| `escape_velocity()` | v_esc; estimates mass from radius (Chen & Kipping 2017) if mass is missing |
| `cosmic_shoreline()` | Retained / stripped / borderline |
| `analyze_planet()` | Runs everything for one planet |
| `write_derived()` | Saves `derived.csv` |

**Run it**
```
python anantara_physics.py --planets planets.csv --stars stars.csv --out derived.csv
python anantara_physics.py --selftest
```

**Cosmic Shoreline as currently coded**
- XUV received (relative to Earth): `I_xuv = (1/a²) × (L/L☉)^0.4`
- The line: `I_xuv = K × v_esc⁴`, with `K ≈ 6.7×10⁻⁴ (km/s)⁻⁴`, anchored on Mars (5.03 km/s, 0.43)
- Verdict from `margin = log10(I_xuv / (K·v_esc⁴))`: above +0.5 stripped, below −0.5 retained, otherwise borderline
- Measured XUV (e.g. MUSCLES) can be passed as `xuv_rel` and overrides the scaling
- Self-test: Earth and Venus fall below the line, Mars sits on it

**Sample results (rounded test numbers, not live data)**

| Planet | In habitable zone? | Shoreline |
|---|---|---|
| TRAPPIST-1 b | No | Stripped |
| K2-18 b | Only by the optimistic definition | Retained |
| LHS 1140 b | Yes | Retained |

**Caveats to state honestly**
- Zahnle & Catling (2017) drew the shoreline by eye and gave no fitted constant. Anchoring K on Mars is a modelling choice, not a published value.
- The `L^0.4` scaling came from reference notes whose details were not independently verified. Check it against the paper before presenting it as theirs.
- The verdicts matching the expected story is a reason to double-check, not to trust them.
- Later empirical fits (e.g. Meni-Gallardo & Pallé) could replace `K_MARS` and `SLOPE`.
- A minimum mass (Msini) understates the true escape velocity; the code adds a note when the mass provenance says so.

---

## 8. Remaining to-do list

1. Run `00_data_access.ipynb` and fix any connection errors.
2. Add your stars/planets CSV and run the physics file on it.
3. Run Model A (simulated) and Model B (small test, then larger).
4. Fill in and verify the planet labels, then run the real-data Model A.
5. Add MUSCLES XUV for demo stars that have it.
6. Write one function combining Model B + physics + Model A per planet.
7. Pick a false-positive demo target (it will not be in the Archive's planet table).
8. Save trained models, cache results for the three demo planets, connect the website.
9. README, demo video, slides.

---

## 9. Suggested GitHub order

1. `README.md` and `requirements.txt`
2. `src/anantara_physics.py`
3. `notebooks/00_data_access.ipynb`
4. `notebooks/model_b_1d_kepler_tess.ipynb`
5. `notebooks/model_a_2d_jwst_hubble.ipynb`
6. `notebooks/model_a_real_2d.ipynb`
7. `models/` (trained `.pt` files)
8. `data/demo/` (cached results for the three demo planets)
9. `web/` (website code)

Commit during the event so the history shows the work was built then.

---

## 10. Website name ideas

Current name: **Anantara** (can stay as is, or shorten to **Anantara**).

**Hindu mythology / Sanskrit**
- **Divya Drishti** – "divine vision": Sanjaya given sight to see the distant battlefield
- **Sanjaya**, **Kshitij** (horizon, fits the shoreline idea), **Trinetra** (Shiva's third eye), **Dhruva** (the pole star)

**Journey and planet**
- **Graha Yatra** – "journey of the planets"
- **Pushpak** – the flying chariot of the Ramayana (check overlap with ISRO's test vehicle name)
- **Parikrama** – circuit / orbit

**Light**
- **Aloka** – light, and also "seeing"
- **Jyotirgamaya** – "lead me to the light" (shorten to Jyotir)
- **Anya Jyoti** – "another light"
- Avoid **Jyotish**, which means astrology

**Fun**
- **Dip Dip Hurray** – "catch the dip, find the planet"
- **Duniya Dhoondo** – "find the worlds"
- **Planet Pakad**, **Blink**, **Jugaad Telescope**

**Top picks:** Divya Drishti (myth), Graha Yatra (journey), Aloka (light), Dip Dip Hurray (fun).

**Before choosing:** search Google, GitHub and domain sites; ask a fluent Hindi or Sanskrit speaker to check meanings and pronunciation; fix one English spelling; check whether the event rules require the name to match the registered project title.
