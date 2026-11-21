"""
Anantara - end-to-end pipeline (closes the "single pipeline" open end from the project notes):

    light curve file -> clean -> BLS period search -> flatten + fold -> Model B (1D CNN)
                     -> physics (orbit, habitable zone, escape velocity, shoreline)
    spectrum file    -> Model A (2D CNN) -> H2O / CO2 / CH4 / SO2 candidate scores

Every function returns plain dicts shaped for web/index.html (see `to_frontend`). Each record says where
each number came from in `source`, so the site never presents a placeholder as a measurement.
"""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

import anantara_physics as phys
import anantara_models as am

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference"
LY_PER_PC = 3.26156
SUN = dict(st_teff=5772.0, st_rad=1.0, st_mass=1.0)      # used (and flagged) when nothing is known about the star


# ------------------------------------------------------------------ reference tables (real NASA Exoplanet Archive values)
_ref = {}


def reference():
    if "t" not in _ref:
        p = pd.read_csv(REF / "planets.csv")
        s = pd.read_csv(REF / "stars.csv")
        t = p.merge(s, on="hostname", how="left")
        t["_key"] = t.pl_name.map(norm_name)
        _ref["t"] = t
    return _ref["t"]


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_planet(name):
    """'k2-18b', 'K2-18 b', 'wasp39b_jwst_nirspec.csv' -> reference row (dict) or None."""
    if not name:
        return None
    t, k = reference(), norm_name(name)
    hit = t[t._key == k]
    if len(hit) == 0:                                    # file names carry extra words: find a planet key inside the name
        cand = [x for x in t._key.unique() if len(x) >= 6 and x in k]
        if cand:
            hit = t[t._key == max(cand, key=len)]
    return None if len(hit) == 0 else hit.iloc[0].to_dict()


# ------------------------------------------------------------------ file readers
_TIME = ("time", "bjd", "btjd", "bkjd", "jd", "mjd", "t")
_FLUX = ("pdcsap_flux", "sap_flux", "flux", "normalized_flux", "norm_flux", "f")
_ERR = ("pdcsap_flux_err", "sap_flux_err", "flux_err", "flux_error", "err", "e")


def read_table(data: bytes, filename: str):
    """bytes -> (DataFrame, header dict). Handles FITS (Kepler/TESS light curves), CSV, TSV, whitespace text."""
    fn = filename.lower()
    if fn.endswith((".fits", ".fit", ".fits.gz")) or data[:6] == b"SIMPLE":
        from astropy.io import fits
        with fits.open(io.BytesIO(data)) as h:
            hdr = {k: h[0].header.get(k) for k in ("OBJECT", "TEFF", "RADIUS", "LOGG", "KEPLERID", "TICID")}
            tab = next(x for x in h if getattr(x, "columns", None) is not None and "TIME" in x.columns.names)
            df = pd.DataFrame({c: np.asarray(tab.data[c]).astype(float) for c in tab.columns.names
                               if np.asarray(tab.data[c]).ndim == 1})
        return df, hdr
    text = data.decode("utf-8", "ignore")
    sep = "\t" if "\t" in text[:2000] else (r"\s+" if ("," not in text[:2000] and ";" not in text[:2000]) else None)
    df = pd.read_csv(io.StringIO(text), comment="#", sep=sep, engine="python")
    if all(re.match(r"^-?[\d.eE+-]+$", str(c)) for c in df.columns[:2]):   # headerless two-column file
        df = pd.read_csv(io.StringIO(text), comment="#", sep=sep, engine="python", header=None)
        df.columns = ["time", "flux", "flux_err"][:df.shape[1]] + [f"c{i}" for i in range(max(0, df.shape[1] - 3))]
    return df, {}


def classify(df):
    """'spectrum' | 'lightcurve' | None"""
    if am.parse_spectrum(df) is not None:
        return "spectrum"
    low = [c.lower() for c in df.columns]
    if any(c in _TIME for c in low) and any(c in _FLUX for c in low):
        return "lightcurve"
    return None


def _col(df, names):
    low = {c.lower(): c for c in df.columns}
    return next((low[n] for n in names if n in low), None)


# ------------------------------------------------------------------ light curve -> Model B
def clean_lightcurve(df):
    tc, fc = _col(df, _TIME), _col(df, _FLUX)
    t = pd.to_numeric(df[tc], errors="coerce").values
    f = pd.to_numeric(df[fc], errors="coerce").values
    ok = np.isfinite(t) & np.isfinite(f)
    t, f = t[ok], f[ok]
    o = np.argsort(t)
    t, f = t[o], f[o]
    if len(t) < 200:
        raise ValueError("light curve needs at least 200 valid points")
    f = f / np.nanmedian(f)
    return t, f


def bls_search(t, f, period=None, pmin=0.5, pmax=40.0):
    """Box Least Squares period search. Returns dict(period, t0, duration, depth, snr)."""
    from astropy.timeseries import BoxLeastSquares
    base = t.max() - t.min()
    if period:
        periods = np.linspace(period * 0.995, period * 1.005, 400)
    else:
        pmax = min(pmax, base / 2)
        if pmax <= pmin:
            raise ValueError("light curve is too short for a period search; pass the period explicitly")
        periods = np.exp(np.linspace(np.log(pmin), np.log(pmax), 6000))
    # bin to ~10 min so BLS stays fast on long Kepler/TESS curves
    step = 10 / 1440
    if np.median(np.diff(t)) < step * 0.7:
        idx = np.floor((t - t[0]) / step).astype(int)
        u, inv = np.unique(idx, return_inverse=True)
        tb = np.bincount(inv, t) / np.bincount(inv)
        fb = np.bincount(inv, f) / np.bincount(inv)
    else:
        tb, fb = t, f
    # pre-flatten (2.5-day running median) so slow stellar variability does not beat the transit; keeps transits up to ~1 day
    from scipy.ndimage import median_filter
    cadb = np.median(np.diff(tb))
    wb = int(2.5 / cadb) // 2 * 2 + 1
    if wb < len(fb) // 2:
        fb = fb / median_filter(fb, size=wb, mode="nearest")
    durs = np.array([0.04, 0.07, 0.1, 0.15, 0.22, 0.3, 0.45, 0.7, 1.0])
    sig = 1.4826 * np.median(np.abs(np.diff(fb))) / np.sqrt(2) or 1e-4      # robust point-to-point scatter
    res = BoxLeastSquares(tb, fb, dy=np.full(len(tb), sig)).power(periods, durs[durs < periods.min() * 0.5], objective="snr")
    i = int(np.argmax(res.power))
    return dict(period=float(res.period[i]), t0=float(res.transit_time[i]), duration=float(res.duration[i]),
                depth=float(res.depth[i]), snr=float(res.depth_snr[i]))


def _binned(ph, f, lo, hi, nb):
    idx = np.digitize(ph, np.linspace(lo, hi, nb + 1)) - 1
    out = np.full(nb, np.nan)
    for i in range(nb):
        m = idx == i
        if m.any():
            out[i] = np.median(f[m])
    ok = ~np.isnan(out)
    if ok.sum() < nb // 2:
        return None
    out[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], out[ok])
    return out


def make_views(t, f, P, t0, dur_d):
    """Same recipe as the Model B notebook: transit-masked flatten, fold, global (-P/2..P/2) + local (+-3 durations) views."""
    from scipy.ndimage import median_filter
    cad = np.median(np.diff(t))
    win = int(np.clip(4 * dur_d, 0.75, 3) / cad) // 2 * 2 + 1
    ph0 = ((t - t0 + 0.5 * P) % P) - 0.5 * P
    intr = np.abs(ph0) < dur_d                           # mask 2x duration like the notebook
    trend = np.interp(t, t[~intr], median_filter(f[~intr], size=min(win, max(3, (~intr).sum() // 2 * 2 - 1)), mode="nearest"))
    flat = f / trend
    keep = flat < 1 + 4 * np.nanstd(flat[~intr])         # remove upward outliers (4 sigma), keep deep dips
    flat, ph = flat[keep], ph0[keep]
    y = flat / np.nanmedian(flat) - 1
    g, l = _binned(ph, y, -P / 2, P / 2, am.BINS), _binned(ph, y, -3 * dur_d, 3 * dur_d, am.BINS)
    if g is None or l is None:
        raise ValueError("not enough in-phase coverage to build the folded views")
    s = abs(min(g.min(), l.min())) + 1e-8
    return np.stack([g / s, l / s]).astype(np.float32), ph, y


def analyse_lightcurve(df, hdr=None, period=None):
    t, f = clean_lightcurve(df)
    b = bls_search(t, f, period)
    views, ph, y = make_views(t, f, b["period"], b["t0"], b["duration"])
    mb = am.predict_transit(views)
    return dict(P=b["period"], t0=b["t0"], dur_d=b["duration"], depth_frac=max(b["depth"], 1e-6), bls_snr=b["snr"],
                cf=mb["score"], cf_members=mb["members"], cf_spread=mb["spread"], n_points=int(len(t)),
                baseline_d=float(t.max() - t.min()))


# ------------------------------------------------------------------ spectrum -> Model A
def analyse_spectrum(df, filename=""):
    r = am.parse_spectrum(df)
    if r is None:
        raise ValueError("could not find wavelength / depth / error columns (wavelength 0.3-6 um, >=10 rows)")
    out = am.predict_molecules(*r, name=filename)
    out["wl_range"] = [float(r[0].min()), float(r[0].max())]
    return out


# ------------------------------------------------------------------ physics + frontend record
def star_color(teff):
    return 0xff7a45 if teff < 3000 else 0xffa066 if teff < 3900 else 0xffd08a if teff < 5200 else 0xfff0c8 if teff < 6200 else 0xdfe8ff


def planet_color(rp):
    return 0xc98b6b if rp < 1.3 else 0x9bb88f if rp < 2.0 else 0x7fb3d5 if rp < 4 else 0x8fa4c8


def build_record(name, star, planet=None, lc=None, spec=None, notes=None, cached_source="archive"):
    """
    star:   dict with st_teff, st_rad, st_mass (+ optional st_lum, sy_dist)
    planet: dict with pl_rade, pl_bmasse, pl_orbper, pl_orbsmax (any may be missing)
    lc:     result of analyse_lightcurve (supplies P, depth, cf) or None
    spec:   result of analyse_spectrum or None
    """
    notes = list(notes or [])
    planet = dict(planet or {})
    src = {"cf": "none", "mol": "none", "orbit": "archive" if planet.get("pl_orbper") else "none",
           "star": cached_source}
    teff, rs = float(star["st_teff"]), float(star["st_rad"])
    ms = star.get("st_mass")
    if lc:
        ap = planet.get("pl_orbper")
        if ap and abs(lc["P"] - ap) / ap < 0.02:
            notes.append("BLS period agrees with the archive period; archive orbit used")
        else:
            if ap:
                notes.append(f"BLS period {lc['P']:.3f} d differs from archive {ap:.3f} d; orbit recomputed from the BLS period")
            planet.pop("pl_orbsmax", None)
            planet["pl_orbper"] = lc["P"]
        src["orbit"] = "bls"
        if not planet.get("pl_rade"):
            planet["pl_rade"] = float(np.sqrt(lc["depth_frac"]) * rs * 109.076)
            notes.append("planet radius from transit depth and the star radius")
        src["cf"] = "model_b"
    p = {**planet, **dict(st_teff=teff, st_rad=rs, st_mass=ms, st_lum=star.get("st_lum"), pl_name=name)}
    res = phys.analyze_planet(p)
    if "error" in res:
        raise ValueError(res["error"])
    sl = res["shoreline"]
    L, a, rp = res["luminosity_lsun"], res["a_au"], res["radius_earth"]
    hz = [*phys.habitable_zone(L, teff, optimistic=True)[:1], *phys.habitable_zone(L, teff)[:2], phys.habitable_zone(L, teff, optimistic=True)[1]]
    dep_pct = float((rp / (rs * 109.076)) ** 2 * 100)
    rec = dict(n=name, T=teff, Rs=rs, L=float(L), a=float(a), P=float(res_period(p, a, ms)), M=float(res["mass_earth"]),
               d=float(star["sy_dist"] * LY_PER_PC) if star.get("sy_dist") else None,
               dep=dep_pct, pc=planet_color(rp), sc=star_color(teff),
               srv=dict(hz=[float(x) for x in hz], Rp=float(rp), ve=float(res["v_esc_kms"]), mg=float(sl["log10_margin"]),
                        I=float(sl["i_xuv_rel"]), in_hz=res["in_hz"], in_hz_opt=res["in_hz_optimistic"],
                        shoreline=sl["verdict"], mass_is_estimate=res["mass_is_estimate"]))
    if lc:
        rec["cf"] = lc["cf"]
        rec["lc"] = {k: lc[k] for k in ("bls_snr", "n_points", "baseline_d", "cf_members", "cf_spread")}
    if spec:
        rec["mol"] = {am.MOL_LABEL[m]: spec["probs"][m] for m in am.MOLS}
        rec["molv"] = {am.MOL_LABEL[m]: spec["visible"][m] for m in am.MOLS}
        rec["thr"] = {am.MOL_LABEL[m]: spec["threshold"][m] for m in am.MOLS}
        rec["spec_mode"] = spec["mode"]
        src["mol"] = "model_a"
    rec["source"] = src
    rec["notes"] = notes + res["notes"]
    return rec


def res_period(p, a, ms):
    if p.get("pl_orbper"):
        return float(p["pl_orbper"])
    return float(365.25 * np.sqrt(a ** 3 / (ms or 1.0)))


def star_from(row=None, hdr=None, overrides=None):
    """Pick stellar parameters: user overrides > archive row > FITS header > Sun-like (flagged)."""
    notes, s, source = [], dict(SUN), "assumed Sun-like"
    if hdr and hdr.get("TEFF") and hdr.get("RADIUS"):
        s["st_teff"], s["st_rad"] = float(hdr["TEFF"]), float(hdr["RADIUS"])
        if hdr.get("LOGG"):
            s["st_mass"] = float(10 ** float(hdr["LOGG"]) * (s["st_rad"] * 6.957e8) ** 2 / 6.674e-11 / 1.989e30 * 1e-2)
        source = "FITS header"
    if row and not pd.isna(row.get("st_teff")):
        for k in ("st_teff", "st_rad", "st_mass", "st_lum", "sy_dist"):
            v = row.get(k)
            if v is not None and not pd.isna(v):
                s[k] = float(v)
        source = "NASA Exoplanet Archive"
    for k, v in (overrides or {}).items():
        if v:
            s[k] = float(v)
            source = "user supplied"
    if source != "assumed Sun-like" and not (row is not None and not pd.isna(row.get("st_mass"))) and not (overrides or {}).get("st_mass"):
        r = s["st_rad"]
        s["st_mass"] = r if r < 1 else r ** 1.25            # rough main-sequence mass-radius relation
        notes.append("star mass is a rough estimate from its radius; give st_mass for a precise orbit")
    if source == "assumed Sun-like":
        notes.append("No star parameters supplied, so a Sun-like star is ASSUMED: orbit, habitable zone and shoreline are indicative only. "
                     "Enter the planet name (e.g. 'K2-18 b') or the star's Teff / radius / mass.")
    return s, source, notes
