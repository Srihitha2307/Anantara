"""
Anantara - model definitions + inference helpers (the glue between the notebooks and the website).

  Net2D          Model A: 2D CNN on a wavelength x time image -> H2O / CO2 / CH4 / SO2 candidate scores
  TransitCNN1D   Model B: 1D CNN on a 2x201 (global + local) folded light curve -> planet-like score

Both architectures are copied exactly from the training notebooks so the saved .pt files load.
Model A thresholds come from models/modelA_thresholds.json (tuned on validation data only).

Wording rule: these are CANDIDATE INDICATIONS, never detections.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

# ------------------------------------------------------------------ Model A constants (must match training)
NL, NT = 128, 64
WL = np.linspace(0.7, 5.2, NL)
MOLS = ["H2O", "CO2", "CH4", "SO2"]
MOL_LABEL = {"H2O": "H₂O", "CO2": "CO₂", "CH4": "CH₄", "SO2": "SO₂"}
BANDS = {"H2O": [(0.94, .03, .3), (1.13, .04, .4), (1.38, .05, .7), (1.87, .08, .8), (2.70, .20, 1.0)],
         "CO2": [(2.00, .03, .15), (2.70, .04, .3), (4.26, .07, 1.0)],
         "CH4": [(1.16, .03, .2), (1.38, .03, .3), (1.68, .04, .4), (2.32, .05, .5), (3.30, .07, .9)],
         "SO2": [(4.05, .05, 1.0)]}
MODES = {"jwst": (0.7, 5.2), "nirspec": (2.87, 5.2), "hst": (1.1, 1.7)}
BINS = 201                                               # Model B view length


def coverage(mode):
    lo, hi = MODES[mode]
    return (WL >= lo) & (WL <= hi)


def vis_mask(lo, hi):
    """A molecule is only scored if a strong band lies inside the observed wavelength range."""
    return np.array([float(any((c >= lo - s) and (c <= hi + s) and w >= 0.3 for c, s, w in BANDS[m])) for m in MOLS])


# ------------------------------------------------------------------ architectures
def _blk2(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


class Net2D(nn.Module):
    def __init__(s):
        super().__init__()
        s.f = nn.Sequential(_blk2(2, 32), _blk2(32, 64), _blk2(64, 128),
                            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
        s.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 4))

    def forward(s, x):
        c = torch.linspace(-1, 1, x.shape[2], device=x.device)[None, None, :, None].expand(x.shape[0], 1, x.shape[2], x.shape[3])
        return s.fc(s.f(torch.cat([x, c], 1)).mean((2, 3)))


class TransitCNN1D(nn.Module):
    def __init__(s):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv1d(i, o, 5, padding=2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        s.features = nn.Sequential(blk(2, 16), blk(16, 32), blk(32, 64), nn.Conv1d(64, 64, 5, padding=2), nn.ReLU())
        s.head = nn.Sequential(nn.Dropout(.4), nn.Linear(64, 1))

    def forward(s, x):
        return s.head(s.features(x).mean(-1)).squeeze(-1)


# ------------------------------------------------------------------ lazy loading
_cache = {}


def _load(cls, path):
    m = cls()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.eval()


def model_a():
    if "a" not in _cache:
        _cache["a"] = _load(Net2D, MODELS / "modelA_v4.pt")
    return _cache["a"]


def model_b_members():
    """Primary Model B plus the 5 cross-validation folds (if present) -> averaged score."""
    if "b" not in _cache:
        paths = [MODELS / "model_b_1d.pt"] + sorted(MODELS.glob("transit_cnn_fold_*.pt"))
        _cache["b"] = [_load(TransitCNN1D, p) for p in paths if p.exists()]
    return _cache["b"]


def thresholds():
    p = MODELS / "modelA_thresholds.json"
    t = json.loads(p.read_text()) if p.exists() else {}
    return np.array([t.get(m, 0.5) for m in MOLS])


def models_ready():
    return {"model_a": (MODELS / "modelA_v4.pt").exists(), "model_b": (MODELS / "model_b_1d.pt").exists(),
            "model_b_members": len(list(MODELS.glob("transit_cnn_fold_*.pt"))) + int((MODELS / "model_b_1d.pt").exists())}


# ------------------------------------------------------------------ Model A: spectrum file -> image -> scores
def shape_curve(rp0, a, b, P=3.0):
    rp0 = float(np.clip(rp0, 0.01, 0.3))
    inc = np.degrees(np.arccos(b / a))
    T14 = P / np.pi / a * np.sqrt((1 + rp0) ** 2 - b ** 2)
    t = np.linspace(-T14, T14, NT)
    try:
        import batman
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = 0., P, rp0, a, inc
        p.ecc, p.w, p.u, p.limb_dark = 0., 90., [0.4, 0.25], "quadratic"
        f0 = batman.TransitModel(p, t).light_curve(p)
    except ImportError:                                  # trapezoid fallback (slightly off the training shape)
        x = np.abs(t) / T14
        f0 = 1 - rp0 ** 2 * np.clip((1 - x) / 0.15, 0, 1)
    return (1 - f0) / max(1 - f0.min(), 1e-9)


def to_image(depth, sig_pix, shape, cov, rng):
    flux = 1 - depth[:, None] * shape[None, :] + rng.normal(0, 1, (NL, NT)) * sig_pix[:, None]
    flux = flux / np.median(flux[:, shape < 0.02], axis=1, keepdims=True)
    img = (flux - flux[cov].mean(0)[None, :]) / sig_pix[:, None]
    img[~cov] = 0
    return img.astype(np.float32)


def spectrum_to_image(wl, depth_ppm, err_ppm, mode, seed=0):
    """Identical to the notebook's real-file path (section 5/6)."""
    rng = np.random.default_rng(seed)
    cov = coverage(mode) & (WL >= wl.min()) & (WL <= wl.max())
    d = np.interp(WL, wl, depth_ppm) * 1e-6
    e = np.interp(WL, wl, err_ppm) * 1e-6
    sig_pix = np.maximum(e * np.sqrt(NT / 2), 1e-9)      # err is DEPTH error -> per-pixel noise
    return to_image(d, sig_pix, shape_curve(np.sqrt(max(d[cov].mean(), 1e-4)), a=10., b=0.3), cov, rng)


_BAD = ("err", "unc", "lim", "upper", "lower", "neg", "pos", "flag", "ref", "name")


def parse_spectrum(df):
    """DataFrame -> (wl_um, depth_ppm, err_ppm) or None if it does not look like a transmission spectrum."""
    import pandas as pd
    low = {c.lower(): c for c in df.columns}
    wc = next((low[c] for c in low if "wave" in c and not any(b in c for b in _BAD)), None)
    dc = next((low[c] for c in low if ("dep" in c or "rp" in c or "radius" in c) and not any(b in c for b in _BAD)), None)
    ec = next((low[c] for c in low if ("err" in c or "unc" in c)), None)
    if not (wc and dc and ec):
        return None
    num = df[[wc, dc, ec]].apply(pd.to_numeric, errors="coerce").dropna().sort_values(wc)
    if len(num) < 10:
        return None
    wl, d, e = num.iloc[:, 0].values, num.iloc[:, 1].values, np.abs(num.iloc[:, 2].values)
    if wl.min() < 0.3 or wl.max() > 6.0:
        return None
    m = np.median(d)
    k = 1.0 if m >= 100 else (1e4 if m >= 0.3 else 1e6)  # ppm / percent / fraction
    return wl, d * k, e * k


def pick_mode(wl, name=""):
    if "hst" in name.lower() or "wfc3" in name.lower() or wl.max() <= 1.8:
        return "hst"
    return "nirspec" if wl.min() > 2.8 else "jwst"


def predict_molecules(wl, depth_ppm, err_ppm, name=""):
    """Returns {mode, probs, decisions, visible, threshold} using Model A."""
    mode = pick_mode(wl, name)
    img = spectrum_to_image(wl, depth_ppm, err_ppm, mode)
    with torch.no_grad():
        p = torch.sigmoid(model_a()(torch.from_numpy(img)[None, None])).numpy()[0]
    vis = vis_mask(max(wl.min(), MODES[mode][0]), min(wl.max(), MODES[mode][1]))
    thr = thresholds()
    return {"mode": mode, "probs": {m: float(x) for m, x in zip(MOLS, p)},
            "decisions": {m: bool(x > t and v > 0) for m, x, t, v in zip(MOLS, p, thr, vis)},
            "visible": {m: bool(v) for m, v in zip(MOLS, vis)},
            "threshold": {m: float(t) for m, t in zip(MOLS, thr)}}


# ------------------------------------------------------------------ Model B: folded views -> planet-like score
def predict_transit(views):
    """views: float32 array (2, 201) -> {score, members, spread}. Averages primary model + CV folds."""
    x = torch.from_numpy(np.asarray(views, np.float32))[None]
    with torch.no_grad():
        s = [float(torch.sigmoid(m(x))[0]) for m in model_b_members()]
    return {"score": float(np.mean(s)), "members": len(s), "spread": float(np.std(s))}
