"""Offline smoke tests (no internet needed).  Run:  pytest -q"""
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import anantara_physics as phys
import anantara_models as am
import anantara_pipeline as pipe


def synth_lightcurve(P=3.2, depth=0.01, days=60, seed=0):
    import batman
    rng = np.random.default_rng(seed)
    t = np.arange(0, days, 2 / 1440)
    p = batman.TransitParams()
    p.t0, p.per, p.rp, p.a, p.inc, p.ecc, p.w, p.u, p.limb_dark = 1.0, P, np.sqrt(depth), 9.0, 89.0, 0., 90., [0.4, 0.25], "quadratic"
    f = batman.TransitModel(p, t).light_curve(p) * (1 + 0.002 * np.sin(2 * np.pi * t / 7)) + rng.normal(0, 4e-4, len(t))
    return pd.DataFrame({"time": t, "flux": f, "flux_err": 4e-4})


def synth_spectrum(mol="CH4", seed=1):
    rng = np.random.default_rng(seed)
    wl = np.linspace(2.9, 5.2, 90)
    depth = np.full_like(wl, 3600.0)
    for c, s, w in am.BANDS[mol]:
        depth += 2500 * w * np.exp(-0.5 * ((wl - c) / s) ** 2)
    depth += rng.normal(0, 60, len(wl))
    return pd.DataFrame({"wavelength_um": wl, "depth_ppm": depth, "depth_err_ppm": 60.0})


def test_physics_selftest():
    phys.self_test()


def test_weights_load_and_run():
    assert am.models_ready()["model_a"] and am.models_ready()["model_b"]
    r = am.predict_transit(np.zeros((2, am.BINS), np.float32))
    assert 0 <= r["score"] <= 1 and r["members"] >= 1


def test_lightcurve_period_recovered():
    out = pipe.analyse_lightcurve(synth_lightcurve(P=3.2))
    assert abs(out["P"] - 3.2) / 3.2 < 0.01, out
    assert 0 <= out["cf"] <= 1


def test_long_period_with_stellar_variability():
    out = pipe.analyse_lightcurve(synth_lightcurve(P=32.9, depth=0.0028, days=90))
    assert abs(out["P"] - 32.9) / 32.9 < 0.02, out


def test_spectrum_runs():
    out = pipe.analyse_spectrum(synth_spectrum(), "x_jwst_nirspec.csv")
    assert set(out["probs"]) == set(am.MOLS) and out["mode"] == "nirspec"


def test_demo_planets_physics():
    import json
    d = {r["n"]: r for r in json.loads((ROOT / "web/data/planets.json").read_text())}
    assert d["TRAPPIST-1 b"]["srv"]["shoreline"] == "stripped"
    assert d["K2-18 b"]["srv"]["shoreline"] == "retained"


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)


def _csv(df):
    return io.BytesIO(df.to_csv(index=False).encode())


def test_api_health_and_site(client):
    assert client.get("/api/health").json()["status"] == "ok"
    assert "Anantara" in client.get("/").text


def test_api_lightcurve_with_name(client):
    r = client.post("/api/analyse", files=[("files", ("lc.csv", _csv(synth_lightcurve()), "text/csv"))], data={"name": "K2-18 b"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n"] == "K2-18 b" and j["source"]["cf"] == "model_b" and "srv" in j


def test_api_lightcurve_unknown_star_is_flagged(client):
    r = client.post("/api/analyse", files=[("files", ("lc.csv", _csv(synth_lightcurve()), "text/csv"))])
    assert r.status_code == 200
    assert any("ASSUMED" in n for n in r.json()["notes"])


def test_api_spectrum_needs_name(client):
    r = client.post("/api/analyse", files=[("files", ("spec.csv", _csv(synth_spectrum()), "text/csv"))])
    assert r.status_code == 422 and "name" in r.json()["detail"].lower()


def test_api_spectrum_with_name(client):
    r = client.post("/api/analyse", files=[("files", ("spec.csv", _csv(synth_spectrum()), "text/csv"))], data={"name": "K2-18 b"})
    assert r.status_code == 200, r.text
    assert r.json()["source"]["mol"] == "model_a" and set(r.json()["mol"]) == {"H₂O", "CO₂", "CH₄", "SO₂"}


def test_api_garbage_rejected(client):
    r = client.post("/api/analyse", files=[("files", ("x.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"), "text/csv"))])
    assert r.status_code == 422
