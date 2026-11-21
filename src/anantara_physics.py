"""
Anantara - planet/star parameters + physics (no machine learning in this file)

  load_tables()        -> read YOUR planets.csv (+ optional stars.csv)       [or fetch_params() for live Archive data]
  orbit_distance()     -> Kepler's third law
  habitable_zone()     -> Kopparapu et al. (2013)
  escape_velocity()    -> v_esc, with Chen & Kipping (2017) mass estimate if mass is missing
  cosmic_shoreline()   -> I_xuv = K * v_esc^4  (Mars-anchored; see notes below)
  analyze_planet()     -> everything for one planet, as a dict
  write_derived()      -> saves results to derived.csv

Run:
  python anantara_physics.py --planets planets.csv --stars stars.csv --out derived.csv
  python anantara_physics.py --planets all_in_one.csv           (one CSV that already has star columns)
  python anantara_physics.py --live                             (fetch 3 demo planets from the Archive)
  python anantara_physics.py --selftest                         (checks Earth/Venus/Mars sit where expected)

COSMIC SHORELINE - what is and isn't established
  * Zahnle & Catling (2017) draw the shoreline "by eye": slope 4 in log-log (I_xuv ~ v_esc^4).
    They give no single fitted constant.
  * The zero point K below is anchored on Mars (v_esc 5.03 km/s, I_xuv 0.43) - a modelling CHOICE,
    not a published fit. K is a parameter: replace it if you adopt a fitted value (e.g. Meni-Gallardo & Palle).
  * I_xuv = (a_earth/a)^2 * (L*/Lsun)^0.4 is the star-type scaling from the reference notes you gave me.
    Please check it against Zahnle & Catling (2017) before presenting it as theirs.
  * Real measured XUV (MUSCLES/Hubble) can be passed as xuv_rel (Earth = 1) and overrides the scaling.
"""
import io, sys, argparse
import numpy as np

G, M_EARTH, R_EARTH = 6.674e-11, 5.972e24, 6.371e6
V_ESC_EARTH = 11.19                                # km/s

# ------------------------------------------------------------------ 1. DATA IN
CANON = ["pl_name","hostname","pl_rade","pl_bmasse","pl_bmassprov","pl_orbper","pl_orbsmax",
         "st_teff","st_rad","st_mass","st_lum","st_age","sy_dist"]
# friendly column names -> canonical Archive names (add yours here if your CSV uses other headers)
ALIASES = {"planet":"pl_name","planet_name":"pl_name","name":"pl_name",
           "star":"hostname","host":"hostname","star_name":"hostname",
           "radius_earth":"pl_rade","planet_radius":"pl_rade","radius":"pl_rade",
           "mass_earth":"pl_bmasse","planet_mass":"pl_bmasse","mass":"pl_bmasse",
           "period":"pl_orbper","period_days":"pl_orbper","orbital_period":"pl_orbper",
           "semi_major_axis":"pl_orbsmax","a_au":"pl_orbsmax",
           "teff":"st_teff","star_teff":"st_teff","star_radius":"st_rad","star_mass":"st_mass",
           "log_lum":"st_lum","age_gyr":"st_age","distance_pc":"sy_dist"}

def _read(path):
    import pandas as pd
    df = pd.read_csv(path, comment="#")                       # '#' lines = NASA Archive download header
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=ALIASES)
    if "lum_lsun" in df.columns and "st_lum" not in df.columns:   # linear luminosity -> log10
        df["st_lum"] = np.log10(df["lum_lsun"])
    return df

def load_tables(planets_csv, stars_csv=None):
    """planets.csv (+ stars.csv joined on 'hostname') -> one DataFrame, one row per planet."""
    df = _read(planets_csv)
    if stars_csv:
        s = _read(stars_csv)
        if "hostname" not in df.columns or "hostname" not in s.columns:
            raise ValueError("To join planets.csv and stars.csv both need a 'hostname' column")
        s = s[["hostname"] + [c for c in s.columns if c not in df.columns]]
        df = df.merge(s, on="hostname", how="left")
    missing = [c for c in ("pl_name","pl_rade","st_teff") if c not in df.columns]
    if missing: raise ValueError(f"CSV is missing required columns {missing}. Found: {list(df.columns)}")
    return df

def fetch_params(names):
    """Live alternative: names like ['K2-18 b'] -> DataFrame from the NASA Exoplanet Archive."""
    import requests, pandas as pd
    lst = ",".join("'%s'" % n for n in names)
    q = f"select {','.join(c for c in CANON if c != 'pl_bmassprov')} from pscomppars where pl_name in ({lst})"
    r = requests.get("https://exoplanetarchive.ipac.caltech.edu/TAP/sync", params={"query": q, "format": "csv"}, timeout=60)
    r.raise_for_status(); return pd.read_csv(io.StringIO(r.text))

# ------------------------------------------------------------------ 2. ORBIT
def orbit_distance(period_days, star_mass_msun):
    """Kepler's third law: a^3 = M * P^2  (a in AU, P in years, M in solar masses)."""
    return (star_mass_msun * (period_days / 365.25) ** 2) ** (1 / 3)

# ------------------------------------------------------------------ 3. HABITABLE ZONE
# Kopparapu et al. 2013, Table 3: Seff_sun, a, b, c, d  (re-check against the paper)
SEFF = {"recent_venus":       (1.7753, 1.4316e-4,  2.9875e-9, -7.5702e-12, -1.1635e-15),
        "runaway_greenhouse": (1.0512, 1.3242e-4,  1.5418e-8, -7.9895e-12, -1.8328e-15),
        "maximum_greenhouse": (0.3438, 5.8942e-5,  1.6558e-9, -3.0045e-12, -5.2983e-16),
        "early_mars":         (0.3179, 5.4513e-5,  1.5313e-9, -2.7786e-12, -4.8997e-16)}
def _seff(teff, key):
    s0, a, b, c, d = SEFF[key]; t = teff - 5780.0
    return s0 + a*t + b*t**2 + c*t**3 + d*t**4

def luminosity(lum_log10=None, radius_rsun=None, teff=None):
    """Solar units. Catalogue log10(L) if present, else L = R^2 (T/5772)^4."""
    if lum_log10 is not None: return 10 ** lum_log10
    return radius_rsun ** 2 * (teff / 5772.0) ** 4

def habitable_zone(lum_lsun, teff, optimistic=False):
    """(inner_AU, outer_AU, warning). d = sqrt(L/Seff). Valid for 2600-7200 K."""
    warn = None
    if not 2600 <= teff <= 7200:
        warn = f"Teff {teff:.0f} K outside Kopparapu's valid range (2600-7200 K); clipped"
        teff = min(max(teff, 2600), 7200)
    ki, ko = ("recent_venus", "early_mars") if optimistic else ("runaway_greenhouse", "maximum_greenhouse")
    return np.sqrt(lum_lsun / _seff(teff, ki)), np.sqrt(lum_lsun / _seff(teff, ko)), warn

# ------------------------------------------------------------------ 4. ESCAPE VELOCITY
def mass_from_radius(r_earth):
    """Chen & Kipping (2017) median mass-radius relation (Earth units). EMPIRICAL ESTIMATE."""
    if r_earth < 1.23: return (r_earth / 1.008) ** (1 / 0.279)      # terran
    if r_earth < 14.3: return (r_earth / 0.808) ** (1 / 0.589)      # neptunian
    return 318.0                                                    # jovian
def escape_velocity(m_earth, r_earth):
    """v_esc = sqrt(2GM/R) in km/s."""
    return np.sqrt(2 * G * m_earth * M_EARTH / (r_earth * R_EARTH)) / 1000.0

# ------------------------------------------------------------------ 5. COSMIC SHORELINE
SLOPE = 4                                      # I_xuv ~ v_esc^4  (Zahnle & Catling 2017)
A_MARS, V_MARS = 1.524, 5.03                   # AU, km/s  (Mars: L = 1 Lsun)
K_MARS = (1.0 / A_MARS**2) / V_MARS**SLOPE     # ~6.7e-4 (km/s)^-4 : zero point anchored on Mars

def xuv_relative(lum_lsun, a_au):
    """Cumulative XUV instellation relative to Earth:  (a_earth/a)^2 * (L/Lsun)^0.4"""
    return (1.0 / a_au**2) * lum_lsun ** 0.4

def cosmic_shoreline(lum_lsun, a_au, v_esc, K=None, xuv_rel=None, band=0.5):
    """
    margin = log10( I_xuv / (K * v_esc^4) )
        margin > +band -> above the line: atmosphere likely STRIPPED
        margin < -band -> below the line: atmosphere can be RETAINED
        otherwise      -> BORDERLINE   (the line was drawn by eye; +-0.5 dex ~ factor 3)
    xuv_rel: measured cumulative XUV relative to Earth (e.g. from MUSCLES); overrides the scaling.
    """
    K = K or K_MARS
    i_xuv = xuv_rel if xuv_rel is not None else xuv_relative(lum_lsun, a_au)
    margin = float(np.log10(i_xuv / (K * v_esc ** SLOPE)))
    verdict = "stripped" if margin > band else ("retained" if margin < -band else "borderline")
    return {"verdict": verdict, "log10_margin": margin, "i_xuv_rel": float(i_xuv),
            "i_xuv_on_line": float(K * v_esc ** SLOPE), "K": K,
            "xuv_source": "measured" if xuv_rel is not None else "scaling (L^0.4 / a^2)"}

def self_test():
    """Solar-system sanity check: Mars ~ on the line, Venus and Earth below it."""
    for name, a, v in [("Earth", 1.0, 11.19), ("Venus", 0.723, 10.36), ("Mars", A_MARS, V_MARS)]:
        r = cosmic_shoreline(1.0, a, v); print(f"{name:6s} margin {r['log10_margin']:+.2f}  -> {r['verdict']}")
    assert abs(cosmic_shoreline(1.0, A_MARS, V_MARS)["log10_margin"]) < 1e-9
    assert cosmic_shoreline(1.0, 1.0, 11.19)["verdict"] == "retained"
    print("self-test OK; K =", f"{K_MARS:.2e}")

# ------------------------------------------------------------------ 6. EVERYTHING FOR ONE PLANET
def _num(x):
    try:
        x = float(x); return x if np.isfinite(x) else None
    except (TypeError, ValueError): return None

def analyze_planet(p, K=None, xuv_rel=None):
    g = lambda k: _num(p.get(k))
    out = {"planet": p.get("pl_name"), "notes": []}
    teff, r = g("st_teff"), g("pl_rade")
    if teff is None or r is None:
        out["error"] = "need st_teff and pl_rade"; return out
    if g("st_lum") is None and g("st_rad") is None:
        out["error"] = "need st_lum or st_rad"; return out
    L = luminosity(g("st_lum"), g("st_rad"), teff)
    a = g("pl_orbsmax")
    if a is None:
        if g("pl_orbper") is None or g("st_mass") is None:
            out["error"] = "need pl_orbsmax, or pl_orbper + st_mass"; return out
        a = orbit_distance(g("pl_orbper"), g("st_mass")); out["notes"].append("a from Kepler's 3rd law")
    inner, outer, warn = habitable_zone(L, teff)
    oi, oo, _ = habitable_zone(L, teff, optimistic=True)
    if warn: out["notes"].append(warn)
    m, est = g("pl_bmasse"), False
    if m is None:
        m, est = mass_from_radius(r), True; out["notes"].append("mass is an empirical estimate (Chen & Kipping 2017)")
    if "msini" in str(p.get("pl_bmassprov", "")).lower():
        out["notes"].append("mass is a MINIMUM (Msini): true v_esc is higher")
    v = escape_velocity(m, r)
    out.update(luminosity_lsun=L, a_au=a, hz_inner_au=inner, hz_outer_au=outer,
               in_hz=bool(inner <= a <= outer), in_hz_optimistic=bool(oi <= a <= oo),
               radius_earth=r, mass_earth=m, mass_is_estimate=est, v_esc_kms=v,
               shoreline=cosmic_shoreline(L, a, v, K=K, xuv_rel=xuv_rel))
    return out

def write_derived(results, path="derived.csv"):
    import pandas as pd
    rows = []
    for r in results:
        sl = r.get("shoreline", {})
        rows.append({"pl_name": r["planet"], "a_au": r.get("a_au"), "hz_inner_au": r.get("hz_inner_au"),
                     "hz_outer_au": r.get("hz_outer_au"), "in_hz": r.get("in_hz"), "in_hz_optimistic": r.get("in_hz_optimistic"),
                     "v_esc_kms": r.get("v_esc_kms"), "mass_is_estimate": r.get("mass_is_estimate"),
                     "shoreline_verdict": sl.get("verdict"), "shoreline_log10_margin": sl.get("log10_margin"),
                     "notes": "; ".join(r["notes"] + ([r["error"]] if "error" in r else []))})
    pd.DataFrame(rows).to_csv(path, index=False); return path

# ------------------------------------------------------------------ demo / command line
OFFLINE = [  # rounded values for TESTING ONLY
    dict(pl_name="K2-18 b",      pl_rade=2.61,  pl_bmasse=8.63,  pl_orbper=32.94,   pl_orbsmax=0.143,  st_teff=3457, st_rad=0.445, st_mass=0.495, st_lum=-1.64),
    dict(pl_name="TRAPPIST-1 b", pl_rade=1.116, pl_bmasse=1.374, pl_orbper=1.51088, pl_orbsmax=0.0115, st_teff=2566, st_rad=0.119, st_mass=0.090, st_lum=-3.26),
    dict(pl_name="LHS 1140 b",   pl_rade=1.73,  pl_bmasse=5.6,   pl_orbper=24.74,   pl_orbsmax=0.0875, st_teff=3096, st_rad=0.216, st_mass=0.184, st_lum=-2.42)]

def report(results):
    for res in results:
        if "error" in res: print(f"\n{res['planet']}: SKIPPED - {res['error']}"); continue
        sl = res["shoreline"]
        print(f"\n{res['planet']}: a={res['a_au']:.4f} AU, HZ={res['hz_inner_au']:.3f}-{res['hz_outer_au']:.3f} AU, "
              f"in HZ: {res['in_hz']} (optimistic: {res['in_hz_optimistic']})")
        print(f"  Rp={res['radius_earth']:.2f} Re, Mp={res['mass_earth']:.2f} Me, v_esc={res['v_esc_kms']:.1f} km/s")
        print(f"  shoreline: {sl['verdict']} (log10 margin {sl['log10_margin']:+.2f}; XUV from {sl['xuv_source']})")
        for n in res["notes"]: print("  note:", n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planets"); ap.add_argument("--stars"); ap.add_argument("--out", default="derived.csv")
    ap.add_argument("--live", action="store_true"); ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest: self_test(); sys.exit()
    if a.planets: rows = load_tables(a.planets, a.stars).to_dict("records")
    elif a.live:  rows = fetch_params(["K2-18 b", "TRAPPIST-1 b", "LHS 1140 b"]).to_dict("records")
    else:         rows = OFFLINE; print("(no CSV given: using rounded sample numbers, testing only)")
    results = [analyze_planet(r) for r in rows]; report(results)
    print("\nsaved", write_derived(results, a.out))
