# -*- coding: utf-8 -*-
"""Build derived tables + verified headline statistics for the geometry-over-alignment analysis.
Outputs land in data_derived/ ; a machine-readable summary goes to data_derived/verified_stats.json.
All numbers come straight from the task CSVs -- no hand-entered values."""
import os, json
import numpy as np
import pandas as pd

REPO = r"d:/onedrive/01_科研与论文/08_风向建设/wind-direction-to-electricity-transition"
OUT  = r"d:/onedrive/01_科研与论文/08_风向建设/分析材料_几何主导杠杆/data_derived"
os.makedirs(OUT, exist_ok=True)

T0   = os.path.join(REPO, "offshore-task0-HuTingxian/output/task0")
farms = pd.read_csv(os.path.join(T0, "farms_master.csv"))
para  = pd.read_csv(os.path.join(REPO, "task1_output/task1_paradigm_classification.csv"))
bal   = pd.read_csv(os.path.join(REPO, "Task3-output/s4_balanced_gauss.csv"))
dev   = pd.read_csv(os.path.join(REPO, "Task3-output/s4_farm_deviation.csv"))
ann   = pd.read_csv(os.path.join(REPO, "offshore-task2/output/task2_annual_floris.csv"))

stats = {}

# ---------- 1. Wake-loss distribution (Task 2, Gauss, farm-year) ----------
g = ann[ann.wake_model == "gauss"].copy()
wl = g.WakeLoss.dropna() * 100  # to %
stats["wakeloss_gauss"] = {
    "n_farmyear": int(len(wl)),
    "mean_pct": round(float(wl.mean()), 2),
    "median_pct": round(float(wl.median()), 2),
    "p90_pct": round(float(wl.quantile(.90)), 2),
    "max_pct": round(float(wl.max()), 2),
    "share_over_25pct": round(float((wl > 25).mean()) * 100, 1),
    "share_over_20pct": round(float((wl > 20).mean()) * 100, 1),
}
# farm-level mean wake loss (average over years) for the map
farm_wl = g.groupby("farm_id").agg(
    wl_mean=("WakeLoss", "mean"), cf_mean=("CF", "mean"),
    n_turb=("n_turb", "first"), country=("country", "first"), region=("region", "first")
).reset_index()
farm_wl["wl_mean_pct"] = farm_wl.wl_mean * 100
farm_wl["cf_mean_pct"] = farm_wl.cf_mean * 100
farm_wl = farm_wl.merge(farms[["farm_id", "centroid_lon", "centroid_lat", "area_km2"]], on="farm_id", how="left")
farm_wl.to_csv(os.path.join(OUT, "farm_wakeloss_map.csv"), index=False)

# regional means
reg = g.groupby("region").agg(cf=("CF", "mean"), wl=("WakeLoss", "mean"), n=("CF", "size")).reset_index()
reg["cf"] *= 100; reg["wl"] *= 100
stats["regional"] = {r.region: {"cf_pct": round(r.cf, 1), "wl_pct": round(r.wl, 1), "n": int(r.n)} for r in reg.itertuples()}

# ---------- 2. Spacing distribution + spacing vs wake loss ----------
sp = para[["farm_id", "n_turb", "country", "layout_type", "spacing_d"]].dropna(subset=["spacing_d"]).copy()
stats["spacing_d"] = {
    "n": int(len(sp)),
    "median_D": round(float(sp.spacing_d.median()), 2),
    "mean_D": round(float(sp.spacing_d.mean()), 2),
    "p25_D": round(float(sp.spacing_d.quantile(.25)), 2),
    "p75_D": round(float(sp.spacing_d.quantile(.75)), 2),
    "min_D": round(float(sp.spacing_d.min()), 2),
    "max_D": round(float(sp.spacing_d.max()), 2),
    "share_under_5D": round(float((sp.spacing_d < 5).mean()) * 100, 1),
    "share_under_7D": round(float((sp.spacing_d < 7).mean()) * 100, 1),
}
sp_wl = sp.merge(farm_wl[["farm_id", "wl_mean_pct", "cf_mean_pct"]], on="farm_id", how="inner")
sp_wl.to_csv(os.path.join(OUT, "spacing_vs_wakeloss.csv"), index=False)
# simple correlation (spacing vs wake loss)
if len(sp_wl) > 5:
    from scipy.stats import spearmanr, pearsonr
    rho, pv = spearmanr(sp_wl.spacing_d, sp_wl.wl_mean_pct)
    stats["spacing_vs_wakeloss"] = {"n": int(len(sp_wl)), "spearman_rho": round(float(rho), 3), "p": float(pv)}

# ---------- 3. Proximity / cross-farm neighbours ----------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1); dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlmb/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

f = farms.copy()
f["r_km"] = np.sqrt(f.area_km2 / np.pi)   # equivalent-circle radius (footprint proxy)
pairs = []
arr = f.reset_index(drop=True)
for i in range(len(arr)):
    for j in range(i+1, len(arr)):
        a, b = arr.iloc[i], arr.iloc[j]
        d = haversine(a.centroid_lat, a.centroid_lon, b.centroid_lat, b.centroid_lon)
        edge = d - a.r_km - b.r_km
        pairs.append((int(a.farm_id), int(b.farm_id), a.country, b.country, round(d, 2), round(edge, 2)))
pp = pd.DataFrame(pairs, columns=["farm_i", "farm_j", "country_i", "country_j", "center_km", "edge_gap_km"])
pp.to_csv(os.path.join(OUT, "farm_pairs_all.csv"), index=False)
near = pp[pp.edge_gap_km < 10].sort_values("edge_gap_km")
near.to_csv(os.path.join(OUT, "farm_pairs_neighbours.csv"), index=False)
farms_with_neighbour = set(near.farm_i) | set(near.farm_j)
stats["proximity"] = {
    "n_farms": int(len(f)),
    "pairs_edge_under_10km": int((pp.edge_gap_km < 10).sum()),
    "pairs_center_under_20km": int((pp.center_km < 20).sum()),
    "pairs_center_under_30km": int((pp.center_km < 30).sum()),
    "farms_with_neighbour_edge_under_10km": int(len(farms_with_neighbour)),
    "share_farms_with_neighbour_pct": round(len(farms_with_neighbour) / len(f) * 100, 1),
}

# ---------- 4. Orientation optimisation (Task 3 balanced panel) ----------
b = bal.copy()
piv = b.pivot_table(index=["farm_id", "year"], columns="layout_group", values=["AEP_kWh", "WakeLoss"])
real_aep = piv[("AEP_kWh", "real")]; opt_aep = piv[("AEP_kWh", "s1_opt")]
rel = ((opt_aep - real_aep) / real_aep * 100).dropna()
stats["orientation_opt"] = {
    "n_farmyear": int(len(rel)),
    "mean_gain_pct": round(float(rel.mean()), 2),
    "median_gain_pct": round(float(rel.median()), 2),
    "p25_pct": round(float(rel.quantile(.25)), 2),
    "p75_pct": round(float(rel.quantile(.75)), 2),
    "win_rate_pct": round(float((rel > 0).mean()) * 100, 1),
}
rel.reset_index().rename(columns={0: "gain_pct"}).to_csv(os.path.join(OUT, "orientation_gain.csv"), index=False)
# deviation
stats["deviation"] = {
    "n": int(len(dev)),
    "mean_abs_deg": round(float(dev.axis_deviation.abs().mean()), 1),
    "median_abs_deg": round(float(dev.axis_deviation.abs().median()), 1),
    "share_over_30deg": round(float((dev.axis_deviation.abs() > 30).mean()) * 100, 1),
    "share_over_60deg": round(float((dev.axis_deviation.abs() > 60).mean()) * 100, 1),
}

with open(os.path.join(OUT, "verified_stats.json"), "w", encoding="utf-8") as fh:
    json.dump(stats, fh, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
