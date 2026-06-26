"""Stratified pvlib yield simulation on SAMPLE_SIZE buildings.

Stratifies the feature space (slope x azimuth sector x SVF band) so the
surrogate model sees good coverage rather than just common cases.

Reads:   data/processed/buildings_features.geojson
Writes:  data/processed/pvlib_sample.csv   (slope, azimuth, svf, area, yield)
         data/processed/tmy_montreal.csv   (cached TMY, reused by later scripts)

Run:
    uv run python src/pvlib_sample.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pvlib

from config import DATA_PROCESSED, PVLIB_LAT, PVLIB_LON, SAMPLE_SIZE, SYSTEM_LOSSES

FEATURES_PATH = DATA_PROCESSED / "buildings_features.geojson"
OUT_PATH      = DATA_PROCESSED / "pvlib_sample.csv"
TMY_CACHE     = DATA_PROCESSED / "tmy_montreal.csv"


# ---------------------------------------------------------------------------
# TMY
# ---------------------------------------------------------------------------

def fetch_tmy() -> pd.DataFrame:
    if TMY_CACHE.exists():
        tmy = pd.read_csv(TMY_CACHE, index_col=0, parse_dates=True)
        if tmy.index.tz is None:
            tmy.index = tmy.index.tz_localize("UTC")
        print(f"  TMY loaded from cache ({len(tmy)} hours)")
        return tmy
    print("  Fetching TMY from PVGIS …", end="", flush=True)
    tmy, _ = pvlib.iotools.get_pvgis_tmy(PVLIB_LAT, PVLIB_LON, map_variables=True)
    if tmy.index.tz is None:
        tmy.index = tmy.index.tz_localize("UTC")
    else:
        tmy.index = tmy.index.tz_convert("UTC")
    tmy.to_csv(TMY_CACHE)
    print(f" {len(tmy)} hours saved to cache")
    return tmy


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(gdf: gpd.GeoDataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Sample n buildings, stratified by slope × azimuth sector × SVF band."""
    df = pd.DataFrame(gdf[["slope_deg", "azimuth_deg", "svf", "area_m2"]])
    df["_slope_bin"] = pd.cut(
        df["slope_deg"], bins=[0, 15, 30, 45, 90], labels=False, include_lowest=True
    )
    df["_az_bin"] = pd.cut(
        df["azimuth_deg"] % 360,
        bins=np.linspace(0, 360, 9), labels=False, include_lowest=True
    )
    df["_svf_bin"] = pd.qcut(df["svf"], q=3, labels=False, duplicates="drop")
    df["_stratum"] = (
        df["_slope_bin"].astype(str) + "_"
        + df["_az_bin"].astype(str) + "_"
        + df["_svf_bin"].astype(str)
    )

    strata = df["_stratum"].dropna().unique()
    n_per = max(1, n // len(strata))

    parts = []
    for i, s in enumerate(strata):
        pool = df[df["_stratum"] == s]
        parts.append(pool.sample(min(n_per, len(pool)), random_state=seed + i))

    result = pd.concat(parts)
    if len(result) > n:
        result = result.sample(n, random_state=seed)

    drop_cols = [c for c in result.columns if c.startswith("_")]
    return result.drop(columns=drop_cols)


# ---------------------------------------------------------------------------
# pvlib simulation
# ---------------------------------------------------------------------------

def simulate_all(
    sample: pd.DataFrame,
    tmy: pd.DataFrame,
) -> np.ndarray:
    """Return annual yield (kWh/kWp) for each row in sample."""
    location  = pvlib.location.Location(PVLIB_LAT, PVLIB_LON, tz="UTC", altitude=30)
    solar_pos = location.get_solarposition(tmy.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy.index)
    airmass   = location.get_airmass(solar_position=solar_pos)

    yields = np.empty(len(sample))
    for i, (_, row) in enumerate(sample.iterrows()):
        poa = pvlib.irradiance.get_total_irradiance(
            surface_tilt=float(row["slope_deg"]),
            surface_azimuth=float(row["azimuth_deg"]),
            solar_zenith=solar_pos["apparent_zenith"],
            solar_azimuth=solar_pos["azimuth"],
            dni=tmy["dni"],
            ghi=tmy["ghi"],
            dhi=tmy["dhi"],
            dni_extra=dni_extra,
            airmass=airmass["airmass_relative"],
            model="perez",
        )
        gen = poa["poa_global"].clip(lower=0).fillna(0)
        yields[i] = gen.sum() / 1000 * (1 - SYSTEM_LOSSES)

        if (i + 1) % 300 == 0:
            print(f"  {i + 1:,}/{len(sample):,}  "
                  f"avg yield so far: {yields[:i+1].mean():.0f} kWh/kWp", flush=True)

    return yields


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading building features …")
    gdf = gpd.read_file(FEATURES_PATH)
    print(f"  {len(gdf):,} buildings")

    print(f"Stratified sampling (target {SAMPLE_SIZE:,}) …")
    sample = stratified_sample(gdf, SAMPLE_SIZE)
    print(f"  {len(sample):,} buildings across {sample.shape[0]} strata entries")

    print("Fetching TMY …")
    tmy = fetch_tmy()

    print(f"Running pvlib simulations ({len(sample):,} buildings) …")
    sample = sample.copy()
    sample["yield_kwh_kwp"] = simulate_all(sample, tmy)
    sample.to_csv(OUT_PATH)
    print(f"\nSaved → {OUT_PATH}")

    y = sample["yield_kwh_kwp"]
    print(f"\nYield (kWh/kWp/yr):  median={y.median():.0f}  "
          f"p5={y.quantile(0.05):.0f}  p95={y.quantile(0.95):.0f}")
    eligible = (sample["slope_deg"] <= 45) & (y >= 800)
    print(f"Eligible in sample:  {eligible.sum():,}/{len(sample):,} "
          f"({100 * eligible.mean():.1f}%)")


if __name__ == "__main__":
    main()
