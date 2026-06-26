"""Borough-level aggregation of solar potential.

Spatial-joins buildings to borough polygons, then rolls up eligible
buildings into per-borough capacity and generation estimates.

Reads:
  data/processed/buildings_predicted.geojson
  data/raw/boroughs.geojson

Writes:
  data/processed/borough_aggregates.geojson

Run:
    uv run python src/aggregate.py
"""

import geopandas as gpd
import pandas as pd

from config import DATA_PROCESSED, DATA_RAW

PREDICTED_PATH = DATA_PROCESSED / "buildings_predicted.geojson"
BOROUGHS_PATH  = DATA_RAW / "boroughs.geojson"
OUT_PATH       = DATA_PROCESSED / "borough_aggregates.geojson"


def main() -> None:
    print("Loading data ...")
    bldg     = gpd.read_file(PREDICTED_PATH)
    boroughs = gpd.read_file(BOROUGHS_PATH).to_crs(bldg.crs)
    print(f"  {len(bldg):,} buildings, {len(boroughs)} borough polygons")

    print("Spatial join (centroid-within) ...")
    # Centroid-within avoids silent boundary losses; project to MTM zone 8 for metric centroids.
    bldg_cols = ["slope_deg", "yield_kwh_kwp", "capacity_kwp",
                 "annual_kwh", "area_m2", "eligible", "geometry"]
    bldg_pts = bldg[bldg_cols].copy()
    bldg_pts["geometry"] = bldg_pts.to_crs("EPSG:32188").geometry.centroid
    bldg_pts = bldg_pts.to_crs(bldg.crs)

    joined = gpd.sjoin(
        bldg_pts,
        boroughs[["NOM", "geometry"]],
        how="left",
        predicate="within",
    )
    n_unmatched = joined["NOM"].isna().sum()
    if n_unmatched:
        print(f"  {n_unmatched:,} buildings outside borough polygons (water, edge) -- dropped")
    joined = joined.dropna(subset=["NOM"])

    print("Aggregating ...")
    total_per = joined.groupby("NOM").size().rename("n_buildings")
    eligible  = joined[joined["eligible"]].copy()

    elig_agg = eligible.groupby("NOM").agg(
        n_eligible    = ("eligible",        "count"),
        capacity_mw   = ("capacity_kwp",    lambda x: x.sum() / 1000),
        energy_gwh_yr = ("annual_kwh",      lambda x: x.sum() / 1e6),
        avg_yield     = ("yield_kwh_kwp",   "mean"),
        avg_slope_deg = ("slope_deg",       "mean"),
        avg_area_m2   = ("area_m2",         "mean"),
    )

    result = (
        boroughs
        .merge(total_per.reset_index(), on="NOM", how="left")
        .merge(elig_agg.reset_index(),  on="NOM", how="left")
    )
    result["n_buildings"]  = result["n_buildings"].fillna(0).astype(int)
    result["n_eligible"]   = result["n_eligible"].fillna(0).astype(int)
    result["capacity_mw"]  = result["capacity_mw"].fillna(0.0)
    result["energy_gwh_yr"]= result["energy_gwh_yr"].fillna(0.0)
    result["pct_eligible"] = (
        100 * result["n_eligible"] / result["n_buildings"].replace(0, pd.NA)
    )

    result.to_file(OUT_PATH, driver="GeoJSON")
    print(f"\nSaved -> {OUT_PATH}")

    island_mw  = result["capacity_mw"].sum()
    island_gwh = result["energy_gwh_yr"].sum()
    print(f"\nIsland total:  {island_mw:,.0f} MW  |  {island_gwh:,.0f} GWh/yr")

    print("\nTop 10 boroughs by capacity:")
    top = (
        result.nlargest(10, "capacity_mw")
        [["NOM", "n_buildings", "n_eligible", "pct_eligible",
          "capacity_mw", "energy_gwh_yr", "avg_yield"]]
        .rename(columns={"energy_gwh_yr": "GWh/yr", "capacity_mw": "MW",
                         "avg_yield": "avg kWh/kWp"})
    )
    print(top.to_string(index=False, float_format="{:.1f}".format))


if __name__ == "__main__":
    main()
