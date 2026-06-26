"""Interactive Folium map of Montreal rooftop solar potential.

Three layers:
  1. Borough choropleth     — coloured by total eligible capacity (MW)
  2. Heatmap                — density of ALL eligible buildings on the island
  3. Detail sample          — 5 000 buildings colour-coded by payback (hidden by default)

The building dataset is first clipped to the island boundary (union of borough
polygons) to remove NRCan buildings outside Montreal that fall within the
collection bounding box (Laval, South Shore, etc.).

Reads:
  data/processed/buildings_economics.geojson
  data/processed/borough_aggregates.geojson

Writes:
  data/outputs/solar_map.html

Run:
    uv run python src/map.py
"""

import geopandas as gpd
import folium
import branca.colormap as cm
import numpy as np
from folium.plugins import HeatMap

from config import DATA_OUTPUTS, DATA_PROCESSED

ECONOMICS_PATH  = DATA_PROCESSED / "buildings_economics.geojson"
AGGREGATES_PATH = DATA_PROCESSED / "borough_aggregates.geojson"
OUT_PATH        = DATA_OUTPUTS / "solar_map.html"

MAX_DETAIL = 5_000   # CircleMarker sample for tooltip layer
MTL_CENTER = [45.508, -73.587]


# ---------------------------------------------------------------------------
# Colourmaps
# ---------------------------------------------------------------------------

def make_payback_cmap(vmin: float = 10, vmax: float = 26) -> cm.LinearColormap:
    return cm.LinearColormap(
        ["#2ecc71", "#f39c12", "#e74c3c"],
        vmin=vmin, vmax=vmax,
        caption="Payback period (yr)",
    )


def make_capacity_cmap(vmax: float) -> cm.LinearColormap:
    return cm.LinearColormap(
        ["#ffffd4", "#fd8d3c", "#800026"],
        vmin=0, vmax=vmax,
        caption="Eligible solar capacity (MW)",
    )


# ---------------------------------------------------------------------------
# Island filter
# ---------------------------------------------------------------------------

def clip_to_island(
    bldg: gpd.GeoDataFrame,
    boroughs: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Remove buildings outside the island (Laval, South Shore) captured by the NRCan bbox."""
    island = boroughs.to_crs("EPSG:32188").geometry.union_all()
    centroids = bldg.to_crs("EPSG:32188").geometry.centroid
    mask = centroids.within(island)
    n_dropped = int((~mask).sum())
    if n_dropped:
        print(f"  {n_dropped:,} buildings outside island dropped (NRCan bbox overflows to Laval/South Shore)")
    return bldg[mask].copy()


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

def add_borough_layer(m: folium.Map, boroughs: gpd.GeoDataFrame) -> None:
    cap_cmap = make_capacity_cmap(float(boroughs["capacity_mw"].max()))
    cap_cmap.add_to(m)

    tooltip_cols = ["NOM", "n_buildings", "n_eligible", "pct_eligible",
                    "capacity_mw", "energy_gwh_yr", "avg_yield"]
    clean = boroughs[
        [c for c in tooltip_cols if c in boroughs.columns] + ["geometry"]
    ].to_crs("EPSG:4326")

    folium.GeoJson(
        clean.__geo_interface__,
        name="Boroughs -- capacity (MW)",
        style_function=lambda feat: {
            "fillColor": cap_cmap(feat["properties"].get("capacity_mw") or 0),
            "color": "#333333",
            "weight": 1,
            "fillOpacity": 0.65,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["NOM", "n_buildings", "n_eligible", "pct_eligible",
                    "capacity_mw", "energy_gwh_yr", "avg_yield"],
            aliases=["Borough", "Total buildings", "Eligible", "% eligible",
                     "Capacity (MW)", "Generation (GWh/yr)", "Avg yield (kWh/kWp)"],
            localize=True,
        ),
    ).add_to(m)


def add_heatmap_layer(m: folium.Map, bldg: gpd.GeoDataFrame) -> None:
    """Density heatmap over all eligible buildings on the island."""
    eligible = bldg[bldg["eligible"]]
    centroids = eligible.to_crs("EPSG:32188").geometry.centroid.to_crs("EPSG:4326")
    heat_data = list(zip(centroids.y, centroids.x))
    HeatMap(
        heat_data,
        name=f"Density -- {len(eligible):,} eligible buildings",
        radius=10,
        blur=8,
        min_opacity=0.25,
        max_zoom=16,
    ).add_to(m)
    print(f"  Heatmap: {len(eligible):,} buildings")


def add_detail_layer(m: folium.Map, bldg: gpd.GeoDataFrame) -> None:
    """Sample of buildings with tooltip (hidden by default)."""
    eligible = bldg[bldg["eligible"]].copy()
    if len(eligible) > MAX_DETAIL:
        eligible = eligible.sample(MAX_DETAIL, random_state=42)

    eligible["geometry"] = eligible.to_crs("EPSG:32188").geometry.centroid
    eligible = eligible.to_crs("EPSG:4326")

    pb_cmap = make_payback_cmap()
    pb_cmap.add_to(m)

    layer = folium.FeatureGroup(
        name=f"Building detail (sample {len(eligible):,})",
        show=False,
    )

    for _, row in eligible.iterrows():
        lat, lon = row.geometry.y, row.geometry.x
        pb  = row.get("payback_yr", np.nan)
        npv = row.get("npv_cad",    np.nan)

        pb_val = float(pb) if (not np.isnan(pb) and not np.isinf(pb)) else None
        color  = pb_cmap(np.clip(pb_val, 10, 26)) if pb_val else "#888888"

        pb_str  = f"{pb_val:.1f} yr" if pb_val else "--"
        npv_str = f"{npv:,.0f} CAD"  if (npv and not np.isnan(npv)) else "--"

        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=0,
            tooltip=(
                f"<b>Payback:</b> {pb_str}<br>"
                f"<b>NPV:</b> {npv_str}<br>"
                f"<b>Yield:</b> {row['yield_kwh_kwp']:.0f} kWh/kWp<br>"
                f"<b>Capacity:</b> {row['capacity_kwp']:.1f} kWp<br>"
                f"<b>Area:</b> {row['area_m2']:.0f} m2"
            ),
        ).add_to(layer)

    layer.add_to(m)
    print(f"  Detail layer: {len(eligible):,} markers (hidden by default)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading data ...")
    bldg     = gpd.read_file(ECONOMICS_PATH)
    boroughs = gpd.read_file(AGGREGATES_PATH)
    print(f"  {len(bldg):,} buildings, {len(boroughs)} boroughs")

    print("Filtering to island boundary ...")
    bldg = clip_to_island(bldg, boroughs)
    print(f"  {len(bldg):,} buildings on island  |  "
          f"eligible: {bldg['eligible'].sum():,}")

    m = folium.Map(location=MTL_CENTER, zoom_start=11, tiles="CartoDB positron")

    print("Adding borough choropleth ...")
    add_borough_layer(m, boroughs)

    print("Adding heatmap (all eligible) ...")
    add_heatmap_layer(m, bldg)

    print(f"Adding detail layer (sample {MAX_DETAIL:,}) ...")
    add_detail_layer(m, bldg)

    folium.LayerControl(collapsed=False).add_to(m)

    title_html = """
    <div style="position:fixed; top:10px; left:60px; z-index:1000;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #ccc; font-family:sans-serif; font-size:13px;">
        <b>Residential rooftop solar potential -- Montreal Island</b><br>
        <span style="color:#666; font-size:11px;">
            Slope &le; 45 deg &middot; yield &ge; 800 kWh/kWp/yr &middot; area &le; 500 m2
        </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    m.save(str(OUT_PATH))
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
