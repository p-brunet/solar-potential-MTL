"""Validate raw collected data and produce a map for visual inspection."""

from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from shapely.geometry import box

from config import DATA_RAW, DATA_OUTPUTS, MONTREAL_BBOX, CRS_PROJECT

BUILDINGS_PATH = DATA_RAW / "buildings_mtl.geojson"
BOROUGHS_PATH = DATA_RAW / "boroughs.geojson"
OUT_PNG = DATA_OUTPUTS / "validation_raw.png"

EXPECTED_MIN_BUILDINGS = 50_000
EXPECTED_BOROUGHS = 34


def _check(label: str, condition: bool, detail: str = "") -> bool:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return condition


def validate() -> bool:
    ok = True

    for path in (BUILDINGS_PATH, BOROUGHS_PATH):
        if not path.exists():
            print(f"[MISSING] {path} — run src/collect.py first.")
            return False

    buildings = gpd.read_file(BUILDINGS_PATH)
    boroughs = gpd.read_file(BOROUGHS_PATH)

    print("\n=== Buildings ===")
    ok &= _check(
        f"count > {EXPECTED_MIN_BUILDINGS:,}",
        len(buildings) > EXPECTED_MIN_BUILDINGS,
        f"got {len(buildings):,}",
    )
    bbox = buildings.to_crs("EPSG:4326").total_bounds  # minx, miny, maxx, maxy
    bbox_geom = box(*MONTREAL_BBOX)
    ok &= _check(
        "bbox within MONTREAL_BBOX",
        box(*bbox).within(bbox_geom.buffer(0.01)),
        f"bbox={bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}",
    )
    ok &= _check("CRS set", buildings.crs is not None, str(buildings.crs))
    print(f"  columns : {list(buildings.columns)}")

    print("\n=== Boroughs ===")
    ok &= _check(
        f"count == {EXPECTED_BOROUGHS}",
        len(boroughs) == EXPECTED_BOROUGHS,
        f"got {len(boroughs)}",
    )
    ok &= _check("CRS set", boroughs.crs is not None, str(boroughs.crs))
    print(f"  columns : {list(boroughs.columns)}")

    return ok


def plot(out_path: Path) -> None:
    buildings = gpd.read_file(BUILDINGS_PATH).to_crs("EPSG:4326")
    boroughs = gpd.read_file(BOROUGHS_PATH).to_crs("EPSG:4326")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Session 01 — Raw data validation", fontsize=13, fontweight="bold")

    # Left: borough outlines + building centroids (fast for large counts)
    ax = axes[0]
    boroughs.boundary.plot(ax=ax, color="steelblue", linewidth=1.2)
    buildings.centroid.plot(ax=ax, color="tomato", markersize=0.3, alpha=0.4)
    minlon, minlat, maxlon, maxlat = MONTREAL_BBOX
    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_title(f"Buildings ({len(buildings):,}) + Boroughs ({len(boroughs)})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(
        handles=[
            mpatches.Patch(color="steelblue", label="Borough boundaries"),
            mpatches.Patch(color="tomato", label="Building centroids"),
        ],
        fontsize=8,
    )

    # Right: borough fill coloured by area
    ax = axes[1]
    boroughs_proj = boroughs.to_crs(CRS_PROJECT)
    boroughs_proj["area_km2"] = boroughs_proj.area / 1e6
    boroughs_proj.to_crs("EPSG:4326").plot(
        ax=ax,
        column="area_km2",
        cmap="YlOrRd",
        legend=True,
        legend_kwds={"label": "Area (km²)", "shrink": 0.6},
        edgecolor="white",
        linewidth=0.5,
    )
    col_name = next(
        (c for c in boroughs_proj.columns if "nom" in c.lower() or "name" in c.lower()),
        None,
    )
    if col_name:
        boroughs_wgs = boroughs_proj.to_crs("EPSG:4326")
        for _, row in boroughs_wgs.iterrows():
            cx, cy = row.geometry.centroid.x, row.geometry.centroid.y
            ax.annotate(row[col_name], (cx, cy), fontsize=5, ha="center", color="black")
    ax.set_title("Boroughs — area (km²)")
    ax.set_xlabel("Longitude")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Map saved → {out_path}")
    plt.show()


def main() -> None:
    print("Validating raw data…")
    ok = validate()
    print(f"\nOverall: {'PASS' if ok else 'FAIL'}")
    if ok:
        print("\nGenerating map…")
        plot(OUT_PNG)


if __name__ == "__main__":
    main()
