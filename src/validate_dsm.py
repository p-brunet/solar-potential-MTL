"""Quick visual validation of the DSM on a small neighbourhood window.

Two panels:
  Left  — hillshade (directional shading) + building footprint outlines
  Right — raw elevation colourmap + building footprint outlines

Usage:
    python src/validate_dsm.py                        # auto-detect valid area
    python src/validate_dsm.py 45.495 -73.570 800     # lat lon half-size-m
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import box

from config import CRS_PROJECT, DATA_RAW, DATA_OUTPUTS

DSM_PATH = DATA_RAW / "dsm_montreal.tif"
BUILDINGS_PATH = DATA_RAW / "buildings_mtl.geojson"
OUT_PNG = DATA_OUTPUTS / "dsm_validation.png"

DEFAULT_HALF_M = 600  # metres each side → 1.2 km × 1.2 km patch


def _find_valid_centre(dsm_path: Path) -> tuple[float, float]:
    """Return (lat, lon) of the median valid-data pixel, sampled at 1/100 resolution."""
    with rasterio.open(dsm_path) as src:
        decimated = src.read(1, out_shape=(src.height // 100, src.width // 100))
        nodata = src.nodata
        rows, cols = np.where(decimated != nodata)
        if not len(rows):
            raise ValueError("DSM has no valid pixels.")
        r = int(np.median(rows)) * 100
        c = int(np.median(cols)) * 100
        x, y = src.xy(r, c)
    tr = Transformer.from_crs(CRS_PROJECT, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return lat, lon


def _centre_window(lat: float, lon: float, half_m: float, crs: str):
    """Return (window, transform, bounds_32188) for a square centred on lat/lon."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = tr.transform(lon, lat)
    left, bottom = cx - half_m, cy - half_m
    right, top = cx + half_m, cy + half_m
    return (left, bottom, right, top)


def _hillshade(dem: np.ndarray, res: float = 1.0, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    az = np.radians(360 - azimuth)
    alt = np.radians(altitude)
    dz_dy, dz_dx = np.gradient(dem, res)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shade, 0, 1)


def main(lat: float | None = None, lon: float | None = None, half_m: float = DEFAULT_HALF_M) -> None:
    if not DSM_PATH.exists():
        print(f"[MISSING] {DSM_PATH} — run src/collect_dsm.py first.")
        return

    if lat is None or lon is None:
        lat, lon = _find_valid_centre(DSM_PATH)
        print(f"Auto-detected valid centre: lat={lat:.4f}, lon={lon:.4f}")

    bounds = _centre_window(lat, lon, half_m, CRS_PROJECT)
    left, bottom, right, top = bounds

    # --- Read DSM window ---
    with rasterio.open(DSM_PATH) as src:
        win = from_bounds(left, bottom, right, top, src.transform)
        dem = src.read(1, window=win)
        win_transform = src.window_transform(win)
        nodata = src.nodata
        res = src.res[0]

    dem = dem.astype(float)
    if nodata is not None:
        dem[dem == nodata] = np.nan

    # --- Load buildings clipped to window ---
    bbox_wgs = box(*_centre_window(lat, lon, half_m * 1.05, "EPSG:4326"))  # slight buffer for clip
    tr_back = Transformer.from_crs(CRS_PROJECT, "EPSG:4326", always_xy=True)
    x1, y1 = tr_back.transform(left, bottom)
    x2, y2 = tr_back.transform(right, top)
    buildings_raw = gpd.read_file(BUILDINGS_PATH, bbox=(x1, y1, x2, y2))
    buildings = buildings_raw.to_crs(CRS_PROJECT) if not buildings_raw.empty else None

    # --- Extent for imshow (metres) ---
    extent = [left, right, bottom, top]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        f"DSM validation  ({lat:.3f}°N, {lon:.3f}°W, {int(half_m*2)} m window)",
        fontsize=12, fontweight="bold",
    )

    # Left — hillshade
    ax = axes[0]
    shade = _hillshade(np.where(np.isnan(dem), 0, dem), res=res)
    ax.imshow(shade, cmap="gray", origin="upper", extent=extent, vmin=0, vmax=1)
    if buildings is not None:
        buildings.boundary.plot(ax=ax, color="tomato", linewidth=0.5, alpha=0.8)
    ax.set_title(f"Hillshade (az=315°, alt=45°)  +  {len(buildings_raw):,} footprints")
    ax.set_xlabel("Easting (m, EPSG:32188)")
    ax.set_ylabel("Northing (m)")
    ax.ticklabel_format(style="plain")

    # Right — raw elevation
    ax = axes[1]
    valid_dem = np.where(np.isnan(dem), np.nan, dem)
    im = ax.imshow(valid_dem, cmap="terrain", origin="upper", extent=extent)
    if buildings is not None:
        buildings.boundary.plot(ax=ax, color="red", linewidth=0.5, alpha=0.7)
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)
    ax.set_title("Elevation (m ASL)")
    ax.set_xlabel("Easting (m, EPSG:32188)")
    ax.ticklabel_format(style="plain")

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PNG}")

    valid = dem[~np.isnan(dem)]
    print(f"Window : {dem.shape[0]} × {dem.shape[1]} px  ({res:.0f} m res)")
    print(f"Elev   : min={valid.min():.1f} m  max={valid.max():.1f} m  mean={valid.mean():.1f} m")
    print(f"Bldgs  : {len(buildings_raw):,} footprints in window")

    plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        main(float(args[0]), float(args[1]), float(args[2]) if len(args) > 2 else DEFAULT_HALF_M)
    else:
        main(None, None)
