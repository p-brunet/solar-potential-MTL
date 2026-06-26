"""Compute per-building roof features from NRCan HRDEM DSM tiles.

For each building footprint the script finds the most-recent DSM tile that
covers its centroid, reads a small window around the footprint, and derives:
  slope_deg   — median roof slope (degrees)
  azimuth_deg — dominant roof aspect (compass degrees, 0 = North)
  svf         — sky-view fraction proxy [0, 1]
  area_m2     — footprint area in metric CRS

Buildings whose centroids are not covered by any tile are dropped (no data
for ~19% of buildings, concentrated in the SE of the island).

Run from project root:
    python src/roof_features.py [--sample N]
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.windows import from_bounds

from config import CRS_PROJECT, DATA_PROCESSED, DATA_RAW

MANIFEST_PATH = DATA_RAW / "dsm_manifest.json"
OUT_PATH = DATA_PROCESSED / "buildings_features.geojson"
NODATA = -32767.0
SVF_RADIUS_M = 50  # horizon scan radius (metres)


# ---------------------------------------------------------------------------
# Low-level DSM feature functions
# ---------------------------------------------------------------------------

def compute_slope(patch: np.ndarray, res: float) -> float:
    """Return median slope in degrees over valid pixels of patch."""
    if patch.ndim < 2 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return np.nan
    if np.sum(~np.isnan(patch)) < 4:
        return np.nan
    dz_dy, dz_dx = np.gradient(patch, res)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    valid = slope[~np.isnan(slope)]
    return float(np.median(valid)) if valid.size > 0 else np.nan


def compute_azimuth(patch: np.ndarray, res: float) -> float:
    """Return circular-mean aspect in compass degrees (0=N, 90=E) over valid pixels."""
    if patch.ndim < 2 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return np.nan
    if np.sum(~np.isnan(patch)) < 4:
        return np.nan
    dz_dy, dz_dx = np.gradient(patch, res)
    aspect_rad = np.arctan2(-dz_dx, dz_dy)
    aspect_deg = np.degrees(aspect_rad) % 360
    valid_mask = ~np.isnan(patch) & ~np.isnan(aspect_deg)
    if not valid_mask.any():
        return np.nan
    rad = np.radians(aspect_deg[valid_mask])
    return float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)


def estimate_svf(patch: np.ndarray, res: float, n_dir: int = 8) -> float:
    """Estimate sky-view fraction via vectorized horizon-angle scan in n_dir directions.

    For each direction, extracts the ray as a numpy array and finds the maximum
    horizon elevation angle. SVF = 1 - mean(sin²(max_horizon)).
    """
    h, w = patch.shape
    cy, cx = h // 2, w // 2
    if np.isnan(patch[cy, cx]):
        return np.nan
    centre_z = patch[cy, cx]
    max_radius = min(cy, cx, h - cy - 1, w - cx - 1)
    if max_radius < 1:
        return 1.0

    rs = np.arange(1, max_radius + 1, dtype=float)
    dists = rs * res
    horizon_sin2 = []

    for i in range(n_dir):
        az = 2 * np.pi * i / n_dir
        row_idx = np.clip((cy - np.cos(az) * rs).astype(int), 0, h - 1)
        col_idx = np.clip((cx + np.sin(az) * rs).astype(int), 0, w - 1)
        z_ray = patch[row_idx, col_idx]
        dz = z_ray - centre_z
        angles = np.where(np.isnan(z_ray), -np.inf, np.arctan2(dz, dists))
        max_angle = float(angles.max())
        horizon_sin2.append(np.sin(max(max_angle, 0.0)) ** 2)

    return float(np.clip(1.0 - np.mean(horizon_sin2), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Tile lookup
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> list[dict]:
    """Load manifest and attach a coarse validity grid to each tile record.

    All clipped tiles share the same bounding box, so a bounds check alone
    cannot distinguish coverage. A decimated mask read once per tile fixes this.
    Paths in the manifest are relative to the project root and resolved here.
    """
    root = manifest_path.parent.parent
    records = json.loads(manifest_path.read_text())
    for rec in records:
        rec["path"] = str(root / rec["path"])
    scale = 50
    for rec in records:
        with rasterio.open(rec["path"]) as src:
            lh = max(1, src.height // scale)
            lw = max(1, src.width // scale)
            small = src.read(1, out_shape=(lh, lw),
                             resampling=rasterio.enums.Resampling.nearest)
            nd = src.nodata if src.nodata is not None else NODATA
            rec["valid_grid"] = (small != nd)
            left, bottom, right, top = rec["bounds"]
            rec["grid_res_x"] = (right - left) / lw
            rec["grid_res_y"] = (top - bottom) / lh
            rec["grid_lw"] = lw
            rec["grid_lh"] = lh
    return records


def _assign_tiles(centroids_x: np.ndarray, centroids_y: np.ndarray,
                  manifest: list[dict]) -> np.ndarray:
    """Return tile index (newest-first) for each centroid, -1 if none."""
    indices = np.full(len(centroids_x), -1, dtype=int)
    for ti, rec in enumerate(manifest):
        left, bottom, right, top = rec["bounds"]
        grid = rec["valid_grid"]
        rx, ry = rec["grid_res_x"], rec["grid_res_y"]
        lw, lh = rec["grid_lw"], rec["grid_lh"]

        cols = ((centroids_x - left) / rx).astype(int)
        rows = ((top - centroids_y) / ry).astype(int)
        in_bounds = (cols >= 0) & (cols < lw) & (rows >= 0) & (rows < lh)
        valid_in_tile = np.zeros(len(centroids_x), dtype=bool)
        valid_in_tile[in_bounds] = grid[rows[in_bounds], cols[in_bounds]]

        indices[valid_in_tile & (indices == -1)] = ti
    return indices


def _clamp_window(win, src_width: int, src_height: int):
    """Clamp a rasterio Window to raster extent. Returns None if no overlap."""
    col_start = max(0, int(win.col_off))
    row_start = max(0, int(win.row_off))
    col_stop = min(src_width, int(win.col_off + win.width))
    row_stop = min(src_height, int(win.row_off + win.height))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return rasterio.windows.Window(col_start, row_start,
                                   col_stop - col_start,
                                   row_stop - row_start)


# ---------------------------------------------------------------------------
# Main extraction driver
# ---------------------------------------------------------------------------

def extract_roof_features(buildings: gpd.GeoDataFrame,
                          manifest_path: Path) -> gpd.GeoDataFrame:
    """Add slope_deg, azimuth_deg, svf, area_m2 to buildings from DSM tiles."""
    manifest = load_manifest(manifest_path)

    proj = buildings.to_crs(CRS_PROJECT)
    cx = proj.geometry.centroid.x.values
    cy = proj.geometry.centroid.y.values
    tile_idx = _assign_tiles(cx, cy, manifest)

    n_no_tile = int((tile_idx == -1).sum())
    if n_no_tile:
        print(f"  Dropping {n_no_tile:,} buildings with no DSM coverage "
              f"(~{100*n_no_tile/len(buildings):.1f}%)")

    covered_mask = tile_idx != -1
    proj = proj[covered_mask].copy().reset_index(drop=True)
    tile_idx = tile_idx[covered_mask]

    slope_arr = np.full(len(proj), np.nan)
    azimuth_arr = np.full(len(proj), np.nan)
    svf_arr = np.full(len(proj), np.nan)
    area_arr = proj.geometry.area.values

    shard_dir = DATA_PROCESSED / "dsm_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    unique_tiles = np.unique(tile_idx)
    for ti in unique_tiles:
        shard_path = shard_dir / f"tile_{ti}.npz"
        rec = manifest[ti]
        group_mask = tile_idx == ti
        group_idx = np.where(group_mask)[0]
        n_group = group_mask.sum()

        # Resume from shard if available and size matches current run
        if shard_path.exists():
            with np.load(shard_path) as shard:
                shard_size = shard["slope"].shape[0]
                if shard_size == n_group:
                    slope_arr[group_idx] = shard["slope"]
                    azimuth_arr[group_idx] = shard["azimuth"]
                    svf_arr[group_idx] = shard["svf"]
            # file is closed here — safe to delete on Windows
            if shard_size != n_group:
                print(f"  Tile {ti}: shard size mismatch "
                      f"({shard_size} vs {n_group}), recomputing")
                shard_path.unlink()
            else:
                done = int(np.sum(~np.isnan(slope_arr[group_idx])))
                print(f"  Tile {ti} ({rec['date']}) — {n_group:,} buildings … "
                      f"{done:,} loaded from shard")
                continue

        tile_path = Path(rec["path"])
        print(f"  Tile {ti} ({rec['date']}) — {n_group:,} buildings …", flush=True)

        with rasterio.open(tile_path) as src:
            res = src.res[0]
            svf_px = max(1, int(SVF_RADIUS_M / res))
            buf = svf_px * res
            nd = src.nodata if src.nodata is not None else NODATA

            for k, gi in enumerate(group_idx):
                geom = proj.geometry.iloc[gi]
                minx, miny, maxx, maxy = geom.bounds

                # Buffered window for SVF horizon scan
                raw_win = from_bounds(minx - buf, miny - buf,
                                      maxx + buf, maxy + buf, src.transform)
                raw_win = _clamp_window(raw_win, src.width, src.height)
                if raw_win is None:
                    continue

                raw = src.read(1, window=raw_win).astype(float)
                raw[raw == nd] = np.nan
                if np.sum(~np.isnan(raw)) < 10:
                    continue

                # Inner footprint window for slope/azimuth
                inner_win = from_bounds(minx, miny, maxx, maxy, src.transform)
                inner_win = _clamp_window(inner_win, src.width, src.height)
                inner = (src.read(1, window=inner_win).astype(float)
                         if inner_win is not None else raw)
                if inner_win is not None:
                    inner[inner == nd] = np.nan

                slope_arr[gi] = compute_slope(inner, res)
                azimuth_arr[gi] = compute_azimuth(inner, res)
                svf_arr[gi] = estimate_svf(raw, res)

                if (k + 1) % 5000 == 0:
                    done = int(np.sum(~np.isnan(slope_arr[group_idx[:k+1]])))
                    print(f"    {k+1:,}/{n_group:,}  computed={done:,}", flush=True)

        # Save shard for crash recovery
        np.savez(shard_path,
                 slope=slope_arr[group_idx],
                 azimuth=azimuth_arr[group_idx],
                 svf=svf_arr[group_idx])
        done = int(np.sum(~np.isnan(slope_arr[group_idx])))
        print(f"    done — {done:,}/{n_group:,} computed, shard saved")

    proj = proj.copy()
    proj["slope_deg"] = slope_arr
    proj["azimuth_deg"] = azimuth_arr
    proj["svf"] = svf_arr
    proj["area_m2"] = area_arr

    before = len(proj)
    proj = proj.dropna(subset=["slope_deg"])
    dropped = before - len(proj)
    if dropped:
        print(f"  Dropped {dropped:,} buildings with no valid DSM pixels in footprint")

    return proj.to_crs(buildings.crs)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(sample: int | None = None) -> None:
    buildings_path = DATA_RAW / "buildings_mtl.geojson"
    if not buildings_path.exists():
        print(f"[MISSING] {buildings_path} -- run src/collect_polygones.py first.")
        return
    if not MANIFEST_PATH.exists():
        print(f"[MISSING] {MANIFEST_PATH} -- run src/collect_dsm.py first.")
        return

    gdf = gpd.read_file(buildings_path)
    if sample:
        gdf = gdf.sample(min(sample, len(gdf)), random_state=42)
        print(f"Loaded {len(gdf):,} buildings (sample of {sample})")
    else:
        print(f"Loaded {len(gdf):,} buildings")

    print("Extracting roof features from DSM...")
    result = extract_roof_features(gdf, MANIFEST_PATH)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(OUT_PATH, driver="GeoJSON")
    print(f"\nSaved -> {OUT_PATH}  ({len(result):,} buildings)")

    print(f"\nSummary ({len(result):,} buildings):")
    for col, label in [("slope_deg", "slope_deg  "),
                        ("azimuth_deg", "azimuth_deg"),
                        ("svf", "svf        "),
                        ("area_m2", "area_m2    ")]:
        s = result[col]
        print(f"  {label}: median={s.median():.1f}  "
              f"p5={s.quantile(0.05):.1f}  p95={s.quantile(0.95):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()
    main(sample=args.sample)
