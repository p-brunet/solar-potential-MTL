"""Download and clip NRCan HRDEM DSM (1 m) for Montreal via STAC API.

Tiles are clipped individually (COG range requests — no full download)
and saved to data/raw/dsm_tiles/. A JSON manifest records each tile's
path, date, and bounds so roof_features.py can look up tiles on demand.

Outputs
-------
data/raw/dsm_tiles/<tile_id>.tif   — one clipped tile per STAC item (cached)
data/raw/dsm_manifest.json         — tile index [{path, date, bounds}] newest first
"""

import json
import os
from pathlib import Path

import numpy as np
import requests
import rasterio
from pyproj import Transformer
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box, mapping

from config import CRS_PROJECT, DATA_RAW, MONTREAL_BBOX

STAC_URL = "https://datacube.services.geo.ca/stac/api/search"
COLLECTION = "hrdem-lidar"
TILES_DIR = DATA_RAW / "dsm_tiles"
MANIFEST_OUT = DATA_RAW / "dsm_manifest.json"
NODATA = -32767.0

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")


def _query_stac(bbox: tuple) -> list[tuple[str, str, str]]:
    """Return (tile_id, date, dsm_url) sorted newest → oldest, >= 2018."""
    r = requests.get(
        STAC_URL,
        params={"collections": COLLECTION, "bbox": ",".join(str(x) for x in bbox), "limit": 100},
        timeout=30,
    )
    if r.status_code != 200:
        raise ValueError(f"STAC query failed: HTTP {r.status_code}")
    features = r.json().get("features", [])
    if not features:
        raise ValueError(f"No HRDEM tiles found for bbox {bbox}")
    features = [f for f in features if f["properties"]["datetime"][:4] >= "2018"]
    features.sort(key=lambda f: f["properties"]["datetime"], reverse=True)
    return [(f["id"], f["properties"]["datetime"][:10], f["assets"]["dsm"]["href"]) for f in features]


def _bbox_in_crs(bbox: tuple, src_crs) -> object:
    minlon, minlat, maxlon, maxlat = bbox
    if src_crs.to_epsg() == 4326:
        return box(minlon, minlat, maxlon, maxlat)
    tr = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
    x1, y1 = tr.transform(minlon, minlat)
    x2, y2 = tr.transform(maxlon, maxlat)
    return box(x1, y1, x2, y2)


def clip_and_reproject(dsm_url: str, bbox: tuple, out_path: Path) -> bool:
    """Clip remote COG to bbox, reproject to CRS_PROJECT, save. Returns False if no data."""
    with rasterio.open(dsm_url) as src:
        src_crs = src.crs
        src_nodata = src.nodata if src.nodata is not None else NODATA
        clip_geom = _bbox_in_crs(bbox, src_crs)
        try:
            clipped, clip_transform = mask(src, [mapping(clip_geom)], crop=True, nodata=src_nodata)
        except Exception:
            return False

    if np.all(clipped == src_nodata):
        return False

    h, w = clipped.shape[1], clipped.shape[2]
    dst_crs = rasterio.crs.CRS.from_string(CRS_PROJECT)
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, w, h, *array_bounds(h, w, clip_transform)
    )
    reprojected = np.full((1, dst_h, dst_w), NODATA, dtype=np.float32)
    reproject(
        source=clipped.astype(np.float32),
        destination=reprojected,
        src_transform=clip_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=src_nodata, dst_nodata=NODATA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, "w", driver="GTiff", crs=dst_crs, transform=dst_transform,
        width=dst_w, height=dst_h, count=1, dtype="float32",
        nodata=NODATA, compress="lzw", tiled=True, blockxsize=512, blockysize=512,
    ) as dst:
        dst.write(reprojected)
    valid = int(np.sum(reprojected != NODATA))
    print(f"    {valid:,} valid pixels")
    return True


def write_manifest(tile_entries: list[tuple[str, str, Path]], out_path: Path) -> None:
    """Write JSON manifest [{path, date, bounds}] for all valid tiles, newest first.

    Paths are stored relative to the project root so the manifest is portable.
    """
    root = Path(__file__).parent.parent
    records = []
    for tile_id, date, tile_path in tile_entries:
        with rasterio.open(tile_path) as src:
            b = src.bounds
        records.append({
            "tile_id": tile_id,
            "date": date,
            "path": str(Path(tile_path).relative_to(root)),
            "bounds": [b.left, b.bottom, b.right, b.top],
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2))
    print(f"  Manifest: {out_path}  ({len(records)} tiles)")


def main() -> None:
    print("Step 1/3 — Querying NRCan STAC for HRDEM DSM (>= 2018)…")
    tiles = _query_stac(MONTREAL_BBOX)
    print(f"  Found {len(tiles)} tiles")
    for tile_id, date, _ in tiles:
        print(f"    {date}  {tile_id}")

    print("\nStep 2/3 — Clipping and reprojecting each tile…")
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    tile_entries = []
    for i, (tile_id, date, url) in enumerate(tiles, 1):
        out = TILES_DIR / f"{tile_id}.tif"
        label = f"  [{i}/{len(tiles)}] {tile_id} ({date})"
        if out.exists():
            print(f"{label} — cached")
            tile_entries.append((tile_id, date, out))
        else:
            print(label)
            if clip_and_reproject(url, MONTREAL_BBOX, out):
                tile_entries.append((tile_id, date, out))
            else:
                print("    no data in bbox — skipped")

    if not tile_entries:
        raise ValueError("No tiles with valid data.")

    print(f"\nStep 3/3 — Writing tile manifest ({len(tile_entries)} tiles)…")
    write_manifest(tile_entries, MANIFEST_OUT)


if __name__ == "__main__":
    main()
