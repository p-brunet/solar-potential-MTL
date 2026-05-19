"""Download NRCan building footprints and Montreal borough polygons to data/raw/."""

import tempfile
from pathlib import Path

import fsspec
import geopandas as gpd
import requests
from shapely.geometry import box

from config import DATA_RAW, CRS_PROJECT, MONTREAL_BBOX


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params, timeout=60)
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code} from {r.url}")
    return r.json()


def _stream(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise ValueError(f"HTTP {r.status_code} downloading {url}")
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)


def _ckan_url(api: str, pkg: str, fmt_hints: list[str]) -> str:
    data = _get(f"{api}/action/package_show", {"id": pkg})
    for hint in fmt_hints:
        for res in data["result"]["resources"]:
            url = res.get("url", "")
            last_segment = url.rstrip("/").rsplit("/", 1)[-1]
            if "." not in last_segment:  # skip directory listings
                continue
            if hint.lower() in res.get("format", "").lower() or hint.lower() in res.get("name", "").lower():
                return url
    raise ValueError(f"No {fmt_hints} resource in {pkg}")


def download_buildings(out_path: Path) -> gpd.GeoDataFrame:
    print("  Querying NRCan CKAN for building footprints…")
    url = _ckan_url(
        "https://open.canada.ca/data/api",
        "7a5cda52-c7df-427f-9ced-26f19a8a64d6",
        ["geoparquet", "gpkg"],
    )

    print(f"  Reading {url[:80]}…")
    if url.endswith(".parquet"):
        with fsspec.open(url, "rb") as f:
            gdf = gpd.read_parquet(f, bbox=MONTREAL_BBOX).to_crs("EPSG:4326")
    else:
        with tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            _stream(url, tmp_path)
            gdf = gpd.read_file(tmp_path).to_crs("EPSG:4326")
        finally:
            tmp_path.unlink(missing_ok=True)
        gdf = gdf[gdf.intersects(box(*MONTREAL_BBOX))].copy()

    if len(gdf) == 0:
        raise ValueError(f"No buildings after clipping to Montreal bbox. Source: {url}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf


def download_boroughs(out_path: Path) -> gpd.GeoDataFrame:
    url = (
        "https://donnees.montreal.ca/fr/dataset/9797a946-9da8-41ec-8815-f6b276dec7e9"
        "/resource/e18bfd07-edc8-4ce8-8a5a-3b617662a794"
        "/download/limites-administratives-agglomeration.geojson"
    )

    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _stream(url, tmp_path)
        gdf = gpd.read_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if len(gdf) == 0:
        raise ValueError(f"Borough polygons empty. Source: {url}")

    gdf = gdf.to_crs(CRS_PROJECT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    return gdf


def main() -> None:
    buildings_path = DATA_RAW / "buildings_mtl.geojson"
    boroughs_path = DATA_RAW / "boroughs.geojson"

    print("Step 1/4 — Discovering NRCan building footprints…")
    buildings = download_buildings(buildings_path)
    print(f"Step 2/4 — {len(buildings):,} buildings saved -> {buildings_path}")

    print("\nStep 3/4 — Downloading Montreal borough polygons…")
    boroughs = download_boroughs(boroughs_path)
    print(f"Step 4/4 — {len(boroughs):,} boroughs saved → {boroughs_path}")

    bbox = buildings.to_crs("EPSG:4326").total_bounds
    print(f"\nBuildings : {len(buildings):,} | bbox {bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f},{bbox[3]:.3f}")
    print(f"  columns : {list(buildings.columns)}")
    print(f"Boroughs  : {len(boroughs):,} | crs {boroughs.crs}")
    print(f"  columns : {list(boroughs.columns)}")


if __name__ == "__main__":
    main()
