# Session 01 — Building Footprints + Borough Polygons

## Goal
Download and validate all raw geodata needed for the pipeline.

## Data sources

- **NRCan building footprints (GeoParquet)**
  https://open.canada.ca/data/en/dataset/7a5cda52-c7df-427f-9ced-26f19a8a64d6
  URL resolved at runtime via CKAN API (NRCan maintains a stable, versioned catalog).
  Resource `auto_building_opti_2.parquet` — read directly from URL, bbox-filtered via pyarrow/fsspec.

- **Montreal administrative limits (GeoJSON WGS 84)**
  https://donnees.montreal.ca/en/dataset/limites-administratives-agglomeration
  URL hardcoded (Montreal open data not reliably harmonized — CKAN API returns 403).
  Resource `e18bfd07` — 34 entities: 19 boroughs (arrondissements) + 15 municipalities on the island.

## Scripts

- `src/collect.py` — downloads both datasets, clips buildings to bbox, saves GeoJSON
- `src/validate.py` — checks quality criteria and generates a control map
- `src/roof_features.py` — stub (loads buildings, prints schema)

## Outputs

| File | CRS | Rows | Key columns |
|------|-----|------|-------------|
| `data/raw/buildings_mtl.geojson` | EPSG:4326 | ~450k | TBD after next run |
| `data/raw/boroughs.geojson` | EPSG:32188 | 34 | `NOM`, `NOM_OFFICIEL`, `TYPE`, `geometry` |
| `data/outputs/validation_raw.png` | — | — | control map |

## Validation criteria

- Buildings count > 50 000, bbox contained within `MONTREAL_BBOX`, the extracted buildings area is larger than Mont
- Administrative limits count == 34 (boroughs + island municipalities), CRS set on both files

## Issues

| # | Problem | Fix |
|---|---------|-----|
| 1 | National GPKG is several GB | Prefer GeoParquet with `bbox` filter — only relevant row groups fetched |
| 2 | `gpd.read_parquet("https://…")` raises `ArrowInvalid` | Wrap with `fsspec.open(url, "rb")` for HTTP range-request support |
| 3 | Montreal CKAN API returns 403 | Use direct download URL instead |
| 4 | Borough fallback URL 404 (wrong dataset + resource ID) | Dataset `limites-administratives-agglomeration`, resource `e18bfd07` |
