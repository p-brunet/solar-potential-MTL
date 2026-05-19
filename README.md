# solar-potential-MTL

End-to-end analysis of residential rooftop solar potential on the island of Montreal.
Quantifies how much of Hydro-Québec's 3 000 MW solar target by 2035 could be met by
behind-the-meter rooftop installations.

## Project overview

The pipeline starts from raw geodata (building footprints + LiDAR), computes roof geometry
and solar yield for every building on the island, then scales up with a surrogate model to
produce borough-level estimates. Economic viability (payback, NPV) and grid impact (net load
reduction at various penetration rates) are analyzed against Hydro-Québec demand data.

### Key questions
- What fraction of Montreal's ~300 000 residential rooftops are eligible (slope < 45°, yield > 800 kWh/kWp/year)?
- How many MW of capacity could realistically be installed?
- Which boroughs have the highest potential density?
- At what penetration rate does rooftop solar meaningfully dent HQ's 3 000 MW target?

## Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `src/collect.py` | Download NRCan building footprints (GeoParquet, bbox-filtered) and Montreal borough polygons |
| 2 | `src/validate.py` | Validate raw data quality and generate a control map |
| 3 | `src/roof_features.py` | Extract slope, azimuth, sky-view fraction from LiDAR DSM tiles |
| 4 | `src/pvlib_sample.py` | pvlib full-year yield simulation on a stratified sample of 3 000 buildings |
| 5 | `src/surrogate_model.py` | XGBoost surrogate: predict yield for all buildings from roof features |
| 6 | `src/aggregate.py` | Borough aggregation + comparison with HQ solar benchmark |
| 7 | `src/economics.py` | Payback period, NPV, net metering scenarios |
| 8 | `src/demand_impact.py` | Net load reduction at 5 / 15 / 30 % penetration |
| 9 | `src/map.py` | Interactive Folium map (GitHub Pages) |

## Data sources

| Dataset | Source |
|---------|--------|
| NRCan building footprints | [open.canada.ca](https://open.canada.ca/data/en/dataset/7a5cda52-c7df-427f-9ced-26f19a8a64d6) — GeoParquet, national coverage, bbox-filtered at read time |
| Montreal borough polygons | [donnees.montreal.ca](https://donnees.montreal.ca/en/dataset/limites-administratives-agglomeration) — GeoJSON WGS 84 |
| Solar irradiance (TMY) | [PVGIS API](https://re.jrc.ec.europa.eu/api/v5_2/tmy?lat=45.508&lon=-73.587&outputformat=json) |
| HQ historical demand | [Hydro-Québec open data](https://donnees.hydroquebec.com/api/explore/v2.1/catalog/datasets/historique-demande-electricite-quebec/records) |

## Setup

```bash
# Install uv (https://docs.astral.sh/uv/)
uv sync

# Collect raw data
uv run python src/collect.py

# Validate and visualize
uv run python src/validate.py
```

Or create a virtual environment with VSCode (`Ctrl+Shift+P` → *Python: Create Environment*),
then run scripts directly with `python src/collect.py`.

## Configuration

All thresholds, paths, and economic parameters are centralized in `config.py`:

- `MONTREAL_BBOX` — bounding box (WGS 84) used to clip all spatial data
- `CRS_PROJECT` — EPSG:32188 (MTM zone 8) for area and distance calculations
- `SLOPE_MAX_ELIGIBLE`, `YIELD_MIN_ELIGIBLE` — roof eligibility thresholds
- `COST_PER_WATT`, `ITC_FEDERAL`, `TARIFF_D_AVG` — economic parameters (QC 2024)
