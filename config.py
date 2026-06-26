from pathlib import Path

ROOT = Path(__file__).parent

# --- Geographic ---
MONTREAL_BBOX = (-73.985, 45.402, -73.467, 45.707)  # (minlon, minlat, maxlon, maxlat) WGS84
CRS_PROJECT = "EPSG:32188"  # MTM zone 8, local Quebec projection

# --- PVLib ---
PVLIB_LAT = 45.508
PVLIB_LON = -73.587
PANEL_EFFICIENCY = 0.20
SYSTEM_LOSSES = 0.14

# --- Eligibility thresholds ---
SLOPE_MAX_ELIGIBLE = 45   # degrees
YIELD_MIN_ELIGIBLE = 800  # kWh/kWp/year

# --- Economics ---
# Sources: HQ Master Plan 2035 (Feb. 2024), Regie de l'energie decisions 2023-2025,
#          HQ PGIR (global investment and renewal plan) filed with the Regie.
# HQ must invest ~$185B by 2035 (+45% demand) -> structural tariff pressure.
# Residential tariff D is politically protected (frequent freezes, capped increases),
# hence a conservative 3.5%/yr assumption vs the estimated real need of 5-7%/yr.
COST_PER_WATT          = 3.00   # CAD/W installed QC 2025 (declining ~5%/yr since 2020)
ITC_FEDERAL            = 0.30   # federal investment tax credit (Bill C-69 2024)
TARIFF_D_AVG           = 0.075  # CAD/kWh average HQ residential tariff D 2024 (base)
TARIFF_ESCALATION      = 0.035  # +3.5%/yr -- Regie-approved increases 2023-25 + PGIR projection
PANEL_DEGRADATION      = 0.005  # 0.5%/yr -- typical crystalline module degradation (NREL 2023)
DISCOUNT_RATE          = 0.05   # nominal discount rate over 25 years
SYSTEM_LIFE_YR         = 25     # guaranteed module lifetime
SELF_CONSUMPTION_RATE  = 0.60   # self-consumed fraction at full tariff (no battery, NRCan 2022)
HQ_BUYBACK_RATE        = 0.035  # CAD/kWh -- HQ buyback rate (Option Autoconsommation 2024)
MAINTENANCE_COST_YR    = 200    # CAD/yr -- O&M + inverter replacement provision over 25 years

# --- Surrogate model ---
SAMPLE_SIZE = 3000                  # buildings for surrogate training
ROOF_USABLE_FRACTION_FLAT    = 0.65 # flat roofs (slope < 5 deg) -- spaced rows
ROOF_USABLE_FRACTION_PITCHED = 0.32 # pitched roofs (slope >= 5 deg) -- one usable face
BLDG_AREA_MAX_M2 = 500              # residential proxy: excludes warehouses and malls

# --- Paths ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS = ROOT / "data" / "outputs"
