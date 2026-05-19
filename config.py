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
COST_PER_WATT = 3.50   # CAD/W installed QC 2024
ITC_FEDERAL = 0.30     # federal tax credit
TARIFF_D_AVG = 0.075   # CAD/kWh HQ residential avg
DISCOUNT_RATE = 0.05   # NPV 25 years

# --- Surrogate model ---
SAMPLE_SIZE = 3000  # buildings for surrogate training

# --- Paths ---
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_OUTPUTS = ROOT / "data" / "outputs"
