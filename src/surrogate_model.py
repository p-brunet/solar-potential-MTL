"""XGBoost surrogate: roof features → annual yield for all buildings.

Azimuth is encoded as (sin, cos) to handle its circularity (0°=360°).
5-fold CV R² is printed as a quality check.

Reads:
  data/processed/pvlib_sample.csv
  data/processed/buildings_features.geojson

Writes:
  data/processed/buildings_predicted.geojson
    added columns: yield_kwh_kwp, capacity_kwp, annual_kwh, eligible

Run:
    uv run python src/surrogate_model.py
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from config import (
    BLDG_AREA_MAX_M2,
    DATA_OUTPUTS,
    DATA_PROCESSED,
    PANEL_EFFICIENCY,
    ROOF_USABLE_FRACTION_FLAT,
    ROOF_USABLE_FRACTION_PITCHED,
    SLOPE_MAX_ELIGIBLE,
    YIELD_MIN_ELIGIBLE,
)

SAMPLE_PATH   = DATA_PROCESSED / "pvlib_sample.csv"
FEATURES_PATH = DATA_PROCESSED / "buildings_features.geojson"
OUT_PATH      = DATA_PROCESSED / "buildings_predicted.geojson"
SHAP_PNG      = DATA_OUTPUTS / "shap_importance.png"

# area_m2 exclu : le yield en kWh/kWp est un ratio, indépendant de la surface
FEATURE_COLS = ["slope_deg", "sin_az", "cos_az", "svf"]


def add_az_encoding(df: pd.DataFrame) -> pd.DataFrame:
    rad = np.radians(df["azimuth_deg"])
    out = df.copy()
    out["sin_az"] = np.sin(rad)
    out["cos_az"] = np.cos(rad)
    return out


def main() -> None:
    # --- Training data ---
    print("Loading training sample ...")
    sample = pd.read_csv(SAMPLE_PATH, index_col=0)
    sample = add_az_encoding(sample)
    X_train = sample[FEATURE_COLS].values
    y_train = sample["yield_kwh_kwp"].values
    print(f"  {len(sample):,} rows, yield range "
          f"[{y_train.min():.0f}, {y_train.max():.0f}] kWh/kWp")

    print("Training XGBoost surrogate ...")
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    cv_r2 = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"  5-fold CV R² = {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")

    model.fit(X_train, y_train)

    print("Computing SHAP values ...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    shap.summary_plot(shap_values, X_train, feature_names=FEATURE_COLS,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(SHAP_PNG, dpi=150)
    plt.close()
    print(f"  SHAP plot → {SHAP_PNG}")

    print("Loading all buildings ...")
    gdf = gpd.read_file(FEATURES_PATH)
    before = len(gdf)
    gdf = gdf[gdf["area_m2"] <= BLDG_AREA_MAX_M2].copy().reset_index(drop=True)
    print(f"  {len(gdf):,} buildings after area filter "
          f"(dropped {before - len(gdf):,} > {BLDG_AREA_MAX_M2} m2)")
    gdf = add_az_encoding(gdf)
    X_all = gdf[FEATURE_COLS].values

    print("Predicting ...")
    gdf["yield_kwh_kwp"] = model.predict(X_all)
    # Usable fraction depends on slope: flat roofs allow spaced rows, pitched roofs one face
    usable = np.where(
        gdf["slope_deg"] < 5,
        ROOF_USABLE_FRACTION_FLAT,
        ROOF_USABLE_FRACTION_PITCHED,
    )
    gdf["capacity_kwp"] = gdf["area_m2"] * usable * PANEL_EFFICIENCY
    gdf["annual_kwh"]   = gdf["capacity_kwp"] * gdf["yield_kwh_kwp"]
    gdf["eligible"]      = (
        (gdf["slope_deg"] <= SLOPE_MAX_ELIGIBLE)
        & (gdf["yield_kwh_kwp"] >= YIELD_MIN_ELIGIBLE)
    )

    n_elig       = int(gdf["eligible"].sum())
    total_cap_mw = gdf.loc[gdf["eligible"], "capacity_kwp"].sum() / 1000
    total_gwh    = gdf.loc[gdf["eligible"], "annual_kwh"].sum() / 1e6

    print(f"\n  Eligible: {n_elig:,} / {len(gdf):,} ({100 * n_elig / len(gdf):.1f}%)")
    print(f"  Total capacity:    {total_cap_mw:,.0f} MW")
    print(f"  Annual generation: {total_gwh:,.0f} GWh/yr")

    # NRCan solar atlas reports 1,100-1,200 kWh/kWp for optimal Montreal roofs.
    south_optimal = gdf[
        gdf["azimuth_deg"].between(135, 225) & gdf["slope_deg"].between(28, 42)
    ]
    print("\n--- NRCan validation ---")
    print(f"  NRCan reference Montreal (optimal) : 1,100-1,200 kWh/kWp")
    print(f"  Model -- south-optimal roofs (135-225 deg, 28-42 deg) : "
          f"median {south_optimal['yield_kwh_kwp'].median():.0f} kWh/kWp  "
          f"(n={len(south_optimal):,})")
    print(f"  Model -- all orientations          : "
          f"median {gdf['yield_kwh_kwp'].median():.0f} kWh/kWp  "
          f"(n={len(gdf):,})")

    # Drop intermediate encoding columns -- not needed downstream
    gdf = gdf.drop(columns=["sin_az", "cos_az"])
    gdf.to_file(OUT_PATH, driver="GeoJSON")
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
