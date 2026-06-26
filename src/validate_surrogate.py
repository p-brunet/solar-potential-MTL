"""Validation hors-échantillon du modèle surrogate XGBoost.

Sélectionne 200 bâtiments qui N'ont PAS été utilisés pour entraîner le surrogate,
simule leur rendement avec pvlib (vérité terrain), puis compare aux prédictions.

Métriques produites : R², RMSE, MAPE, biais médian.
Un scatter plot annotated est sauvegardé.

Reads:
  data/processed/pvlib_sample.csv            (bâtiments d'entraînement — exclus)
  data/processed/buildings_features.geojson
  data/processed/tmy_montreal.csv

Writes:
  data/outputs/surrogate_validation.png

Run:
    uv run python src/validate_surrogate.py
"""

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# Permet d'importer pvlib_sample et surrogate_model depuis le même dossier src/
sys.path.insert(0, str(Path(__file__).parent))
from pvlib_sample import fetch_tmy, simulate_all, stratified_sample  # noqa: E402
from surrogate_model import add_az_encoding, FEATURE_COLS            # noqa: E402

from config import BLDG_AREA_MAX_M2, DATA_OUTPUTS, DATA_PROCESSED    # noqa: E402

SAMPLE_PATH   = DATA_PROCESSED / "pvlib_sample.csv"
FEATURES_PATH = DATA_PROCESSED / "buildings_features.geojson"
OUT_PNG       = DATA_OUTPUTS / "surrogate_validation.png"

N_HOLDOUT = 200


def train_surrogate(sample: pd.DataFrame) -> XGBRegressor:
    s = add_az_encoding(sample)
    model = XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(s[FEATURE_COLS].values, s["yield_kwh_kwp"].values)
    return model


def main() -> None:
    # --- Training set ---
    print("Loading training sample ...")
    sample = pd.read_csv(SAMPLE_PATH, index_col=0)
    train_idx = set(sample.index)
    print(f"  {len(sample):,} training rows (indices excluded from holdout)")

    print("Loading building features ...")
    gdf = gpd.read_file(FEATURES_PATH)
    gdf = gdf[gdf["area_m2"] <= BLDG_AREA_MAX_M2]
    holdout_pool = gdf[~gdf.index.isin(train_idx)]
    print(f"  {len(holdout_pool):,} buildings available for holdout")

    print(f"Sampling {N_HOLDOUT} holdout buildings ...")
    holdout = stratified_sample(holdout_pool, n=N_HOLDOUT, seed=99)

    print("Fetching TMY ...")
    tmy = fetch_tmy()

    print(f"Running pvlib on {len(holdout)} holdout buildings (ground truth) ...")
    yields_true = simulate_all(holdout, tmy)

    print("Training surrogate ...")
    model = train_surrogate(sample)

    holdout_enc = add_az_encoding(holdout)
    yields_pred = model.predict(holdout_enc[FEATURE_COLS].values)

    # --- Métriques ---
    r2   = r2_score(yields_true, yields_pred)
    rmse = float(np.sqrt(np.mean((yields_pred - yields_true) ** 2)))
    mape = float(np.mean(np.abs((yields_pred - yields_true) / yields_true)) * 100)
    bias = float(np.median(yields_pred - yields_true))

    print(f"\n--- Out-of-sample validation ({N_HOLDOUT} buildings) ---")
    print(f"  R2   = {r2:.3f}")
    print(f"  RMSE = {rmse:.1f} kWh/kWp")
    print(f"  MAPE = {mape:.1f}%")
    print(f"  Median bias = {bias:+.1f} kWh/kWp")

    # --- Scatter plot ---
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    vmin = min(yields_true.min(), yields_pred.min())
    vmax = max(yields_true.max(), yields_pred.max())
    sc = ax.scatter(yields_true, yields_pred,
                    c=holdout["slope_deg"], cmap="plasma",
                    alpha=0.75, s=30, edgecolors="none")
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1, label="y = x (perfect)")
    ax.set_xlabel("pvlib yield -- ground truth (kWh/kWp)")
    ax.set_ylabel("XGBoost surrogate yield (kWh/kWp)")
    ax.set_title(
        f"Out-of-sample validation (n={N_HOLDOUT})\n"
        f"R2={r2:.3f}  RMSE={rmse:.1f} kWh/kWp  MAPE={mape:.1f}%"
    )
    ax.legend(fontsize=9)
    plt.colorbar(sc, ax=ax, label="Slope (deg)")

    ax = axes[1]
    residuals = yields_pred - yields_true
    ax.hist(residuals, bins=25, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0,    color="black", linestyle="--", linewidth=1)
    ax.axvline(bias, color="orange", linestyle="--",
               label=f"Median bias {bias:+.1f} kWh/kWp")
    ax.set_xlabel("Residual: surrogate - pvlib (kWh/kWp)")
    ax.set_ylabel("Buildings")
    ax.set_title("Residual distribution")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"\nSaved -> {OUT_PNG}")


if __name__ == "__main__":
    main()
