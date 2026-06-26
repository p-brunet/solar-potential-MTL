"""Solar generation vs Hydro-Québec 3 000 MW 2035 target.

Computes island-level solar capacity and annual generation at 5 / 15 / 30 %
rooftop penetration rates, and plots the seasonal generation profile.

Reads:
  data/processed/borough_aggregates.geojson
  data/processed/tmy_montreal.csv

Writes:
  data/outputs/demand_impact.csv
  data/outputs/demand_impact.png

Run:
    uv run python src/demand_impact.py
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

from config import DATA_OUTPUTS, DATA_PROCESSED, PVLIB_LAT, PVLIB_LON, SYSTEM_LOSSES

AGGREGATES_PATH = DATA_PROCESSED / "borough_aggregates.geojson"
TMY_CACHE       = DATA_PROCESSED / "tmy_montreal.csv"
OUT_CSV         = DATA_OUTPUTS / "demand_impact.csv"
OUT_PNG         = DATA_OUTPUTS / "demand_impact.png"

HQ_TARGET_MW       = 3_000   # HQ 2035 rooftop solar target
PENETRATION_RATES  = [0.05, 0.15, 0.30]
MONTH_LABELS       = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load_tmy() -> pd.DataFrame:
    tmy = pd.read_csv(TMY_CACHE, index_col=0, parse_dates=True)
    if tmy.index.tz is None:
        tmy.index = tmy.index.tz_localize("UTC")
    return tmy


def monthly_generation_profile(tmy: pd.DataFrame) -> pd.Series:
    """Monthly fraction of annual generation for a representative MTL roof.

    Uses a south-facing surface at the median roof slope (30°) with Perez model.
    Returns a Series indexed 1..12 summing to 1.
    """
    location  = pvlib.location.Location(PVLIB_LAT, PVLIB_LON, tz="UTC", altitude=30)
    solar_pos = location.get_solarposition(tmy.index)
    dni_extra = pvlib.irradiance.get_extra_radiation(tmy.index)
    airmass   = location.get_airmass(solar_position=solar_pos)

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=30,
        surface_azimuth=180,    # South-facing
        solar_zenith=solar_pos["apparent_zenith"],
        solar_azimuth=solar_pos["azimuth"],
        dni=tmy["dni"],
        ghi=tmy["ghi"],
        dhi=tmy["dhi"],
        dni_extra=dni_extra,
        airmass=airmass["airmass_relative"],
        model="perez",
    )
    gen = poa["poa_global"].clip(lower=0).fillna(0) * (1 - SYSTEM_LOSSES)
    monthly = gen.groupby(tmy.index.month).sum()
    return monthly / monthly.sum()


def main() -> None:
    print("Loading borough aggregates ...")
    agg = gpd.read_file(AGGREGATES_PATH)
    island_mw  = agg["capacity_mw"].sum()
    island_gwh = agg["energy_gwh_yr"].sum()
    cf         = island_gwh * 1e6 / (island_mw * 1e3 * 8_760)
    print(f"  Island eligible: {island_mw:,.0f} MW  {island_gwh:,.0f} GWh/yr  CF={cf:.2f}")

    print("Computing monthly generation profile ...")
    tmy     = load_tmy()
    monthly = monthly_generation_profile(tmy)

    # Penetration scenarios
    rows = []
    for p in PENETRATION_RATES:
        cap_mw  = island_mw  * p
        ann_gwh = island_gwh * p
        rows.append({
            "penetration_pct":    int(p * 100),
            "capacity_mw":        round(cap_mw,  1),
            "annual_gwh":         round(ann_gwh, 1),
            "pct_hq_target_cap":  round(100 * cap_mw  / HQ_TARGET_MW, 1),
        })
    df = pd.DataFrame(rows)
    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved → {OUT_CSV}")
    print(df.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#4e91d9", "#f5a623", "#e84040"]

    ax = axes[0]
    x = np.arange(12)
    width = 0.25
    for i, (p, color) in enumerate(zip(PENETRATION_RATES, colors)):
        gwh_monthly = monthly.values * island_gwh * p
        ax.bar(x + i * width, gwh_monthly, width,
               label=f"{int(p*100)}% penetration", color=color, alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.set_ylabel("Monthly generation (GWh)")
    ax.set_title("Seasonal generation profile by penetration rate")
    ax.legend(fontsize=9)

    ax = axes[1]
    caps = [island_mw * p for p in PENETRATION_RATES]
    bars = ax.bar([f"{int(p*100)}%" for p in PENETRATION_RATES],
                  caps, color=colors, alpha=0.85, width=0.5)
    ax.axhline(HQ_TARGET_MW, color="red", linestyle="--", linewidth=1.5,
               label=f"HQ 2035 target ({HQ_TARGET_MW:,} MW)")
    for bar, cap in zip(bars, caps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{cap:.0f} MW\n({100*cap/HQ_TARGET_MW:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Rooftop penetration rate")
    ax.set_ylabel("Installed capacity (MW)")
    ax.set_title("Montreal rooftop solar vs HQ 3 000 MW target")
    ax.legend(fontsize=9)
    ax.set_ylim(0, HQ_TARGET_MW * 1.2)

    plt.suptitle("Impact of rooftop solar on Montreal's grid", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {OUT_PNG}")


if __name__ == "__main__":
    main()
