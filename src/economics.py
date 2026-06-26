"""Per-building economics with net metering, maintenance and sensitivity curve.

Net metering: 60% self-consumed at full tariff, 40% sold back to HQ at buyback rate.
Maintenance: 200 CAD/yr (O&M + inverter replacement provision).
Sensitivity curve: NPV = f(tariff escalation rate) to identify breakeven threshold.

Sources:
  - SELF_CONSUMPTION_RATE : NRCan (2022), residential without battery
  - HQ_BUYBACK_RATE       : HQ Option Autoconsommation (2024) -- 3.5 c/kWh
  - TARIFF_ESCALATION     : HQ Master Plan 2035 / PGIR (conservative 3.5%/yr)

Reads:   data/processed/buildings_predicted.geojson
Writes:  data/processed/buildings_economics.geojson
         data/outputs/economics_hist.png
         data/outputs/economics_sensitivity.png

Run:
    uv run python src/economics.py
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    COST_PER_WATT,
    DATA_OUTPUTS,
    DATA_PROCESSED,
    DISCOUNT_RATE,
    HQ_BUYBACK_RATE,
    ITC_FEDERAL,
    MAINTENANCE_COST_YR,
    PANEL_DEGRADATION,
    SELF_CONSUMPTION_RATE,
    SYSTEM_LIFE_YR,
    TARIFF_D_AVG,
    TARIFF_ESCALATION,
)

PREDICTED_PATH = DATA_PROCESSED / "buildings_predicted.geojson"
OUT_PATH       = DATA_PROCESSED / "buildings_economics.geojson"
HIST_PNG       = DATA_OUTPUTS / "economics_hist.png"
SENS_PNG       = DATA_OUTPUTS / "economics_sensitivity.png"


def compute_economics(
    annual_kwh: np.ndarray,
    net_capex: np.ndarray,
    escalation: float = TARIFF_ESCALATION,
    buyback_rate: float = HQ_BUYBACK_RATE,
    self_consumption: float = SELF_CONSUMPTION_RATE,
    base_tariff: float = TARIFF_D_AVG,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (payback_yr, npv_cad) with net metering and annual maintenance."""
    years        = np.arange(1, SYSTEM_LIFE_YR + 1)
    prod_factor  = (1 - PANEL_DEGRADATION) ** (years - 1)
    tariff_t     = base_tariff * (1 + escalation) ** (years - 1)

    effective_rate = (
        self_consumption * tariff_t
        + (1 - self_consumption) * buyback_rate
    )
    savings = (
        annual_kwh[:, None] * prod_factor[None, :] * effective_rate[None, :]
        - MAINTENANCE_COST_YR
    )  # shape (N, 25)

    discount_factors = 1 / (1 + DISCOUNT_RATE) ** years
    npv = (savings * discount_factors[None, :]).sum(axis=1) - net_capex

    cum_savings = savings.cumsum(axis=1)
    exceeded    = cum_savings >= net_capex[:, None]
    payback     = np.where(
        exceeded.any(axis=1), exceeded.argmax(axis=1) + 1.0, np.inf
    )
    return payback, npv


POLICY_SCENARIOS = [
    {"label": "Baseline -- PGIR 3.5%/yr",        "escalation": 0.035, "buyback": 0.035, "itc": 0.30, "self_cons": 0.60},
    {"label": "HQ buyback doubled (7 c/kWh)",     "escalation": 0.035, "buyback": 0.070, "itc": 0.30, "self_cons": 0.60},
    {"label": "High escalation 5%/yr (PGIR high)","escalation": 0.050, "buyback": 0.035, "itc": 0.30, "self_cons": 0.60},
    {"label": "Federal ITC 45%",                  "escalation": 0.035, "buyback": 0.035, "itc": 0.45, "self_cons": 0.60},
    {"label": "Combined optimistic",              "escalation": 0.050, "buyback": 0.070, "itc": 0.45, "self_cons": 0.80},
]


def run_policy_scenarios(
    annual_kwh: np.ndarray,
    capex_cad: np.ndarray,
) -> pd.DataFrame:
    """Evaluate each policy scenario and return a comparative summary table."""
    rows = []
    for sc in POLICY_SCENARIOS:
        net_capex = capex_cad * (1 - sc["itc"])
        pb, npv = compute_economics(
            annual_kwh, net_capex,
            escalation=sc["escalation"],
            buyback_rate=sc["buyback"],
            self_consumption=sc["self_cons"],
            base_tariff=sc.get("base_tariff", TARIFF_D_AVG),
        )
        viable = npv > 0
        within = pb <= SYSTEM_LIFE_YR
        med_pb = float(np.median(pb[within])) if within.any() else float("inf")
        rows.append({
            "Scenario":              sc["label"],
            "Median payback (yr)":   round(med_pb, 1),
            "% payback < 25 yr":     round(100 * within.mean(), 1),
            "% NPV > 0":             round(100 * viable.mean(), 1),
            "Viable buildings":      int(viable.sum()),
            "Median NPV (k CAD)":    round(float(np.median(npv)) / 1000, 1),
        })
    return pd.DataFrame(rows)


def sensitivity_curve(
    annual_kwh: np.ndarray,
    net_capex: np.ndarray,
    escalation_range: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median NPV and % viable buildings across a range of escalation rates."""
    median_npv = np.empty(len(escalation_range))
    pct_viable = np.empty(len(escalation_range))
    for i, esc in enumerate(escalation_range):
        _, npv = compute_economics(annual_kwh, net_capex, escalation=esc)
        median_npv[i] = np.median(npv)
        pct_viable[i] = 100 * (npv > 0).mean()
    return median_npv, pct_viable


def main() -> None:
    print("Loading buildings ...")
    gdf  = gpd.read_file(PREDICTED_PATH)
    elig = gdf[gdf["eligible"]].copy()
    print(f"  {len(elig):,} eligible buildings")

    elig["capex_cad"] = elig["capacity_kwp"] * 1_000 * COST_PER_WATT
    elig["net_capex"] = elig["capex_cad"] * (1 - ITC_FEDERAL)
    elig["annual_savings_cad"] = elig["annual_kwh"] * (
        SELF_CONSUMPTION_RATE * TARIFF_D_AVG
        + (1 - SELF_CONSUMPTION_RATE) * HQ_BUYBACK_RATE
    )

    elig["payback_yr"], elig["npv_cad"] = compute_economics(
        elig["annual_kwh"].values,
        elig["net_capex"].values,
    )

    print("Assumptions:")
    print(f"  Installed cost       : {COST_PER_WATT:.2f} CAD/W")
    print(f"  Federal ITC          : {ITC_FEDERAL*100:.0f}%")
    print(f"  Base tariff D        : {TARIFF_D_AVG*100:.1f} c/kWh")
    print(f"  Tariff escalation    : +{TARIFF_ESCALATION*100:.1f}%/yr (HQ PGIR 2035)")
    print(f"  Self-consumption     : {SELF_CONSUMPTION_RATE*100:.0f}% at full tariff, "
          f"{(1-SELF_CONSUMPTION_RATE)*100:.0f}% at {HQ_BUYBACK_RATE*100:.1f} c/kWh (HQ buyback)")
    print(f"  Maintenance          : {MAINTENANCE_COST_YR} CAD/yr")
    print(f"  Panel degradation    : -{PANEL_DEGRADATION*100:.1f}%/yr")

    print("\nEconomics (eligible buildings):")
    fmt = "{:>12,.0f}"
    summary_rows = [
        ("Gross capex",             "capex_cad",          "CAD"),
        ("Net capex (after ITC)",   "net_capex",          "CAD"),
        ("Year-1 savings",          "annual_savings_cad", "CAD/yr"),
        ("Payback",                 "payback_yr",         "yr"),
        ("NPV 25yr @ 5%",           "npv_cad",            "CAD"),
    ]
    for label, col, unit in summary_rows:
        s = elig[col].replace(np.inf, np.nan).dropna()
        print(f"  {label:26s}  "
              f"p25={fmt.format(s.quantile(0.25))}  "
              f"median={fmt.format(s.median())}  "
              f"p75={fmt.format(s.quantile(0.75))}  {unit}")

    viable    = elig["npv_cad"] > 0
    within_25 = elig["payback_yr"] <= SYSTEM_LIFE_YR
    tariff_25 = TARIFF_D_AVG * (1 + TARIFF_ESCALATION) ** 24
    print(f"\n  NPV > 0 (viable)     : {viable.sum():,} / {len(elig):,} ({100*viable.mean():.1f}%)")
    print(f"  Payback < 25 yr      : {within_25.sum():,} / {len(elig):,} ({100*within_25.mean():.1f}%)")
    print(f"  Projected tariff yr25: {tariff_25*100:.1f} c/kWh")

    DATA_OUTPUTS.mkdir(parents=True, exist_ok=True)

    # Distribution histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    pb = elig["payback_yr"].replace(np.inf, SYSTEM_LIFE_YR + 1).clip(upper=SYSTEM_LIFE_YR + 1)
    ax.hist(pb, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(SYSTEM_LIFE_YR, color="red", linestyle="--",
               label=f"System life ({SYSTEM_LIFE_YR} yr)")
    med_pb = pb[pb <= SYSTEM_LIFE_YR].median()
    ax.axvline(med_pb, color="orange", linestyle="--", label=f"Median {med_pb:.1f} yr")
    ax.set_xlabel("Payback period (yr)")
    ax.set_ylabel("Buildings")
    ax.set_title("Payback period distribution\n(net metering + maintenance)")
    ax.legend(fontsize=9)

    ax = axes[1]
    npv_k = elig["npv_cad"].clip(-25_000, 30_000) / 1_000
    ax.hist(npv_k, bins=40, color="seagreen", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linestyle="--", label="NPV = 0")
    ax.set_xlabel("NPV (k CAD)")
    ax.set_ylabel("Buildings")
    ax.set_title(f"NPV 25yr @ {DISCOUNT_RATE*100:.0f}%\n"
                 f"(tariff {TARIFF_D_AVG*100:.1f}c -> {tariff_25*100:.1f}c/kWh yr 25)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(HIST_PNG, dpi=150)
    plt.close()
    print(f"\nHistograms -> {HIST_PNG}")

    # Sensitivity curve: NPV = f(escalation rate)
    print("Computing sensitivity curve ...")
    esc_range  = np.linspace(0.01, 0.09, 60)
    med_npv, pct_v = sensitivity_curve(
        elig["annual_kwh"].values,
        elig["net_capex"].values,
        esc_range,
    )

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(esc_range * 100, med_npv / 1_000, color="steelblue", linewidth=2,
             label="Median NPV (k CAD)")
    ax1.axhline(0, color="steelblue", linestyle=":", linewidth=1)
    ax1.set_xlabel("Annual tariff escalation rate (%)")
    ax1.set_ylabel("Median NPV (k CAD)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2.plot(esc_range * 100, pct_v, color="seagreen", linewidth=2, linestyle="--",
             label="% viable buildings")
    ax2.set_ylabel("Buildings with NPV > 0 (%)", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax2.set_ylim(0, 100)

    ax1.axvline(TARIFF_ESCALATION * 100, color="orange", linestyle="--", linewidth=1.5,
                label=f"PGIR assumption ({TARIFF_ESCALATION*100:.1f}%/yr)")
    ax1.axvline(5.0, color="red", linestyle=":", linewidth=1,
                label="HQ estimated need (~5%/yr)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    ax1.set_title("NPV sensitivity to HQ tariff escalation rate\n"
                  "(net metering + maintenance included)")
    plt.tight_layout()
    plt.savefig(SENS_PNG, dpi=150)
    plt.close()
    print(f"Sensitivity curve -> {SENS_PNG}")

    # Policy scenarios
    print("Computing policy scenarios ...")
    scenarios_df = run_policy_scenarios(
        elig["annual_kwh"].values,
        elig["capex_cad"].values,
    )
    scenarios_csv = DATA_OUTPUTS / "policy_scenarios.csv"
    scenarios_df.to_csv(scenarios_csv, index=False)
    print(f"\n{scenarios_df.to_string(index=False)}")
    print(f"\nScenarios -> {scenarios_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors = ["#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#1abc9c"]
    labels = [s["label"] for s in POLICY_SCENARIOS]
    pct_viable  = scenarios_df["% NPV > 0"].values
    med_pb_vals = scenarios_df["Median payback (yr)"].values

    ax = axes[0]
    bars = ax.barh(labels, pct_viable, color=colors, alpha=0.85)
    ax.axvline(50, color="black", linestyle="--", linewidth=1, label="50%")
    for bar, val in zip(bars, pct_viable):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("% of buildings with NPV > 0")
    ax.set_title("Viable buildings by scenario")
    ax.set_xlim(0, max(pct_viable) * 1.15 + 5)
    ax.legend(fontsize=9)

    ax = axes[1]
    ax.barh(labels, med_pb_vals, color=colors, alpha=0.85)
    ax.axvline(SYSTEM_LIFE_YR, color="red", linestyle="--", linewidth=1,
               label=f"System life ({SYSTEM_LIFE_YR} yr)")
    for bar, val in zip(ax.patches, med_pb_vals):
        ax.text(val + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} yr", va="center", fontsize=9)
    ax.set_xlabel("Median payback (yr)")
    ax.set_title("Median payback by scenario")
    ax.set_xlim(0, SYSTEM_LIFE_YR * 1.1)
    ax.legend(fontsize=9)

    plt.suptitle("Policy lever impact on residential solar viability", fontsize=12)
    plt.tight_layout()
    scenarios_png = DATA_OUTPUTS / "policy_scenarios.png"
    plt.savefig(scenarios_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart -> {scenarios_png}")

    econ_cols = ["capex_cad", "net_capex", "annual_savings_cad", "payback_yr", "npv_cad"]
    for col in econ_cols:
        gdf[col] = np.nan
    gdf.loc[elig.index, econ_cols] = elig[econ_cols].values

    gdf.to_file(OUT_PATH, driver="GeoJSON")
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
