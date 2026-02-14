"""
Structural Imbalance Model
============================
Based on Michael Pettis's framework for analyzing China's economy.

Core thesis: China's growth model systematically transfers income from
households to producers via three mechanisms:
  1. Undervalued currency (export subsidy / import tax on consumers)
  2. Low wage growth relative to productivity (labor repression)
  3. Financial repression (artificially low interest rates = hidden tax on savers)

This produces:
  - Abnormally LOW consumption share of GDP (~53% vs global ~75%)
  - Abnormally HIGH investment share of GDP (~43% vs global ~22%)
  - Structural current account surpluses
  - Rising debt (investment increasingly unproductive)

The model tracks these ratios against international benchmarks and projects
rebalancing scenarios.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data_fetcher import (
    fetch_china_consumption_pct,
    fetch_china_gov_consumption_pct,
    fetch_china_investment_pct,
    fetch_china_gdp_usd_B,
    fetch_benchmarks_consumption,
    fetch_benchmarks_investment,
)


def build_structural_dataset():
    """
    China GDP expenditure breakdown and comparisons.

    Sources:
    - World Bank WDI (live via data_fetcher, with hardcoded fallback)
    - NBS China Statistical Yearbook
    - Pettis's Carnegie writings for framework interpretation
    """

    years = list(range(2000, 2026))

    # --- China: Household Consumption as % of GDP ---
    # Source: World Bank NE.CON.PRVT.ZS (live) / NBS (fallback)
    # Pettis emphasizes this is THE key metric.
    # Global norm: 55-65% for comparable economies, 74-75% global average
    china_consumption_pct = fetch_china_consumption_pct(years)

    # --- China: Government Consumption as % of GDP ---
    # Source: World Bank NE.CON.GOVT.ZS (live) / NBS (fallback)
    china_gov_consumption_pct = fetch_china_gov_consumption_pct(years)

    # --- China: Total Consumption (household + gov) as % of GDP ---
    china_total_consumption = [h + g for h, g in
                                zip(china_consumption_pct, china_gov_consumption_pct)]

    # --- China: Gross Capital Formation (Investment) as % of GDP ---
    # Source: World Bank NE.GDI.TOTL.ZS (live) / NBS (fallback)
    # Pettis: "40-50% for over a decade - unprecedented"
    china_investment_pct = fetch_china_investment_pct(years)

    # --- Net Exports as % of GDP ---
    china_net_exports_pct = [
        round(100 - c - i, 1)
        for c, i in zip(china_total_consumption, china_investment_pct)
    ]

    # --- Financial Repression Gap (basis points) ---
    # Pettis: deposit rates are 500-800 bps below market-clearing rate
    # Measured as: benchmark 1yr deposit rate minus CPI inflation
    # Negative = savers are being taxed (real return < 0)
    # Source: PBOC benchmark rates, NBS CPI
    real_deposit_rate_bps = [
        -140,  # 2000
        150,   # 2001
        210,   # 2002
        80,    # 2003
        -280,  # 2004
        -20,   # 2005
        -70,   # 2006
        -260,  # 2007
        -360,  # 2008
        140,   # 2009
        -110,  # 2010
        -320,  # 2011
        -30,   # 2012
        40,    # 2013
        80,    # 2014
        150,   # 2015
        60,    # 2016
        -10,   # 2017
        -60,   # 2018
        -140,  # 2019
        -100,  # 2020
        -60,   # 2021
        -70,   # 2022
        150,   # 2023 - deflation pushes real rates up
        180,   # 2024
        160,   # 2025 est
    ]

    # --- Household Income as % of GDP ---
    # Pettis: systematically suppressed vs international norm of 60-65%
    # Source: NBS flow-of-funds accounts, Pettis estimates
    household_income_share = [
        46.0, 45.5, 45.0, 44.0, 42.5, 41.0, 40.0, 39.5, 39.0, 39.5,
        40.0, 41.0, 42.0, 42.5, 43.0, 43.5, 44.0, 44.5, 44.0, 44.5,
        43.5, 44.0, 43.0, 44.0, 44.5, 44.8,
    ]

    # --- Incremental Capital-Output Ratio (ICOR) ---
    # Measures investment efficiency: higher = more wasteful
    # ICOR = Investment / Change in GDP
    # Pettis: rising ICOR proves increasing share of investment is unproductive
    # Source: World Bank NY.GDP.MKTP.CD (live) / fallback
    china_gdp_usd_B = fetch_china_gdp_usd_B(years)
    icor = []
    for i in range(len(years)):
        if i == 0:
            icor.append(np.nan)
        else:
            delta_gdp = china_gdp_usd_B[i] - china_gdp_usd_B[i - 1]
            investment = china_gdp_usd_B[i] * china_investment_pct[i] / 100
            if delta_gdp > 0:
                icor.append(investment / delta_gdp)
            else:
                icor.append(np.nan)

    df = pd.DataFrame({
        'year': years,
        'hh_consumption_pct': china_consumption_pct,
        'gov_consumption_pct': china_gov_consumption_pct,
        'total_consumption_pct': china_total_consumption,
        'investment_pct': china_investment_pct,
        'net_exports_pct': china_net_exports_pct,
        'real_deposit_rate_bps': real_deposit_rate_bps,
        'hh_income_share_pct': household_income_share,
        'gdp_usd_B': china_gdp_usd_B,
        'icor': icor,
    })

    return df


def build_rebalancing_scenarios():
    """
    Pettis rebalancing scenarios.

    From his Sept 2024 Carnegie paper:
    If China wants to raise consumption from 53% -> 63-64% of GDP,
    the math constrains what GDP growth rates are achievable.

    Using TOTAL consumption (household + government):
    Current ~55%, target ~65%, which requires similar dynamics.
    """

    scenarios = []
    current_consumption_share = 0.55  # total consumption / GDP
    target_consumption_share = 0.65

    for gdp_growth in [0.02, 0.03, 0.04, 0.05]:
        for consumption_share_of_growth in [0.60, 0.70, 0.80, 0.90]:
            # Simulate year by year
            gdp = 1.0
            consumption = current_consumption_share * gdp
            years_to_target = None

            for yr in range(1, 51):
                new_gdp = gdp * (1 + gdp_growth)
                delta_gdp = new_gdp - gdp
                delta_consumption = delta_gdp * consumption_share_of_growth

                consumption += delta_consumption
                gdp = new_gdp
                share = consumption / gdp

                if share >= target_consumption_share and years_to_target is None:
                    years_to_target = yr
                    break

            scenarios.append({
                'gdp_growth_pct': gdp_growth * 100,
                'consumption_share_of_growth_pct': consumption_share_of_growth * 100,
                'years_to_65pct': years_to_target if years_to_target else 50,
                'implied_investment_growth': gdp_growth * (1 - consumption_share_of_growth) / (1 - current_consumption_share),
            })

    return pd.DataFrame(scenarios)


# --- International Benchmarks ---
def _build_benchmarks():
    """Build benchmarks dict with live World Bank data where available."""
    bench_consumption = fetch_benchmarks_consumption()
    bench_investment = fetch_benchmarks_investment()

    countries = list(bench_consumption.keys())
    hh_vals = [bench_consumption[c] for c in countries]
    inv_vals = [bench_investment.get(c, 25.0) for c in countries]

    # Add China entries and global average (always from model)
    countries += ['China (official)', 'China (Pettis adj.)', 'Global Average']
    hh_vals += [38.8, 36.0, 58.0]
    inv_vals += [42.2, 44.0, 25.0]

    return {
        'country': countries,
        'hh_consumption_pct': hh_vals,
        'investment_pct': inv_vals,
    }

BENCHMARKS = _build_benchmarks()


def plot_structural_imbalance(df):
    """Multi-panel visualization of China's structural imbalances."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: Consumption vs Investment over time ---
    ax = axes[0, 0]
    ax.plot(df['year'], df['hh_consumption_pct'], 'b-', linewidth=2.5,
            label='Household Consumption / GDP')
    ax.plot(df['year'], df['investment_pct'], 'r-', linewidth=2.5,
            label='Investment / GDP')
    ax.axhline(y=58, color='blue', linestyle=':', alpha=0.5, label='Global avg consumption (58%)')
    ax.axhline(y=25, color='red', linestyle=':', alpha=0.5, label='Global avg investment (25%)')
    ax.fill_between(df['year'], df['hh_consumption_pct'], 58,
                     where=[c < 58 for c in df['hh_consumption_pct']],
                     alpha=0.15, color='blue', label='Consumption gap vs world')
    ax.fill_between(df['year'], df['investment_pct'], 25,
                     where=[i > 25 for i in df['investment_pct']],
                     alpha=0.15, color='red', label='Investment excess vs world')
    ax.set_ylabel('%  of GDP', fontsize=11)
    ax.set_title('Pettis Framework: Consumption vs. Investment\n(The Core Imbalance)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='center right')
    ax.set_ylim(20, 65)
    ax.grid(alpha=0.3)

    # --- Panel 2: Financial Repression ---
    ax = axes[0, 1]
    colors_fr = ['#d73027' if r < 0 else '#4575b4' for r in df['real_deposit_rate_bps']]
    ax.bar(df['year'], df['real_deposit_rate_bps'], color=colors_fr, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=-500, color='red', linestyle='--', alpha=0.4,
               label='Pettis: rates 500-800bps too low')
    ax.set_ylabel('Basis Points', fontsize=11)
    ax.set_title('Financial Repression: Real Deposit Rate\n(Negative = Hidden Tax on Savers)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 3: ICOR (Investment Efficiency) ---
    ax = axes[1, 0]
    icor_clean = df.dropna(subset=['icor'])
    icor_clean = icor_clean[icor_clean['icor'] < 30]  # filter outliers
    ax.bar(icor_clean['year'], icor_clean['icor'], color='#d6604d', alpha=0.7)
    ax.axhline(y=3.5, color='green', linestyle='--', linewidth=1.5,
               label='Healthy ICOR (~3.5)')
    ax.axhline(y=7, color='red', linestyle='--', linewidth=1.5,
               label='Wasteful investment threshold (~7)')
    ax.set_ylabel('ICOR (Investment / delta-GDP)', fontsize=11)
    ax.set_title('Investment Efficiency: ICOR\n(Higher = More Wasteful Investment)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 25)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 4: International Comparison ---
    ax = axes[1, 1]
    bench = pd.DataFrame(BENCHMARKS)
    bench_sorted = bench.sort_values('hh_consumption_pct', ascending=True)

    y_pos = range(len(bench_sorted))
    colors_bar = []
    for c in bench_sorted['country']:
        if 'China' in c:
            colors_bar.append('#d73027')
        elif c == 'Global Average':
            colors_bar.append('#4575b4')
        else:
            colors_bar.append('#92c5de')

    ax.barh(list(y_pos), bench_sorted['hh_consumption_pct'], color=colors_bar, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(bench_sorted['country'], fontsize=9)
    ax.set_xlabel('Household Consumption as % of GDP', fontsize=11)
    ax.set_title('International Comparison\n(China is a massive outlier)',
                 fontsize=12, fontweight='bold')
    ax.axvline(x=58, color='navy', linestyle=':', alpha=0.5, label='Global avg')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.suptitle("China Structural Imbalance Model (Pettis Framework)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('structural_imbalance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: structural_imbalance.png")


def print_pettis_summary(df):
    """Print the core Pettis metrics."""
    latest = df[df.year == 2025].iloc[0]
    peak_inv = df.loc[df['investment_pct'].idxmax()]

    print("=" * 65)
    print("STRUCTURAL IMBALANCE SUMMARY (Pettis Framework) - 2025")
    print("=" * 65)
    print(f"  Household Consumption / GDP:  {latest.hh_consumption_pct:>6.1f}%  (global avg: 58%)")
    print(f"  Investment / GDP:             {latest.investment_pct:>6.1f}%  (global avg: 25%)")
    print(f"  Net Exports / GDP:            {latest.net_exports_pct:>6.1f}%")
    print(f"  Household Income / GDP:       {latest.hh_income_share_pct:>6.1f}%  (global avg: 60-65%)")
    print(f"  Real Deposit Rate:            {latest.real_deposit_rate_bps:>+5.0f} bps")
    print(f"  {'-' * 40}")
    print(f"  Consumption Gap vs World:     {58 - latest.hh_consumption_pct:>6.1f} pp")
    print(f"  Investment Excess vs World:   {latest.investment_pct - 25:>6.1f} pp")
    print(f"  Peak Investment Year:         {int(peak_inv.year)} ({peak_inv.investment_pct:.1f}%)")
    print("=" * 65)
    print()
    print("Pettis thesis: China's GDP growth is overstated because a large")
    print("share of investment creates no economic value. The true productive")
    print("GDP is significantly lower. Rebalancing requires transferring")
    print("income back to households, but this means picking losers among")
    print("SOEs and local governments.")


def print_rebalancing_scenarios():
    """Print Pettis's rebalancing math."""
    scenarios = build_rebalancing_scenarios()

    print("\n" + "=" * 65)
    print("REBALANCING SCENARIOS (Pettis Framework)")
    print("Years to reach 65% total consumption share of GDP")
    print("=" * 65)

    pivot = scenarios.pivot_table(
        index='gdp_growth_pct',
        columns='consumption_share_of_growth_pct',
        values='years_to_65pct'
    )

    print(f"\n{'GDP Growth ->':>20}", end='')
    for col in pivot.columns:
        print(f"  C={col:.0f}%", end='')
    print()
    print("-" * 55)
    for idx, row in pivot.iterrows():
        print(f"  GDP grows {idx:.0f}%/yr:   ", end='')
        for val in row:
            if val >= 50:
                print(f"  Never", end='')
            else:
                print(f"  {val:>3.0f} yr", end='')
        print()

    print("\n  C = consumption's share of each year's GDP growth")
    print("  'Never' = consumption share doesn't reach 65% within 50 years")
    print("  Key takeaway: Even at 5% GDP growth, consumption must claim")
    print("  70%+ of growth to rebalance within a generation.")


if __name__ == '__main__':
    df = build_structural_dataset()
    print_pettis_summary(df)
    print_rebalancing_scenarios()
    plot_structural_imbalance(df)
