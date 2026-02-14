"""
Current Account Forensics
==========================
Based on Brad Setser's methodology for detecting discrepancies between
China's customs-reported trade surplus and BOP-reported current account.

Key finding: Starting in 2022, China introduced a new BOP methodology
that creates an "unexplained variance" -- the customs surplus is much larger
than the BOP goods surplus, and the gap lacks clear IMF justification.

Setser estimates China's TRUE current account surplus is ~$1 trillion (5% of GDP),
far above the officially reported ~$400B (2% of GDP).

This model tracks:
1. Customs trade surplus vs BOP goods surplus (the "missing surplus")
2. Current account surplus: official vs Setser-adjusted
3. Services balance cross-checking
4. Quarterly patterns of data manipulation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data_fetcher import (
    fetch_trade_balance_quarterly,
    fetch_official_ca_annual,
    fetch_china_gdp_T_annual,
    DEFAULTS,
    merge_into_list,
)


def build_ca_forensics_dataset():
    """
    Quarterly current account forensics data.

    Sources:
    - China Customs: monthly trade data (goods exports - imports)
    - SAFE BOP: quarterly current account releases
    - Setser's CFR blog analysis of the gap
    """

    # Quarterly data from Q1 2019 through Q2 2025
    quarters = []
    for y in range(2019, 2026):
        for q in range(1, 5):
            if y == 2025 and q > 2:
                break
            quarters.append(f"{y}Q{q}")

    # --- Customs Goods Trade Surplus ($ billions per quarter) ---
    # Source: FRED XTNTVA01CNM667S aggregated to quarterly (live) / China Customs (fallback)
    _hardcoded_customs = DEFAULTS["customs_surplus_q"]
    _live_trade = fetch_trade_balance_quarterly(start_year=2019)
    if _live_trade is not None and len(_live_trade) > 0:
        # Map live quarterly data into our quarter labels
        customs_surplus = list(_hardcoded_customs)  # start with defaults
        for i, q_label in enumerate(quarters):
            yr = int(q_label[:4])
            qn = int(q_label[-1])
            # FRED quarterly dates are quarter start: YYYY-01-01, YYYY-04-01, etc.
            q_start = f"{yr}-{(qn-1)*3+1:02d}-01"
            if q_start in _live_trade.index:
                customs_surplus[i] = round(float(_live_trade[q_start]), 0)
    else:
        customs_surplus = list(_hardcoded_customs)

    # --- BOP Goods Surplus ($ billions per quarter) ---
    # Source: SAFE BOP releases
    # Post-2022 methodology creates systematic gap vs customs
    bop_goods_surplus = [
        72, 110, 120, 90,           # 2019 Q1-Q4: close to customs
        24, 132, 155, 142,          # 2020 Q1-Q4: close to customs
        113, 130, 155, 172,         # 2021 Q1-Q4: close to customs
        148, 175, 195, 170,         # 2022 Q1-Q4: gap emerges with new methodology
        155, 175, 190, 168,         # 2023 Q1-Q4: systematic gap
        165, 185, 210, 208,         # 2024 Q1-Q4: gap persists
        210, 210,                    # 2025 Q1-Q2: Q2 shows $90B gap (Setser)
    ]

    # --- Services Balance ($ billions per quarter) ---
    # Source: SAFE BOP
    services_balance = [
        -62, -65, -60, -55,         # 2019 Q1-Q4
        -35, -30, -28, -32,         # 2020 Q1-Q4: COVID collapses travel
        -20, -22, -25, -28,         # 2021 Q1-Q4
        -18, -20, -22, -25,         # 2022 Q1-Q4
        -30, -45, -55, -50,         # 2023 Q1-Q4: travel resumes
        -50, -60, -60, -59,         # 2024 Q1-Q4: -$229B annual
        -50, -56,                    # 2025 Q1-Q2
    ]

    # --- Primary Income Balance ($ billions per quarter) ---
    # Source: SAFE BOP
    primary_income = [
        -15, -18, -20, -15,         # 2019
        -12, -15, -18, -15,         # 2020
        -18, -22, -25, -20,         # 2021
        -20, -25, -30, -28,         # 2022
        -25, -30, -38, -39,         # 2023: -$132B annual
        -28, -35, -35, -34,         # 2024
        -30, -33,                    # 2025
    ]

    # --- Official Current Account Surplus ($ billions per quarter) ---
    # Source: SAFE BOP releases
    official_ca = [
        -5, 32, 45, 23,             # 2019 Q1-Q4
        -20, 90, 112, 98,           # 2020 Q1-Q4
        80, 90, 110, 130,           # 2021 Q1-Q4
        114, 132, 148, 120,         # 2022 Q1-Q4
        105, 105, 102, 95,          # 2023 Q1-Q4: mysteriously low vs customs
        90, 50, 120, 162,           # 2024 Q1-Q4: Q2 = only $50B on $250B customs!
        165, 135,                    # 2025 Q1-Q2
    ]

    df = pd.DataFrame({
        'quarter': quarters,
        'customs_surplus': customs_surplus,
        'bop_goods_surplus': bop_goods_surplus,
        'services_balance': services_balance,
        'primary_income': primary_income,
        'official_ca': official_ca,
    })

    # Compute key forensic metrics
    df['customs_bop_gap'] = df['customs_surplus'] - df['bop_goods_surplus']
    df['expected_ca'] = (df['customs_surplus'] + df['services_balance']
                         + df['primary_income'])
    df['ca_gap'] = df['expected_ca'] - df['official_ca']  # "missing" surplus
    df['gap_pct'] = df['ca_gap'] / df['expected_ca'].clip(lower=1) * 100

    # Setser-adjusted CA (use customs-based estimate)
    df['setser_adj_ca'] = df['expected_ca']

    # Year column for annual aggregation
    df['year'] = [q[:4] for q in quarters]

    return df


def build_annual_ca_comparison():
    """
    Annual current account: Official vs Setser estimate vs IMF forecast.
    """

    years = list(range(2018, 2026))

    # Fetch live data where available, fall back to hardcoded
    official_ca_B = fetch_official_ca_annual(years)
    china_gdp_T = fetch_china_gdp_T_annual(years)

    data = {
        'year': years,
        'official_ca_B': official_ca_B,
        # Setser estimates and IMF forecasts are model/analytical - remain hardcoded
        'setser_est_ca_B': [
            24,     # 2018
            110,    # 2019
            290,    # 2020
            340,    # 2021
            480,    # 2022
            550,    # 2023 - Setser: much higher than official
            750,    # 2024 - Setser: ~$750B true surplus
            1000,   # 2025 - Setser: approaching $1T
        ],
        'imf_forecast_B': [
            24,     # 2018
            103,    # 2019
            274,    # 2020
            317,    # 2021
            402,    # 2022
            264,    # 2023
            422,    # 2024
            365,    # 2025 - IMF spring 2025 forecast (naive)
        ],
        'china_gdp_T': china_gdp_T,
    }

    df = pd.DataFrame(data)
    df['official_ca_pct_gdp'] = df['official_ca_B'] / (df['china_gdp_T'] * 10)  # in % of GDP
    df['setser_ca_pct_gdp'] = df['setser_est_ca_B'] / (df['china_gdp_T'] * 10)
    df['imf_ca_pct_gdp'] = df['imf_forecast_B'] / (df['china_gdp_T'] * 10)

    return df


def plot_ca_forensics(df_q, df_a):
    """Visualize current account discrepancies."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Panel 1: Customs vs BOP Goods Surplus (quarterly) ---
    ax = axes[0, 0]
    x = range(len(df_q))
    ax.bar([i - 0.15 for i in x], df_q['customs_surplus'], width=0.3,
           color='#2166ac', alpha=0.8, label='Customs Goods Surplus')
    ax.bar([i + 0.15 for i in x], df_q['bop_goods_surplus'], width=0.3,
           color='#d6604d', alpha=0.8, label='BOP Goods Surplus')

    # Show every 4th label
    tick_idx = list(range(0, len(df_q), 4))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([df_q['quarter'].iloc[i] for i in tick_idx], rotation=45, fontsize=8)
    ax.set_ylabel('$ Billions', fontsize=11)
    ax.set_title('Customs vs BOP Goods Surplus\n(Gap = data manipulation)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # --- Panel 2: The "Missing" Surplus (quarterly) ---
    ax = axes[0, 1]
    colors_gap = ['#d73027' if g > 10 else '#4575b4' for g in df_q['ca_gap']]
    ax.bar(x, df_q['ca_gap'], color=colors_gap, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=1)
    tick_idx = list(range(0, len(df_q), 4))
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([df_q['quarter'].iloc[i] for i in tick_idx], rotation=45, fontsize=8)
    ax.set_ylabel('$ Billions', fontsize=11)
    ax.set_title('"Missing" Current Account Surplus\n(Expected CA minus Official CA)',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Annotate 2024 Q2
    q2_2024_idx = list(df_q['quarter']).index('2024Q2')
    ax.annotate(f'2024Q2: ${df_q.iloc[q2_2024_idx]["ca_gap"]:.0f}B\n"missing"',
                xy=(q2_2024_idx, df_q.iloc[q2_2024_idx]['ca_gap']),
                xytext=(q2_2024_idx - 5, df_q.iloc[q2_2024_idx]['ca_gap'] + 30),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontweight='bold')

    # --- Panel 3: Annual CA - Official vs Setser vs IMF ---
    ax = axes[1, 0]
    ax.plot(df_a['year'], df_a['official_ca_B'], 'b-o', linewidth=2.5,
            markersize=8, label='Official SAFE Report')
    ax.plot(df_a['year'], df_a['setser_est_ca_B'], 'r-s', linewidth=2.5,
            markersize=8, label='Setser Estimate (customs-based)')
    ax.plot(df_a['year'], df_a['imf_forecast_B'], 'g--^', linewidth=1.5,
            markersize=6, label='IMF Forecast', alpha=0.7)
    ax.fill_between(df_a['year'], df_a['official_ca_B'], df_a['setser_est_ca_B'],
                     alpha=0.2, color='red', label='Hidden surplus')
    ax.set_ylabel('$ Billions', fontsize=11)
    ax.set_title('Annual Current Account: Three Views\n(Official vs Setser vs IMF)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # --- Panel 4: CA as % of GDP ---
    ax = axes[1, 1]
    ax.plot(df_a['year'], df_a['official_ca_pct_gdp'], 'b-o', linewidth=2.5,
            label='Official (% GDP)')
    ax.plot(df_a['year'], df_a['setser_ca_pct_gdp'], 'r-s', linewidth=2.5,
            label='Setser Est. (% GDP)')
    ax.axhline(y=3, color='orange', linestyle='--', linewidth=1.5,
               label='US Treasury "manipulation" threshold (3%)')
    ax.fill_between(df_a['year'], df_a['official_ca_pct_gdp'],
                     df_a['setser_ca_pct_gdp'],
                     alpha=0.2, color='red')
    ax.set_ylabel('% of GDP', fontsize=11)
    ax.set_title('Current Account as % of GDP\n(Setser: China is above manipulation threshold)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("Current Account Forensics (Setser Framework)",
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('current_account_forensics.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: current_account_forensics.png")


def print_ca_summary(df_q, df_a):
    """Print CA forensics summary."""

    latest_annual = df_a[df_a.year == 2025].iloc[0]

    print("=" * 65)
    print("CURRENT ACCOUNT FORENSICS (Setser Framework) - 2025")
    print("=" * 65)
    print(f"  Official CA Surplus:          ${latest_annual.official_ca_B:>7,.0f}B")
    print(f"  Setser Estimated CA:          ${latest_annual.setser_est_ca_B:>7,.0f}B")
    print(f"  IMF Forecast:                 ${latest_annual.imf_forecast_B:>7,.0f}B")
    print(f"  {'-' * 40}")
    print(f"  Official CA / GDP:            {latest_annual.official_ca_pct_gdp:>6.1f}%")
    print(f"  Setser CA / GDP:              {latest_annual.setser_ca_pct_gdp:>6.1f}%")
    print(f"  US Treasury threshold:         3.0%")
    print(f"  {'-' * 40}")
    print(f"  Hidden surplus:               ${latest_annual.setser_est_ca_B - latest_annual.official_ca_B:>7,.0f}B")
    print("=" * 65)
    print()
    print("Key Setser finding: China's customs data shows ~$1T surplus but")
    print("BOP methodology 'adjustments' reduce it to ~$590B officially.")
    print("The ~$400B gap has no clear IMF methodological justification.")
    print("If the true CA surplus is ~5% of GDP, China clearly exceeds the")
    print("US Treasury's 3% threshold for currency manipulation.")


if __name__ == '__main__':
    df_q = build_ca_forensics_dataset()
    df_a = build_annual_ca_comparison()
    print_ca_summary(df_q, df_a)
    plot_ca_forensics(df_q, df_a)
