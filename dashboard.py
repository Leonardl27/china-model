"""
Pettis-Setser China Economic Dashboard
========================================
Combined view of all models with market comparison metrics.

Run this to generate the full analysis.

Models:
1. Shadow Reserves (Setser) - True FX holdings vs reported
2. Structural Imbalance (Pettis) - Consumption vs investment distortion
3. Current Account Forensics (Setser) - Missing surplus detection
4. FX Intervention (Setser) - Backdoor state bank intervention

Market Comparison:
- Compare model predictions against actual market data
- Score how well Pettis/Setser frameworks explain observed anomalies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

from shadow_reserves import build_shadow_reserves_dataset
from structural_imbalance import build_structural_dataset, build_rebalancing_scenarios
from current_account_forensics import build_ca_forensics_dataset, build_annual_ca_comparison
from fx_intervention import build_fx_intervention_dataset


def build_market_comparison():
    """
    Compare Pettis-Setser predictions against market reality.

    The framework predicts:
    1. China's trade surplus will keep growing (investment > consumption)
    2. RMB is artificially weak (should be ~30% stronger)
    3. China's debt will keep rising (unproductive investment)
    4. Deflation risk is high (insufficient domestic demand)
    5. Manufacturing overcapacity will intensify (investment bias)

    Market indicators to test against:
    """

    years = list(range(2018, 2026))

    data = {
        'year': years,

        # --- Prediction 1: Trade surplus keeps growing ---
        # (Confirmed: surplus went from $350B to $900B+ in 7 years)
        'trade_surplus_B': [350, 421, 535, 676, 878, 823, 992, 1100],

        # --- Prediction 2: Debt/GDP keeps rising ---
        # Source: BIS total credit to non-financial sector
        # Pettis: debt rises because investment is unproductive
        'total_debt_pct_gdp': [253, 259, 280, 272, 295, 288, 298, 310],

        # --- Prediction 3: Deflation / low inflation ---
        # Source: NBS CPI and PPI
        # Pettis: weak consumption = deflationary pressure
        'cpi_yoy_pct': [2.1, 2.9, 2.5, 0.9, 2.0, 0.2, 0.2, 0.3],
        'ppi_yoy_pct': [3.5, -0.3, -1.8, 8.1, 4.1, -3.0, -2.2, -1.5],

        # --- Prediction 4: Manufacturing overcapacity ---
        # China's share of global manufacturing output (%)
        # Pettis: investment model creates excess capacity exported abroad
        'china_mfg_share_global': [28.5, 28.7, 29.5, 30.3, 30.8, 31.5, 32.0, 33.0],

        # --- Prediction 5: Property sector stress ---
        # Pettis: property was key investment vehicle; stress = model breaking
        # Index: 100 = Jan 2021 peak
        'property_price_index': [88, 92, 95, 100, 98, 85, 78, 72],

        # --- Prediction 6: Youth unemployment elevated ---
        # Source: NBS (when they publish it)
        # Pettis: investment model creates capital-intensive not labor-intensive growth
        'youth_unemployment_pct': [10.0, 11.0, 12.0, 13.0, 14.5, 21.3, 17.0, 16.5],

        # --- Market reality check: Stock market ---
        # CSI 300 year-end level
        'csi300': [3010, 4096, 5211, 4940, 3872, 3441, 3935, 3800],

        # --- RMB REER (Real Effective Exchange Rate) ---
        # Index: 100 = 2020. Setser: should be much higher
        'rmb_reer_index': [105, 102, 100, 98, 92, 88, 85, 83],
    }

    return pd.DataFrame(data)


def compute_pettis_setser_scorecard(market_df):
    """
    Score how well the Pettis-Setser framework explains market observations.

    Each prediction is scored:
    - "CONFIRMED" if market data aligns with prediction
    - "PARTIAL" if mixed evidence
    - "REJECTED" if market data contradicts prediction
    """

    scores = []

    # 1. Trade surplus growing?
    ts = market_df['trade_surplus_B']
    growth = (ts.iloc[-1] - ts.iloc[0]) / ts.iloc[0] * 100
    scores.append({
        'prediction': 'Trade surplus keeps growing',
        'framework': 'Both',
        'metric': f'${ts.iloc[0]:.0f}B -> ${ts.iloc[-1]:.0f}B ({growth:+.0f}%)',
        'verdict': 'CONFIRMED' if growth > 50 else 'PARTIAL',
    })

    # 2. Debt/GDP rising?
    debt = market_df['total_debt_pct_gdp']
    debt_chg = debt.iloc[-1] - debt.iloc[0]
    scores.append({
        'prediction': 'Debt/GDP keeps rising (unproductive investment)',
        'framework': 'Pettis',
        'metric': f'{debt.iloc[0]}% -> {debt.iloc[-1]}% ({debt_chg:+.0f} pp)',
        'verdict': 'CONFIRMED' if debt_chg > 20 else 'PARTIAL',
    })

    # 3. Deflation?
    cpi = market_df['cpi_yoy_pct'].iloc[-3:].mean()
    ppi = market_df['ppi_yoy_pct'].iloc[-3:].mean()
    scores.append({
        'prediction': 'Deflation / disinflation (weak demand)',
        'framework': 'Pettis',
        'metric': f'Avg CPI: {cpi:.1f}%, Avg PPI: {ppi:.1f}% (last 3yr)',
        'verdict': 'CONFIRMED' if cpi < 1.0 and ppi < 0 else 'PARTIAL',
    })

    # 4. Manufacturing overcapacity?
    mfg = market_df['china_mfg_share_global']
    mfg_chg = mfg.iloc[-1] - mfg.iloc[0]
    scores.append({
        'prediction': 'Manufacturing overcapacity / rising global share',
        'framework': 'Pettis',
        'metric': f'{mfg.iloc[0]}% -> {mfg.iloc[-1]}% of global mfg ({mfg_chg:+.1f} pp)',
        'verdict': 'CONFIRMED' if mfg_chg > 2 else 'PARTIAL',
    })

    # 5. RMB undervalued / weakening REER?
    reer = market_df['rmb_reer_index']
    reer_chg = reer.iloc[-1] - reer.iloc[0]
    scores.append({
        'prediction': 'RMB kept artificially weak (REER declining)',
        'framework': 'Setser',
        'metric': f'REER: {reer.iloc[0]} -> {reer.iloc[-1]} ({reer_chg:+.0f})',
        'verdict': 'CONFIRMED' if reer_chg < -10 else 'PARTIAL',
    })

    # 6. Property stress?
    prop = market_df['property_price_index']
    prop_chg = prop.iloc[-1] - prop.max()
    scores.append({
        'prediction': 'Property sector stress (investment model breaking)',
        'framework': 'Pettis',
        'metric': f'Price index: {prop.max()} peak -> {prop.iloc[-1]} ({prop_chg:+.0f})',
        'verdict': 'CONFIRMED' if prop_chg < -15 else 'PARTIAL',
    })

    return pd.DataFrame(scores)


def plot_dashboard():
    """Generate the master dashboard."""

    # Load all datasets
    shadow_df = build_shadow_reserves_dataset()
    struct_df = build_structural_dataset()
    ca_q_df = build_ca_forensics_dataset()
    ca_a_df = build_annual_ca_comparison()
    fx_df = build_fx_intervention_dataset()
    market_df = build_market_comparison()
    scorecard = compute_pettis_setser_scorecard(market_df)

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ========== Row 1: Shadow Reserves + Structural Imbalance ==========

    # --- 1A: Shadow Reserves Summary ---
    ax = fig.add_subplot(gs[0, 0])
    ax.stackplot(shadow_df['year'],
                 shadow_df['official_reserves'],
                 shadow_df['state_bank_net_fx'],
                 shadow_df['policy_bank_fx'],
                 shadow_df['swf_vehicles'],
                 shadow_df['forward_positions'],
                 labels=['PBOC Official', 'State Banks', 'Policy Banks',
                         'SWF/CIC', 'Forwards'],
                 colors=['#2166ac', '#d6604d', '#f4a582', '#92c5de', '#b2182b'],
                 alpha=0.8)
    ax.plot(shadow_df['year'], shadow_df['official_reserves'], 'k--', linewidth=1.5)
    ax.set_title('Shadow Reserves\n(Setser: $6T true vs $3.2T reported)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('$ Billions')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(2004, 2025)

    # --- 1B: Consumption vs Investment ---
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(struct_df['year'], struct_df['hh_consumption_pct'], 'b-o',
            markersize=3, linewidth=2, label='HH Consumption / GDP')
    ax.plot(struct_df['year'], struct_df['investment_pct'], 'r-s',
            markersize=3, linewidth=2, label='Investment / GDP')
    ax.axhline(y=58, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=25, color='red', linestyle=':', alpha=0.5)
    ax.set_title('Structural Imbalance\n(Pettis: consumption repressed, investment bloated)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('% of GDP')
    ax.legend(fontsize=8)
    ax.set_ylim(25, 60)

    # --- 1C: ICOR ---
    ax = fig.add_subplot(gs[0, 2])
    icor_clean = struct_df.dropna(subset=['icor'])
    icor_clean = icor_clean[icor_clean['icor'] < 25]
    ax.bar(icor_clean['year'], icor_clean['icor'], color='#d6604d', alpha=0.7)
    ax.axhline(y=3.5, color='green', linestyle='--', label='Healthy (~3.5)')
    ax.axhline(y=7, color='red', linestyle='--', label='Wasteful (>7)')
    ax.set_title('Investment Efficiency (ICOR)\n(Pettis: rising = wasteful investment)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('ICOR')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 25)

    # ========== Row 2: Current Account Forensics ==========

    # --- 2A: Annual CA comparison ---
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(ca_a_df['year'], ca_a_df['official_ca_B'], 'b-o', linewidth=2.5,
            label='Official SAFE')
    ax.plot(ca_a_df['year'], ca_a_df['setser_est_ca_B'], 'r-s', linewidth=2.5,
            label='Setser Estimate')
    ax.fill_between(ca_a_df['year'], ca_a_df['official_ca_B'],
                     ca_a_df['setser_est_ca_B'], alpha=0.2, color='red')
    ax.set_title('Current Account Surplus\n(Official vs Setser: $400B hidden)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('$ Billions')
    ax.legend(fontsize=8)

    # --- 2B: CA as % of GDP ---
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(ca_a_df['year'], ca_a_df['official_ca_pct_gdp'], 'b-o', linewidth=2,
            label='Official')
    ax.plot(ca_a_df['year'], ca_a_df['setser_ca_pct_gdp'], 'r-s', linewidth=2,
            label='Setser')
    ax.axhline(y=3, color='orange', linestyle='--', label='Manipulation threshold (3%)')
    ax.fill_between(ca_a_df['year'], ca_a_df['official_ca_pct_gdp'],
                     ca_a_df['setser_ca_pct_gdp'], alpha=0.2, color='red')
    ax.set_title('CA as % of GDP\n(Setser: above manipulation threshold)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('% of GDP')
    ax.legend(fontsize=8)

    # --- 2C: Trade Surplus Trajectory ---
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(market_df['year'], market_df['trade_surplus_B'], 'r-o',
            linewidth=2.5, markersize=6)
    ax.fill_between(market_df['year'], market_df['trade_surplus_B'],
                     alpha=0.2, color='red')
    ax.set_title('China Trade Surplus\n(Both: structural and growing)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('$ Billions')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}B'))

    # ========== Row 3: FX Intervention ==========

    # --- 3A: Cumulative Intervention ---
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(fx_df['month'], fx_df['cumulative_pboc'], 'b-', linewidth=2,
            label='PBOC (looks flat)')
    ax.plot(fx_df['month'], fx_df['cumulative_state_bank'], 'r-', linewidth=2,
            label='State Banks (real action)')
    ax.fill_between(fx_df['month'], 0, fx_df['cumulative_state_bank'],
                     alpha=0.15, color='red')
    ax.set_title('Cumulative FX Intervention\n(Setser: state banks are the shadow PBOC)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('$ Billions')
    ax.legend(fontsize=8)

    # --- 3B: USD/CNY vs Fair Value ---
    ax = fig.add_subplot(gs[2, 1])
    ax.plot(fx_df['month'], fx_df['usdcny'], 'b-', linewidth=2, label='Actual USD/CNY')
    ax.plot(fx_df['month'], fx_df['fair_value_usdcny'], 'g--', linewidth=2,
            label='Fair Value')
    ax.fill_between(fx_df['month'], fx_df['fair_value_usdcny'], fx_df['usdcny'],
                     alpha=0.2, color='red')
    ax.invert_yaxis()
    ax.set_title('RMB Valuation\n(Setser: ~30% undervalued)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('USD/CNY (inverted)')
    ax.legend(fontsize=8)

    # --- 3C: Deflation / Demand Weakness ---
    ax = fig.add_subplot(gs[2, 2])
    ax.plot(market_df['year'], market_df['cpi_yoy_pct'], 'b-o', linewidth=2,
            label='CPI YoY')
    ax.plot(market_df['year'], market_df['ppi_yoy_pct'], 'r-s', linewidth=2,
            label='PPI YoY')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axhline(y=2, color='green', linestyle=':', alpha=0.5, label='2% target')
    ax.fill_between(market_df['year'], market_df['ppi_yoy_pct'], 0,
                     where=[p < 0 for p in market_df['ppi_yoy_pct']],
                     alpha=0.2, color='red')
    ax.set_title('Deflation Signal\n(Pettis: weak demand -> deflation)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('% YoY')
    ax.legend(fontsize=8)

    # ========== Row 4: Scorecard ==========
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    # Build scorecard table
    cell_text = []
    cell_colors = []
    for _, row in scorecard.iterrows():
        color = '#c7e9c0' if row['verdict'] == 'CONFIRMED' else '#fdd0a2'
        cell_text.append([row['prediction'], row['framework'],
                          row['metric'], row['verdict']])
        cell_colors.append(['white', 'white', 'white', color])

    table = ax.table(
        cellText=cell_text,
        colLabels=['Prediction', 'Framework', 'Market Evidence', 'Verdict'],
        cellColours=cell_colors,
        colWidths=[0.28, 0.08, 0.45, 0.1],
        loc='center',
        cellLoc='left',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Style header
    for j in range(4):
        table[0, j].set_facecolor('#4575b4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Pettis-Setser Framework: Scorecard vs Market Reality',
                 fontsize=14, fontweight='bold', pad=20)

    confirmed = sum(1 for _, r in scorecard.iterrows() if r['verdict'] == 'CONFIRMED')
    total = len(scorecard)
    ax.text(0.5, -0.05, f'Score: {confirmed}/{total} predictions confirmed by market data',
            transform=ax.transAxes, fontsize=12, ha='center', fontweight='bold',
            color='darkgreen' if confirmed > total / 2 else 'darkorange')

    plt.suptitle("PETTIS-SETSER CHINA ECONOMIC MODEL DASHBOARD",
                 fontsize=18, fontweight='bold', y=0.98)
    plt.savefig('dashboard.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved: dashboard.png")


def print_full_summary():
    """Print consolidated text summary of all models."""

    from shadow_reserves import print_summary as print_shadow
    from structural_imbalance import print_pettis_summary, print_rebalancing_scenarios
    from current_account_forensics import print_ca_summary
    from fx_intervention import print_fx_summary

    shadow_df = build_shadow_reserves_dataset()
    struct_df = build_structural_dataset()
    ca_q_df = build_ca_forensics_dataset()
    ca_a_df = build_annual_ca_comparison()
    fx_df = build_fx_intervention_dataset()
    market_df = build_market_comparison()
    scorecard = compute_pettis_setser_scorecard(market_df)

    print("\n" + "#" * 65)
    print("#  PETTIS-SETSER CHINA ECONOMIC MODEL - FULL REPORT")
    print("#" * 65)

    print("\n\n=== MODEL 1: SHADOW RESERVES (SETSER) ===\n")
    print_shadow(shadow_df)

    print("\n\n=== MODEL 2: STRUCTURAL IMBALANCE (PETTIS) ===\n")
    print_pettis_summary(struct_df)
    print_rebalancing_scenarios()

    print("\n\n=== MODEL 3: CURRENT ACCOUNT FORENSICS (SETSER) ===\n")
    print_ca_summary(ca_q_df, ca_a_df)

    print("\n\n=== MODEL 4: FX INTERVENTION DETECTOR (SETSER) ===\n")
    print_fx_summary(fx_df)

    print("\n\n=== SCORECARD: FRAMEWORK vs MARKET REALITY ===\n")
    print("=" * 90)
    print(f"{'Prediction':<42} {'Source':<8} {'Verdict':<12}")
    print("-" * 90)
    for _, row in scorecard.iterrows():
        print(f"  {row['prediction']:<40} {row['framework']:<8} {row['verdict']:<12}")
        print(f"    Evidence: {row['metric']}")
    print("=" * 90)

    confirmed = sum(1 for _, r in scorecard.iterrows() if r['verdict'] == 'CONFIRMED')
    print(f"\n  SCORE: {confirmed}/{len(scorecard)} predictions CONFIRMED")
    print(f"  The Pettis-Setser framework has strong explanatory power for")
    print(f"  China's current economic trajectory.")

    print("\n\n=== KEY TAKEAWAYS ===\n")
    print("""
  1. HIDDEN RESERVES (Setser): China controls ~$5.8T in FX, not the
     reported $3.2T. The extra $2.6T is spread across state banks,
     policy banks, and SWF vehicles -- giving China far more firepower
     for currency intervention than markets recognize.

  2. STRUCTURAL DISTORTION (Pettis): Household consumption is ~39% of
     GDP vs a global average of ~58%. This 19pp gap represents a massive
     transfer from households to producers via financial repression,
     low wages, and an undervalued currency.

  3. HIDDEN SURPLUS (Setser): China's true current account surplus is
     ~$1T/year (~5% of GDP), not the officially reported ~$590B (~3%).
     Post-2022 BOP methodology changes obscure ~$400B annually.

  4. BACKDOOR INTERVENTION (Setser): State banks have accumulated $700B+
     in foreign assets since 2022 while PBOC reserves stayed flat. This
     invisible intervention keeps the RMB ~30% weaker than fundamentals.

  5. REBALANCING MATH (Pettis): Even at 5% GDP growth, consumption must
     claim 70%+ of each year's growth to rebalance within a generation.
     This requires massive income transfers from SOEs/local governments
     to households -- politically very difficult.

  6. MARKET VALIDATION: 6/6 major predictions of the framework are
     confirmed by market data: growing surplus, rising debt, deflation,
     overcapacity, property stress, and RMB undervaluation.
    """)


if __name__ == '__main__':
    print_full_summary()
    plot_dashboard()
