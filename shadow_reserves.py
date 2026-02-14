"""
Shadow Reserves Tracker
========================
Based on Brad Setser's framework for estimating China's TRUE foreign exchange
holdings vs. the official PBOC-reported reserves.

Key insight: China moved ~$3 trillion off the PBOC balance sheet into state banks,
policy banks, and sovereign wealth vehicles starting in 2010. The official $3.2T
in PBOC reserves is only HALF the story.

Vehicles tracked:
- PBOC official reserves (reported by SAFE)
- State commercial banks foreign assets (BOC, ICBC, CCB, ABC)
- Policy banks (CDB, Ex-Im Bank) - entrusted loans from SAFE
- CIC / SAFE co-investment vehicles (Silk Road Fund, etc.)
- Forward/swap positions (off-balance-sheet)

Data sources: PBOC balance sheet, bank financial statements, SAFE annual reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from data_fetcher import fetch_official_reserves


def build_shadow_reserves_dataset():
    """
    Construct time series of official vs. estimated true reserves.

    Data assembled from:
    - FRED TRESEGCNM052N (live via data_fetcher, with hardcoded fallback)
    - Setser's estimates from CFR "Follow the Money" blog
    - State bank annual reports (foreign asset lines)
    - SAFE annual report disclosures on entrusted lending
    """

    years = list(range(2004, 2026))

    # --- Official PBOC Reserves ($ billions) ---
    # Source: FRED TRESEGCNM052N (live) / SAFE monthly releases (fallback)
    official_reserves = fetch_official_reserves(years)

    # --- State Commercial Bank Foreign Assets ($ billions) ---
    # Source: Bank annual reports, Setser estimates
    # BOC + ICBC + CCB + ABC combined foreign assets minus foreign liabilities
    # Key: These are funded domestically (Chinese depositors' USD deposits)
    # which Setser argues function as "invisible reserves"
    state_bank_net_foreign = [
        80,    # 2004 - pre-recapitalization
        150,   # 2005 - after PBOC injected $60B into banks
        200,   # 2006
        280,   # 2007
        350,   # 2008
        400,   # 2009
        500,   # 2010 - diversification begins
        600,   # 2011
        680,   # 2012
        750,   # 2013
        800,   # 2014
        850,   # 2015
        900,   # 2016 - state banks deployed to defend RMB
        920,   # 2017
        880,   # 2018
        850,   # 2019
        900,   # 2020
        950,   # 2021
        1000,  # 2022
        1050,  # 2023
        1160,  # 2024 - $110B added per Setser Dec 2024 data
        1250,  # 2025 (est, continued accumulation)
    ]

    # --- Policy Bank & BRI Lending ($ billions) ---
    # CDB + Ex-Im Bank entrusted loans from SAFE
    # Source: SAFE annual reports, Setser's reverse-engineering
    # In 2015, $93B of entrusted loans were converted to equity
    policy_bank_fx = [
        10,    # 2004
        20,    # 2005
        30,    # 2006
        50,    # 2007
        80,    # 2008
        120,   # 2009 - BRI precursors begin
        180,   # 2010
        250,   # 2011
        320,   # 2012
        400,   # 2013
        480,   # 2014
        500,   # 2015 - $93B converted to equity
        520,   # 2016
        550,   # 2017
        570,   # 2018
        580,   # 2019
        570,   # 2020 - BRI lending slows
        560,   # 2021
        550,   # 2022
        540,   # 2023
        530,   # 2024
        520,   # 2025 (est)
    ]

    # --- Sovereign Wealth / SAFE Co-Investment Vehicles ($ billions) ---
    # CIC, Silk Road Fund, China-Africa Fund, China-LAC Fund, CNIC
    # Source: Vehicle disclosures, Setser reconstruction
    # Silk Road Fund: $40B (65% from SAFE = $26B)
    # China-Africa: $10B (80% SAFE = $8B)
    # China-LAC: $30B (85% SAFE = $25.5B)
    # CIC International: ~$11B initial + growth
    swf_vehicles = [
        0,     # 2004
        0,     # 2005
        0,     # 2006
        200,   # 2007 - CIC established ($200B)
        220,   # 2008
        300,   # 2009
        380,   # 2010
        410,   # 2011
        480,   # 2012
        530,   # 2013
        560,   # 2014
        600,   # 2015 - Silk Road Fund launched
        620,   # 2016
        640,   # 2017
        650,   # 2018
        640,   # 2019
        660,   # 2020
        680,   # 2021
        670,   # 2022
        700,   # 2023
        720,   # 2024
        740,   # 2025 (est)
    ]

    # --- Forward/Swap Positions ($ billions) ---
    # Off-balance-sheet FX commitments
    # Source: PBOC FX settlement data adjusted for forwards
    # Setser notes Dec 2024: $120B when adjusted for forwards vs $100B spot
    forward_positions = [
        0, 0, 0, 0, 0, 0,  # 2004-2009: minimal
        20,   # 2010
        30,   # 2011
        40,   # 2012
        50,   # 2013
        80,   # 2014
        150,  # 2015 - heavy forward selling to defend RMB
        120,  # 2016
        80,   # 2017
        60,   # 2018
        50,   # 2019
        40,   # 2020
        30,   # 2021
        50,   # 2022
        60,   # 2023
        80,   # 2024
        100,  # 2025 (est)
    ]

    df = pd.DataFrame({
        'year': years,
        'official_reserves': official_reserves,
        'state_bank_net_fx': state_bank_net_foreign,
        'policy_bank_fx': policy_bank_fx,
        'swf_vehicles': swf_vehicles,
        'forward_positions': forward_positions,
    })

    df['total_shadow'] = (df['state_bank_net_fx'] + df['policy_bank_fx']
                          + df['swf_vehicles'] + df['forward_positions'])
    df['total_true_reserves'] = df['official_reserves'] + df['total_shadow']
    df['shadow_ratio'] = df['total_shadow'] / df['total_true_reserves']
    df['official_share'] = df['official_reserves'] / df['total_true_reserves']

    return df


def plot_shadow_reserves(df):
    """Stacked area chart: official vs shadow reserve components."""

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})

    # --- Top panel: Stacked reserves ---
    ax = axes[0]

    components = ['official_reserves', 'state_bank_net_fx', 'policy_bank_fx',
                  'swf_vehicles', 'forward_positions']
    labels = ['PBOC Official Reserves', 'State Bank Net Foreign Assets',
              'Policy Bank FX (CDB/Ex-Im)', 'SWF Vehicles (CIC etc.)',
              'Forward/Swap Positions']
    colors = ['#2166ac', '#d6604d', '#f4a582', '#92c5de', '#b2182b']

    ax.stackplot(df['year'],
                 [df[c] for c in components],
                 labels=labels, colors=colors, alpha=0.85)

    # Overlay official reserves line for contrast
    ax.plot(df['year'], df['official_reserves'], 'k--', linewidth=2,
            label='Official Reserves (what China reports)')
    ax.plot(df['year'], df['total_true_reserves'], 'w-', linewidth=2.5)
    ax.plot(df['year'], df['total_true_reserves'], 'k-', linewidth=1.5,
            label='Setser Estimated TRUE Reserves')

    ax.set_ylabel('USD Billions', fontsize=12)
    ax.set_title("China's Shadow Reserves: Official vs. Estimated True FX Holdings\n"
                 "(Setser Framework)", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'${x:,.0f}B'))
    ax.set_xlim(2004, 2025)
    ax.grid(axis='y', alpha=0.3)

    # Annotations
    ax.annotate('2015: Capital flight\nPBOC burns $800B+\nbut shadow reserves\nabsorb pressure',
                xy=(2015, 3330), xytext=(2008, 4500),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red', fontweight='bold')

    ax.annotate(f'2025: Official = $3.2T\nTrue ~ ${df[df.year==2025]["total_true_reserves"].values[0]/1000:.1f}T',
                xy=(2025, df[df.year==2025]['total_true_reserves'].values[0]),
                xytext=(2020, 6500),
                arrowprops=dict(arrowstyle='->', color='darkblue'),
                fontsize=10, color='darkblue', fontweight='bold')

    # --- Bottom panel: Shadow share ---
    ax2 = axes[1]
    ax2.fill_between(df['year'], df['shadow_ratio'] * 100, alpha=0.4, color='#d6604d')
    ax2.plot(df['year'], df['shadow_ratio'] * 100, color='#d6604d', linewidth=2)
    ax2.set_ylabel('Shadow Share (%)', fontsize=11)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_title('Share of True Reserves Held Outside PBOC', fontsize=11)
    ax2.set_ylim(0, 60)
    ax2.set_xlim(2004, 2025)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=50, color='red', linestyle=':', alpha=0.5, label='50% threshold')
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('shadow_reserves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: shadow_reserves.png")


def print_summary(df):
    """Print key shadow reserve metrics."""
    latest = df[df.year == 2025].iloc[0]

    print("=" * 65)
    print("SHADOW RESERVES SUMMARY (Setser Framework) - 2025 Estimate")
    print("=" * 65)
    print(f"  PBOC Official Reserves:      ${latest.official_reserves:>7,.0f}B")
    print(f"  State Bank Net Foreign:       ${latest.state_bank_net_fx:>7,.0f}B")
    print(f"  Policy Bank FX:               ${latest.policy_bank_fx:>7,.0f}B")
    print(f"  SWF Vehicles:                 ${latest.swf_vehicles:>7,.0f}B")
    print(f"  Forward/Swap Positions:       ${latest.forward_positions:>7,.0f}B")
    print(f"  {'-' * 40}")
    print(f"  TOTAL SHADOW RESERVES:        ${latest.total_shadow:>7,.0f}B")
    print(f"  TOTAL TRUE RESERVES:          ${latest.total_true_reserves:>7,.0f}B")
    print(f"  Official as % of True:        {latest.official_share*100:>7.1f}%")
    print(f"  Hidden as % of True:          {latest.shadow_ratio*100:>7.1f}%")
    print("=" * 65)
    print()
    print("Key Setser thesis: China reports ~$3.2T but controls ~$5.8T")
    print("The gap represents firepower for currency intervention that")
    print("markets don't see in headline reserve numbers.")


if __name__ == '__main__':
    df = build_shadow_reserves_dataset()
    print_summary(df)
    plot_shadow_reserves(df)
