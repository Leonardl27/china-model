"""
Generate GitHub Pages site for the Pettis-Setser China Economic Model.

Reads model outputs, builds docs/index.html with live data, and copies
chart PNGs into docs/ so GitHub Pages can serve everything statically.

Usage:
    python generate_site.py
"""

import os
import shutil
from datetime import datetime, timezone

from dashboard import (
    build_market_comparison,
    compute_pettis_setser_scorecard,
)
from shadow_reserves import build_shadow_reserves_dataset
from structural_imbalance import build_structural_dataset
from current_account_forensics import build_ca_forensics_dataset, build_annual_ca_comparison
from fx_intervention import build_fx_intervention_dataset


DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

CHARTS = [
    "dashboard.png",
    "shadow_reserves.png",
    "structural_imbalance.png",
    "current_account_forensics.png",
    "fx_intervention.png",
]


def collect_metrics():
    """Run all models and collect key summary data."""

    shadow_df = build_shadow_reserves_dataset()
    struct_df = build_structural_dataset()
    ca_a_df = build_annual_ca_comparison()
    fx_df = build_fx_intervention_dataset()
    market_df = build_market_comparison()
    scorecard = compute_pettis_setser_scorecard(market_df)

    latest_shadow = shadow_df.iloc[-1]
    latest_struct = struct_df.iloc[-1]
    latest_ca = ca_a_df.iloc[-1]
    latest_fx = fx_df.iloc[-1]

    metrics = {
        # Shadow reserves
        "official_reserves": f"${latest_shadow['official_reserves']:,.0f}B",
        "true_reserves": f"${latest_shadow['total_true_reserves']:,.0f}B",
        "hidden_pct": f"{latest_shadow['shadow_ratio'] * 100:.1f}%",

        # Structural imbalance
        "consumption_pct": f"{latest_struct['hh_consumption_pct']:.1f}%",
        "investment_pct": f"{latest_struct['investment_pct']:.1f}%",
        "consumption_gap": f"{58 - latest_struct['hh_consumption_pct']:.1f} pp",

        # Current account
        "official_ca": f"${latest_ca['official_ca_B']:,.0f}B",
        "setser_ca": f"${latest_ca['setser_est_ca_B']:,.0f}B",
        "ca_pct_gdp": f"{latest_ca['setser_ca_pct_gdp']:.1f}%",

        # FX intervention
        "usdcny": f"{latest_fx['usdcny']:.2f}",
        "fair_value": f"{latest_fx['fair_value_usdcny']:.2f}",
        "undervaluation": f"{latest_fx['undervaluation_pct']:.1f}%",
        "state_bank_cumul": f"${latest_fx['cumulative_state_bank']:,.0f}B",

        # Trade surplus
        "trade_surplus_start": f"${market_df['trade_surplus_B'].iloc[0]:,.0f}B",
        "trade_surplus_end": f"${market_df['trade_surplus_B'].iloc[-1]:,.0f}B",

        # Debt
        "debt_gdp_start": f"{market_df['total_debt_pct_gdp'].iloc[0]:.0f}%",
        "debt_gdp_end": f"{market_df['total_debt_pct_gdp'].iloc[-1]:.0f}%",
    }

    # Scorecard rows
    scorecard_rows = []
    for _, row in scorecard.iterrows():
        scorecard_rows.append({
            "prediction": row["prediction"],
            "framework": row["framework"],
            "metric": row["metric"],
            "verdict": row["verdict"],
        })

    confirmed = sum(1 for r in scorecard_rows if r["verdict"] == "CONFIRMED")
    total = len(scorecard_rows)

    return metrics, scorecard_rows, confirmed, total


def build_scorecard_html(scorecard_rows):
    """Generate the scorecard table HTML."""
    rows_html = ""
    for r in scorecard_rows:
        if r["verdict"] == "CONFIRMED":
            badge = '<span class="badge confirmed">CONFIRMED</span>'
        elif r["verdict"] == "PARTIAL":
            badge = '<span class="badge partial">PARTIAL</span>'
        else:
            badge = '<span class="badge rejected">REJECTED</span>'

        rows_html += f"""            <tr>
              <td>{r['prediction']}</td>
              <td><span class="framework-tag">{r['framework']}</span></td>
              <td class="metric-cell">{r['metric']}</td>
              <td>{badge}</td>
            </tr>
"""
    return rows_html


def build_html(metrics, scorecard_rows, confirmed, total, timestamp):
    """Build the full HTML page."""

    scorecard_html = build_scorecard_html(scorecard_rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pettis-Setser China Economic Model</title>
  <style>
    :root {{
      --bg: #0d1117;
      --surface: #161b22;
      --border: #30363d;
      --text: #e6edf3;
      --text-muted: #8b949e;
      --accent: #58a6ff;
      --green: #3fb950;
      --orange: #d29922;
      --red: #f85149;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 0 24px; }}

    /* Header */
    header {{
      background: linear-gradient(135deg, #1a2332 0%, #0d1117 100%);
      border-bottom: 1px solid var(--border);
      padding: 32px 0;
      text-align: center;
    }}
    header h1 {{
      font-size: 28px;
      font-weight: 700;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
    }}
    header .subtitle {{
      color: var(--text-muted);
      font-size: 15px;
    }}
    .score-banner {{
      display: inline-block;
      margin-top: 16px;
      padding: 8px 24px;
      background: rgba(63, 185, 80, 0.15);
      border: 1px solid rgba(63, 185, 80, 0.4);
      border-radius: 20px;
      color: var(--green);
      font-weight: 600;
      font-size: 16px;
    }}
    .timestamp {{
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 12px;
    }}

    /* Metrics grid */
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 16px;
      margin: 32px 0;
    }}
    .metric-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
    }}
    .metric-card .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
      margin-bottom: 4px;
    }}
    .metric-card .value {{
      font-size: 28px;
      font-weight: 700;
      color: var(--accent);
    }}
    .metric-card .detail {{
      font-size: 13px;
      color: var(--text-muted);
      margin-top: 4px;
    }}

    /* Section headers */
    .section-title {{
      font-size: 20px;
      font-weight: 600;
      margin: 40px 0 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}

    /* Charts */
    .chart-full {{
      margin: 24px 0;
      text-align: center;
    }}
    .chart-full img {{
      max-width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
      margin: 24px 0;
    }}
    .chart-grid img {{
      width: 100%;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .chart-caption {{
      font-size: 13px;
      color: var(--text-muted);
      text-align: center;
      margin-top: 6px;
    }}

    /* Scorecard table */
    .scorecard-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 32px;
      font-size: 14px;
    }}
    .scorecard-table th {{
      background: #1f2937;
      color: var(--text);
      padding: 12px 16px;
      text-align: left;
      font-weight: 600;
      border-bottom: 2px solid var(--border);
    }}
    .scorecard-table td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    .scorecard-table tr:hover {{
      background: rgba(88, 166, 255, 0.04);
    }}
    .metric-cell {{
      font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
      font-size: 13px;
    }}
    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 12px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge.confirmed {{
      background: rgba(63, 185, 80, 0.15);
      color: var(--green);
      border: 1px solid rgba(63, 185, 80, 0.4);
    }}
    .badge.partial {{
      background: rgba(210, 153, 34, 0.15);
      color: var(--orange);
      border: 1px solid rgba(210, 153, 34, 0.4);
    }}
    .badge.rejected {{
      background: rgba(248, 81, 73, 0.15);
      color: var(--red);
      border: 1px solid rgba(248, 81, 73, 0.4);
    }}
    .framework-tag {{
      display: inline-block;
      padding: 2px 8px;
      background: rgba(88, 166, 255, 0.1);
      border: 1px solid rgba(88, 166, 255, 0.3);
      border-radius: 4px;
      font-size: 12px;
      color: var(--accent);
    }}

    /* Footer */
    footer {{
      text-align: center;
      padding: 32px 0;
      margin-top: 48px;
      border-top: 1px solid var(--border);
      color: var(--text-muted);
      font-size: 13px;
    }}
    footer a {{
      color: var(--accent);
      text-decoration: none;
    }}
    footer a:hover {{
      text-decoration: underline;
    }}

    /* Responsive */
    @media (max-width: 768px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: 1fr; }}
      header h1 {{ font-size: 22px; }}
      .metric-card .value {{ font-size: 22px; }}
    }}
  </style>
</head>
<body>

  <header>
    <div class="container">
      <h1>Pettis-Setser China Economic Model</h1>
      <div class="subtitle">
        Quantitative analysis of China's structural imbalances, hidden reserves, and FX intervention
      </div>
      <div class="score-banner">{confirmed}/{total} predictions confirmed by market data</div>
      <div class="timestamp">Last updated: {timestamp}</div>
    </div>
  </header>

  <main class="container">

    <!-- Key Metrics -->
    <h2 class="section-title">Key Metrics</h2>
    <div class="metrics">
      <div class="metric-card">
        <div class="label">True FX Reserves (Setser)</div>
        <div class="value">{metrics['true_reserves']}</div>
        <div class="detail">Official: {metrics['official_reserves']} &middot; {metrics['hidden_pct']} hidden</div>
      </div>
      <div class="metric-card">
        <div class="label">Household Consumption / GDP (Pettis)</div>
        <div class="value">{metrics['consumption_pct']}</div>
        <div class="detail">Global avg: 58% &middot; Gap: {metrics['consumption_gap']}</div>
      </div>
      <div class="metric-card">
        <div class="label">True Current Account Surplus (Setser)</div>
        <div class="value">{metrics['setser_ca']}</div>
        <div class="detail">Official: {metrics['official_ca']} &middot; {metrics['ca_pct_gdp']} of GDP</div>
      </div>
      <div class="metric-card">
        <div class="label">RMB Undervaluation (Setser)</div>
        <div class="value">{metrics['undervaluation']}</div>
        <div class="detail">USD/CNY: {metrics['usdcny']} actual vs {metrics['fair_value']} fair value</div>
      </div>
      <div class="metric-card">
        <div class="label">State Bank FX Accumulation</div>
        <div class="value">{metrics['state_bank_cumul']}</div>
        <div class="detail">Since 2022 &middot; PBOC reserves flat</div>
      </div>
      <div class="metric-card">
        <div class="label">Trade Surplus Trajectory</div>
        <div class="value">{metrics['trade_surplus_end']}</div>
        <div class="detail">Up from {metrics['trade_surplus_start']} in 2018</div>
      </div>
      <div class="metric-card">
        <div class="label">Investment / GDP (Pettis)</div>
        <div class="value">{metrics['investment_pct']}</div>
        <div class="detail">Global avg: 25% &middot; Excess: {float(metrics['investment_pct'].rstrip('%')) - 25:.1f} pp</div>
      </div>
      <div class="metric-card">
        <div class="label">Debt / GDP</div>
        <div class="value">{metrics['debt_gdp_end']}</div>
        <div class="detail">Up from {metrics['debt_gdp_start']} in 2018</div>
      </div>
    </div>

    <!-- Dashboard Chart -->
    <h2 class="section-title">Combined Dashboard</h2>
    <div class="chart-full">
      <img src="dashboard.png" alt="Pettis-Setser Combined Dashboard" loading="lazy">
    </div>

    <!-- Individual Model Charts -->
    <h2 class="section-title">Individual Models</h2>
    <div class="chart-grid">
      <div>
        <img src="shadow_reserves.png" alt="Shadow Reserves (Setser)" loading="lazy">
        <div class="chart-caption">Model 1: Shadow Reserves &mdash; True FX holdings vs reported (Setser)</div>
      </div>
      <div>
        <img src="structural_imbalance.png" alt="Structural Imbalance (Pettis)" loading="lazy">
        <div class="chart-caption">Model 2: Structural Imbalance &mdash; Consumption vs investment distortion (Pettis)</div>
      </div>
      <div>
        <img src="current_account_forensics.png" alt="Current Account Forensics (Setser)" loading="lazy">
        <div class="chart-caption">Model 3: Current Account Forensics &mdash; Missing surplus detection (Setser)</div>
      </div>
      <div>
        <img src="fx_intervention.png" alt="FX Intervention (Setser)" loading="lazy">
        <div class="chart-caption">Model 4: FX Intervention &mdash; Backdoor state bank intervention (Setser)</div>
      </div>
    </div>

    <!-- Scorecard -->
    <h2 class="section-title">Scorecard: Framework vs Market Reality</h2>
    <table class="scorecard-table">
      <thead>
        <tr>
          <th>Prediction</th>
          <th>Framework</th>
          <th>Market Evidence</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
{scorecard_html}
      </tbody>
    </table>

  </main>

  <footer>
    <div class="container">
      Based on the frameworks of
      <a href="https://carnegieendowment.org/people/michael-pettis" target="_blank" rel="noopener">Michael Pettis</a> and
      <a href="https://www.cfr.org/expert/brad-w-setser" target="_blank" rel="noopener">Brad Setser</a>.
      Data from FRED, World Bank, PBOC, SAFE, NBS.
      <br>
      <a href="https://github.com/leonardl27/china-model" target="_blank" rel="noopener">View source on GitHub</a>
    </div>
  </footer>

</body>
</html>
"""


def generate_site():
    """Main entry point: collect data, build HTML, copy charts."""

    os.makedirs(DOCS_DIR, exist_ok=True)

    # Bypass Jekyll on GitHub Pages
    nojekyll_path = os.path.join(DOCS_DIR, ".nojekyll")
    open(nojekyll_path, "a").close()

    print("Collecting model data...")
    metrics, scorecard_rows, confirmed, total = collect_metrics()

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print("Building HTML...")
    html = build_html(metrics, scorecard_rows, confirmed, total, timestamp)

    index_path = os.path.join(DOCS_DIR, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Wrote {index_path}")

    # Copy chart PNGs into docs/
    root = os.path.dirname(os.path.abspath(__file__))
    for chart in CHARTS:
        src = os.path.join(root, chart)
        dst = os.path.join(DOCS_DIR, chart)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  Copied {chart}")
        else:
            print(f"  WARNING: {chart} not found, skipping")

    print(f"\nSite generated in {DOCS_DIR}/")


if __name__ == "__main__":
    generate_site()
