# China Economic Model: Pettis-Setser Framework

Measurable models based on the analytical frameworks of **Michael Pettis** (Carnegie Endowment)
and **Brad Setser** (Council on Foreign Relations).

## Models

### 1. Shadow Reserves Tracker (`shadow_reserves.py`)
Estimates China's **true** foreign exchange holdings vs. official PBOC reserves.
Tracks hidden FX across state banks, policy banks, and sovereign wealth vehicles.

### 2. Structural Imbalance Model (`structural_imbalance.py`)
Pettis's framework: consumption vs. investment share of GDP, financial repression gap,
and rebalancing scenarios.

### 3. Current Account Forensics (`current_account_forensics.py`)
Detects discrepancies between customs trade data and BOP-reported current account.
Setser's methodology for finding "missing" surplus.

### 4. FX Intervention Detector (`fx_intervention.py`)
Uses FX settlement data and state bank balance sheet changes to detect
backdoor currency intervention.

### 5. Dashboard (`dashboard.py`)
Combined visualization of all models with market comparison.

## Data Sources
- PBOC balance sheet data
- SAFE foreign exchange settlement data
- China customs trade data (via SAFE BOP reports)
- NBS GDP expenditure breakdown
- State bank financial statements
- World Bank / IMF consumption benchmarks

## Usage
```bash
python dashboard.py          # Run full dashboard
python shadow_reserves.py    # Individual model
```
