"""
Data Fetcher Module
====================
Central data source with 3-tier fallback:
  1. Live API (FRED, World Bank, yfinance)
  2. Local JSON cache (data_cache/ directory)
  3. Hardcoded defaults (original model values)

Caching TTLs:
  - FRED: 7 days
  - World Bank: 30 days
  - yfinance: 1 day

Config:
  - FRED_API_KEY: set via environment variable or .env file
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import requests
except ImportError:
    requests = None

try:
    import wbgapi as wb
except ImportError:
    wb = None

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

# TTLs in seconds
TTL_FRED = 7 * 86400      # 7 days
TTL_WB = 30 * 86400       # 30 days
TTL_YFINANCE = 1 * 86400  # 1 day


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    """Return path for a cache entry."""
    safe = key.replace("/", "_").replace(":", "_").replace("?", "_")
    return CACHE_DIR / f"{safe}.json"


def _read_cache(key: str, ttl: float) -> pd.Series | None:
    """Read from JSON cache if not expired. Returns pd.Series or None."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if time.time() - payload.get("ts", 0) > ttl:
            return None
        data = payload["data"]
        s = pd.Series(data["values"], index=data["index"], name=data.get("name", key))
        # Restore numeric index if it was numeric
        if data.get("index_numeric"):
            s.index = pd.to_numeric(s.index, errors="coerce")
        return s
    except Exception:
        return None


def _write_cache(key: str, series: pd.Series) -> None:
    """Write a pd.Series to JSON cache."""
    try:
        idx = series.index.tolist()
        index_numeric = all(isinstance(i, (int, float, np.integer, np.floating)) for i in idx)
        payload = {
            "ts": time.time(),
            "data": {
                "values": series.tolist(),
                "index": [str(i) for i in idx],
                "name": series.name,
                "index_numeric": index_numeric,
            },
        }
        with open(_cache_path(key), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.debug(f"Cache write failed for {key}: {e}")


def _read_dict_cache(key: str, ttl: float) -> dict | None:
    """Read a dict-shaped cache entry."""
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if time.time() - payload.get("ts", 0) > ttl:
            return None
        return payload["data"]
    except Exception:
        return None


def _write_dict_cache(key: str, data: dict) -> None:
    """Write a dict to JSON cache."""
    try:
        payload = {"ts": time.time(), "data": data}
        with open(_cache_path(key), "w") as f:
            json.dump(payload, f)
    except Exception as e:
        logger.debug(f"Dict cache write failed for {key}: {e}")


# ---------------------------------------------------------------------------
# FRED fetching
# ---------------------------------------------------------------------------

def get_fred_series(
    series_id: str,
    start_date: str = "2000-01-01",
    frequency: str | None = None,
    aggregation: str = "avg",
    unit_divisor: float = 1.0,
) -> pd.Series | None:
    """
    Fetch a FRED series. Returns pd.Series indexed by date string, or None.

    Parameters
    ----------
    series_id : FRED series ID (e.g. 'TRESEGCNM052N')
    start_date : earliest observation
    frequency : 'a' (annual), 'q' (quarterly), 'm' (monthly) or None (native)
    aggregation : 'avg', 'sum', 'eop' (end of period)
    unit_divisor : divide values by this (e.g. 1000 for millions->billions)
    """
    cache_key = f"fred_{series_id}_{frequency}_{aggregation}"
    cached = _read_cache(cache_key, TTL_FRED)
    if cached is not None:
        logger.info(f"FRED cache hit: {series_id}")
        return cached

    if not FRED_API_KEY or requests is None:
        logger.info(f"FRED unavailable for {series_id} (no key or requests)")
        return None

    try:
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
            "sort_order": "asc",
        }
        if frequency:
            params["frequency"] = frequency
            params["aggregation_method"] = aggregation

        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/observations",
            params=params,
            timeout=30,
        )
        resp.raise_for_status()
        obs = resp.json().get("observations", [])

        dates = []
        values = []
        for o in obs:
            val = o["value"]
            if val == ".":
                continue
            dates.append(o["date"])
            values.append(float(val) / unit_divisor)

        s = pd.Series(values, index=dates, name=series_id)
        _write_cache(cache_key, s)
        logger.info(f"FRED fetched: {series_id} ({len(s)} obs)")
        return s

    except Exception as e:
        logger.warning(f"FRED fetch failed for {series_id}: {e}")
        return None


def get_fred_annual_yearend(
    series_id: str,
    start_year: int = 2000,
    unit_divisor: float = 1.0,
) -> pd.Series | None:
    """Fetch FRED series aggregated to annual year-end values."""
    return get_fred_series(
        series_id,
        start_date=f"{start_year}-01-01",
        frequency="a",
        aggregation="eop",
        unit_divisor=unit_divisor,
    )


def get_fred_annual_avg(
    series_id: str,
    start_year: int = 2000,
    unit_divisor: float = 1.0,
) -> pd.Series | None:
    """Fetch FRED series aggregated to annual average."""
    return get_fred_series(
        series_id,
        start_date=f"{start_year}-01-01",
        frequency="a",
        aggregation="avg",
        unit_divisor=unit_divisor,
    )


def get_fred_monthly(
    series_id: str,
    start_date: str = "2022-01-01",
    unit_divisor: float = 1.0,
) -> pd.Series | None:
    """Fetch FRED monthly series."""
    return get_fred_series(
        series_id,
        start_date=start_date,
        frequency="m",
        aggregation="avg",
        unit_divisor=unit_divisor,
    )


def get_fred_quarterly(
    series_id: str,
    start_date: str = "2019-01-01",
    aggregation: str = "sum",
    unit_divisor: float = 1.0,
) -> pd.Series | None:
    """Fetch FRED series aggregated to quarterly."""
    return get_fred_series(
        series_id,
        start_date=start_date,
        frequency="q",
        aggregation=aggregation,
        unit_divisor=unit_divisor,
    )


# ---------------------------------------------------------------------------
# World Bank fetching
# ---------------------------------------------------------------------------

def get_wb_series(
    indicator: str,
    country: str = "CHN",
    start_year: int = 2000,
    unit_divisor: float = 1.0,
) -> pd.Series | None:
    """
    Fetch a World Bank indicator for a single country.
    Returns pd.Series indexed by year (int), or None.
    """
    cache_key = f"wb_{indicator}_{country}_{start_year}"
    cached = _read_cache(cache_key, TTL_WB)
    if cached is not None:
        logger.info(f"WB cache hit: {indicator} {country}")
        return cached

    if wb is None:
        logger.info(f"WB unavailable (wbgapi not installed)")
        return None

    try:
        data = wb.data.DataFrame(indicator, country, time=range(start_year, 2026))
        # wbgapi returns a DataFrame with YRxxxx columns
        if data.empty:
            return None

        # Flatten: row for the country, columns are YRyyyy
        row = data.iloc[0]
        years = []
        values = []
        for col in row.index:
            yr_str = str(col).replace("YR", "")
            try:
                yr = int(yr_str)
            except ValueError:
                continue
            if yr < start_year:
                continue
            val = row[col]
            if pd.notna(val):
                years.append(yr)
                values.append(float(val) / unit_divisor)

        if not values:
            return None

        s = pd.Series(values, index=years, name=indicator)
        # Forward-fill for 1-2yr data lag
        full_index = list(range(min(years), 2026))
        s = s.reindex(full_index).ffill()
        _write_cache(cache_key, s)
        logger.info(f"WB fetched: {indicator} {country} ({len(s)} obs)")
        return s

    except Exception as e:
        logger.warning(f"WB fetch failed for {indicator} {country}: {e}")
        return None


def get_wb_benchmarks(
    indicator: str,
    countries: dict[str, str] | None = None,
) -> dict[str, float] | None:
    """
    Fetch latest value of a World Bank indicator for multiple countries.
    Returns dict {display_name: value} or None.

    countries: {iso3_code: display_name}
    """
    if countries is None:
        countries = {
            "USA": "United States",
            "JPN": "Japan",
            "DEU": "Germany",
            "KOR": "South Korea",
            "IND": "India",
            "BRA": "Brazil",
            "GBR": "UK",
            "FRA": "France",
            "IDN": "Indonesia",
            "THA": "Thailand",
        }

    cache_key = f"wb_bench_{indicator}_{'_'.join(sorted(countries.keys()))}"
    cached = _read_dict_cache(cache_key, TTL_WB)
    if cached is not None:
        logger.info(f"WB benchmark cache hit: {indicator}")
        return cached

    if wb is None:
        logger.info("WB unavailable (wbgapi not installed)")
        return None

    try:
        codes = list(countries.keys())
        data = wb.data.DataFrame(indicator, codes, time=range(2018, 2026), numericTimeKeys=True)
        if data.empty:
            return None

        result = {}
        for code, name in countries.items():
            try:
                row = data.loc[code]
                # Get the latest non-NaN value
                valid = row.dropna()
                if len(valid) > 0:
                    result[name] = float(valid.iloc[-1])
            except (KeyError, IndexError):
                continue

        if not result:
            return None

        _write_dict_cache(cache_key, result)
        logger.info(f"WB benchmarks fetched: {indicator} ({len(result)} countries)")
        return result

    except Exception as e:
        logger.warning(f"WB benchmarks fetch failed for {indicator}: {e}")
        return None


# ---------------------------------------------------------------------------
# yfinance fetching
# ---------------------------------------------------------------------------

def get_yfinance_series(
    ticker: str,
    start_date: str = "2018-01-01",
    column: str = "Close",
) -> pd.Series | None:
    """
    Fetch a price series from yfinance.
    Returns pd.Series indexed by date string, or None.
    """
    cache_key = f"yf_{ticker}_{column}"
    cached = _read_cache(cache_key, TTL_YFINANCE)
    if cached is not None:
        logger.info(f"yfinance cache hit: {ticker}")
        return cached

    if yf is None:
        logger.info("yfinance unavailable (not installed)")
        return None

    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start_date)
        if hist.empty:
            return None

        s = hist[column].rename(ticker)
        s.index = s.index.strftime("%Y-%m-%d")
        _write_cache(cache_key, s)
        logger.info(f"yfinance fetched: {ticker} ({len(s)} obs)")
        return s

    except Exception as e:
        logger.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


def get_yfinance_annual_close(
    ticker: str,
    start_year: int = 2018,
) -> pd.Series | None:
    """Fetch year-end closing prices from yfinance."""
    raw = get_yfinance_series(ticker, start_date=f"{start_year}-01-01", column="Close")
    if raw is None:
        return None

    try:
        df = pd.DataFrame({"date": raw.index, "close": raw.values})
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        # Get last trading day of each year
        annual = df.groupby("year")["close"].last()
        annual.name = ticker
        return annual
    except Exception as e:
        logger.warning(f"yfinance annual aggregation failed for {ticker}: {e}")
        return None


# ---------------------------------------------------------------------------
# Helper: merge live data into hardcoded list
# ---------------------------------------------------------------------------

def merge_into_list(
    live_series: pd.Series | None,
    hardcoded: list,
    index_values: list,
    round_digits: int | None = 1,
) -> list:
    """
    Replace hardcoded values with live data where available.
    Falls back to hardcoded for any missing years/dates.

    Parameters
    ----------
    live_series : pd.Series from an API (may be None)
    hardcoded : original hardcoded list (same length as index_values)
    index_values : the years/dates corresponding to hardcoded entries
    round_digits : round live values to this many decimals (None = no rounding)
    """
    if live_series is None:
        return hardcoded

    result = list(hardcoded)  # copy
    for i, idx in enumerate(index_values):
        key = str(idx) if not isinstance(idx, str) else idx
        # Try exact match, or year-only match for date-indexed series
        val = None
        if key in live_series.index:
            val = live_series[key]
        elif isinstance(idx, int):
            # Try matching as integer index
            if idx in live_series.index:
                val = live_series[idx]
            else:
                # Try matching year from date string like "2020-01-01"
                for si, sv in live_series.items():
                    if str(si).startswith(str(idx)):
                        val = sv
                        break

        if val is not None and pd.notna(val):
            if round_digits is not None:
                val = round(float(val), round_digits)
            else:
                val = float(val)
            result[i] = val

    return result


def merge_into_dict(
    live_data: dict[str, float] | None,
    hardcoded: dict[str, float],
    round_digits: int | None = 1,
) -> dict[str, float]:
    """Merge live benchmark values into hardcoded dict."""
    if live_data is None:
        return hardcoded

    result = dict(hardcoded)
    for key, val in live_data.items():
        if key in result and val is not None:
            if round_digits is not None:
                result[key] = round(float(val), round_digits)
            else:
                result[key] = float(val)

    return result


# ---------------------------------------------------------------------------
# Hardcoded Defaults
# (Original model values, used as last-resort fallback)
# ---------------------------------------------------------------------------

DEFAULTS = {
    # shadow_reserves.py: official_reserves (2004-2025, $ billions, year-end)
    "official_reserves": [
        610, 819, 1066, 1528, 1946, 2399, 2847, 3181, 3312, 3821,
        3843, 3330, 3011, 3140, 3073, 3108, 3217, 3250, 3128, 3238,
        3245, 3200,
    ],

    # structural_imbalance.py: china_consumption_pct (2000-2025)
    "china_consumption_pct": [
        46.7, 45.3, 44.0, 42.2, 40.6, 39.2, 38.2, 36.7, 35.6, 35.4,
        35.6, 36.3, 36.5, 36.6, 37.3, 38.0, 39.3, 39.1, 39.4, 39.2,
        38.1, 38.5, 37.2, 39.7, 38.5, 38.8,
    ],

    # structural_imbalance.py: china_gov_consumption_pct (2000-2025)
    "china_gov_consumption_pct": [
        16.0, 16.2, 15.8, 14.8, 14.0, 14.2, 13.9, 13.5, 13.5, 13.6,
        13.3, 13.7, 13.9, 14.0, 14.3, 14.7, 14.6, 14.4, 14.9, 15.1,
        16.5, 15.8, 16.3, 16.2, 16.0, 16.0,
    ],

    # structural_imbalance.py: china_investment_pct (2000-2025)
    "china_investment_pct": [
        34.3, 36.5, 37.9, 41.0, 43.3, 42.1, 41.8, 41.7, 43.8, 46.3,
        47.2, 47.0, 46.7, 46.5, 46.2, 44.7, 44.1, 44.4, 44.8, 43.1,
        43.0, 43.0, 43.7, 42.0, 42.5, 42.2,
    ],

    # structural_imbalance.py: china_gdp_usd_B (2000-2025)
    "china_gdp_usd_B": [
        1211, 1339, 1471, 1660, 1955, 2286, 2752, 3550, 4594, 5102,
        6087, 7552, 8532, 9570, 10476, 11062, 11233, 12310, 13895, 14280,
        14688, 17734, 17963, 17795, 18533, 19000,
    ],

    # structural_imbalance.py: BENCHMARKS hh_consumption_pct
    "bench_hh_consumption": {
        "United States": 68.0, "Japan": 53.5, "Germany": 52.0,
        "South Korea": 49.0, "India": 60.0, "Brazil": 63.0,
        "UK": 63.0, "France": 54.0, "Indonesia": 57.0, "Thailand": 52.0,
    },

    # structural_imbalance.py: BENCHMARKS investment_pct
    "bench_investment": {
        "United States": 21.0, "Japan": 25.5, "Germany": 22.0,
        "South Korea": 31.0, "India": 32.0, "Brazil": 15.0,
        "UK": 17.0, "France": 24.0, "Indonesia": 30.0, "Thailand": 24.0,
    },

    # current_account_forensics.py: customs_surplus (quarterly, 2019Q1-2025Q2)
    "customs_surplus_q": [
        76, 114, 126, 96,
        26, 137, 160, 148,
        117, 137, 162, 181,
        162, 203, 230, 198,
        180, 215, 230, 198,
        195, 250, 260, 250,
        250, 300,
    ],

    # current_account_forensics.py: official_ca_B (annual, 2018-2025)
    "official_ca_B": [24, 103, 274, 317, 402, 264, 422, 590],

    # current_account_forensics.py: china_gdp_T (annual, 2018-2025)
    "china_gdp_T": [13.9, 14.3, 14.7, 17.7, 18.0, 17.8, 18.5, 19.0],

    # fx_intervention.py: usdcny (monthly, Jan 2022 - Jun 2025)
    "usdcny_monthly": [
        6.36, 6.32, 6.34, 6.61, 6.72, 6.70, 6.75, 6.89, 7.12, 7.25, 7.15, 6.95,
        6.78, 6.88, 6.87, 6.92, 7.08, 7.25, 7.14, 7.29, 7.30, 7.32, 7.15, 7.10,
        7.18, 7.20, 7.22, 7.24, 7.25, 7.27, 7.26, 7.15, 7.02, 7.12, 7.25, 7.30,
        7.28, 7.25, 7.20, 7.22, 7.18, 7.15,
    ],

    # fx_intervention.py: pboc_reserve_change (monthly, Jan 2022 - Jun 2025)
    "pboc_reserve_change": [
        -5, -3, -8, -10, -12, -5, 2, -8, -15, -12, 5, 3,
        -3, 5, 8, -2, -5, 2, 3, -5, -2, 4, -3, 5,
        2, -5, 3, -2, 5, -3, 2, -4, 5, -2, 3, 8,
        -5, 3, -2, 5, -3, 2,
    ],

    # dashboard.py: trade_surplus_B (annual, 2018-2025)
    "trade_surplus_B": [350, 421, 535, 676, 878, 823, 992, 1100],

    # dashboard.py: total_debt_pct_gdp (annual, 2018-2025)
    "total_debt_pct_gdp": [253, 259, 280, 272, 295, 288, 298, 310],

    # dashboard.py: cpi_yoy_pct (annual, 2018-2025)
    "cpi_yoy_pct": [2.1, 2.9, 2.5, 0.9, 2.0, 0.2, 0.2, 0.3],

    # dashboard.py: ppi_yoy_pct (annual, 2018-2025)
    "ppi_yoy_pct": [3.5, -0.3, -1.8, 8.1, 4.1, -3.0, -2.2, -1.5],

    # dashboard.py: rmb_reer_index (annual, 2018-2025)
    "rmb_reer_index": [105, 102, 100, 98, 92, 88, 85, 83],

    # dashboard.py: csi300 (annual, 2018-2025)
    "csi300": [3010, 4096, 5211, 4940, 3872, 3441, 3935, 3800],
}


# ---------------------------------------------------------------------------
# High-level fetch functions for each model
# ---------------------------------------------------------------------------

def fetch_official_reserves(years: list[int]) -> list:
    """Fetch China official FX reserves (year-end, $B) from FRED."""
    live = get_fred_annual_yearend("TRESEGCNM052N", start_year=years[0], unit_divisor=1000)
    return merge_into_list(live, DEFAULTS["official_reserves"], years, round_digits=0)


def fetch_china_consumption_pct(years: list[int]) -> list:
    """Fetch household consumption % GDP from World Bank."""
    live = get_wb_series("NE.CON.PRVT.ZS", "CHN", start_year=years[0])
    return merge_into_list(live, DEFAULTS["china_consumption_pct"], years)


def fetch_china_gov_consumption_pct(years: list[int]) -> list:
    """Fetch government consumption % GDP from World Bank."""
    live = get_wb_series("NE.CON.GOVT.ZS", "CHN", start_year=years[0])
    return merge_into_list(live, DEFAULTS["china_gov_consumption_pct"], years)


def fetch_china_investment_pct(years: list[int]) -> list:
    """Fetch investment (gross capital formation) % GDP from World Bank."""
    live = get_wb_series("NE.GDI.TOTL.ZS", "CHN", start_year=years[0])
    return merge_into_list(live, DEFAULTS["china_investment_pct"], years)


def fetch_china_gdp_usd_B(years: list[int]) -> list:
    """Fetch China GDP in USD (billions) from World Bank."""
    live = get_wb_series("NY.GDP.MKTP.CD", "CHN", start_year=years[0], unit_divisor=1e9)
    return merge_into_list(live, DEFAULTS["china_gdp_usd_B"], years, round_digits=0)


def fetch_benchmarks_consumption() -> dict[str, float]:
    """Fetch household consumption % GDP benchmarks for comparison countries."""
    live = get_wb_benchmarks("NE.CON.PRVT.ZS")
    return merge_into_dict(live, DEFAULTS["bench_hh_consumption"])


def fetch_benchmarks_investment() -> dict[str, float]:
    """Fetch investment % GDP benchmarks for comparison countries."""
    live = get_wb_benchmarks("NE.GDI.TOTL.ZS")
    return merge_into_dict(live, DEFAULTS["bench_investment"])


def fetch_trade_balance_quarterly(start_year: int = 2019) -> pd.Series | None:
    """
    Fetch China trade balance from FRED, aggregated to quarterly sums ($B).
    FRED series XTNTVA01CNM667S is monthly in millions of USD.
    """
    live = get_fred_quarterly(
        "XTNTVA01CNM667S",
        start_date=f"{start_year}-01-01",
        aggregation="sum",
        unit_divisor=1e6,  # millions -> billions
    )
    return live


def fetch_current_account_quarterly(start_year: int = 2019) -> pd.Series | None:
    """
    Fetch China current account balance from FRED (quarterly, $B).
    FRED series CHNB6BLTT02STSAQ is quarterly in millions of USD.
    """
    live = get_fred_quarterly(
        "CHNB6BLTT02STSAQ",
        start_date=f"{start_year}-01-01",
        aggregation="avg",
        unit_divisor=1000,  # millions -> billions
    )
    return live


def fetch_usdcny_monthly(start_date: str = "2022-01-01") -> list | None:
    """Fetch USD/CNY monthly from FRED."""
    live = get_fred_monthly("EXCHUS", start_date=start_date)
    if live is not None and len(live) > 0:
        return live
    return None


def fetch_official_reserves_monthly(start_date: str = "2022-01-01") -> pd.Series | None:
    """Fetch China official reserves monthly from FRED ($B)."""
    return get_fred_monthly("TRESEGCNM052N", start_date=start_date, unit_divisor=1000)


def fetch_trade_surplus_annual(years: list[int]) -> list:
    """Fetch annual trade surplus from FRED ($B)."""
    live = get_fred_series(
        "XTNTVA01CNM667S",
        start_date=f"{years[0]}-01-01",
        frequency="a",
        aggregation="sum",
        unit_divisor=1e6,
    )
    return merge_into_list(live, DEFAULTS["trade_surplus_B"], years, round_digits=0)


def fetch_cpi_annual(years: list[int]) -> list:
    """Fetch China CPI YoY% annual avg from FRED."""
    live = get_fred_annual_avg("CPALTT01CNM657N", start_year=years[0])
    return merge_into_list(live, DEFAULTS["cpi_yoy_pct"], years)


def fetch_ppi_annual(years: list[int]) -> list:
    """Fetch China PPI YoY% annual avg from FRED."""
    live = get_fred_annual_avg("CHNPIEATI01GYM", start_year=years[0])
    return merge_into_list(live, DEFAULTS["ppi_yoy_pct"], years)


def fetch_credit_gdp_annual(years: list[int]) -> list:
    """Fetch credit/GDP% from FRED (BIS total credit to non-financial sector)."""
    live = get_fred_annual_avg("QCNCAM770A", start_year=years[0])
    return merge_into_list(live, DEFAULTS["total_debt_pct_gdp"], years, round_digits=0)


def fetch_reer_annual(years: list[int]) -> list:
    """Fetch RMB REER index annual avg from FRED."""
    live = get_fred_annual_avg("RBCNBIS", start_year=years[0])
    return merge_into_list(live, DEFAULTS["rmb_reer_index"], years, round_digits=0)


def fetch_csi300_annual(years: list[int]) -> list:
    """Fetch CSI 300 year-end close from yfinance."""
    live = get_yfinance_annual_close("000300.SS", start_year=years[0])
    return merge_into_list(live, DEFAULTS["csi300"], years, round_digits=0)


def fetch_official_ca_annual(years: list[int]) -> list:
    """Fetch official current account balance annual ($B) from FRED."""
    live = get_fred_series(
        "CHNB6BLTT02STSAQ",
        start_date=f"{years[0]}-01-01",
        frequency="a",
        aggregation="sum",
        unit_divisor=1000,
    )
    return merge_into_list(live, DEFAULTS["official_ca_B"], years, round_digits=0)


def fetch_china_gdp_T_annual(years: list[int]) -> list:
    """Fetch China GDP in trillions USD from World Bank."""
    live = get_wb_series("NY.GDP.MKTP.CD", "CHN", start_year=years[0], unit_divisor=1e12)
    return merge_into_list(live, DEFAULTS["china_gdp_T"], years)
