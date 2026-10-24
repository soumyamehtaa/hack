"""
REAL-WORLD observations for Massachusetts (ISO New England).

Sources (public, no API key):
  * Demand: U.S. EIA Hourly Electric Grid Monitor (Form EIA-930) sub-region
    file. ISO-NE reports demand for its load zones; Massachusetts is
    4006 SEMASS, 4007 WCMASS, 4008 NEMASSBOST.
    https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/
  * Prices: ISO-NE Real-Time Energy Market Hourly LMP reports (final, with
    preliminary fallback) for MA load zones and the Internal Hub.
    https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/

Both parsers also accept the same official CSV files uploaded by the user.
Nothing in this module is simulated; infrastructure capacities in the
optimization model remain simulated and are kept separate.
"""
from __future__ import annotations

import csv
import io
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"
UA = {"User-Agent": "Mozilla/5.0 (UNBOUND hackathon research prototype)"}

MA_ZONES = {"4006": "SEMASS", "4007": "WCMASS", "4008": "NEMASSBOST"}
LMP_LOCATIONS = {".Z.SEMASS": "SEMASS", ".Z.WCMASS": "WCMASS", ".Z.NEMASSBOST": "NEMASSBOST",
                 ".H.INTERNAL_HUB": "INTERNAL_HUB"}

EIA_URL = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_SUBREGION_{y}_{h}.csv"
LMP_URLS = [
    "https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/lmp_rt_final_{d}.csv",
    "https://www.iso-ne.com/static-transform/csv/histRpts/rt-lmp/lmp_rt_prelim_{d}.csv",
]


# ---------------------------------------------------------------- demand (EIA-930)
def parse_eia_subregion(src) -> pd.DataFrame:
    """Parse an EIA-930 SUBREGION csv -> long df [ts, zone, demand_mw] for MA zones."""
    df = pd.read_csv(src, dtype={"Sub-Region": str}, thousands=",", low_memory=False)
    need = {"Balancing Authority", "Sub-Region", "Demand (MW)", "Local Time at End of Hour"}
    if not need.issubset(df.columns):
        raise ValueError("Not an EIA-930 sub-region file (missing expected columns).")
    df = df[(df["Balancing Authority"] == "ISNE") & (df["Sub-Region"].isin(MA_ZONES))]
    out = pd.DataFrame({
        "ts": pd.to_datetime(df["Local Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce"),
        "zone": df["Sub-Region"].map(MA_ZONES),
        "demand_mw": pd.to_numeric(df["Demand (MW)"], errors="coerce"),
    }).dropna()
    if out.empty:
        raise ValueError("File contains no ISNE Massachusetts sub-region demand rows.")
    return out.drop_duplicates(["ts", "zone"]).sort_values("ts").reset_index(drop=True)


def _eia_halves(today: date, n: int = 2):
    y, h = today.year, 0 if today.month <= 6 else 1
    out = []
    for _ in range(n):
        out.append((y, "Jan_Jun" if h == 0 else "Jul_Dec"))
        y, h = (y, 0) if h == 1 else (y - 1, 1)
    return out


def fetch_eia_demand(today: date | None = None, max_age_h: float = 12, halves: int = 2):
    """Download (or reuse cached) EIA-930 sub-region files for the latest
    `halves` six-month periods. Returns (df, meta)."""
    today = today or date.today()
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, files, errors = [], [], []
    for y, h in _eia_halves(today, halves):
        url = EIA_URL.format(y=y, h=h)
        path = CACHE / f"eia930_subregion_{y}_{h}.csv"
        stale = not path.exists() or (time.time() - path.stat().st_mtime) / 3600 > max_age_h
        if stale:
            try:
                r = requests.get(url, headers=UA, timeout=90)
                r.raise_for_status()
                path.write_bytes(r.content)
            except Exception as e:  # keep any cached copy
                errors.append(f"{url}: {e}")
        if path.exists():
            try:
                frames.append(parse_eia_subregion(path))
                files.append(url)
            except Exception as e:
                errors.append(f"{path.name}: {e}")
    if not frames:
        return None, dict(source="EIA-930", files=files, errors=errors)
    df = pd.concat(frames).drop_duplicates(["ts", "zone"]).sort_values("ts").reset_index(drop=True)
    return df, dict(source="U.S. EIA Form EIA-930 (ISO-NE load zones)", files=files, errors=errors,
                    retrieved=time.strftime("%Y-%m-%d %H:%M"))


# ---------------------------------------------------------------- prices (ISO-NE RT LMP)
def parse_isone_lmp(src) -> pd.DataFrame:
    """Parse an ISO-NE hourly LMP report (final or prelim, RT or DA) ->
    long df [ts, location, lmp, congestion] for MA load zones + hub."""
    text = src if isinstance(src, str) else src.decode("utf-8", "replace")
    header, rows = None, []
    for rec in csv.reader(io.StringIO(text)):
        if not rec:
            continue
        if rec[0] == "H" and header is None:
            header = rec[1:]
        elif rec[0] == "D" and header is not None:
            rows.append(rec[1:])
    if header is None or not rows:
        raise ValueError("Not an ISO-NE LMP report (no H/D records).")
    df = pd.DataFrame(rows, columns=header[: len(rows[0])])
    loc_col = "Location Name" if "Location Name" in df else "Location"
    lmp_col = "Locational Marginal Price" if "Locational Marginal Price" in df else "LMP"
    cong_col = "Congestion Component" if "Congestion Component" in df else "Congestion"
    if loc_col not in df or lmp_col not in df:
        raise ValueError("LMP report missing location / price columns.")
    df = df[df[loc_col].isin(LMP_LOCATIONS)]
    he = pd.to_numeric(df["Hour Ending"].str.extract(r"(\d+)")[0], errors="coerce")
    ts = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce") + pd.to_timedelta(he, unit="h")
    out = pd.DataFrame({
        "ts": ts, "location": df[loc_col].map(LMP_LOCATIONS),
        "lmp": pd.to_numeric(df[lmp_col], errors="coerce"),
        "congestion": pd.to_numeric(df.get(cong_col), errors="coerce"),
    }).dropna(subset=["ts", "lmp"])
    # DST fall-back produces a repeated hour; average it
    return out.groupby(["ts", "location"], as_index=False).mean()


def _fetch_lmp_day(d: date):
    ds = d.strftime("%Y%m%d")
    cached = CACHE / f"lmp_rt_ma_{ds}.csv"
    if cached.exists():
        df = pd.read_csv(cached, parse_dates=["ts"])
        return df, df["report"].iloc[0] if len(df) else "cache"
    for url in LMP_URLS:
        try:
            r = requests.get(url.format(d=ds), headers=UA, timeout=45)
            if r.status_code != 200 or b'"D"' not in r.content[:5000]:
                continue
            df = parse_isone_lmp(r.content)
            kind = "final" if "final" in url else "prelim"
            df["report"] = kind
            if kind == "final":  # prelim files may still change: don't cache
                df.to_csv(cached, index=False)
            return df, kind
        except Exception:
            continue
    return None, None


def fetch_lmp(days: int = 14, today: date | None = None, end: date | None = None):
    """Fetch `days` daily RT LMP reports ending at `end` (default: yesterday)."""
    today = today or date.today()
    end = end or (today - timedelta(days=1))
    CACHE.mkdir(parents=True, exist_ok=True)
    dates = [end - timedelta(days=k) for k in range(days) if end - timedelta(days=k) < today]
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_fetch_lmp_day, dates))
    frames = [df for df, _ in results if df is not None]
    kinds = [k for _, k in results if k]
    if not frames:
        return None, dict(source="ISO-NE RT LMP", errors=["no LMP reports retrieved"])
    df = pd.concat(frames).sort_values("ts").reset_index(drop=True)
    return df, dict(source="ISO New England Real-Time Hourly LMP reports",
                    days=len(frames), prelim_days=kinds.count("prelim"),
                    retrieved=time.strftime("%Y-%m-%d %H:%M"))


def parse_upload(name: str, data: bytes):
    """Detect and parse an uploaded official CSV. Returns (kind, df)."""
    head = data[:3000].decode("utf-8", "replace")
    if "Balancing Authority" in head and "Sub-Region" in head:
        return "demand", parse_eia_subregion(io.BytesIO(data))
    if "LMP" in head and '"H"' in head:
        return "lmp", parse_isone_lmp(data)
    raise ValueError(f"{name}: unrecognized format. Expected an EIA-930 SUBREGION csv or an ISO-NE hourly LMP report.")


# ---------------------------------------------------------------- analytics (real observations only)
def ma_hourly(demand: pd.DataFrame) -> pd.Series:
    """Statewide MA hourly demand: sum of the three zones, only hours where all three report."""
    w = demand.pivot_table(index="ts", columns="zone", values="demand_mw", aggfunc="mean")
    w = w.dropna(how="any")
    return w.sum(axis=1).rename("ma_mw")


def demand_stats(demand: pd.DataFrame) -> dict:
    s = ma_hourly(demand)
    peak = float(s.max())
    return dict(
        start=s.index.min(), end=s.index.max(), hours=len(s),
        peak_mw=peak, peak_ts=s.idxmax(), mean_mw=float(s.mean()), min_mw=float(s.min()),
        load_factor=float(s.mean() / peak),
        hours_within_5pct=int((s >= 0.95 * peak).sum()),
        hours_within_10pct=int((s >= 0.90 * peak).sum()),
        top1pct_threshold=float(s.quantile(0.99)),
    )


def diurnal_shape(demand: pd.DataFrame, top_days: int = 10):
    """Average hour-of-day MA load on the `top_days` highest-peak days, normalized to 1.
    Hour index uses hour-ending converted to 0-23 hour-beginning."""
    s = ma_hourly(demand)
    daily_peak = s.groupby(s.index.date).max().nlargest(top_days)
    sub = s[np.isin(s.index.date, daily_peak.index)]
    hb = (sub.index - pd.Timedelta(hours=1)).hour
    shape = sub.groupby(hb).mean().reindex(range(24)).interpolate().bfill().ffill()
    return (shape / shape.max()).to_numpy(), sorted(daily_peak.index)


def peak_window(shape: np.ndarray, width: int = 5):
    c = int(np.argmax(shape))
    lo = max(0, min(24 - width, c - width // 2))
    return list(range(lo, lo + width))


def lmp_stats(lmp: pd.DataFrame) -> pd.DataFrame:
    g = lmp.groupby("location")
    return pd.DataFrame({
        "mean_lmp": g.lmp.mean(), "p95_lmp": g.lmp.quantile(0.95), "max_lmp": g.lmp.max(),
        "min_lmp": g.lmp.min(), "hours_over_100": g.lmp.apply(lambda x: int((x > 100).sum())),
        "mean_congestion": g.congestion.mean(), "hours": g.lmp.count(),
    }).reset_index()


def flexible_headroom_curve(demand: pd.DataFrame, added_mw) -> pd.DataFrame:
    """For a flat block of new load of X MW, how many observed hours would MA
    zonal demand exceed its observed peak, and what share of the new load's
    energy would need to be curtailed/shifted to keep the observed peak
    unchanged? Pure arithmetic on observed demand; assumes the observed peak
    is the ceiling (an assumption, not a measured grid limit)."""
    s = ma_hourly(demand).to_numpy()
    peak = s.max()
    rows = []
    for x in added_mw:
        over = np.clip(s + x - peak, 0, None)
        rows.append(dict(added_mw=float(x), hours_over_peak=int((over > 0).sum()),
                         curtailed_share=float(over.sum() / (x * len(s))) if x > 0 else 0.0,
                         max_curtail_mw=float(over.max())))
    return pd.DataFrame(rows)


def peak_vs_offpeak_price(lmp: pd.DataFrame, window_hours) -> dict:
    """Mean MA load-zone LMP inside vs outside a peak window (hour-beginning)."""
    z = lmp[lmp.location != "INTERNAL_HUB"].copy()
    z["hb"] = (z.ts - pd.Timedelta(hours=1)).dt.hour
    inside = z[z.hb.isin(window_hours)].lmp.mean()
    outside = z[~z.hb.isin(window_hours)].lmp.mean()
    return dict(peak=float(inside), offpeak=float(outside), spread=float(inside - outside))
