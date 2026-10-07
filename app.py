import os
import io
import re
import hmac
import time
import hashlib
from datetime import datetime, timedelta
from urllib.parse import quote
from flask import (Flask, jsonify, send_from_directory, request, send_file,
                   session, redirect, render_template_string)
from flask_cors import CORS
import numpy as np
import yfinance as yf
from yfinance.exceptions import YFRateLimitError
from statsmodels.tsa.stattools import coint

# static_folder=None so the server never hands out its own source files.
app = Flask(__name__, static_folder=None)
CORS(app)

# ── Password ────────────────────────────────────────────────────────────────
# Optional shared password. Set APP_PASSWORD in Railway's Variables tab to turn it on;
# if it's not set, the site stays open to anyone with the link.
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

# Sign-in cookie key. Derived from the password so changing the password signs everyone out.
app.secret_key = os.environ.get("SECRET_KEY") or hashlib.sha256(f"pair-analysis:{APP_PASSWORD}".encode()).hexdigest()
app.permanent_session_lifetime = timedelta(days=30)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=bool(os.environ.get("RAILWAY_ENVIRONMENT_NAME") or os.environ.get("RAILWAY_ENVIRONMENT")),
)

LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Pair Analysis — Sign In</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
<style>
  body { background:#0a0c10; color:#e8e4dc; font-family:'DM Mono',monospace; min-height:100vh; margin:0;
         display:flex; align-items:center; justify-content:center; padding:24px; box-sizing:border-box; }
  form { background:#111318; border:1px solid #1e2330; border-radius:6px; padding:36px 32px; width:100%; max-width:340px; text-align:center; }
  h1 { font-family:'Playfair Display',serif; color:#c8a96e; font-size:2rem; margin:0 0 6px; }
  p { color:#6b7280; font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; margin:0 0 28px; }
  input { width:100%; box-sizing:border-box; background:#0a0c10; border:1px solid #1e2330; color:#e8e4dc; font-family:inherit;
          font-size:0.9rem; padding:11px 14px; border-radius:4px; outline:none; margin-bottom:16px; }
  input:focus { border-color:#c8a96e; }
  button { width:100%; background:#c8a96e; color:#0a0c10; border:none; font-family:inherit; font-size:0.75rem; font-weight:500;
           letter-spacing:2px; text-transform:uppercase; padding:12px; border-radius:4px; cursor:pointer; }
  .err { color:#f87171; font-size:0.75rem; margin-bottom:16px; }
</style>
</head>
<body>
<form method="post" action="/login">
  <h1>Pair Analysis</h1>
  <p>Sign in to continue</p>
  {% if error %}<div class="err">{{ error }}</div>{% endif %}
  <input type="password" name="password" placeholder="Password" autofocus required autocomplete="current-password">
  <input type="hidden" name="next" value="{{ next }}">
  <button type="submit">Sign In</button>
</form>
</body>
</html>"""


def _safe_next(target):
    """Only allow redirects back into this site (e.g. '/?a=CRM&b=HUBS'), never to another domain."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return "/"


@app.before_request
def require_login():
    if not APP_PASSWORD or request.endpoint in ("login", "logout") or session.get("authed"):
        return None
    if request.path == "/api/compare":
        return jsonify({"error": "Your sign-in has expired. Please reload the page and sign in again."}), 401
    return redirect("/login?next=" + quote(_safe_next(request.full_path.rstrip("?")), safe=""))


@app.route("/login", methods=["GET", "POST"])
def login():
    if not APP_PASSWORD:
        return redirect("/")
    nxt = _safe_next(request.values.get("next", "/"))
    if request.method == "POST":
        if hmac.compare_digest(request.form.get("password", "").encode(), APP_PASSWORD.encode()):
            session.permanent = True
            session["authed"] = True
            return redirect(nxt)
        time.sleep(1)  # slow down password guessing
        return render_template_string(LOGIN_PAGE, error="Incorrect password.", next=nxt), 401
    return render_template_string(LOGIN_PAGE, error=None, next=nxt)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login" if APP_PASSWORD else "/")


# ── Price data (Yahoo Finance, cached) ──────────────────────────────────────
# Remember downloaded prices so repeat tickers (and SPY) aren't re-fetched on every click.
CACHE_SECONDS = 4 * 60 * 60
_price_cache = {}  # ticker -> (fetched_at, dates, prices)

YAHOO_BUSY_MSG = "Yahoo Finance is limiting requests right now. Please wait a few minutes and try again."
YAHOO_DOWN_MSG = "Couldn't get data from Yahoo Finance right now (the service may be down). Please try again shortly."

TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")


class DataSourceError(Exception):
    """Yahoo Finance itself failed (outage or throttling), as opposed to a bad ticker."""


def _download_closes(ticker):
    """Download full-history adjusted closes. Returns (dates, prices); empty lists if Yahoo has nothing."""
    try:
        hist = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=True)
    except YFRateLimitError:
        raise DataSourceError(YAHOO_BUSY_MSG)
    except Exception:
        raise DataSourceError(YAHOO_DOWN_MSG)

    if hist is None or hist.empty or "Close" not in hist:
        return [], []
    closes = hist["Close"].dropna()
    return [d.strftime("%Y-%m-%d") for d in closes.index], [float(p) for p in closes.values]


def _cached(ticker):
    hit = _price_cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1], hit[2]
    return None


def fetch_daily_prices(ticker):
    """Fetch daily adjusted close prices from Yahoo Finance (no API key needed), with caching."""
    hit = _cached(ticker)
    if hit:
        return hit

    dates, prices = _download_closes(ticker)
    if not dates and "." in ticker:
        # Yahoo writes share classes with a dash (BRK-B), but people type BRK.B. Keep the dot
        # version first since foreign listings genuinely use it (VOD.L, SHOP.TO).
        dates, prices = _download_closes(ticker.replace(".", "-"))
    if not dates:
        # Empty result: either a bad ticker or Yahoo is failing quietly. Check a known-good ticker to tell which.
        if ticker != "SPY":
            spy_ok = bool(_cached("SPY")) or bool(_download_closes("SPY")[0])
            if spy_ok:
                raise ValueError(f"No price data found for {ticker}. Check the ticker symbol.")
        raise DataSourceError(YAHOO_DOWN_MSG)

    _price_cache[ticker] = (time.time(), dates, prices)
    return dates, prices


# ── Request parsing ─────────────────────────────────────────────────────────
PERIOD_DAYS = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
ALLOWED_WINDOWS = {10, 20, 30, 60, 90}


def parse_request_args():
    """Validate query parameters shared by /api/compare and /api/export. Raises ValueError with a friendly message."""
    ticker_a = request.args.get("a", "").upper().strip()
    ticker_b = request.args.get("b", "").upper().strip()
    period = request.args.get("period", "1y")

    if not ticker_a or not ticker_b:
        raise ValueError("Both tickers are required.")
    for t in (ticker_a, ticker_b):
        if not TICKER_RE.match(t):
            raise ValueError(f"'{t}' doesn't look like a ticker symbol.")
    if ticker_a == ticker_b:
        raise ValueError("Please enter two different tickers.")

    try:
        window = int(request.args.get("window", 30))
    except ValueError:
        window = 0
    if window not in ALLOWED_WINDOWS:
        raise ValueError("Invalid roll window.")

    today = datetime.today().date()
    if period == "custom":
        try:
            start = datetime.strptime(request.args.get("start", ""), "%Y-%m-%d").date()
            end = datetime.strptime(request.args.get("end", "") or today.isoformat(), "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Please pick both a start and an end date.")
        if start >= end:
            raise ValueError("The start date must be before the end date.")
        end = min(end, today)
    elif period in PERIOD_DAYS:
        start, end = today - timedelta(days=PERIOD_DAYS[period]), today
    else:
        raise ValueError("Invalid period.")

    return ticker_a, ticker_b, period, window, start.isoformat(), end.isoformat()


def filter_by_range(dates, prices, start, end):
    """Keep observations with start <= date <= end (ISO date strings compare correctly as text)."""
    kept = [(d, p) for d, p in zip(dates, prices) if start <= d <= end]
    return [d for d, _ in kept], [p for _, p in kept]


def align_three(dates_a, prices_a, dates_b, prices_b, dates_c, prices_c):
    map_a = dict(zip(dates_a, prices_a))
    map_b = dict(zip(dates_b, prices_b))
    map_c = dict(zip(dates_c, prices_c))
    common = sorted(set(map_a.keys()) & set(map_b.keys()) & set(map_c.keys()))
    return common, [map_a[d] for d in common], [map_b[d] for d in common], [map_c[d] for d in common]


def load_pair(ticker_a, ticker_b, start, end):
    """Fetch both tickers plus SPY, cut to the date range and keep only days all three traded."""
    series = [filter_by_range(*fetch_daily_prices(t), start, end) for t in (ticker_a, ticker_b, "SPY")]
    dates, prices_a, prices_b, prices_spy = align_three(*series[0], *series[1], *series[2])
    if len(dates) < 10:
        raise ValueError("Not enough overlapping data for those dates. Check your tickers or pick a longer range.")
    return dates, prices_a, prices_b, prices_spy


# ── Statistics ──────────────────────────────────────────────────────────────
def daily_returns(prices):
    arr = np.array(prices, dtype=float)
    return np.diff(arr) / arr[:-1]


def compute_stats(prices_a, prices_b, ticker_a, ticker_b):
    ret_a = daily_returns(prices_a)
    ret_b = daily_returns(prices_b)

    corr = float(np.corrcoef(ret_a, ret_b)[0, 1])
    r2 = corr ** 2

    cov = float(np.cov(ret_a, ret_b)[0, 1])
    var_b = float(np.var(ret_b, ddof=1))
    beta = cov / var_b if var_b != 0 else 0.0

    vol_a = float(np.std(ret_a, ddof=1) * np.sqrt(252))
    vol_b = float(np.std(ret_b, ddof=1) * np.sqrt(252))

    spread_returns = ret_a - ret_b
    tracking_error = float(np.std(spread_returns, ddof=1) * np.sqrt(252))

    total_ret_a = float((prices_a[-1] - prices_a[0]) / prices_a[0])
    total_ret_b = float((prices_b[-1] - prices_b[0]) / prices_b[0])

    def sharpe(rets):
        ann_ret = float(np.mean(rets) * 252)
        ann_vol = float(np.std(rets, ddof=1) * np.sqrt(252))
        return ann_ret / ann_vol if ann_vol != 0 else 0.0

    def max_drawdown(prices):
        arr = np.array(prices, dtype=float)
        peak = np.maximum.accumulate(arr)
        dd = (peak - arr) / peak
        return float(np.max(dd))

    return {
        "correlation": round(corr, 6),
        "r2": round(r2, 6),
        "beta": round(beta, 4),
        "tracking_error": round(tracking_error, 6),
        "ticker_a": {
            "symbol": ticker_a,
            "total_return": round(total_ret_a, 6),
            "ann_vol": round(vol_a, 6),
            "sharpe": round(sharpe(ret_a), 4),
            "max_drawdown": round(max_drawdown(prices_a), 6),
            "best_day": round(float(np.max(ret_a)), 6),
            "worst_day": round(float(np.min(ret_a)), 6),
        },
        "ticker_b": {
            "symbol": ticker_b,
            "total_return": round(total_ret_b, 6),
            "ann_vol": round(vol_b, 6),
            "sharpe": round(sharpe(ret_b), 4),
            "max_drawdown": round(max_drawdown(prices_b), 6),
            "best_day": round(float(np.max(ret_b)), 6),
            "worst_day": round(float(np.min(ret_b)), 6),
        },
    }


def compute_rolling(prices_a, prices_b, window=30):
    """Rolling correlation and rolling beta (A on B) of daily returns. Entry k belongs to price date k+1."""
    ret_a = daily_returns(prices_a)
    ret_b = daily_returns(prices_b)
    corr, beta = [], []
    for i in range(len(ret_a)):
        if i < window - 1:
            corr.append(None)
            beta.append(None)
            continue
        a_slice = ret_a[i - window + 1: i + 1]
        b_slice = ret_b[i - window + 1: i + 1]
        corr.append(round(float(np.corrcoef(a_slice, b_slice)[0, 1]), 4))
        var_b = float(np.var(b_slice, ddof=1))
        beta.append(round(float(np.cov(a_slice, b_slice)[0, 1]) / var_b, 4) if var_b > 0 else None)
    return corr, beta


MIN_COINT_DAYS = 60
MIN_HALF_LIFE_DAYS = 30


def compute_pair_signals(prices_a, prices_b):
    """Spread z-score, Engle-Granger cointegration test, hedge ratio and mean-reversion half-life."""
    a = np.array(prices_a, dtype=float)
    b = np.array(prices_b, dtype=float)

    ratio = a / b
    mean = float(np.mean(ratio))
    std = float(np.std(ratio, ddof=1))
    zscores = ((ratio - mean) / std) if std > 0 else np.zeros_like(ratio)

    log_a, log_b = np.log(a), np.log(b)
    hedge_ratio = float(np.polyfit(log_b, log_a, 1)[0])
    p_value = t_stat = None
    if len(a) >= MIN_COINT_DAYS:
        t, p, _ = coint(log_a, log_b)
        t_stat, p_value = round(float(t), 4), round(float(p), 4)

    # Half-life: regress the daily change in the hedged log spread on its previous level.
    # A negative slope means gaps tend to close; half-life = -ln(2) / slope.
    half_life = None
    if len(a) >= MIN_HALF_LIFE_DAYS:
        spread = log_a - hedge_ratio * log_b
        slope = float(np.polyfit(spread[:-1], np.diff(spread), 1)[0])
        if slope < 0:
            half_life = round(-np.log(2) / slope, 1)

    return {
        "spread_mean": round(mean, 6),
        "spread_std": round(std, 6),
        "zscores": [round(float(z), 4) for z in zscores],
        "current_z": round(float(zscores[-1]), 4),
        "coint_pvalue": p_value,
        "coint_tstat": t_stat,
        "hedge_ratio": round(hedge_ratio, 4),
        "half_life": half_life,
    }


def coint_verdict(signals):
    if signals["coint_pvalue"] is None:
        return f"Needs {MIN_COINT_DAYS}+ trading days (try a longer period)"
    return "Likely cointegrated (p < 0.05)" if signals["coint_pvalue"] < 0.05 else "Not cointegrated (p ≥ 0.05)"


def half_life_note(signals, n_days):
    if n_days < MIN_HALF_LIFE_DAYS:
        return f"Needs {MIN_HALF_LIFE_DAYS}+ trading days"
    if signals["half_life"] is None:
        return "No mean reversion detected"
    return "Trading days for a gap to halve"


def period_label(period, start, end):
    return f"{start} to {end}" if period == "custom" else period


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/api/compare")
def compare():
    try:
        ticker_a, ticker_b, period, window, start, end = parse_request_args()
        dates, prices_a, prices_b, prices_spy = load_pair(ticker_a, ticker_b, start, end)

        display_dates = [datetime.strptime(d, "%Y-%m-%d").strftime("%b %d '%y") for d in dates]

        idx_a   = [round(p / prices_a[0] * 100, 4) for p in prices_a]
        idx_b   = [round(p / prices_b[0] * 100, 4) for p in prices_b]
        idx_spy = [round(p / prices_spy[0] * 100, 4) for p in prices_spy]

        spread = [round(a / b, 6) for a, b in zip(prices_a, prices_b)]

        ret_a = [round((prices_a[i] - prices_a[i-1]) / prices_a[i-1] * 100, 4) for i in range(1, len(prices_a))]
        ret_b = [round((prices_b[i] - prices_b[i-1]) / prices_b[i-1] * 100, 4) for i in range(1, len(prices_b))]

        rolling_corr, rolling_beta = compute_rolling(prices_a, prices_b, window=window)
        stats = compute_stats(prices_a, prices_b, ticker_a, ticker_b)
        signals = compute_pair_signals(prices_a, prices_b)
        signals["coint_verdict"] = coint_verdict(signals)
        signals["half_life_note"] = half_life_note(signals, len(dates))

        return jsonify({
            "signals": signals,
            "dates": display_dates,
            "start": dates[0],
            "end": dates[-1],
            "indexed_a": idx_a,
            "indexed_b": idx_b,
            "indexed_spy": idx_spy,
            "spread": spread,
            "returns_a": ret_a,
            "returns_b": ret_b,
            "rolling_corr": rolling_corr,
            "rolling_beta": rolling_beta,
            "rolling_window": window,
            "stats": stats,
            "prices_a": [round(p, 4) for p in prices_a],
            "prices_b": [round(p, 4) for p in prices_b],
            "prices_spy": [round(p, 4) for p in prices_spy],
        })

    except DataSourceError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Data fetch failed: {str(e)}"}), 500


@app.route("/api/export")
def export():
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        return jsonify({"error": "openpyxl not installed."}), 500

    try:
        ticker_a, ticker_b, period, window, start, end = parse_request_args()
        dates, prices_a, prices_b, prices_spy = load_pair(ticker_a, ticker_b, start, end)

        rolling_corr, rolling_beta = compute_rolling(prices_a, prices_b, window=window)
        stats = compute_stats(prices_a, prices_b, ticker_a, ticker_b)
        signals = compute_pair_signals(prices_a, prices_b)

        wb = openpyxl.Workbook()

        dark_fill   = PatternFill("solid", fgColor="0A0C10")
        header_fill = PatternFill("solid", fgColor="C8A96E")
        alt_fill    = PatternFill("solid", fgColor="111318")
        white_font  = Font(color="E8E4DC", name="Courier New", size=10)
        dark_font   = Font(color="0A0C10", bold=True, name="Courier New", size=10)
        gold_font   = Font(color="C8A96E", bold=True, name="Courier New", size=12)
        muted_font  = Font(color="6B7280", name="Courier New", size=9)
        center      = Alignment(horizontal="center", vertical="center")
        left        = Alignment(horizontal="left", vertical="center")

        ws1 = wb.active
        ws1.title = "Summary"
        ws1.sheet_view.showGridLines = False

        ws1.merge_cells("A1:C1")
        ws1["A1"] = f"Pair Analysis: {ticker_a} vs {ticker_b}"
        ws1["A1"].font = gold_font
        ws1["A1"].fill = dark_fill
        ws1["A1"].alignment = left
        ws1.row_dimensions[1].height = 32

        ws1.merge_cells("A2:C2")
        ws1["A2"] = f"Period: {period_label(period, dates[0], dates[-1])}   |   Rolling window: {window} days"
        ws1["A2"].font = muted_font
        ws1["A2"].fill = dark_fill
        ws1["A2"].alignment = left
        ws1.row_dimensions[2].height = 18

        ws1.merge_cells("A4:C4")
        ws1["A4"] = "PAIR STATISTICS"
        ws1["A4"].font = Font(color="C8A96E", bold=True, name="Courier New", size=10)
        ws1["A4"].fill = dark_fill
        ws1["A4"].alignment = left
        ws1.row_dimensions[4].height = 20

        pair_rows = [
            ("Metric", "Value"),
            ("Correlation", stats["correlation"]),
            ("R²", stats["r2"]),
            (f"Beta ({ticker_a}/{ticker_b})", stats["beta"]),
            ("Tracking Error (Ann.)", f"{stats['tracking_error']*100:.2f}%"),
            ("Spread Z-Score (latest)", f"{signals['current_z']:+.2f}"),
            ("Cointegration p-value", signals["coint_pvalue"] if signals["coint_pvalue"] is not None else "n/a"),
            ("Cointegration result", coint_verdict(signals)),
            (f"Hedge Ratio (log {ticker_a} on log {ticker_b})", signals["hedge_ratio"]),
            ("Half-Life (trading days)", signals["half_life"] if signals["half_life"] is not None
                else half_life_note(signals, len(dates))),
        ]
        for i, (label, val) in enumerate(pair_rows, start=5):
            is_hdr = i == 5
            alt = i % 2 == 0
            for j, v in enumerate([label, val], start=1):
                cell = ws1.cell(row=i, column=j, value=v)
                cell.fill = header_fill if is_hdr else (alt_fill if alt else dark_fill)
                cell.font = dark_font if is_hdr else white_font
                cell.alignment = center
            ws1.row_dimensions[i].height = 20

        sr = 5 + len(pair_rows) + 2
        ws1.merge_cells(f"A{sr}:C{sr}")
        ws1[f"A{sr}"] = "INDIVIDUAL STOCK STATISTICS"
        ws1[f"A{sr}"].font = Font(color="C8A96E", bold=True, name="Courier New", size=10)
        ws1[f"A{sr}"].fill = dark_fill
        ws1[f"A{sr}"].alignment = left
        ws1.row_dimensions[sr].height = 20

        stock_rows = [
            ("Metric", ticker_a, ticker_b),
            ("Total Return", f"{stats['ticker_a']['total_return']*100:.2f}%", f"{stats['ticker_b']['total_return']*100:.2f}%"),
            ("Ann. Volatility", f"{stats['ticker_a']['ann_vol']*100:.2f}%", f"{stats['ticker_b']['ann_vol']*100:.2f}%"),
            ("Sharpe (rf=0)", f"{stats['ticker_a']['sharpe']:.2f}", f"{stats['ticker_b']['sharpe']:.2f}"),
            ("Max Drawdown", f"-{stats['ticker_a']['max_drawdown']*100:.2f}%", f"-{stats['ticker_b']['max_drawdown']*100:.2f}%"),
            ("Best Day", f"{stats['ticker_a']['best_day']*100:.2f}%", f"{stats['ticker_b']['best_day']*100:.2f}%"),
            ("Worst Day", f"{stats['ticker_a']['worst_day']*100:.2f}%", f"{stats['ticker_b']['worst_day']*100:.2f}%"),
        ]
        for i, row_data in enumerate(stock_rows, start=sr + 1):
            is_hdr = i == sr + 1
            alt = i % 2 == 0
            for j, v in enumerate(row_data, start=1):
                cell = ws1.cell(row=i, column=j, value=v)
                cell.fill = header_fill if is_hdr else (alt_fill if alt else dark_fill)
                cell.font = dark_font if is_hdr else white_font
                cell.alignment = center
            ws1.row_dimensions[i].height = 20

        for col, w in [(1, 36), (2, 30), (3, 16)]:
            ws1.column_dimensions[get_column_letter(col)].width = w

        ws2 = wb.create_sheet("Price Data")
        ws2.sheet_view.showGridLines = False

        idx_a   = [round(p / prices_a[0] * 100, 4) for p in prices_a]
        idx_b   = [round(p / prices_b[0] * 100, 4) for p in prices_b]
        idx_spy = [round(p / prices_spy[0] * 100, 4) for p in prices_spy]
        spread  = [round(a / b, 6) for a, b in zip(prices_a, prices_b)]

        headers = ["Date", ticker_a, ticker_b, "SPY",
                   f"{ticker_a} Idx", f"{ticker_b} Idx", "SPY Idx",
                   f"Spread ({ticker_a}/{ticker_b})", "Spread Z-Score",
                   f"Rolling Corr ({window}d)", f"Rolling Beta ({window}d)"]
        for j, h in enumerate(headers, start=1):
            cell = ws2.cell(row=1, column=j, value=h)
            cell.fill = header_fill
            cell.font = dark_font
            cell.alignment = center
        ws2.row_dimensions[1].height = 22

        for i, date in enumerate(dates):
            rn = i + 2
            alt = i % 2 == 0
            row_vals = [
                date,
                round(prices_a[i], 4),
                round(prices_b[i], 4),
                round(prices_spy[i], 4),
                idx_a[i], idx_b[i], idx_spy[i],
                spread[i],
                signals["zscores"][i],
                # rolling series are returns-based, so entry k belongs to date k+1
                rolling_corr[i - 1] if i >= 1 else None,
                rolling_beta[i - 1] if i >= 1 else None,
            ]
            for j, val in enumerate(row_vals, start=1):
                cell = ws2.cell(row=rn, column=j, value=val)
                cell.fill = alt_fill if alt else dark_fill
                cell.font = white_font
                cell.alignment = center
            ws2.row_dimensions[rn].height = 18

        for col, w in enumerate([14,12,12,12,12,12,12,22,16,20,20], start=1):
            ws2.column_dimensions[get_column_letter(col)].width = w

        buf = io.BytesIO()
        wb.save(buf)
        buf.seek(0)

        file_period = f"{dates[0]}_to_{dates[-1]}" if period == "custom" else period
        return send_file(
            buf,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name=f"pair_analysis_{ticker_a}_{ticker_b}_{file_period}.xlsx",
        )

    except DataSourceError as e:
        return jsonify({"error": str(e)}), 503
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Export failed: {str(e)}"}), 500


@app.route("/")
def index():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html"), encoding="utf-8") as f:
        html = f.read()
    if APP_PASSWORD:
        html = html.replace("<!--SIGN_OUT-->", '<a href="/logout" class="signout">Sign out</a>')
    return html


if __name__ == "__main__":
    print("\n✓ Pair Analysis server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
