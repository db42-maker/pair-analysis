"""Portfolio macro sensitivity: how a hand-entered book of up to 12 names moves with the
10-year Treasury yield, WTI crude and ULSD diesel.

The book is today's holdings applied to each name's past returns. Long, short and net books are
each regressed on the macro series (alone, and alongside SPY so the market's influence is removed).

Crude and diesel are Yahoo's front-month futures (CL=F, HO=F). Yahoo splices contracts together
without adjusting, so the day it switches to the next contract shows a fake jump. Those days are
found from the exchanges' expiry rules and set to zero change before anything is computed.
"""

import io
import math
import re
from datetime import date, datetime, timedelta

import numpy as np

MAX_HOLDINGS = 12
TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,15}$")
LOOKBACK_DAYS = {"6mo": 182, "1y": 365, "2y": 730}
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}
ROLLING_WINDOW = {"daily": 60, "weekly": 26}
MIN_OBS = {"daily": 60, "weekly": 20}

# Comparison points, shown as if 100% of NAV were invested. ETFs rather than the indexes so dividends
# are included, as they are in the stocks' adjusted prices.
BENCHMARKS = [("spy", "S&P 500 (SPY)", "SPY"), ("iwm", "Russell 2000 (IWM)", "IWM")]

# key, label, Yahoo symbol, roll rule, shock unit used for sensitivities
SERIES = [
    ("rates",  "10-Year Treasury Yield", "^TNX", None, "bp"),
    ("crude",  "WTI Crude",              "CL=F", "cl", "pct"),
    ("diesel", "ULSD Diesel",            "HO=F", "ho", "pct"),
]
RATE_SHOCK_BP = 100    # sensitivities are quoted per +100bp in the 10-year yield…
OIL_SHOCK_PCT = 10     # …and per +10% in crude or diesel


# ── Input ───────────────────────────────────────────────────────────────────
def parse_holdings(raw):
    """'OTIS:L:15,XLY:S:8,BALL:L' -> [(ticker, signed weight as a fraction of NAV)].
    Weights are % of NAV. Leave every weight blank for an equal-weight book (gross 100%)."""
    parts = [p for p in (raw or "").upper().replace(" ", "").split(",") if p]
    if not parts:
        raise ValueError("Enter at least one ticker.")
    if len(parts) > MAX_HOLDINGS:
        raise ValueError(f"Enter at most {MAX_HOLDINGS} names.")
    rows = []
    for p in parts:
        ticker, _, rest = p.partition(":")
        side, _, weight = rest.partition(":")
        side = side or "L"
        if not TICKER_RE.match(ticker):
            raise ValueError(f"'{ticker}' doesn't look like a ticker symbol.")
        if side not in ("L", "S"):
            raise ValueError(f"Pick long or short for {ticker}.")
        if weight:
            try:
                w = float(weight.rstrip("%"))
            except ValueError:
                raise ValueError(f"'{weight}' isn't a valid weight for {ticker}.")
            if w <= 0:
                raise ValueError(f"Enter weights as positive numbers; long/short sets the sign ({ticker}).")
        else:
            w = None
        rows.append((ticker, side, w))

    tickers = [t for t, _, _ in rows]
    dupes = sorted({t for t in tickers if tickers.count(t) > 1})
    if dupes:
        raise ValueError(f"{', '.join(dupes)} is listed more than once.")
    if all(w is None for _, _, w in rows):
        return [(t, (1 if s == "L" else -1) / len(rows)) for t, s, _ in rows]
    if any(w is None for _, _, w in rows):
        raise ValueError("Give a weight for every name, or leave all weights blank for equal weight.")
    return [(t, (1 if s == "L" else -1) * w / 100) for t, s, w in rows]


# ── Futures roll days ───────────────────────────────────────────────────────
def _roll_days(trading_days, rule):
    """First trading day after each front-month expiry, using the series' own trading days as the
    business calendar. WTI (CL) stops trading 3 business days before the 25th of the month before
    delivery (4 if the 25th isn't a business day). ULSD (HO) stops on the last business day of the month."""
    days = sorted(trading_days)
    by_month = {}
    for d in days:
        by_month.setdefault(d[:7], []).append(d)
    rolls = set()
    for month, month_days in by_month.items():
        if rule == "cl":
            the_25th = f"{month}-25"
            if month_days[-1] < the_25th:
                continue  # the current month: expiry hasn't happened yet
            before = [d for d in month_days if d < the_25th]
            back = 3 if the_25th in month_days else 4
            if len(before) < back:
                continue
            expiry = before[-back]
        else:
            expiry = month_days[-1]
        i = days.index(expiry)
        if i + 1 < len(days):
            rolls.add(days[i + 1])
    return rolls


def roll_adjusted(dates, prices, rule):
    """Continuous index (starting at the first price) with each roll day's return set to zero."""
    rolls = _roll_days(dates, rule)
    out = [prices[0]]
    for i in range(1, len(prices)):
        ret = 0.0 if dates[i] in rolls else prices[i] / prices[i - 1] - 1
        out.append(out[-1] * (1 + ret))
    return out, rolls


# ── Statistics ──────────────────────────────────────────────────────────────
def _ols(y, X):
    """OLS of y on [1, X]. Returns (slopes, t-stats, r2)."""
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = len(y) - A.shape[1]
    s2 = float(resid @ resid) / dof if dof > 0 else 0.0
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(A.T @ A)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    t = np.divide(coef, se, out=np.zeros_like(coef), where=se > 0)
    return coef[1:], t[1:], r2


def _corr(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 0.0


def _shock(unit):
    """Size of the move a sensitivity is quoted for, in the regressor's units."""
    return RATE_SHOCK_BP / 100 if unit == "bp" else OIL_SHOCK_PCT / 100


def _rolling_corr(a, b, window):
    out = [None] * len(a)
    for i in range(window - 1, len(a)):
        out[i] = round(_corr(a[i - window + 1:i + 1], b[i - window + 1:i + 1]), 4)
    return out


# ── Main entry point ────────────────────────────────────────────────────────
def analyze(holdings, fetch_prices, lookback="1y", frequency="daily", nav=None):
    """fetch_prices(ticker) -> (dates, prices); raises ValueError for an unknown ticker."""
    if lookback not in LOOKBACK_DAYS:
        raise ValueError("Invalid lookback.")
    if frequency not in PERIODS_PER_YEAR:
        raise ValueError("Invalid return frequency.")
    if frequency == "weekly" and lookback == "6mo":
        raise ValueError("Six months of weekly returns is too few. Use daily, or a longer lookback.")

    start = (date.today() - timedelta(days=LOOKBACK_DAYS[lookback])).isoformat()
    tickers = [t for t, _ in holdings]

    # Calendar: SPY's trading days. Everything else is carried across its own holidays.
    spy_dates, spy_prices = fetch_prices("SPY")
    base = [d for d in spy_dates if d >= start]

    def aligned(dates, prices):
        out, last, j = [], None, 0
        for d in base:
            while j < len(dates) and dates[j] <= d:
                last = prices[j]
                j += 1
            out.append(last)
        return out

    raw_levels, levels, roll_notes = {}, {}, {}
    for key, label, symbol, rule, _ in SERIES:
        dates, prices = fetch_prices(symbol)
        dates, prices = [d for d in dates if d >= start], [p for d, p in zip(dates, prices) if d >= start]
        raw_levels[key] = aligned(dates, prices)
        if rule:
            adj, rolls = roll_adjusted(dates, prices, rule)
            levels[key] = aligned(dates, adj)
            roll_notes[key] = len(rolls)
        else:
            levels[key] = raw_levels[key]

    holding_levels, problems = {}, []
    for t in tickers:
        try:
            dates, prices = fetch_prices(t)
        except ValueError as e:
            problems.append(str(e))
            continue
        lv = aligned(dates, prices)
        if sum(v is None for v in lv) > 0.2 * len(base):
            problems.append(f"{t} doesn't have enough price history for this lookback.")
        holding_levels[t] = lv
    if problems:
        raise ValueError(" ".join(problems))

    # Keep only dates where every series has a value, then sample weekly if asked.
    bench_levels = {sym: aligned(*fetch_prices(sym)) for _, _, sym in BENCHMARKS if sym != "SPY"}
    all_series = {"SPY": aligned(spy_dates, spy_prices), **bench_levels, **holding_levels,
                  **{f"_{k}": v for k, v in levels.items()}}
    keep = [i for i in range(len(base)) if all(s[i] is not None for s in all_series.values())]
    if frequency == "weekly":
        weeks = {}
        for i in keep:
            weeks[datetime.strptime(base[i], "%Y-%m-%d").isocalendar()[:2]] = i
        keep = sorted(weeks.values())
    if len(keep) - 1 < MIN_OBS[frequency]:
        raise ValueError("Not enough overlapping history for these names. Try a longer lookback.")
    sample = [base[i] for i in keep]

    def rets(series):
        lv = np.array([series[i] for i in keep], dtype=float)
        return lv[1:] / lv[:-1] - 1

    spy = rets(all_series["SPY"])
    R = {t: rets(holding_levels[t]) for t in tickers}
    X = {}
    for key, _, _, _, unit in SERIES:
        lv = np.array([levels[key][i] for i in keep], dtype=float)
        X[key] = np.diff(lv) if unit == "bp" else lv[1:] / lv[:-1] - 1  # yield change in percentage points

    w = dict(holdings)
    books = {
        "long": sum((w[t] * R[t] for t in tickers if w[t] > 0), np.zeros(len(spy))),
        "short": sum((w[t] * R[t] for t in tickers if w[t] < 0), np.zeros(len(spy))),
    }
    books["net"] = books["long"] + books["short"]
    has_short, has_long = any(v < 0 for v in w.values()), any(v > 0 for v in w.values())

    bench = {key: rets(all_series[sym]) for key, _, sym in BENCHMARKS}

    def sensitivity(y, key, unit, ex_mkt=True):
        (b,), (t,), r2 = _ols(y, X[key][:, None])
        k = _shock(unit)
        out = {"corr": round(_corr(y, X[key]), 4), "r2": round(r2, 4),
               "impact": round(float(b * k), 6), "t": round(float(t), 2),        # fraction of NAV per shock
               "impact_ex_mkt": None, "t_ex_mkt": None}
        if ex_mkt:  # meaningless for SPY itself: removing the market from the market leaves nothing
            (_, b_x), (_, t_x), _ = _ols(y, np.column_stack([spy, X[key]]))
            out.update(impact_ex_mkt=round(float(b_x * k), 6), t_ex_mkt=round(float(t_x), 2))
        return out

    stats = []
    for book in ("long", "short", "net"):
        if (book == "short" and not has_short) or (book == "long" and not has_long):
            continue
        y = books[book]
        stats.append({"book": book, "label": f"{book.title()} book", "benchmark": False,
                      "spy_corr": round(_corr(y, spy), 4),
                      "series": {key: sensitivity(y, key, unit) for key, _, _, _, unit in SERIES}})
    for key, label, _ in BENCHMARKS:
        y = bench[key]
        stats.append({"book": key, "label": label, "benchmark": True, "spy_corr": round(_corr(y, spy), 4),
                      "series": {s: sensitivity(y, s, unit, ex_mkt=key != "spy") for s, _, _, _, unit in SERIES}})

    # Each position's share of the book's sensitivity (they add up exactly: same sample, same regressors).
    positions = []
    for t in tickers:
        row = {"ticker": t, "weight": w[t], "series": {}}
        for key, _, _, _, unit in SERIES:
            s = sensitivity(R[t], key, unit)
            row["series"][key] = {"corr": s["corr"], "t": s["t"], "t_ex_mkt": s["t_ex_mkt"],
                                  "beta_impact": s["impact"], "beta_impact_ex_mkt": s["impact_ex_mkt"],
                                  "impact": round(w[t] * s["impact"], 6),
                                  "impact_ex_mkt": round(w[t] * s["impact_ex_mkt"], 6)}
        positions.append(row)

    # Scenario model: rates and one oil series fitted together (crude and diesel move too closely to fit jointly).
    scenario = {}
    for oil in ("crude", "diesel"):
        scenario[oil] = {}
        for book, y in {**books, **bench}.items():
            (b_r, b_o), (t_r, t_o), r2 = _ols(y, np.column_stack([X["rates"], X[oil]]))
            scenario[oil][book] = {"per_100bp": round(b_r, 6), "per_10pct": round(b_o * 0.10, 6),
                                   "t_rates": round(float(t_r), 2), "t_oil": round(float(t_o), 2), "r2": round(r2, 4)}

    window = ROLLING_WINDOW[frequency]
    net_index = np.concatenate([[0.0], np.cumsum(books["net"])])  # cumulative P&L, % of NAV (simple sum)
    return {
        "holdings": [{"ticker": t, "weight": w[t]} for t in tickers],
        "summary": {"long": sum(v for v in w.values() if v > 0), "short": sum(v for v in w.values() if v < 0),
                    "net": sum(w.values()), "gross": sum(abs(v) for v in w.values())},
        "nav": nav, "lookback": lookback, "frequency": frequency,
        "start": sample[0], "end": sample[-1], "n_obs": len(sample) - 1,
        "series_defs": [{"key": k, "label": l, "symbol": s, "unit": u,
                         "shock": f"+{RATE_SHOCK_BP}bp" if u == "bp" else f"+{OIL_SHOCK_PCT}%"} for k, l, s, _, u in SERIES],
        "roll_days_removed": roll_notes,
        "stats": stats,
        "positions": positions,
        "scenario": scenario,
        "series_moves": {k: {"start": round(raw_levels[k][keep[0]], 3), "end": round(raw_levels[k][keep[-1]], 3)}
                         for k, *_ in SERIES},
        "chart": {
            "dates": sample,
            "net_cum": [round(float(v), 6) for v in net_index],
            "levels": {k: [round(raw_levels[k][i], 3) for i in keep] for k, *_ in SERIES},
            "rolling_window": window,
            "rolling_corr": {k: _rolling_corr(books["net"], X[k], window) for k, *_ in SERIES},
            "rolling_corr_bench": {b: {k: _rolling_corr(bench[b], X[k], window) for k, *_ in SERIES}
                                   for b, *_ in BENCHMARKS},
            # Each name's own returns (not signed by long/short), so the line describes the stock itself.
            "rolling_corr_names": {t: {k: _rolling_corr(R[t], X[k], window) for k, *_ in SERIES} for t in tickers},
        },
        "benchmark_defs": [{"key": k, "label": l, "symbol": s} for k, l, s in BENCHMARKS],
    }


# ── Excel export ────────────────────────────────────────────────────────────
def build_workbook(result):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter

    dark_fill = PatternFill("solid", fgColor="0A0C10")
    header_fill = PatternFill("solid", fgColor="C8A96E")
    alt_fill = PatternFill("solid", fgColor="111318")
    white_font = Font(color="E8E4DC", name="Courier New", size=10)
    dark_font = Font(color="0A0C10", bold=True, name="Courier New", size=10)
    gold_font = Font(color="C8A96E", bold=True, name="Courier New", size=12)
    muted_font = Font(color="6B7280", name="Courier New", size=9)
    center = Alignment(horizontal="center", vertical="center")
    left = Alignment(horizontal="left", vertical="center")
    PCT, PCT2, NUM = "0.0%", "0.00%", "0.00"

    def table(ws, top, headers, rows, formats, widths=None):
        for j, h in enumerate(headers, start=1):
            c = ws.cell(row=top, column=j, value=h)
            c.fill, c.font, c.alignment = header_fill, dark_font, center
        for i, row in enumerate(rows, start=top + 1):
            for j, (v, fmt) in enumerate(zip(row, formats), start=1):
                c = ws.cell(row=i, column=j, value=v)
                c.fill = alt_fill if i % 2 == 0 else dark_fill
                c.font, c.alignment = white_font, (left if fmt == "text" else center)
                if fmt not in ("text", None):
                    c.number_format = fmt
        for j, wd in enumerate(widths or [], start=1):
            ws.column_dimensions[get_column_letter(j)].width = wd
        return top + len(rows) + 2

    def banner(ws, row, text, font):
        ws.cell(row=row, column=1, value=text).font = font
        for j in range(1, 12):
            ws.cell(row=row, column=j).fill = dark_fill

    defs = result["series_defs"]
    nav = result["nav"]
    wb = openpyxl.Workbook()

    ws = wb.active
    ws.title = "Summary"
    ws.sheet_view.showGridLines = False
    banner(ws, 1, "Portfolio Macro Sensitivity", gold_font)
    banner(ws, 2, f"{result['frequency'].title()} returns {result['start']} to {result['end']} ({result['n_obs']} obs)"
                  f"  |  Impact = % of NAV for {', '.join(d['label'] + ' ' + d['shock'] for d in defs)}", muted_font)
    nxt = table(ws, 4, ["Ticker", "Weight (% NAV)"], [(h["ticker"], h["weight"]) for h in result["holdings"]],
                ["text", PCT], [26, 14, 12, 14, 10, 16, 10, 14, 14])
    for d in defs:
        rows = []
        for s in result["stats"]:
            v = s["series"][d["key"]]
            ex = v["impact_ex_mkt"]
            rows.append((s["label"] + (" — as 100% of NAV" if s["benchmark"] else ""), v["corr"], v["r2"], v["impact"], v["t"],
                         ex if ex is not None else "n/a", v["t_ex_mkt"] if ex is not None else "n/a",
                         v["impact"] * nav if nav else None, ex * nav if nav and ex is not None else None))
        ws.cell(row=nxt, column=1, value=f"{d['label'].upper()} ({d['symbol']}), per {d['shock']}").font = gold_font
        nxt = table(ws, nxt + 1, ["Book", "Corr", "R²", "Impact", "t-stat", "Impact ex-Mkt", "t-stat", "Impact $", "Impact ex-Mkt $"],
                    rows, ["text", NUM, NUM, PCT2, NUM, PCT2, NUM, "#,##0", "#,##0"])

    ws = wb.create_sheet("Positions")
    ws.sheet_view.showGridLines = False
    headers = ["Ticker", "Weight"]
    for d in defs:
        headers += [f"{d['label']} corr", f"Impact {d['shock']}", "Impact ex-Mkt"]
    rows = []
    for p in result["positions"]:
        row = [p["ticker"], p["weight"]]
        for d in defs:
            v = p["series"][d["key"]]
            row += [v["corr"], v["impact"], v["impact_ex_mkt"]]
        rows.append(row)
    for s in result["stats"]:
        if s["benchmark"]:  # comparison rows: the benchmark held at 100% of NAV
            row = [s["label"], 1.0]
            for d in defs:
                v = s["series"][d["key"]]
                row += [v["corr"], v["impact"], v["impact_ex_mkt"] if v["impact_ex_mkt"] is not None else "n/a"]
            rows.append(row)
    table(ws, 1, headers, rows, ["text", PCT] + [NUM, PCT2, PCT2] * len(defs), [22, 10] + [16, 14, 14] * len(defs))

    ws = wb.create_sheet("Data")
    ws.sheet_view.showGridLines = False
    ch = result["chart"]
    headers = ["Date", "Net cumulative P&L (% NAV)"] + [f"{d['label']} ({d['symbol']})" for d in defs]
    rows = [[dt, ch["net_cum"][i]] + [ch["levels"][d["key"]][i] for d in defs] for i, dt in enumerate(ch["dates"])]
    table(ws, 1, headers, rows, ["text", PCT2] + [NUM] * len(defs), [12, 24, 24, 18, 18])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf
