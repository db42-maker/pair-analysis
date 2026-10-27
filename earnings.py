"""Earnings calendar for the companies we follow.

Dates come from Yahoo Finance (cached), with manual overrides layered on top. Overrides and the
written call previews live in a small SQLite database under DATA_DIR. On Railway, DATA_DIR must
point at a mounted volume, or everything saved is lost on the next deploy.
"""

import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import date, datetime, timezone

import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COVERAGE_PATH = os.path.join(BASE_DIR, "coverage.json")
DATA_DIR = os.environ.get("DATA_DIR") or os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "calendar.db")

# On Railway the app's own disk is wiped on every deploy; only a mounted volume (DATA_DIR) survives.
ON_RAILWAY = bool(os.environ.get("RAILWAY_ENVIRONMENT_NAME") or os.environ.get("RAILWAY_ENVIRONMENT"))
STORAGE_PERSISTENT = bool(os.environ.get("DATA_DIR")) or not ON_RAILWAY

TIMINGS = {"bmo": "Before open", "amc": "After close", "dmh": "During market", "unknown": "Time TBD"}
MATCH_DAYS = 21          # a preview or override within this many days of an event belongs to that event
CACHE_SECONDS = 12 * 60 * 60
FETCH_WORKERS = 6
MAX_PREVIEW_CHARS = 100_000


# ── Coverage list ───────────────────────────────────────────────────────────
def load_coverage():
    """[{'ticker': 'OTIS', 'name': 'Otis Worldwide'}, …] from coverage.json, read fresh each time."""
    with open(COVERAGE_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    return [{"ticker": r["ticker"].strip().upper(), "name": r.get("name") or r["ticker"]} for r in rows]


def coverage_tickers():
    return {c["ticker"] for c in load_coverage()}


# ── Storage ─────────────────────────────────────────────────────────────────
def _connect():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS previews (
            ticker TEXT NOT NULL, event_date TEXT NOT NULL, body TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, event_date));
        CREATE TABLE IF NOT EXISTS date_overrides (
            ticker TEXT NOT NULL, event_date TEXT NOT NULL, timing TEXT NOT NULL, updated_at TEXT NOT NULL,
            PRIMARY KEY (ticker, event_date));
    """)
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _days_apart(a, b):
    return abs((date.fromisoformat(a) - date.fromisoformat(b)).days)


def _nearest(rows, event_date):
    """The row whose event_date is closest to event_date, if within MATCH_DAYS."""
    close = [r for r in rows if _days_apart(r["event_date"], event_date) <= MATCH_DAYS]
    return min(close, key=lambda r: _days_apart(r["event_date"], event_date)) if close else None


def get_preview(ticker, event_date):
    with closing(_connect()) as conn:
        rows = conn.execute("SELECT * FROM previews WHERE ticker = ?", (ticker,)).fetchall()
    row = _nearest(rows, event_date)
    return dict(row) if row else None


def save_preview(ticker, event_date, body):
    """Replace the preview for this event (any preview within MATCH_DAYS). An empty body deletes it."""
    body = (body or "").strip()
    with closing(_connect()) as conn, conn:
        rows = conn.execute("SELECT event_date FROM previews WHERE ticker = ?", (ticker,)).fetchall()
        for r in rows:
            if _days_apart(r["event_date"], event_date) <= MATCH_DAYS:
                conn.execute("DELETE FROM previews WHERE ticker = ? AND event_date = ?", (ticker, r["event_date"]))
        if body:
            conn.execute("INSERT INTO previews VALUES (?, ?, ?, ?)", (ticker, event_date, body, _now()))


def set_date_override(ticker, event_date, timing, replaces=None):
    """Pin a confirmed date/time. `replaces` is the date the event showed before (Yahoo's or an earlier pin);
    the old pin and any preview attached to it move to the new date."""
    with closing(_connect()) as conn, conn:
        for old in filter(None, {replaces, event_date}):
            for r in conn.execute("SELECT event_date FROM date_overrides WHERE ticker = ?", (ticker,)).fetchall():
                if _days_apart(r["event_date"], old) <= MATCH_DAYS:
                    conn.execute("DELETE FROM date_overrides WHERE ticker = ? AND event_date = ?", (ticker, r["event_date"]))
        if replaces and replaces != event_date:
            prev = _nearest(conn.execute("SELECT * FROM previews WHERE ticker = ?", (ticker,)).fetchall(), replaces)
            if prev and prev["event_date"] != event_date:
                conn.execute("DELETE FROM previews WHERE ticker = ? AND event_date = ?", (ticker, event_date))
                conn.execute("UPDATE previews SET event_date = ? WHERE ticker = ? AND event_date = ?",
                             (event_date, ticker, prev["event_date"]))
        conn.execute("INSERT INTO date_overrides VALUES (?, ?, ?, ?)", (ticker, event_date, timing, _now()))


def clear_date_override(ticker, event_date):
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM date_overrides WHERE ticker = ? AND event_date = ?", (ticker, event_date))


# ── Yahoo dates (cached) ────────────────────────────────────────────────────
_yahoo_cache = {}  # ticker -> (fetched_at, events)


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)  # NaN -> None


def _timing(ts):
    if ts.hour == 0 and ts.minute == 0:
        return "unknown"
    hm = ts.hour + ts.minute / 60
    return "bmo" if hm < 9.5 else "amc" if hm >= 16 else "dmh"


def yahoo_events(ticker):
    """Past and upcoming earnings dates from Yahoo. None if Yahoo failed (not cached, so it retries)."""
    hit = _yahoo_cache.get(ticker)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        tk = yf.Ticker(ticker)
        df = tk.get_earnings_dates(limit=12)
    except Exception:
        return None

    events = []
    if df is not None and not df.empty:
        for ts, row in df.iterrows():
            if ts.tzinfo is not None:
                ts = ts.tz_convert("America/New_York")
            events.append({"date": ts.date().isoformat(), "timing": _timing(ts),
                           "eps_estimate": _num(row.get("EPS Estimate")),
                           "eps_reported": _num(row.get("Reported EPS")), "estimated": False})

    # Yahoo often stamps an upcoming call with a placeholder time (e.g. 3pm for a company that always reports
    # after the close). If the company has a consistent habit, use it and say so.
    history = [e["timing"] for e in events if e["eps_reported"] is not None][:8]
    habit = max(set(history), key=history.count) if history else None
    if habit in ("bmo", "amc") and history.count(habit) >= 0.75 * len(history):
        for e in events:
            if e["eps_reported"] is None and e["timing"] in ("dmh", "unknown") and e["date"] >= date.today().isoformat():
                e.update(timing=habit, timing_inferred=True)

    # When Yahoo only has a window ("Oct 28 – Nov 3"), flag the upcoming event as an estimate.
    try:
        window = [d for d in (tk.calendar or {}).get("Earnings Date", []) if isinstance(d, date)]
    except Exception:
        window = []
    if len(window) >= 2 and window[0] != window[-1]:
        first, last = window[0].isoformat(), window[-1].isoformat()
        match = [e for e in events if first <= e["date"] <= last]
        if not match:
            events.append({"date": first, "timing": "unknown", "eps_estimate": None, "eps_reported": None})
            match = events[-1:]
        for e in match:
            e.update(estimated=True, window=f"{window[0]:%b %d} – {window[-1]:%b %d}")

    _yahoo_cache[ticker] = (time.time(), events)
    return events


# ── Calendar ────────────────────────────────────────────────────────────────
def events_between(start, end):
    """Every covered company's events with start <= date <= end, overrides applied, preview flags attached."""
    coverage = load_coverage()
    tickers = [c["ticker"] for c in coverage]
    with ThreadPoolExecutor(FETCH_WORKERS) as pool:
        yahoo = dict(zip(tickers, pool.map(yahoo_events, tickers)))
    with closing(_connect()) as conn:
        overrides = [dict(r) for r in conn.execute("SELECT * FROM date_overrides")]
        previews = [dict(r) for r in conn.execute("SELECT ticker, event_date, updated_at FROM previews")]

    events, unavailable = [], []
    for c in coverage:
        t = c["ticker"]
        if yahoo[t] is None:
            unavailable.append(t)
        pins = [o for o in overrides if o["ticker"] == t]
        merged = []
        for e in yahoo[t] or []:
            if not any(_days_apart(e["date"], p["event_date"]) <= MATCH_DAYS for p in pins):
                merged.append({**e, "source": "yahoo"})
        for p in pins:
            replaced = _nearest([{**e, "event_date": e["date"]} for e in yahoo[t] or []], p["event_date"])
            merged.append({"date": p["event_date"], "timing": p["timing"], "source": "manual", "estimated": False,
                           "eps_estimate": replaced["eps_estimate"] if replaced else None,
                           "eps_reported": replaced["eps_reported"] if replaced else None})

        own_previews = [p for p in previews if p["ticker"] == t]
        for e in merged:
            if start <= e["date"] <= end:
                prev = _nearest(own_previews, e["date"])
                label = TIMINGS[e["timing"]] + (" (usual time)" if e.get("timing_inferred") else "")
                events.append({**e, "ticker": t, "name": c["name"], "timing_label": label,
                               "has_preview": bool(prev), "preview_updated": prev["updated_at"] if prev else None})

    order = {"bmo": 0, "dmh": 1, "amc": 2, "unknown": 3}
    events.sort(key=lambda e: (e["date"], order[e["timing"]], e["ticker"]))
    return events, unavailable
