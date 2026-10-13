"""Hedge finder: given a long, rank candidate hedges (industry peers, ETFs, user picks) and build
the best small basket. Pure analytics plus peer lookup; data fetching and routes live in app.py."""
import numpy as np
import yfinance as yf
from scipy.optimize import nnls
from statsmodels.tsa.stattools import coint

BROAD_ETFS = ["SPY", "QQQ", "IWM", "MDY", "RSP"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
# Narrower industry ETFs worth trying for each Yahoo sector.
INDUSTRY_ETFS = {
    "technology": ["IGV", "SMH", "SOXX", "SKYY", "HACK"],
    "communication-services": ["SOCL", "IYZ"],
    "consumer-cyclical": ["XHB", "ITB", "XRT", "IBUY", "PEJ", "CARZ"],
    "consumer-defensive": ["PBJ", "KXI"],
    "energy": ["XOP", "OIH", "AMLP", "URA"],
    "financial-services": ["KRE", "KBE", "IAI", "KIE", "IPAY"],
    "healthcare": ["XBI", "IBB", "IHI", "IHF", "XPH"],
    "industrials": ["XAR", "ITA", "IYT", "JETS", "PAVE", "XTN"],
    "basic-materials": ["XME", "GDX", "LIT", "WOOD", "SLX"],
    "real-estate": ["VNQ", "REZ", "INDS"],
    "utilities": ["ICLN", "TAN"],
}

MAX_PEERS = 15
MAX_BASKET = 5
BASKET_POOL = 12          # best single hedges considered for the basket
MIN_BASKET_SHARE = 0.05   # drop basket members under 5% of the hedge
MIN_COVERAGE = 0.9        # a candidate must have prices on 90% of the long's trading days
ROLL_DAYS = 60


def lookup_universe(ticker):
    """Industry peers and relevant ETFs from Yahoo. Returns (peers, etfs, sector, industry); never raises."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    sector, industry = info.get("sectorKey"), info.get("industryKey")
    peers = []
    if industry:
        try:
            top = yf.Industry(industry).top_companies
            if top is not None and not top.empty:
                peers = [str(t).upper() for t in top.index if str(t).upper() != ticker][:MAX_PEERS]
        except Exception:
            pass
    etfs = BROAD_ETFS + SECTOR_ETFS + INDUSTRY_ETFS.get(sector, [])
    return peers, etfs, info.get("sector"), info.get("industry")


def _returns(prices):
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1


def _fit(r_long, r_hedges):
    """Long-only hedge amounts b (dollars short per $1 long) minimising the variance of r_long - r_hedges @ b."""
    X = r_hedges - r_hedges.mean(axis=0)
    y = r_long - r_long.mean()
    b, _ = nnls(X, y)
    return b


def _risk_removed(r_long, hedged):
    var_long = np.var(r_long, ddof=1)
    return float(1 - np.var(hedged, ddof=1) / var_long) if var_long > 0 else 0.0


def _evaluate(dates, long_px, hedge_px, b):
    """Statistics for a hedge made of the columns of hedge_px with amounts b.
    hedge_px: (n_days, k) prices aligned with long_px."""
    r_long = _returns(long_px)
    r_h = np.column_stack([_returns(hedge_px[:, j]) for j in range(hedge_px.shape[1])])
    hedge_r = r_h @ b
    residual = r_long - hedge_r
    n = len(r_long)

    # Out-of-sample check: refit on the first two-thirds, measure on the last third.
    split = int(n * 2 / 3)
    oos = None
    if n - split >= 20:
        b_train = _fit(r_long[:split], r_h[:split]) if r_h.shape[1] > 1 else _single_beta(r_long[:split], r_h[:split, 0])
        oos = _risk_removed(r_long[split:], r_long[split:] - r_h[split:] @ b_train)

    # Weakest rolling correlation shows whether the relationship ever broke down.
    worst_corr = None
    if n >= ROLL_DAYS:
        corrs = [np.corrcoef(r_long[i - ROLL_DAYS:i], hedge_r[i - ROLL_DAYS:i])[0, 1] for i in range(ROLL_DAYS, n + 1)]
        worst_corr = float(np.nanmin(corrs))

    # Cointegration between the long and the hedge as a fixed-mix basket (daily rebalanced).
    coint_p = None
    total = float(b.sum())
    if n >= 60 and total > 0:
        basket_index = np.concatenate([[1.0], np.cumprod(1 + r_h @ (b / total))])
        try:
            coint_p = float(coint(np.log(np.asarray(long_px, dtype=float)), np.log(basket_index))[1])
        except Exception:
            coint_p = None

    corr = float(np.corrcoef(r_long, hedge_r)[0, 1]) if np.std(hedge_r) > 0 else 0.0
    return {
        "correlation": round(corr, 4),
        "risk_removed": round(_risk_removed(r_long, residual), 4),
        "risk_removed_oos": round(oos, 4) if oos is not None else None,
        "hedge_ratio": round(total, 4),
        "long_vol": round(float(np.std(r_long, ddof=1) * np.sqrt(252)), 4),
        "hedged_vol": round(float(np.std(residual, ddof=1) * np.sqrt(252)), 4),
        "worst_corr": round(worst_corr, 4) if worst_corr is not None else None,
        "coint_pvalue": round(coint_p, 4) if coint_p is not None else None,
    }


def _single_beta(r_long, r_hedge):
    var = np.var(r_hedge, ddof=1)
    return np.array([max(np.cov(r_long, r_hedge)[0, 1] / var, 0.0) if var > 0 else 0.0])


def find_hedges(long_ticker, long_series, candidates):
    """long_series: {date: price}. candidates: {ticker: ({date: price}, source)} with source in
    'peer' / 'etf' / 'yours'. Returns ranked single hedges, the best basket and skipped tickers."""
    dates = sorted(long_series)
    singles, skipped, usable = [], [], {}

    for t, (series, source) in candidates.items():
        common = [d for d in dates if d in series]
        if len(common) < MIN_COVERAGE * len(dates):
            skipped.append({"ticker": t, "reason": "not enough price history for this period"})
            continue
        long_px = np.array([long_series[d] for d in common])
        hedge_px = np.array([[series[d]] for d in common])
        b = _single_beta(_returns(long_px), _returns(hedge_px[:, 0]))
        if b[0] <= 0:
            skipped.append({"ticker": t, "reason": "moves against the long (negative beta)"})
            continue
        stats = _evaluate(common, long_px, hedge_px, b)
        singles.append({"ticker": t, "source": source, **stats})
        usable[t] = series

    singles.sort(key=lambda s: s["risk_removed"], reverse=True)
    return singles, _best_basket(dates, long_series, singles, usable), skipped


def _best_basket(dates, long_series, singles, usable):
    pool = [s["ticker"] for s in singles[:BASKET_POOL]]
    common = [d for d in dates if all(d in usable[t] for t in pool)]
    if len(pool) < 2 or len(common) < 60:
        return None
    long_px = np.array([long_series[d] for d in common])
    r_long = _returns(long_px)

    # Fit, then repeatedly drop the smallest member until at most MAX_BASKET remain, each >= MIN_BASKET_SHARE.
    members = list(pool)
    while True:
        r_h = np.column_stack([_returns([usable[t][d] for d in common]) for t in members])
        b = _fit(r_long, r_h)
        keep = b > 0
        members, b = [t for t, k in zip(members, keep) if k], b[keep]
        if not members:
            return None
        share = b / b.sum()
        if len(members) <= MAX_BASKET and share.min() >= MIN_BASKET_SHARE:
            break
        drop = int(np.argmin(share))
        members.pop(drop)
        if len(members) == 0:
            return None

    if len(members) < 2:
        return None  # the best "basket" is just the top single hedge
    hedge_px = np.array([[usable[t][d] for t in members] for d in common])
    stats = _evaluate(common, long_px, hedge_px, b)
    share = b / b.sum()
    return {
        "members": [{"ticker": t, "amount": round(float(x), 4), "weight": round(float(w), 4)}
                    for t, x, w in sorted(zip(members, b, share), key=lambda m: -m[2])],
        **stats,
    }
