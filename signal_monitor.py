"""
S&P 500 Signal Monitor
======================
Run this each evening AFTER the US market close. It downloads recent ^GSPC data,
checks for two live trading signals (H3: 20-day closing low, H1: 5+ consecutive
red days), reports the current regime, and prints an exact action plan for
tomorrow. Every run writes a row to the Supabase `signal_log` table, and when a
signal fires it also inserts a PENDING row into the Supabase `trade_log` table.

Does NOT import backtest.py for its indicator logic. Uses yfinance, pandas,
numpy, and the Supabase client (credentials loaded from .env).

NOTE: the position-sizing figures are illustrative only and are NOT financial
advice. The EV / win-rate / average-move numbers are the backtested results
from backtest.py (2010-present, realistic open-entry where applicable).
"""

from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# Load .env locally; no-op if the file is absent (e.g. in GitHub Actions, where
# SUPABASE_URL / SUPABASE_KEY are injected from repository secrets instead).
load_dotenv()

from supabase_client import get_client

TICKER = "^GSPC"

# Backtested stats (from backtest.py, 2010-present). Win rate & EV are the
# realistic open-entry figures; avg win/loss are the per-trade averages.
SIGNAL_STATS = {
    "H3": {
        "name": "H3: 20-day closing low (5-day hold)",
        "ev_realistic": 0.6933, "win_realistic": 63.09,
        "avg_win": 2.525, "avg_loss": -2.3661, "hold_days": 5,
    },
    "H1": {
        "name": "H1: 5+ consecutive red days (next-day)",
        "ev_realistic": 0.4084, "win_realistic": 63.16,
        "avg_win": 1.3823, "avg_loss": -1.2587, "hold_days": 1,
    },
    "H11": {
        "name": "H11 COMBINATION: 20-day low AND 5+ red days (5-day hold)",
        "ev_realistic": 0.6817, "win_realistic": 63.19,
        "avg_win": 2.4849, "avg_loss": -2.4725, "hold_days": 5,
    },
}

# H3 regime split (backtest.py): used for the bear-regime note.
H3_BULL_EV = 0.41
H3_BEAR_EV = 1.06


# ---------------------------------------------------------------------------
# Data download helpers
# ---------------------------------------------------------------------------

def download(period: str) -> pd.DataFrame:
    """Download ^GSPC daily OHLCV for the given period and return a clean
    DataFrame indexed by date. Raises RuntimeError on failure/empty data."""
    raw = yf.download(TICKER, period=period, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for period={period}.")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Indicator logic (self-contained, identical rules to backtest.py)
# ---------------------------------------------------------------------------

def red_streak(closes: np.ndarray) -> int:
    """Number of consecutive days (ending on the last row) where the close was
    lower than the previous close."""
    streak = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes[i] < closes[i - 1]:
            streak += 1
        else:
            break
    return streak


def add_business_days(d, n: int):
    """Add n business days (weekends excluded; holidays not modelled)."""
    return np.busday_offset(np.datetime64(d, "D"), n, roll="forward").astype("M8[D]").astype(object)


# ---------------------------------------------------------------------------
# Supabase persistence
# ---------------------------------------------------------------------------

def _iso(d) -> str | None:
    """Serialise a date/None to an ISO string for Supabase (date columns)."""
    return None if d is None else str(d)


def log_signal_run(client, run_date, signal_date, signal_type, regime, spx_close,
                   change_pct, ma_200, triggered, entry_date, exit_date) -> None:
    """Insert one row into signal_log for every monitor run (full history)."""
    stats = SIGNAL_STATS.get(signal_type)
    client.table("signal_log").insert({
        "run_date": _iso(run_date),
        "signal_date": _iso(signal_date),
        "signal_type": signal_type,
        "regime": regime,
        "spx_close": round(float(spx_close), 2),
        "change_pct": round(float(change_pct), 4),
        "ma_200": round(float(ma_200), 2),
        "triggered": bool(triggered),
        "entry_date": _iso(entry_date),
        "exit_date": _iso(exit_date),
        "realistic_ev_pct": stats["ev_realistic"] if stats else None,
        "win_rate_pct": stats["win_realistic"] if stats else None,
    }).execute()


def log_trade(client, signal_date, signal_type, regime, entry_date, exit_date) -> None:
    """Insert a PENDING trade row into trade_log when a signal fires."""
    client.table("trade_log").insert({
        "signal_date": _iso(signal_date),
        "signal_type": signal_type,
        "regime": regime,
        "entry_date": _iso(entry_date),
        "exit_date": _iso(exit_date),
        "entry_price": None,
        "exit_price": None,
        "actual_return_pct": None,
        "status": "PENDING",
    }).execute()


# ---------------------------------------------------------------------------
# Output blocks
# ---------------------------------------------------------------------------

def print_action_block(signal_key, signal_date, regime, today_close, avg_gap_pct,
                       entry_date, exit_date):
    s = SIGNAL_STATS[signal_key]
    win = s["win_realistic"] / 100.0
    entry_est = today_close * (1 + avg_gap_pct / 100.0)

    print("\n\u26a1 SIGNAL DETECTED")
    print("==================")
    print(f"Signal: {s['name']}")
    if regime == "BEAR":
        print(f"Regime: BEAR — Bear regime — historically stronger signal "
              f"(H3 bear EV: {H3_BEAR_EV:.2f}% vs bull EV: {H3_BULL_EV:.2f}%)")
    else:
        print("Regime: BULL")
    print(f"Realistic EV per trade: {s['ev_realistic']:.4f}%")

    print("\nACTION FOR TOMORROW")
    print("-------------------")
    print("Entry: Buy ^GSPC CFD / spread bet at market open tomorrow")
    print(f"Entry price estimate: {entry_est:.2f}  "
          f"(today's close {today_close:.2f} {avg_gap_pct:+.2f}% avg 20-day overnight gap)")
    if s["hold_days"] == 1:
        print(f"Exit: {exit_date} (tomorrow's close — 1 trading day)")
    else:
        print(f"Exit: {exit_date} (close of day 5 — {s['hold_days']} trading days from entry)")
    print(f"Expected move by exit (avg win scenario): {s['avg_win']:+.2f}% from entry")
    print(f"Expected move by exit (avg loss scenario): {s['avg_loss']:+.2f}% from entry")
    print(f"Win rate (realistic): {s['win_realistic']:.2f}%")

    print("\nPOSITION SIZING (for reference only — not financial advice)")
    print("------------------")
    for stake in (100, 250, 500):
        ev_cash = stake * s["ev_realistic"] / 100.0
        print(f"At \u00a3{stake} stake: expected value = \u00a3{ev_cash:.2f} per trade")

    print("\nLOG THIS TRADE")
    print("--------------")
    print(f"Signal date: {signal_date}")
    print(f"Entry date: {entry_date}")
    print(f"Exit date: {exit_date}")
    print("Paste this block into your trade log when you enter tomorrow.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    now = datetime.now()
    today = now.date()

    # If today is a weekend, it is not a trading day at all.
    if today.weekday() >= 5:
        print("Market closed today — no signals to check")
        return

    try:
        df = download("90d")          # ~60+ trading days for signal indicators
        today_dl = download("5d")     # explicit recent download to confirm close
        sma_df = download("450d")     # ~300 trading days so the 200-day MA is valid
    except Exception as exc:  # network/parse failures
        print(f"ERROR: could not download data — {exc}")
        return

    last_date = today_dl.index[-1].date()
    # Confirm today's close is available (market has closed and data published).
    if last_date != today:
        print("S&P 500 SIGNAL MONITOR")
        print("======================")
        print(f"Run date: {now:%Y-%m-%d %H:%M}")
        print(f"\nWARNING: today's close ({today}) is not yet available.")
        print(f"Last available close is {last_date}. The US market may still be open, "
              "or today is a public holiday.")
        print("No signals checked — re-run after the close. Exiting cleanly.")
        return

    df = df.tail(60)
    if len(df) < 21 or len(sma_df) < 200:
        print("ERROR: not enough clean trading days returned to compute indicators.")
        return

    closes = df["Close"].to_numpy()
    today_close = float(closes[-1])
    prev_close = float(closes[-2])
    change_today = (today_close / prev_close - 1) * 100

    sma200 = float(sma_df["Close"].tail(200).mean())
    regime = "BULL" if today_close > sma200 else "BEAR"

    # 20-day low: lowest close over the last 20 trading days (incl. today).
    twenty_day_low = float(np.min(closes[-20:]))
    low_gap_pct = (today_close / twenty_day_low - 1) * 100
    direction = "above" if low_gap_pct >= 0 else "below"

    # Level today's close must break for H3 to fire (min of last 19 closes —
    # these become the "previous 19" tomorrow).
    h3_trigger_level = float(np.min(closes[-19:]))

    streak = red_streak(closes)

    # 20-day average overnight gap = open[t] / close[t-1] - 1.
    opens = df["Open"].to_numpy()
    gaps = opens[1:] / closes[:-1] - 1.0
    avg_gap_pct = float(np.mean(gaps[-20:])) * 100

    # ---- Header -------------------------------------------------------------
    print("S&P 500 SIGNAL MONITOR")
    print("======================")
    print(f"Run date: {now:%Y-%m-%d %H:%M}")
    print(f"Today's close: {today_close:.2f}")
    print(f"Change today: {change_today:+.2f}%")
    print(f"200-day MA: {sma200:.2f}")
    print(f"Regime: {regime}")
    print(f"20-day low: {twenty_day_low:.2f} (today's close is {abs(low_gap_pct):.2f}% {direction})")
    print(f"Current red streak: {streak} days")

    # ---- Signal evaluation --------------------------------------------------
    # H3: today's close is the lowest of the last 20 trading days.
    h3_fired = today_close <= twenty_day_low
    # H1: 5+ consecutive red closes ending today.
    h1_fired = streak >= 5

    entry_date = add_business_days(today, 1)            # tomorrow (next business day)
    exit_5d = add_business_days(entry_date, 4)          # close of 5th trading day
    exit_1d = entry_date                               # tomorrow's close

    triggered = h3_fired or h1_fired
    if not triggered:
        print("\nNO SIGNALS TODAY")
        print("================")
        print("Nothing to trade tomorrow.")
        print("Next levels to watch:")
        print(f"  H3 triggers if close falls below: {h3_trigger_level:.2f}")
        more = max(0, 5 - streak)
        if more == 0:
            print("  H1 (5-day): already at 5+ red days context — see signal logic")
        else:
            day_word = "day" if more == 1 else "days"
            print(f"  H1 (5-day) triggers if {more} more consecutive red {day_word}")
        signal_key = "NONE"
        log_entry_date = None
        log_exit_date = None
    else:
        # One or more signals fired. Combination takes priority.
        if h3_fired and h1_fired:
            signal_key = "H11"
            print("\nNote: BOTH H3 and H1 fired today — this is an H11 COMBINATION signal "
                  "(historically the strongest of the three).")
            exit_date = exit_5d
        elif h3_fired:
            signal_key = "H3"
            exit_date = exit_5d
        else:
            signal_key = "H1"
            exit_date = exit_1d

        print_action_block(signal_key, today, regime, today_close, avg_gap_pct,
                           entry_date, exit_date)
        log_entry_date = entry_date
        log_exit_date = exit_date

    # ---- Persist to Supabase ------------------------------------------------
    try:
        client = get_client()
        log_signal_run(client, today, today, signal_key, regime, today_close,
                       change_today, sma200, triggered, log_entry_date, log_exit_date)
        print("\nResults written to Supabase signal_log")
        if triggered:
            log_trade(client, today, signal_key, regime, log_entry_date, log_exit_date)
            print("Trade logged to Supabase trade_log — fill in entry_price tomorrow at open")
    except Exception as exc:
        print(f"\nWARNING: could not write to Supabase — {str(exc)[:200]}")


if __name__ == "__main__":
    main()
