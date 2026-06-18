"""
S&P 500 Daily Signal Backtest
=============================
Downloads daily OHLCV data for ^GSPC (2010-01-01 to today) via yfinance and
tests 10 predefined hypotheses about predictable directional patterns.

Libraries used: pandas, numpy, yfinance only.
"""

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

START_DATE = "2010-01-01"
TICKER = "^GSPC"
MIN_SAMPLE = 30  # below this, results are flagged as statistically unreliable
RESULTS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_results.csv")


# ---------------------------------------------------------------------------
# Data loading & cleaning
# ---------------------------------------------------------------------------

def load_data(start: str = START_DATE) -> pd.DataFrame:
    """Download daily OHLCV for ^GSPC and return a clean DataFrame with
    columns: Date, Open, High, Low, Close, Volume."""
    raw = yf.download(TICKER, start=start, auto_adjust=False, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("No data returned from yfinance — check connectivity.")

    # yfinance may return MultiIndex columns (field, ticker) — flatten them.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()

    # Edge cases: coerce to numeric, drop rows with missing/zero prices
    # (covers NaNs, halted/delisted-style gaps and bad rows).
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df[df["Close"] > 0].reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df = df.sort_values("Date").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def compute_stats(outcome_returns: pd.Series) -> dict:
    """Given the forward returns observed after a trigger, compute the
    standard stat block used by every hypothesis."""
    outcome_returns = outcome_returns.dropna()
    n = len(outcome_returns)
    if n == 0:
        return {"sample": 0, "wins": 0, "win_rate": np.nan,
                "avg_win": np.nan, "avg_loss": np.nan, "avg_all": np.nan}
    wins = outcome_returns[outcome_returns > 0]
    losses = outcome_returns[outcome_returns <= 0]
    return {
        "sample": n,
        "wins": len(wins),
        "win_rate": 100.0 * len(wins) / n,
        "avg_win": 100.0 * wins.mean() if len(wins) else np.nan,
        "avg_loss": 100.0 * losses.mean() if len(losses) else np.nan,
        "avg_all": 100.0 * outcome_returns.mean(),
    }


def expected_value(s: dict) -> float:
    """EV per trade in %: (win_rate x avg_win) + ((1 - win_rate) x avg_loss)."""
    if s["sample"] == 0 or np.isnan(s["win_rate"]):
        return np.nan
    p = s["win_rate"] / 100.0
    avg_win = 0.0 if np.isnan(s["avg_win"]) else s["avg_win"]
    avg_loss = 0.0 if np.isnan(s["avg_loss"]) else s["avg_loss"]
    return round(p * avg_win + (1.0 - p) * avg_loss, 4)


def ev_verdict(ev: float) -> str:
    """EV-based edge classification (EV in % per trade)."""
    if ev > 0.10:
        return "positive directional edge observed"
    if ev < -0.10:
        return "negative expected value — fading this signal loses money"
    return "no meaningful directional edge (close to coin flip)"


def fmt_pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}%"


def slippage_cost(s_cc: dict, s_oc: dict) -> float:
    """EV lost (in % points) by entering at next open instead of trigger close."""
    ev_cc = expected_value(s_cc)
    ev_oc = expected_value(s_oc)
    if np.isnan(ev_cc) or np.isnan(ev_oc):
        return np.nan
    return round(ev_cc - ev_oc, 4)


def print_entry_comparison(s_cc: dict, s_oc: dict):
    """Theoretical (close-to-close) vs realistic (open-entry) comparison block."""
    ev_cc = expected_value(s_cc)
    ev_oc = expected_value(s_oc)
    slip = slippage_cost(s_cc, s_oc)
    print("  [Entry price comparison]")
    print(f"  Close-to-close (theoretical):  win rate={fmt_pct(s_cc['win_rate'])}, "
          f"EV={'n/a' if np.isnan(ev_cc) else f'{ev_cc:.4f}%'}")
    print(f"  Open-to-close (realistic):     win rate={fmt_pct(s_oc['win_rate'])}, "
          f"EV={'n/a' if np.isnan(ev_oc) else f'{ev_oc:.4f}%'}")
    print(f"  Slippage cost: {'n/a' if np.isnan(slip) else f'{slip:.4f}%'} "
          f"EV lost by waiting for open")


def print_hypothesis(num: int, title: str, conditions: list, results: list,
                     extra_notes: list | None = None):
    """results: list of (label, stats_cc, stats_oc) tuples — close-to-close
    (theoretical) and open-to-close (realistic entry) stats side by side."""
    print(f"\nHYPOTHESIS {num}: {title}")
    print(f"Trigger conditions tested: {', '.join(conditions)}")
    print("Results:")
    for label, s, s_oc in results:
        notes = []
        if s["sample"] < MIN_SAMPLE:
            notes.append(f"WARNING: sample size under {MIN_SAMPLE} — statistically unreliable")
        ev = expected_value(s)
        if not np.isnan(ev):
            notes.append(f"{ev_verdict(ev)} (EV={ev:.4f}%)")
        print(f"  [{label}]")
        print(f"  - Sample size: {s['sample']} occurrences")
        print(f"  - Win rate: {fmt_pct(s['win_rate'])}")
        print(f"  - Avg return on win days: {fmt_pct(s['avg_win'])}")
        print(f"  - Avg return on loss days: {fmt_pct(s['avg_loss'])}")
        print(f"  - Notes: {'; '.join(notes) if notes else 'none'}")
        print_entry_comparison(s, s_oc)
    for note in (extra_notes or []):
        print(f"  Note: {note}")


def regime_verdict(bull: dict, bear: dict) -> str:
    """Derive a one-sentence, EV-based verdict comparing the signal by regime
    (EV > 0.10% = edge, -0.10% to 0.10% = no edge, < -0.10% = negative EV)."""
    if bull["sample"] == 0 or bear["sample"] == 0:
        return "Insufficient data in one regime to compare."
    caveat = ""
    if bull["sample"] < MIN_SAMPLE or bear["sample"] < MIN_SAMPLE:
        caveat = f" (caution: at least one regime has fewer than {MIN_SAMPLE} samples)"
    ev_bull = expected_value(bull)
    ev_bear = expected_value(bear)
    edge_bull = ev_bull > 0.10
    edge_bear = ev_bear > 0.10
    if edge_bull and edge_bear:
        if abs(ev_bull - ev_bear) <= 0.10:
            return f"Edge holds in both regimes{caveat}"
        side = "bull" if ev_bull > ev_bear else "bear"
        return f"Edge is stronger in {side} regime{caveat}"
    if edge_bull:
        return f"Edge disappears in bear regime{caveat}"
    if edge_bear:
        return f"Edge disappears in bull regime{caveat}"
    if ev_bull < -0.10 and ev_bear < -0.10:
        return f"Negative expected value in both regimes — fading this signal loses money{caveat}"
    return f"No meaningful directional edge in either regime{caveat}"


def regime_split(df: pd.DataFrame, outcome: pd.Series, signal_label: str,
                 regime_data: dict):
    """Split a signal's forward returns by market regime (trigger day above or
    below the 200-day SMA) and print the comparison block.

    `outcome` must be the forward returns indexed by the trigger-day row index
    in `df`. Rows where the 200-day MA is still NaN (first 200 days) are
    excluded from regime classification."""
    outcome = outcome.dropna()
    close = df.loc[outcome.index, "Close"]
    sma = df.loc[outcome.index, "SMA200"]
    valid = sma.notna()
    bull = compute_stats(outcome[valid & (close > sma)])
    bear = compute_stats(outcome[valid & (close < sma)])
    regime_data[signal_label] = (bull["win_rate"], bear["win_rate"])

    print("  [Regime split]")
    for name, s in (("Bull (above 200-day MA)", bull), ("Bear (below 200-day MA)", bear)):
        print(f"  {name}: sample={s['sample']}, win rate={fmt_pct(s['win_rate'])}, "
              f"avg win={fmt_pct(s['avg_win'])}, avg loss={fmt_pct(s['avg_loss'])}, "
              f"EV={fmt_pct(expected_value(s))}")
    print(f"  Verdict: {regime_verdict(bull, bear)}")


# ---------------------------------------------------------------------------
# Shared series
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["NextReturn"] = df["Return"].shift(-1)
    df["Red"] = df["Return"] < 0
    df["Green"] = df["Return"] > 0
    # 200-day SMA for regime classification; NaN for the first 200 days so
    # those rows are automatically excluded from regime splits.
    df["SMA200"] = df["Close"].rolling(200, min_periods=200).mean()
    # Realistic-entry returns: missing/zero opens become NaN so they drop out
    # of the stats rather than producing inf/garbage returns.
    open_safe = df["Open"].where(df["Open"] > 0)
    df["OpenToClose"] = df["Close"] / open_safe - 1.0      # same-day open -> close
    df["NextOpen"] = open_safe.shift(-1)                   # next day's open (entry price)
    df["NextOpenToClose"] = df["OpenToClose"].shift(-1)    # next day open -> next day close
    return df


def consecutive_streak(flags: pd.Series) -> pd.Series:
    """Length of the current run of True values ending at each row."""
    streak = np.zeros(len(flags), dtype=int)
    run = 0
    for i, f in enumerate(flags.to_numpy()):
        run = run + 1 if f else 0
        streak[i] = run
    return pd.Series(streak, index=flags.index)


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------

def hypothesis_1(df, ranking, regime_data):
    # Logic: after N consecutive red closes (close < previous close) the market
    # may be short-term oversold; test whether the NEXT day closes positive.
    streak = consecutive_streak(df["Red"])
    results = []
    for n in (3, 4, 5):
        mask = streak >= n
        s = compute_stats(df.loc[mask, "NextReturn"])
        s_oc = compute_stats(df.loc[mask, "NextOpenToClose"])
        results.append((f"{n}+ consecutive red days", s, s_oc))
        ranking.append((f"H1: next day after {n}+ red days", s, s_oc))
    print_hypothesis(1, "Mean reversion after consecutive red days",
                     ["3, 4 and 5 consecutive red closes -> next day positive"],
                     results)
    # Regime split for the top-ranked variant of this signal (5+ red days).
    regime_split(df, df.loc[streak >= 5, "NextReturn"],
                 "H1: next day after 5+ red days", regime_data)
    return results


def hypothesis_2(df, ranking):
    # Logic: a large single-day drop often triggers panic selling; test whether
    # the next day bounces (closes positive).
    results = []
    for thresh in (-0.015, -0.02, -0.03):
        mask = df["Return"] <= thresh
        s = compute_stats(df.loc[mask, "NextReturn"])
        s_oc = compute_stats(df.loc[mask, "NextOpenToClose"])
        label = f"single-day drop > {abs(thresh) * 100:.1f}%"
        results.append((label, s, s_oc))
        ranking.append((f"H2: next day after {label}", s, s_oc))
    print_hypothesis(2, "Bounce after a large single-day drop",
                     ["drop > 1.5%, > 2%, > 3% -> next day positive"],
                     results)
    return results


def hypothesis_3(df, ranking, regime_data):
    # Logic: a 20-day closing low marks short-term capitulation; test whether
    # the close 5 trading days later is above the trigger-day close.
    low20 = df["Close"].rolling(20, min_periods=20).min()
    trigger = df["Close"] <= low20
    fwd5 = df["Close"].shift(-5) / df["Close"] - 1.0
    # Realistic 5-day window: enter at next day's open, exit at day +5 close.
    fwd5_oc = df["Close"].shift(-5) / df["NextOpen"] - 1.0
    s = compute_stats(fwd5[trigger])
    s_oc = compute_stats(fwd5_oc[trigger])
    ranking.append(("H3: 5 days after 20-day low", s, s_oc))
    print_hypothesis(3, "Recovery in the 5 days after a 20-day closing low",
                     ["close at a 20-day low -> close on day +5 higher than trigger close"],
                     [("20-day low, 5-day forward return", s, s_oc)])
    regime_split(df, fwd5[trigger], "H3: 5 days after 20-day low", regime_data)
    return s


def hypothesis_4(df, ranking):
    # Logic: after N consecutive green closes, does momentum carry the next day
    # higher (continuation) or does the streak revert?
    streak = consecutive_streak(df["Green"])
    results = []
    extra = []
    for n in (3, 4, 5):
        mask = streak >= n
        s = compute_stats(df.loc[mask, "NextReturn"])
        s_oc = compute_stats(df.loc[mask, "NextOpenToClose"])
        results.append((f"{n}+ consecutive green days", s, s_oc))
        ranking.append((f"H4: next day after {n}+ green days", s, s_oc))
        if not np.isnan(s["win_rate"]):
            verdict = "momentum continues" if s["win_rate"] > 50 else "momentum reverses"
            extra.append(f"after {n}+ green days: {verdict} ({s['win_rate']:.1f}% next-day win rate)")
    print_hypothesis(4, "Momentum after consecutive green days",
                     ["3, 4 and 5 consecutive green closes -> next day positive"],
                     results, extra)


def hypothesis_5(df, ranking, regime_data):
    # Logic: a fresh 52-week (252 trading day) closing high signals trend
    # strength; test whether the index is net positive 30 calendar days later.
    high252 = df["Close"].rolling(252, min_periods=252).max()
    trigger_idx = df.index[df["Close"] >= high252]
    dates = df["Date"].to_numpy()
    closes = df["Close"].to_numpy()
    next_opens = df["NextOpen"].to_numpy()
    rets = {}
    rets_oc = {}
    for i in trigger_idx:
        target_date = dates[i] + np.timedelta64(30, "D")
        j = np.searchsorted(dates, target_date)  # first trading day >= +30 calendar days
        if j < len(dates):
            rets[i] = closes[j] / closes[i] - 1.0
            if not np.isnan(next_opens[i]):  # realistic entry at next day's open
                rets_oc[i] = closes[j] / next_opens[i] - 1.0
    # Keep returns indexed by trigger row so they can be regime-classified.
    fwd30 = pd.Series(rets, dtype=float)
    s = compute_stats(fwd30)
    s_oc = compute_stats(pd.Series(rets_oc, dtype=float))
    ranking.append(("H5: 30 calendar days after 52-week high", s, s_oc))
    print_hypothesis(5, "Follow-through after a new 52-week high",
                     ["close at a 252-day high -> net positive 30 calendar days later"],
                     [("52-week high, 30-calendar-day forward return", s, s_oc)])
    regime_split(df, fwd30, "H5: 30 calendar days after 52-week high", regime_data)


def hypothesis_6(df, ranking):
    # Logic: a red close on volume >50% above its 20-day average suggests a
    # high-volume flush; test whether the next day closes positive.
    avg_vol20 = df["Volume"].rolling(20, min_periods=20).mean()
    trigger = (df["Volume"] > 1.5 * avg_vol20) & df["Red"] & avg_vol20.notna()
    s = compute_stats(df.loc[trigger, "NextReturn"])
    s_oc = compute_stats(df.loc[trigger, "NextOpenToClose"])
    ranking.append(("H6: next day after high-volume red day", s, s_oc))
    print_hypothesis(6, "Bounce after a high-volume red day",
                     ["volume > 150% of 20-day average AND red close -> next day positive"],
                     [("high-volume red day", s, s_oc)])


def hypothesis_7(df, ranking):
    # Logic: a narrow-range day (High-Low in the bottom decile of all daily
    # ranges) implies compression; test next-day direction and whether the
    # following move is larger or smaller than a normal day.
    day_range = df["High"] - df["Low"]
    threshold = day_range.quantile(0.10)
    trigger = day_range <= threshold
    nxt = df.loc[trigger, "NextReturn"].dropna()
    s = compute_stats(nxt)
    s_oc = compute_stats(df.loc[trigger, "NextOpenToClose"])
    ranking.append(("H7: next day after narrow-range day", s, s_oc))

    down_rate = 100.0 * (nxt < 0).sum() / len(nxt) if len(nxt) else np.nan
    avg_abs_after = 100.0 * nxt.abs().mean() if len(nxt) else np.nan
    avg_abs_normal = 100.0 * df["Return"].abs().mean()
    extra = [
        f"next-day down-move rate: {fmt_pct(down_rate)} (up-move rate is the win rate above)",
        f"avg |next-day move| after narrow range day: {fmt_pct(avg_abs_after)} "
        f"vs normal day: {fmt_pct(avg_abs_normal)}",
    ]
    print_hypothesis(7, "Behaviour after a narrow-range (compression) day",
                     ["High-Low in bottom 10% of all daily ranges -> next day direction & magnitude"],
                     [("narrow-range day", s, s_oc)], extra)


def hypothesis_8(df, ranking):
    # Logic: test the classic day-of-week effect — average daily return and
    # win rate grouped by weekday to spot any consistent directional bias.
    df = df.copy()
    df["Weekday"] = df["Date"].dt.day_name()
    results = []
    extra = []
    best = (None, -np.inf)
    worst = (None, np.inf)
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
        mask = df["Weekday"] == day
        s = compute_stats(df.loc[mask, "Return"])
        # Realistic capture of a weekday bias is that day's open -> close.
        s_oc = compute_stats(df.loc[mask, "OpenToClose"])
        results.append((day, s, s_oc))
        ranking.append((f"H8: {day} daily return", s, s_oc))
        if not np.isnan(s["avg_all"]):
            if s["avg_all"] > best[1]:
                best = (day, s["avg_all"])
            if s["avg_all"] < worst[1]:
                worst = (day, s["avg_all"])
    extra.append(f"best average day: {best[0]} ({best[1]:+.3f}%/day); "
                 f"worst: {worst[0]} ({worst[1]:+.3f}%/day)")
    print_hypothesis(8, "Day-of-week directional bias",
                     ["average return & win rate for each weekday (Mon-Fri)"],
                     results, extra)


def fomc_announcement_days(end: date) -> list:
    # Approximation per spec: 8 scheduled FOMC meetings per year in Jan, Mar,
    # May, Jun, Jul, Sep, Nov, Dec; the meeting runs second Tuesday/Wednesday
    # and the announcement lands on the second Wednesday of the month.
    months = [1, 3, 5, 6, 7, 9, 11, 12]
    days = []
    for year in range(2010, end.year + 1):
        for month in months:
            d = date(year, month, 1)
            # advance to first Wednesday (weekday 2), then add 7 days
            d += timedelta(days=(2 - d.weekday()) % 7)
            d += timedelta(days=7)
            if d <= end:
                days.append(np.datetime64(d))
    return days


def hypothesis_9(df, ranking):
    # Logic: Fed announcement days inject volatility; test the return of the
    # first trading day AFTER each (approximated) FOMC announcement vs all
    # other (non-Fed) trading days.
    dates = df["Date"].to_numpy()
    fed_days = fomc_announcement_days(pd.Timestamp(dates[-1]).date())

    after_fed_idx = set()
    for fd in fed_days:
        j = np.searchsorted(dates, fd + np.timedelta64(1, "D"))  # first trading day after announcement
        if j < len(dates):
            after_fed_idx.add(j)
    after_fed_idx = sorted(after_fed_idx)

    is_after_fed = df.index.isin(after_fed_idx)
    fed_stats = compute_stats(df.loc[is_after_fed, "Return"])
    # Realistic capture of the day-after-Fed move is that day's open -> close.
    fed_stats_oc = compute_stats(df.loc[is_after_fed, "OpenToClose"])
    normal_stats = compute_stats(df.loc[~is_after_fed, "Return"])
    ranking.append(("H9: day after Fed announcement", fed_stats, fed_stats_oc))

    extra = [
        f"approximated FOMC announcement dates used: {len(fed_days)}",
        f"non-Fed days for comparison — win rate: {fmt_pct(normal_stats['win_rate'])}, "
        f"avg return: {fmt_pct(normal_stats['avg_all'])}",
        f"day-after-Fed avg return: {fmt_pct(fed_stats['avg_all'])} "
        f"vs non-Fed: {fmt_pct(normal_stats['avg_all'])}",
    ]
    print_hypothesis(9, "Day after Fed (FOMC) announcement vs normal days",
                     ["first trading day after approximated FOMC announcement (2nd Wed of "
                      "Jan/Mar/May/Jun/Jul/Sep/Nov/Dec)"],
                     [("day after Fed announcement", fed_stats, fed_stats_oc)], extra)


def hypothesis_10(df, ranking, h1_results):
    # Logic: 3 consecutive red days where each day's low undercuts the prior
    # day's low (accelerating decline) is a stronger capitulation signal than
    # plain 3 red days; compare next-day stats against Hypothesis 1's 3-red signal.
    accel = df["Red"] & (df["Low"] < df["Low"].shift(1))
    streak = consecutive_streak(accel)
    trigger = streak >= 3
    s = compute_stats(df.loc[trigger, "NextReturn"])
    s_oc = compute_stats(df.loc[trigger, "NextOpenToClose"])
    ranking.append(("H10: next day after 3 accelerating red days", s, s_oc))

    h1_3red = h1_results[0][1]  # stats for the 3+ red days signal from H1
    if not np.isnan(s["win_rate"]) and not np.isnan(h1_3red["win_rate"]):
        diff = s["win_rate"] - h1_3red["win_rate"]
        verdict = ("acceleration ADDS predictive value" if diff > 0
                   else "acceleration does NOT add predictive value")
        comparison = (f"vs simple 3-red-days (H1): win rate {h1_3red['win_rate']:.2f}% -> "
                      f"{s['win_rate']:.2f}% ({diff:+.2f} pts); {verdict}")
    else:
        comparison = "comparison unavailable (insufficient data)"
    print_hypothesis(10, "Accelerating 3-day decline (lower lows) vs simple 3 red days",
                     ["3 consecutive red days, each with a lower low than the prior day -> next day positive"],
                     [("3 accelerating red days", s, s_oc)], [comparison])


def hypothesis_11(df, ranking, regime_data, h1_results, h3_stats):
    # Logic: combination signal — the close sits at a 20-day low AND the last
    # 3 closes were each lower than the prior close (H3 and H1-3 conditions
    # true on the same day). Same 5-day forward window as H3; the question is
    # whether stacking two overlapping oversold signals lifts the EV above
    # either signal alone.
    label = "H11: 5 days after 20-day low + 3+ red days"
    streak = consecutive_streak(df["Red"])
    low20 = df["Close"].rolling(20, min_periods=20).min()
    trigger = (df["Close"] <= low20) & (streak >= 3)
    fwd5 = df["Close"].shift(-5) / df["Close"] - 1.0
    # Realistic 5-day window: enter at next day's open, exit at day +5 close.
    fwd5_oc = df["Close"].shift(-5) / df["NextOpen"] - 1.0
    s = compute_stats(fwd5[trigger])
    s_oc = compute_stats(fwd5_oc[trigger])
    ranking.append((label, s, s_oc))

    ev11 = expected_value(s)
    ev3 = expected_value(h3_stats)
    ev1_3 = expected_value(h1_results[0][1])  # 3+ red days signal from H1
    if any(np.isnan(v) for v in (ev11, ev3, ev1_3)):
        comparison = "EV comparison unavailable (insufficient data)"
    else:
        adds = ev11 > ev3 and ev11 > ev1_3
        conclusion = ("combining ADDS value over both components"
                      if adds else "combining does NOT add value over the best single signal")
        comparison = (f"EV comparison: H3 alone={ev3:.4f}% (5-day window), "
                      f"H1 3-red alone={ev1_3:.4f}% (next-day window), "
                      f"H11 combined={ev11:.4f}%; {conclusion}")
    print_hypothesis(11, "Combination: 20-day low AND 3+ consecutive red days",
                     ["close at a 20-day low AND 3+ consecutive red closes -> "
                      "close on day +5 higher than trigger close"],
                     [("20-day low + 3+ red days, 5-day forward return", s, s_oc)],
                     [comparison])
    regime_split(df, fwd5[trigger], label, regime_data)


# ---------------------------------------------------------------------------
# CSV result logging
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "run_date", "data_start", "data_end", "trading_days", "hypothesis",
    "win_rate_pct", "sample_size", "avg_win_return_pct", "avg_loss_return_pct",
    "expected_value_pct", "reliability", "bull_win_rate_pct", "bear_win_rate_pct",
    "realistic_ev_pct", "slippage_est_pct",
]


def append_results_csv(ranked, regime_data, data_start, data_end, trading_days) -> int:
    """Append one row per ranked signal to backtest_results.csv. Creates the
    file with headers if missing; never overwrites previous runs."""
    run_date = date.today().isoformat()
    rows = []
    for label, s, s_oc in ranked:
        bull_wr, bear_wr = regime_data.get(label, (np.nan, np.nan))
        rows.append({
            "run_date": run_date,
            "data_start": data_start,
            "data_end": data_end,
            "trading_days": trading_days,
            "hypothesis": label,
            "win_rate_pct": round(s["win_rate"], 4),
            "sample_size": s["sample"],
            "avg_win_return_pct": round(s["avg_win"], 4) if not np.isnan(s["avg_win"]) else np.nan,
            "avg_loss_return_pct": round(s["avg_loss"], 4) if not np.isnan(s["avg_loss"]) else np.nan,
            "expected_value_pct": expected_value(s),
            "reliability": "low sample" if s["sample"] < MIN_SAMPLE else "ok",
            # Blank for signals without a regime split.
            "bull_win_rate_pct": round(bull_wr, 4) if not np.isnan(bull_wr) else np.nan,
            "bear_win_rate_pct": round(bear_wr, 4) if not np.isnan(bear_wr) else np.nan,
            "realistic_ev_pct": expected_value(s_oc),
            "slippage_est_pct": slippage_cost(s, s_oc),
        })
    out = pd.DataFrame(rows, columns=CSV_COLUMNS)
    # Write the header only on first creation; an existing-but-empty file also
    # counts as "first time" so the log never ends up headerless.
    needs_header = not os.path.exists(RESULTS_CSV) or os.path.getsize(RESULTS_CSV) == 0
    if not needs_header:
        # If a previous run logged with an older column set, migrate the
        # existing rows (new columns blank) so the file stays one valid CSV.
        with open(RESULTS_CSV) as f:
            existing_header = f.readline().strip()
        if existing_header != ",".join(CSV_COLUMNS):
            old = pd.read_csv(RESULTS_CSV).reindex(columns=CSV_COLUMNS)
            old.to_csv(RESULTS_CSV, index=False)
    out.to_csv(RESULTS_CSV, mode="a", header=needs_header, index=False)
    return len(rows)


# ---------------------------------------------------------------------------
# Realistic trading summary
# ---------------------------------------------------------------------------

def print_realistic_summary(ranked):
    """Derive (programmatically) which signals survive open-price entry."""
    print("\nREALISTIC TRADING SUMMARY")
    print("=" * 25)

    positive = [(label, expected_value(s_oc))
                for label, s, s_oc in ranked if expected_value(s_oc) > 0]
    print("Signals with positive realistic EV after open-entry:")
    if positive:
        for label, ev in positive:
            print(f"  {label}: realistic EV={ev:.4f}%")
    else:
        print("  (none)")

    losers = []
    for label, s, s_oc in ranked:
        ev_cc = expected_value(s)
        ev_oc = expected_value(s_oc)
        if not np.isnan(ev_cc) and ev_cc > 0 and (ev_cc - ev_oc) > 0.5 * ev_cc:
            losers.append((label, ev_cc, ev_oc))
    print("\nSignals that lose meaningful edge at open:")
    if losers:
        for label, ev_cc, ev_oc in losers:
            print(f"  {label}: theoretical EV={ev_cc:.4f}% -> realistic EV={ev_oc:.4f}%")
    else:
        print("  (none)")

    reliable = [(label, s, s_oc) for label, s, s_oc in ranked
                if s["sample"] >= MIN_SAMPLE and expected_value(s_oc) > 0]
    if reliable:
        label, s, s_oc = max(reliable, key=lambda r: expected_value(r[2]))
        ev_cc = expected_value(s)
        ev_oc = expected_value(s_oc)
        retained = f", retaining {100.0 * ev_oc / ev_cc:.0f}% of its theoretical edge" \
            if ev_cc > 0 else ""
        print(f"\nBest single signal for live trading: {label} — highest realistic EV "
              f"({ev_oc:.4f}% per trade) over a reliable sample of "
              f"{s['sample']} occurrences{retained}.")
    else:
        print("\nBest single signal for live trading: none — no signal keeps a "
              "positive realistic EV on a reliable sample after open entry.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_data()
    df = build_features(df)

    start = df["Date"].iloc[0].date()
    end = df["Date"].iloc[-1].date()

    print(f"BACKTEST COMPLETE — S&P 500 Daily Signal Analysis, "
          f"Data range: {start} to {end}, Total trading days analysed: {len(df)}")
    print("=" * 100)

    ranking = []      # list of (label, stats) collected across all hypotheses
    regime_data = {}  # label -> (bull_win_rate, bear_win_rate) for top signals
    h1_results = hypothesis_1(df, ranking, regime_data)
    hypothesis_2(df, ranking)
    h3_stats = hypothesis_3(df, ranking, regime_data)
    hypothesis_4(df, ranking)
    hypothesis_5(df, ranking, regime_data)
    hypothesis_6(df, ranking)
    hypothesis_7(df, ranking)
    hypothesis_8(df, ranking)
    hypothesis_9(df, ranking)
    hypothesis_10(df, ranking, h1_results)
    hypothesis_11(df, ranking, regime_data, h1_results, h3_stats)

    print("\n" + "=" * 100)
    print("OVERALL SUMMARY — signals ranked by realistic EV (open-entry, descending)")
    print("=" * 100)
    ranked = sorted(
        (r for r in ranking if not np.isnan(expected_value(r[2]))),
        key=lambda r: expected_value(r[2]), reverse=True,
    )
    print(f"{'Rank':<5} {'Signal':<50} {'Win rate':>9} {'Real EV':>9} {'Sample':>7}  Reliability")
    print("-" * 105)
    for rank, (label, s, s_oc) in enumerate(ranked, 1):
        ev_cc = expected_value(s)
        ev_oc = expected_value(s_oc)
        flag = "UNRELIABLE (n < 30)" if s["sample"] < MIN_SAMPLE else "ok"
        if not np.isnan(ev_cc) and ev_cc > 0 and (ev_cc - ev_oc) > 0.5 * ev_cc:
            flag += "  ⚠ slippage"
        print(f"{rank:<5} {label:<50} {s['win_rate']:>8.2f}% {ev_oc:>8.4f}% "
              f"{s['sample']:>7}  {flag}")

    print_realistic_summary(ranked)

    n_rows = append_results_csv(ranked, regime_data, start, end, len(df))
    print(f"\nResults appended to backtest_results.csv — {n_rows} rows added")


if __name__ == "__main__":
    main()
