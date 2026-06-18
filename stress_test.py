"""
S&P 500 Stress Test — H3 and H2 >3% only
========================================
Validation exercise: extends the data range back to 2000-01-01 and re-runs the
two strongest signals through the dot-com crash, the 2008 financial crisis and
the 2020 COVID crash, split into six market eras.

Signals tested:
  H3     — close at a 20-day low -> close on day +5 higher than trigger close
  H2 >3% — single-day drop of more than 3% -> next day positive

Reuses the data loading, feature engineering, stats and regime logic from
backtest.py. Does NOT write to backtest_results.csv.
"""

import numpy as np
import pandas as pd

import backtest as bt

STRESS_START = "2000-01-01"
LOW_SAMPLE = 15  # per-era threshold below which results are flagged

# (label, description, start, end) — end None means "to present"
ERAS = [
    ("2000-2002", "dot-com crash", "2000-01-01", "2002-12-31"),
    ("2003-2007", "recovery/bull", "2003-01-01", "2007-12-31"),
    ("2008-2009", "financial crisis", "2008-01-01", "2009-12-31"),
    ("2010-2019", "post-crisis bull", "2010-01-01", "2019-12-31"),
    ("2020-2021", "COVID crash and recovery", "2020-01-01", "2021-12-31"),
    ("2022-present", "rate hike cycle and recent", "2022-01-01", None),
]


def build_signals(df: pd.DataFrame) -> dict:
    """Triggers and forward returns (theoretical + realistic open entry) for
    the two signals under test. Same logic as H3 / H2 in backtest.py."""
    low20 = df["Close"].rolling(20, min_periods=20).min()
    h3_trigger = df["Close"] <= low20
    h3_cc = df["Close"].shift(-5) / df["Close"] - 1.0
    h3_oc = df["Close"].shift(-5) / df["NextOpen"] - 1.0

    h2_trigger = df["Return"] <= -0.03
    return {
        "H3": (h3_trigger, h3_cc, h3_oc),
        "H2 >3%": (h2_trigger, df["NextReturn"], df["NextOpenToClose"]),
    }


def fmt_ev(ev: float) -> str:
    return "n/a" if np.isnan(ev) else f"{ev:.4f}%"


def signal_stats(trigger: pd.Series, cc: pd.Series, oc: pd.Series) -> tuple:
    s_cc = bt.compute_stats(cc[trigger])
    s_oc = bt.compute_stats(oc[trigger])
    return s_cc, s_oc, bt.expected_value(s_cc), bt.expected_value(s_oc)


def print_signal_line(name: str, s_cc: dict, s_oc: dict, ev_cc: float, ev_oc: float):
    flag = "  [LOW SAMPLE — treat with caution]" if s_cc["sample"] < LOW_SAMPLE else ""
    print(f"{name} — sample={s_cc['sample']}, win rate={bt.fmt_pct(s_cc['win_rate'])}, "
          f"EV={fmt_ev(ev_cc)} (theoretical), EV={fmt_ev(ev_oc)} (realistic/open entry){flag}")


def bear_share(df: pd.DataFrame, trigger: pd.Series) -> tuple:
    """% of trigger days below the 200-day MA, among triggers with enough MA
    history; also returns the count that could not be classified."""
    t = df[trigger]
    classifiable = t["SMA200"].notna()
    unclassified = int((~classifiable).sum())
    if classifiable.sum() == 0:
        return np.nan, unclassified
    bear_pct = 100.0 * (t.loc[classifiable, "Close"] < t.loc[classifiable, "SMA200"]).mean()
    return bear_pct, unclassified


def summarize_signal(title: str, era_results: list):
    """era_results: list of (era_label, ev_realistic, sample)."""
    print(f"\n{title}")
    usable = [(era, ev, n) for era, ev, n in era_results if not np.isnan(ev)]
    if not usable:
        print("  No usable era results.")
        return False
    best = max(usable, key=lambda r: r[1])
    worst = min(usable, key=lambda r: r[1])
    negative = [era for era, ev, _ in usable if ev < 0]
    low = [era for era, ev, n in usable if n < LOW_SAMPLE]
    print(f"  Best era: {best[0]} (realistic EV={best[1]:.4f}%)")
    print(f"  Worst era: {worst[0]} (realistic EV={worst[1]:.4f}%)")
    print(f"  Negative EV eras: {', '.join(negative) if negative else 'none'}")
    caveat = f" — but {', '.join(low)} below {LOW_SAMPLE} samples, treat with caution" if low else ""
    if not negative:
        print(f"  Verdict: Edge kept a positive realistic EV in all "
              f"{len(usable)} eras — it survives crash and bull conditions alike{caveat}.")
    else:
        print(f"  Verdict: Edge went negative in {', '.join(negative)} — "
              f"it does NOT hold across all market conditions{caveat}.")
    return not negative


def main():
    df = bt.load_data(start=STRESS_START)
    df = bt.build_features(df)
    signals = build_signals(df)

    start = df["Date"].iloc[0].date()
    end = df["Date"].iloc[-1].date()
    print(f"STRESS TEST — S&P 500, H3 and H2 >3% only, "
          f"Data range: {start} to {end}, Total trading days: {len(df)}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # Analysis 1 — full period
    # ------------------------------------------------------------------
    print("\nANALYSIS 1 — FULL PERIOD (2000 to present)")
    print("-" * 100)
    for name, (trigger, cc, oc) in signals.items():
        s_cc, s_oc, ev_cc, ev_oc = signal_stats(trigger, cc, oc)
        print_signal_line(name, s_cc, s_oc, ev_cc, ev_oc)
        # Full regime split (200-day MA), same format as the main backtest.
        # Throwaway dict: stress test does not log regime data anywhere.
        bt.regime_split(df, cc[trigger], name, {})
        print()

    # ------------------------------------------------------------------
    # Analysis 2 — sub-period breakdown
    # ------------------------------------------------------------------
    print("\nANALYSIS 2 — SUB-PERIOD BREAKDOWN")
    print("-" * 100)
    era_results = {name: [] for name in signals}
    for label, desc, era_start, era_end in ERAS:
        era_mask = df["Date"] >= pd.Timestamp(era_start)
        if era_end is not None:
            era_mask &= df["Date"] <= pd.Timestamp(era_end)
        print(f"\n[Era: {label} | {desc}]")
        bear_parts = []
        for name, (trigger, cc, oc) in signals.items():
            era_trigger = trigger & era_mask
            s_cc, s_oc, ev_cc, ev_oc = signal_stats(era_trigger, cc, oc)
            print_signal_line(name, s_cc, s_oc, ev_cc, ev_oc)
            era_results[name].append((label, ev_oc, s_cc["sample"]))
            bear_pct, unclassified = bear_share(df, era_trigger)
            part = f"{name}: {'n/a' if np.isnan(bear_pct) else f'{bear_pct:.1f}%'}"
            if unclassified:
                part += f" ({unclassified} unclassifiable — insufficient MA history)"
            bear_parts.append(part)
        print(f"Regime: triggers in bear market (below 200-day MA) — {'; '.join(bear_parts)}")

    # ------------------------------------------------------------------
    # Stress test summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STRESS TEST SUMMARY")
    print("=" * 19)
    h3_robust = summarize_signal("H3 (20-day low, 5-day hold)", era_results["H3"])
    h2_robust = summarize_signal("H2 >3% single-day crash bounce", era_results["H2 >3%"])

    if h3_robust and h2_robust:
        conclusion = ("Both signals kept a positive realistic EV through the dot-com crash, "
                      "2008 and COVID — robust enough to paper trade.")
    elif h3_robust or h2_robust:
        survivor = "H3" if h3_robust else "H2 >3%"
        failed = "H2 >3%" if h3_robust else "H3"
        conclusion = (f"Only {survivor} held up across all eras; {failed} failed in at least "
                      f"one era — paper trade {survivor} only.")
    else:
        conclusion = ("Neither signal held a positive realistic EV across all eras — "
                      "not robust enough to paper trade.")
    print(f"\nOverall conclusion: {conclusion}")

    print("\nStress test complete — no rows written to CSV")


if __name__ == "__main__":
    main()
