"""
Trade Updater
=============
Run this AFTER you close a trade that signal_monitor.py logged. It shows your
PENDING trades, lets you enter the actual entry/exit prices, marks the trade
CLOSED, and prints a running P&L summary compared with the backtested EV.

Uses pandas / numpy only.
"""

import os

import numpy as np
import pandas as pd

TRADE_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trade_log.csv")

TRADE_LOG_COLUMNS = [
    "signal_date", "signal_type", "regime", "entry_date", "exit_date",
    "entry_price", "exit_price", "actual_return", "status",
]

# Backtested realistic EV per signal type (from backtest.py) for comparison.
BACKTESTED_EV = {"H3": 0.6933, "H1": 0.4084, "H11": 0.6817}


def prompt_float(prompt: str):
    """Read a positive float from the user; return None to cancel."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            val = float(raw)
        except ValueError:
            print("  Please enter a number (or blank to cancel).")
            continue
        if val <= 0:
            print("  Price must be greater than zero.")
            continue
        return val


def print_pnl_summary(df: pd.DataFrame):
    closed = df[df["status"] == "CLOSED"].copy()
    print("\nRUNNING P&L SUMMARY")
    print("===================")
    if closed.empty:
        print("No closed trades yet.")
        return

    returns = pd.to_numeric(closed["actual_return"], errors="coerce").dropna()
    total = len(returns)
    wins = int((returns > 0).sum())
    win_rate = 100.0 * wins / total if total else 0.0
    avg_return = returns.mean()
    total_return = returns.sum()

    print(f"Total closed trades: {total}")
    print(f"Win rate: {win_rate:.2f}% ({wins}/{total})")
    print(f"Average return per trade: {avg_return:+.4f}%")
    print(f"Total cumulative return: {total_return:+.2f}%")

    # Compare actual avg return to the sample-weighted backtested EV.
    evs = [BACKTESTED_EV.get(t, np.nan) for t in closed["signal_type"]]
    evs = [e for e in evs if not np.isnan(e)]
    if evs:
        expected_ev = float(np.mean(evs))
        diff = avg_return - expected_ev
        verdict = ("outperforming" if diff > 0.05 else
                   "underperforming" if diff < -0.05 else "in line with")
        print(f"Backtested expected EV (per trade, blended): {expected_ev:+.4f}%")
        print(f"Live vs backtest: {diff:+.4f}% per trade — you are {verdict} the backtest.")


def main():
    if not os.path.exists(TRADE_LOG) or os.path.getsize(TRADE_LOG) == 0:
        print(f"No trade log found at {TRADE_LOG}. Run signal_monitor.py first.")
        return

    df = pd.read_csv(TRADE_LOG, dtype=str).reindex(columns=TRADE_LOG_COLUMNS)
    pending = df[df["status"] == "PENDING"]

    if pending.empty:
        print("No PENDING trades to update.")
        print_pnl_summary(df)
        return

    print("PENDING TRADES")
    print("==============")
    for pos, (idx, row) in enumerate(pending.iterrows(), 1):
        print(f"  [{pos}] {row['signal_date']}  {row['signal_type']:<4}  "
              f"entry {row['entry_date']} -> exit {row['exit_date']}  ({row['regime']})")

    choice = input("\nWhich trade are you closing? (number, or blank to cancel): ").strip()
    if choice == "":
        print("Cancelled — no changes made.")
        return
    try:
        sel = int(choice)
    except ValueError:
        print("Invalid selection — no changes made.")
        return
    pending_idx = pending.index.tolist()
    if not (1 <= sel <= len(pending_idx)):
        print("Selection out of range — no changes made.")
        return
    target = pending_idx[sel - 1]

    entry_price = prompt_float("Enter entry_price (fill at open): ")
    if entry_price is None:
        print("Cancelled — no changes made.")
        return
    exit_price = prompt_float("Enter exit_price (fill at exit): ")
    if exit_price is None:
        print("Cancelled — no changes made.")
        return

    actual_return = (exit_price - entry_price) / entry_price * 100.0
    df.loc[target, "entry_price"] = f"{entry_price:.2f}"
    df.loc[target, "exit_price"] = f"{exit_price:.2f}"
    df.loc[target, "actual_return"] = f"{actual_return:.4f}"
    df.loc[target, "status"] = "CLOSED"

    df.to_csv(TRADE_LOG, index=False)
    print(f"\nTrade closed: {df.loc[target, 'signal_type']} "
          f"({df.loc[target, 'signal_date']}) — actual return {actual_return:+.2f}%")

    print_pnl_summary(df)


if __name__ == "__main__":
    main()
