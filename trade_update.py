"""
Trade Updater
=============
Run this AFTER you close a trade that signal_monitor.py logged. It reads PENDING
trades from the Supabase `trade_log` table, lets you enter the actual
entry/exit prices, marks the trade CLOSED, and prints a running P&L summary
compared with the backtested EV.

Uses the Supabase client (credentials from .env), pandas and numpy.
"""

import numpy as np
import pandas as pd

from supabase_client import get_client

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


def print_pnl_summary(client):
    """Pull all CLOSED trades from Supabase and print the running P&L."""
    print("\nRUNNING P&L SUMMARY")
    print("===================")
    try:
        resp = client.table("trade_log").select("*").eq("status", "CLOSED").execute()
    except Exception as exc:
        print(f"Could not read closed trades from Supabase — {str(exc)[:200]}")
        return

    closed = pd.DataFrame(resp.data)
    if closed.empty:
        print("No closed trades yet.")
        return

    returns = pd.to_numeric(closed["actual_return_pct"], errors="coerce").dropna()
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
    try:
        client = get_client()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    try:
        resp = client.table("trade_log").select("*").eq("status", "PENDING") \
            .order("signal_date").execute()
    except Exception as exc:
        print(f"ERROR: could not read PENDING trades from Supabase — {str(exc)[:200]}")
        return

    pending = resp.data
    if not pending:
        print("No PENDING trades to update.")
        print_pnl_summary(client)
        return

    print("PENDING TRADES")
    print("==============")
    for pos, row in enumerate(pending, 1):
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
    if not (1 <= sel <= len(pending)):
        print("Selection out of range — no changes made.")
        return
    target = pending[sel - 1]

    entry_price = prompt_float("Enter entry_price (fill at open): ")
    if entry_price is None:
        print("Cancelled — no changes made.")
        return
    exit_price = prompt_float("Enter exit_price (fill at exit): ")
    if exit_price is None:
        print("Cancelled — no changes made.")
        return

    actual_return = (exit_price - entry_price) / entry_price * 100.0
    try:
        client.table("trade_log").update({
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "actual_return_pct": round(actual_return, 4),
            "status": "CLOSED",
        }).eq("id", target["id"]).execute()
    except Exception as exc:
        print(f"ERROR: could not update trade in Supabase — {str(exc)[:200]}")
        return

    print(f"\nTrade closed: {target['signal_type']} "
          f"({target['signal_date']}) — actual return {actual_return:+.2f}%")

    print_pnl_summary(client)


if __name__ == "__main__":
    main()
