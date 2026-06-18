# S&P 500 Daily Signal Backtest

Backtests 10 hypotheses about predictable directional patterns in daily S&P 500
(`^GSPC`) price data from 2010-01-01 to today, using `yfinance`, `pandas` and `numpy`.

## Hypotheses tested

1. Mean reversion after 3/4/5 consecutive red days
2. Bounce after a single-day drop of >1.5% / >2% / >3%
3. 5-day recovery after a 20-day closing low
4. Momentum after 3/4/5 consecutive green days
5. 30-calendar-day follow-through after a new 52-week high
6. Next-day bounce after a high-volume (>150% of 20-day avg) red day
7. Next-day direction & magnitude after a narrow-range (bottom-decile range) day
8. Day-of-week directional bias (Mon–Fri)
9. Day after approximated FOMC announcement vs normal days
10. 3 accelerating red days (lower lows) vs simple 3 red days
11. Combination signal: 20-day low AND 3+ consecutive red days (5-day forward)

## Usage

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python backtest.py
```

`stress_test.py` is a separate validation run: it extends the data back to
2000-01-01 and re-tests H3 and H2 >3% only, across six market eras (dot-com
crash, 2003-07 bull, 2008-09 crisis, 2010-19 bull, COVID, 2022-present), with
regime splits and a stress-test summary. It never writes to the CSV log.

Output is a formatted console report: per-hypothesis sample size, win rate,
average win/loss returns and notes, followed by an overall ranking of all
signals by win rate. Signals with fewer than 30 occurrences are flagged as
statistically unreliable.

The top-ranked signals (H1 5+ red days, H3 20-day low, H5 52-week high, and
the H11 combination) also get a bull/bear regime split based on the trigger
day's position relative to the 200-day moving average. Edge verdicts are
EV-based: EV > 0.10% per trade counts as an edge, below -0.10% as negative
expected value.

Every signal is measured two ways: close-to-close (theoretical, you'd need to
fill at the trigger day's close) and open-to-close (realistic, entering at the
next day's open). The overall summary ranks signals by realistic EV and flags
any signal that loses more than half its theoretical EV to the overnight gap
with a slippage warning, and a final REALISTIC TRADING SUMMARY lists which
signals survive open-price entry.

Each run appends one row per ranked signal to `backtest_results.csv`
(created with headers on first run, never overwritten), including win rate,
sample size, theoretical and realistic EV per trade, estimated slippage and —
for the regime-split signals — bull/bear win rates. Rerunning on different
dates builds a historical log automatically.
