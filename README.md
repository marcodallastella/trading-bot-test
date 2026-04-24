# Trading Bot

A mathematically-sized DCA (dollar-cost averaging) trading bot that runs on GitHub Actions, uses Alpaca for order execution, and applies Kelly-criterion position sizing with regime awareness and trailing stops.

> **Educational project.** Built with $100 starting capital to study systematic trading mechanics. Do not deploy real capital without understanding every line of code.

---

## Table of Contents

1. [What this does](#1-what-this-does)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [GitHub Secrets and Variables](#6-github-secrets-and-variables)
7. [Gist Setup (State Persistence)](#7-gist-setup-state-persistence)
8. [Running Locally](#8-running-locally)
9. [GitHub Actions Deployment](#9-github-actions-deployment)
10. [Safety Model](#10-safety-model)
11. [Strategy Reference](#11-strategy-reference)
12. [Backtesting](#12-backtesting)
13. [Monitoring](#13-monitoring)
14. [Going Live](#14-going-live)
15. [Disclaimer](#15-disclaimer)

---

## 1. What this does

Each trading day at 11 AM ET the bot:

1. Loads persistent state from a private GitHub Gist (trailing stops, wash-sale blacklist, rolling equity peak, cooldowns).
2. Evaluates every position for stop-loss (-10%), trailing-stop activation (+20% gain), or trailing-stop trigger (-10% from peak).
3. Evaluates every ticker in the 25-stock universe for a buy signal (Position-in-Range + 50-SMA trend filter).
4. Sizes each buy using a Kelly-informed pipeline: `kelly_base × signal_multiplier × volatility_factor × regime_multiplier`, capped by per-ticker exposure, total deployment, and position count limits.
5. Submits fractional market orders via Alpaca.
6. Saves updated state back to the Gist.
7. Posts a daily summary to Slack (optional).

**Universe (25 tickers, alphabetical):**
`AAPL ABNB AMD AMZN AVGO COIN CRWD DDOG DIS ENPH GOOGL JNJ JPM KO META MRVL MSFT NFLX NET NVDA PLTR SHOP SNOW TSLA UBER`

---

## 2. Architecture

```
main.py  ←  orchestrator (runs once per day)
  ├── state.py       load / save JSON state via GitHub Gist
  ├── risk.py        drawdown halt · kill switch · wash-sale · trailing stops  (pure)
  ├── strategy.py    PIR · SMA · ATR · buy signal · sell signal · regime  (pure)
  ├── sizing.py      Kelly pipeline → notional USD  (pure)
  ├── alpaca_client.py  clock · account · quotes · bars · orders
  ├── notify.py      format and post daily summary
  └── logbook.py     append per-ticker decisions to decisions.csv Gist

backtest/
  ├── data.py        Alpaca primary → yfinance fallback → parquet cache
  ├── metrics.py     Sharpe · Sortino · Kelly · drawdown · win rate
  ├── run_backtest.py  daily-bar replay → equity curve PNG + Kelly recommendation
  └── sensitivity.py   parameter sweep → sensitivity.csv

config.py  ←  single source of truth for every tunable parameter
models.py  ←  shared dataclasses (State, Position, BuySignal, SizingResult, …)
```

**Key invariants:**
- `strategy.py` and `risk.py` are pure — they take DataFrames and return values; no I/O.
- `state.py` never raises — on any Gist error it returns a sell-only state with `drawdown_halt_active=True`.
- All `risk.py` state transitions use `dataclasses.replace` — input State is never mutated.

---

## 3. Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| Alpaca account | Paper account (free at alpaca.markets) |
| GitHub account | For Actions + Gist |
| Slack workspace | Optional (for daily summary) |

---

## 4. Installation

```bash
git clone https://github.com/<your-username>/trading-bot.git
cd trading-bot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Verify the install:

```bash
python -c "import alpaca; import pandas; import numpy; print('OK')"
```

---

## 5. Configuration

All tunable parameters live in `src/config.py`. The values below are defaults — do not change them unless you have backtest evidence justifying the change.

### Key parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BASE_KELLY_FRACTION` | `0.08` | Half-Kelly base; update from backtest output |
| `BASE_KELLY_FRACTION_MIN` | `0.04` | Floor after tier + signal adjustments |
| `BASE_KELLY_FRACTION_MAX` | `0.12` | Ceiling after tier + signal adjustments |
| `MAX_PER_TICKER_EXPOSURE_PCT` | `0.15` | Max 15% of equity in any single ticker |
| `MAX_CONCURRENT_POSITIONS` | `5` | Tier 1 ($0–$500) max open positions |
| `STOP_LOSS_PCT` | `-0.10` | Hard stop at -10% from entry |
| `TRAILING_STOP_ACTIVATION_PCT` | `0.20` | Activate trailing stop at +20% gain |
| `TRAILING_STOP_TRAIL_PCT` | `0.10` | Trail 10% below peak once activated |
| `DRAWDOWN_HALT_THRESHOLD` | `0.25` | Halt all buying if portfolio drawdown ≥ 25% |
| `DRAWDOWN_RESUME_THRESHOLD` | `0.15` | Resume buying when drawdown drops below 15% |
| `WASH_SALE_DAYS` | `31` | IRS wash-sale blackout after a loss sell |
| `PER_TICKER_COOLDOWN_DAYS` | `3` | Minimum days between buys of same ticker |
| `LOOKBACK_DAYS` | `60` | Bar history for PIR + ATR calculation |
| `PIR_STANDARD_BUY_THRESHOLD` | `0.30` | Buy if PIR < 0.30 (price in lower 30th percentile) |
| `PIR_DEEP_DIP_THRESHOLD` | `0.15` | Deep dip bypasses 50-SMA trend filter |
| `PAPER_BURN_IN_DAYS` | `30` | Minimum paper-trading days before going live |

### Equity tiers

The bot automatically adjusts position sizing and limits based on total account equity:

| Tier | Equity range | Per-ticker cap | Max positions | Kelly range |
|------|-------------|----------------|---------------|-------------|
| 1 | $0 – $500 | 15% | 5 | 8%–12% |
| 2 | $500 – $2,500 | 10% | 6 | 6%–10% |
| 3 | $2,500 – $10,000 | 7% | 8 | 5%–8% |
| 4 | $10,000+ | 5% | 10 | 4%–6% |

### Updating `BASE_KELLY_FRACTION` from backtest

After running a backtest, the output block prints:

```
>>> Kelly recommendation (half-Kelly, clamped [0.04, 0.12]):
    BASE_KELLY_FRACTION = 0.0712
```

Copy that value into `src/config.py` and commit before re-deploying.

---

## 6. GitHub Secrets and Variables

### Secrets (encrypted — never visible after entry)

Go to: **Repo → Settings → Secrets and variables → Actions → Secrets**

| Secret | Description |
|--------|-------------|
| `ALPACA_API_KEY` | Alpaca API key (paper or live) |
| `ALPACA_SECRET_KEY` | Alpaca secret key (paper or live) |
| `GIST_ID` | GitHub Gist ID for bot state (see §7) |
| `GIST_TOKEN` | GitHub PAT with `gist` scope only (see §7) |
| `DECISIONS_GIST_ID` | GitHub Gist ID for decisions log (optional) |
| `SLACK_WEBHOOK_URL` | Incoming webhook URL (optional) |

### Variables (plain text — visible in UI, easy to flip)

Go to: **Repo → Settings → Secrets and variables → Actions → Variables**

| Variable | Initial value | Description |
|----------|--------------|-------------|
| `ALPACA_PAPER` | `true` | `true` = paper account, `false` = live |
| `DRY_RUN` | `true` | `true` = evaluate signals but do not submit orders |
| `LIVE_READY` | `false` | Must be `true` for live trading (three-gate check) |
| `KILL_SWITCH` | `0` | Set to `1` to halt all buys immediately |

---

## 7. Gist Setup (State Persistence)

The bot stores all runtime state in two private GitHub Gists — no database required.

### 7.1 State Gist

1. Go to [gist.github.com](https://gist.github.com) → **New secret Gist**
2. Filename: `trading_bot_state.json`
3. Initial content:

```json
{
  "rolling_peak_equity": 100.0,
  "rolling_peak_timestamp": "2026-04-20",
  "last_run_timestamp": "",
  "ticker_last_buy_date": {},
  "wash_sale_blacklist": {},
  "trailing_stops_active": {},
  "cumulative_realized_pnl": 0.0,
  "drawdown_halt_active": false
}
```

4. Click **Create secret gist**. Copy the Gist ID from the URL.

### 7.2 Decisions Gist (optional but recommended)

1. Go to [gist.github.com](https://gist.github.com) → **New secret Gist**
2. Filename: `decisions.csv`
3. Initial content (header row only):

```
timestamp,ticker,price,sma_50,pir,atr_pct,signal_mult,vol_factor,regime_mult,kelly_base,final_pct,action,order_notional,fill_price,reason
```

4. Click **Create secret gist**. Copy the Gist ID.

### 7.3 Personal Access Token

1. Go to [github.com/settings/tokens](https://github.com/settings/tokens) → **Generate new token (classic)**
2. Select scope: **`gist` only**
3. Copy the token — it is shown only once.
4. Add as `GIST_TOKEN` secret in the repo (§6).

---

## 8. Running Locally

### 8.1 Set environment variables

```bash
export ALPACA_API_KEY="your-paper-key"
export ALPACA_SECRET_KEY="your-paper-secret"
export ALPACA_PAPER="true"
export DRY_RUN="true"
export LIVE_READY="false"
export GIST_ID="your-gist-id"
export GIST_TOKEN="your-pat"
export DECISIONS_GIST_ID="your-decisions-gist-id"   # optional
```

### 8.2 Dry-run (no orders submitted)

```bash
python -m src.main
```

Expected output:

```
[INFO] Loaded state from Gist
[INFO] Market is open
[INFO] Regime: BULL (multiplier=1.00)
[INFO] Evaluating sells for 0 positions
[INFO] Evaluating buys for 25 tickers
[INFO] AAPL: PIR=0.24 signal=STANDARD_DIP DRY_RUN_BUY notional=$8.00
...
[INFO] State saved to Gist
[INFO] Daily summary sent
```

### 8.3 Run tests

```bash
# Unit tests (fast, no network)
pytest tests/ -v --ignore=tests/test_integration.py

# Integration tests (requires real credentials)
pytest tests/test_integration.py -v -s
```

### 8.4 Paper-mode local run

Remove `DRY_RUN=true` (or set it to `false`) and run again. Orders are submitted to the Alpaca paper account.

```bash
export DRY_RUN="false"
python -m src.main
```

---

## 9. GitHub Actions Deployment

### 9.1 Daily trading (`trade.yml`)

- **Schedule:** 11:00 AM ET, Monday–Friday
- **Manual trigger:** Actions → "Daily Trade" → Run workflow
- **Concurrency:** Only one run at a time; a second trigger waits (never cancels in progress)
- **Timeout:** 15 minutes hard kill

The workflow uses the `vars.*` flags (§6) to control paper/live and dry-run mode. Change a variable in the UI to take effect on the next scheduled run.

### 9.2 Backtest (`backtest.yml`)

- **Trigger:** Manual only (Actions → "Backtest" → Run workflow)
- **Inputs:** `start_date`, `end_date`, `run_sweep` (true/false)
- **Output:** Artifacts retained 30 days — download `backtest_output.zip` for the equity curve PNG and (if sweep=true) `sensitivity.csv`
- **Bar cache:** Alpaca bars are cached in the runner using `actions/cache`. Re-runs over the same date range skip re-downloading.

---

## 10. Safety Model

### 10.1 Three-gate live check

All three conditions must be true for a live order to be submitted:

```
ALPACA_PAPER=false   AND
DRY_RUN=false        AND
LIVE_READY=true
```

If any gate is closed, `main.py` exits with code 1 before touching the market. This makes it impossible to accidentally trade live when testing configuration changes.

### 10.2 Kill switch

Set repository variable `KILL_SWITCH=1` at any time. Takes effect on the next run. All buys are skipped; existing positions continue to be evaluated for sells normally.

To re-enable buying: set `KILL_SWITCH=0`.

### 10.3 Drawdown halt

If the portfolio falls 25% or more from its rolling 365-day peak, all buying halts automatically. Selling continues normally. Buying resumes automatically when the drawdown drops below 15% (hysteresis prevents flapping).

| Threshold | Behaviour |
|-----------|-----------|
| Drawdown ≥ 25% | `drawdown_halt_active = True` — no new buys |
| Drawdown < 15% | `drawdown_halt_active = False` — buys resume |

### 10.4 Sell-only mode (Gist failure)

If the state Gist cannot be read (network error, invalid JSON, missing fields), the bot enters sell-only mode: existing positions are evaluated for stops, but no new buys are placed. This is belt-and-suspenders — the sell-only flag AND `drawdown_halt_active=True` are both set, so a caller that ignores the flag is still protected by the halt state.

### 10.5 Spread guard

Orders are skipped if the bid-ask spread is too wide:
- Buy: spread > 20 bps skips the ticker
- Sell: spread > 30 bps delays the sell (logged as `SPREAD_WIDE`)

### 10.6 Wash-sale guard

After selling at a loss, the ticker is blacklisted for 31 calendar days. A buy attempted during the blackout period is skipped and logged. This is an IRS-compliance safeguard — it does not guarantee tax compliance; consult a tax professional.

---

## 11. Strategy Reference

### 11.1 Buy signal

A buy signal fires when **both** conditions are met:

1. **PIR (Position-in-Range):** Current price is below the 30th percentile of the last 60 closes.
   ```
   PIR = clip((price - 10th_pct) / (90th_pct - 10th_pct), 0, 1)
   Buy if PIR < 0.30
   ```

2. **Trend filter:** Current price is above the 50-day SMA.
   - Exception: if `PIR < 0.15` (deep dip), the trend filter is bypassed.

**Signal multipliers** (applied to Kelly base):

| Condition | Multiplier |
|-----------|-----------|
| PIR < 0.10 (very deep dip) | 1.5× |
| PIR < 0.20 (strong dip) | 1.2× |
| PIR < 0.30 (standard dip) | 1.0× |

### 11.2 Sell signal priority

Evaluated in order (first match wins):

1. **Stop loss:** unrealized return ≤ -10% → sell immediately
2. **Trailing stop trigger:** trailing stop active AND price fell ≥ 10% from trailing peak → sell
3. **Trailing stop activation:** unrealized return ≥ +20% → activate trailing stop (no sell yet)
4. **Hold:** none of the above

### 11.3 Regime detection (SPY)

Uses SPY 200-SMA and 20-day slope to classify market regime:

| Regime | SPY vs 200SMA | Slope | Kelly multiplier |
|--------|--------------|-------|-----------------|
| BULL | Above | Rising (>0.5%) | 1.00× |
| LATE_CYCLE | Above | Flat | 0.85× |
| TOPPING | Above | Falling (<-0.5%) | 0.60× |
| RECOVERY | Below | Rising (>0.5%) | 0.60× |
| BEAR | Below | Falling or flat | 0.30× |

### 11.4 Volatility factor (ATR)

ATR(14) as a percentage of price:

| ATR% | Factor |
|------|--------|
| < 2% (low vol) | 1.2× (size up) |
| 2%–4% (normal) | 1.0× |
| > 4% (high vol) | 0.7× (size down) |

### 11.5 Kelly position size pipeline

```
kelly_base        = BASE_KELLY_FRACTION (from tier, bounded [0.04, 0.12])
× signal_mult     = 1.0 / 1.2 / 1.5 (based on PIR depth)
× vol_factor      = 0.7 / 1.0 / 1.2 (based on ATR%)
× regime_mult     = 0.30–1.00 (based on SPY regime)
= position_pct

buy_amount_usd = total_equity × position_pct
```

Capped in order:
1. Position count ≥ tier max → skip
2. Per-ticker exposure would exceed 15% → reduce to fill remaining headroom
3. Total deployment would exceed 90% → reduce to fill remaining headroom
4. Result < $1.00 minimum notional → skip

---

## 12. Backtesting

### 12.1 Run a backtest

```bash
# Activate venv and set Alpaca credentials
python - <<'EOF'
from datetime import date
from backtest.run_backtest import run_backtest
from src import config

result = run_backtest(
    universe=config.UNIVERSE,
    start=date(2023, 1, 1),
    end=date(2024, 12, 31),
    api_key="your-alpaca-key",
    secret_key="your-alpaca-secret",
    output_dir="backtest_output",
)
EOF
```

Output files in `backtest_output/`:
- `equity_curve.png` — strategy vs SPY buy-and-hold
- Console block with Sharpe, Sortino, Kelly recommendation

### 12.2 Run a sensitivity sweep

```bash
python - <<'EOF'
from datetime import date
from backtest.sensitivity import run_sweep, SweepConfig
from src import config

rows = run_sweep(
    universe=config.UNIVERSE,
    start=date(2023, 1, 1),
    end=date(2024, 12, 31),
    sweep_config=SweepConfig(
        stop_loss_pcts=[-0.08, -0.10, -0.12],
        pir_thresholds=[0.20, 0.25, 0.30],
        lookback_days=[45, 60, 90],
        regime_bear_mults=[0.20, 0.30, 0.40],
    ),
    output_dir="backtest_output/sweep",
)
# Results sorted by Sharpe descending; saved to backtest_output/sweep/sensitivity.csv
print(rows[0])  # best combo
EOF
```

### 12.3 Interpreting results

| Metric | What to look for |
|--------|-----------------|
| Sharpe ratio | > 0.5 is acceptable; > 1.0 is good |
| Max drawdown | Should be < 30% (halt fires at 25%) |
| Kelly fraction | Copy half-Kelly into `BASE_KELLY_FRACTION` in `config.py` |
| Profit factor | > 1.2 indicates system edge |

**Bar cache:** Fetched bars are stored in `.backtest_cache/` as parquet files. Re-runs over the same date range reuse the cache. Pass `force_refresh=True` to `fetch_bars()` to bypass.

---

## 13. Monitoring

### 13.1 Daily Slack summary

If `SLACK_WEBHOOK_URL` is set, the bot posts a summary after each run:

```
Trading Bot — 2026-04-21 (PAPER / DRY_RUN)
──────────────────────────────────────────────
Equity:       $103.42  (+3.42%)
Peak equity:  $104.30  (2026-03-15)
Drawdown:     0.84%    [no halt]
Regime:       BULL  (mult=1.00)
Kelly base:   0.0800

SELLS (0):    —
BUYS (2):
  AAPL  PIR=0.21  notional=$8.02  (STANDARD_DIP)
  NVDA  PIR=0.14  notional=$9.41  (DEEP_DIP, trend bypass)

Trailing stops active: PLTR (peak=$28.15, entry=$22.40)
Wash-sale blacklist:   0
Realized PnL (all-time): +$8.42
```

### 13.2 Decisions log (CSV Gist)

The decisions Gist accumulates one row per ticker per run. Columns:

```
timestamp, ticker, price, sma_50, pir, atr_pct,
signal_mult, vol_factor, regime_mult, kelly_base,
final_pct, action, order_notional, fill_price, reason
```

Open the raw Gist URL or download it for analysis in a spreadsheet.

### 13.3 GitHub Actions run history

Each workflow run is retained in the Actions tab. Failure runs upload a `trade-failure-logs-<run_id>.zip` artifact (7-day retention) containing any `.log` files from the working directory.

---

## 14. Going Live

**Do not go live until all of the following are true:**

- [ ] Paper mode has run for ≥ 30 calendar days (`PAPER_BURN_IN_DAYS`) with exit code 0 every run
- [ ] State Gist has never contained invalid JSON during the paper period
- [ ] No single position has exceeded 15% of equity at any point
- [ ] A backtest on the last 2 years of data has been run and `BASE_KELLY_FRACTION` updated in `config.py`
- [ ] The backtest Sharpe ratio is > 0 (system has positive expected value)
- [ ] You understand every parameter in `config.py` and can explain what happens if it is doubled or halved

**Promotion steps:**

1. Generate a live Alpaca API key at `app.alpaca.markets` (not paper).
2. Update secrets `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` in the repo.
3. Set variables: `ALPACA_PAPER=false`, `LIVE_READY=true`, keep `DRY_RUN=true`.
4. Trigger a manual run. Verify the bot loads, evaluates, and exits 0 without submitting orders (three-gate: `DRY_RUN=true` still blocks).
5. Set `DRY_RUN=false`.
6. Trigger a manual run. Monitor in real time. Verify orders appear in the Alpaca live dashboard.
7. The scheduled workflow takes over from here.

**Rolling back to paper:**

Set `ALPACA_PAPER=true` and swap `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` back to paper credentials. No code change required.

---

## 15. Disclaimer

This project is for **educational purposes only**.

- Past backtest performance does not predict future results.
- The Kelly criterion produces aggressive position sizes for high win-rate systems. The half-Kelly implementation reduces sizing, but drawdowns can still be severe.
- The wash-sale logic is a best-effort safeguard and does not constitute tax advice. Consult a qualified tax professional before claiming losses.
- The authors are not registered investment advisers. Nothing in this repository constitutes investment advice.
- Starting capital of $100 is intentional — size your real deployment accordingly.

**Use at your own risk.**
