# Volatility-Based Quantitative Trading Strategy

A quantitative trading strategy that exploits the volatility risk premium (VRP) by trading straddles based on the relationship between implied volatility (IV) and realized volatility (RV). The strategy includes both delta-hedged and unhedged implementations for performance comparison.

## Overview

This project implements a volatility trading strategy that:
- Uses **Bollinger Bands** to detect the trading sign (when to go long or short straddles)
- **Shorts straddles** when Implied Volatility (IV) > Realized Volatility (RV) - betting that IV is overpriced
- **Longs straddles** when IV < RV - betting that IV is underpriced
- Compares **delta-hedged** vs **unhedged** strategies to demonstrate the impact of delta hedging on risk management

Example backtest results (from a typical run; actual numbers depend on data and parameters):

| Strategy | Sharpe Ratio | Annual Return | Annual Volatility |
|----------|--------------|---------------|-------------------|
| **Delta-Hedged** | ~2.46 | ~139.83% | ~55.21% |
| **Pure Straddle** | ~1.57 | ~166.72% | ~103.56% |

### Performance Visualization

The backtest results demonstrate the comparative performance of delta-hedged and unhedged strategies:

![Portfolio Value and Returns](demo/value_and_return_rate.png)
*Portfolio value evolution and cumulative returns comparison between delta-hedged and unhedged strategies. The chart shows how delta hedging affects portfolio performance, volatility, and risk-adjusted returns over the backtest period.*

![Greeks Monitoring](demo/Greeks_monitor.png)
*Portfolio Greeks (Delta, Gamma, Vega, Theta) exposure over time for both strategies. This visualization highlights how delta hedging maintains near-zero delta exposure while the unhedged strategy retains directional risk. The charts also show vega and theta exposure, which are key drivers of volatility trading profitability.*

### Risk-Return Metrics
**Key Findings:**
- **Delta-Hedged Strategy**: Typically achieves a higher Sharpe Ratio with lower annual volatility, demonstrating better risk-adjusted returns.
- **Pure Straddle Strategy**: Usually shows a lower Sharpe Ratio and higher volatility due to unhedged directional exposure.
- **Delta Hedging Benefit**: Delta hedging reduces volatility and improves risk-adjusted returns by keeping net delta near zero.

## Strategy Logic

### Bollinger Bands (VRP Bands)

The strategy uses a **Bollinger-Bands-style** construction on the **Volatility Risk Premium (VRP)**:

- **VRP** = IV − RV (implied volatility minus realized volatility, both in level form).
- **VRP_mean**: 20-day rolling mean of VRP (the “middle band”).
- **VRP_std**: 20-day rolling standard deviation of VRP.
- **Bands**: **VRP_mean + k × VRP_std** (upper band) and **VRP_mean − k × VRP_std** (lower band), where **k** is the threshold parameter (`vrp_threshold` for entry, `vrp_close_threshold` for exit).

The **z-score** is (VRP − VRP_mean) / VRP_std. When VRP is above the upper band, the z-score is high (IV rich vs history); when VRP is below the lower band, the z-score is low (IV cheap vs history). Entry and exit rules are expressed in terms of this z-score and the bands.

### How the Agent Opens a Position

The agent opens a position **only when the VRP z-score leaves the band** (i.e. |z-score| > `vrp_threshold`):

1. **Short straddle** (sell options): When **VRP > VRP_mean + vrp_threshold × VRP_std** (z-score above the upper band).  
   Interpretation: IV is high relative to recent VRP; the agent sells volatility (collects premium).

2. **Long straddle** (buy options): When **VRP < VRP_mean − vrp_threshold × VRP_std** (z-score below the lower band).  
   Interpretation: IV is low relative to recent VRP; the agent buys volatility (pays premium).

3. **No trade**: When VRP is **inside the bands** (between the lower and upper band), the agent does not open a new position.

So the “Bollinger bands” on VRP directly drive the **direction** of the trade (long vs short) and the **timing** of entry.

### How the Agent Closes a Position

A position is closed in any of these cases:

1. **Near-expiry**: TTM &lt; min_ttm/2 (e.g. &lt; ~0.5 days). The position is closed to avoid expiration risk.

2. **Roll (monthly switch)**: On the **first Tuesday of each month** (trade date), if the agent has an open position, it closes that position and may open the **next month’s option** on the same day. So the “old” option is closed on the roll date.

3. **VRP close (bands):**  
   - **Long position** (holding long straddle): Close when **VRP > VRP_mean + vrp_close_threshold × VRP_std** (VRP moves back above the upper band). Idea: IV has repriced higher; take profit or cut risk.  
   - **Short position** (holding short straddle): Close when **VRP < VRP_mean − vrp_close_threshold × VRP_std** (VRP moves back below the lower band). Idea: IV has repriced lower; take profit or cut risk.

4. **Rehedge (delta-hedged only)**: This is **not** a full close. When |net_delta| > `delta_rehedge_threshold`, the agent **only adjusts the underlying share position** (same option size). Option count is unchanged; balance is updated for the underlying buy/sell (PnL from the hedge adjustment). No new option is opened.

### Long vs Short: What’s Different

| Aspect | Long position | Short position |
|--------|----------------|----------------|
| **Entry condition** | VRP &lt; VRP_mean − k×VRP_std (below lower band) | VRP &gt; VRP_mean + k×VRP_std (above upper band) |
| **Action at entry** | Buy straddle (pay premium), buy/sell underlying to delta-hedge | Sell straddle (receive premium), buy/sell underlying to delta-hedge |
| **Sizing cap** | **max_invest × NAV** (max fraction of NAV spent: premium + hedge cost) | **max_leverage × NAV** (max fraction of NAV as exposure: premium + \|delta\|×S) |
| **Close condition** | Close when VRP moves **back up** above upper band (VRP &gt; VRP_mean + k×VRP_std) | Close when VRP moves **back down** below lower band (VRP &lt; VRP_mean − k×VRP_std) |
| **Intuition** | Buy cheap vol when VRP is low; exit when vol reprices higher | Sell rich vol when VRP is high; exit when vol reprices lower |

So: **long** = enter when VRP is low (below band), exit when VRP is high (above band). **Short** = enter when VRP is high (above band), exit when VRP is low (below band). The bands define both entry and exit.

### Delta Hedging
- **Delta-Hedged Strategy**: `Agent_DDH` with `delta_hedge=True` — hedges delta at entry and can rehedge when |net_delta| exceeds a threshold.
- **Unhedged (Pure Straddle) Strategy**: `Agent_DDH` with `delta_hedge=False` — no underlying position; used in the backtest for comparison. An alternative is `Agent_Straddles` (raw VRP, no z-score).

### Capital Allocation

Position sizing is **NAV-based** and scales with account value:

#### Key Components
- **Total Account Value (NAV)**: Cash + option positions (marked-to-market) + underlying position (for delta-hedged strategy).

#### Sizing logic (Agent_DDH)
1. **Long straddle**: Cost per unit = premium + max(0, −delta × S) (underlying hedge cost when delta is negative). Units = floor((max_invest × NAV) / cost_per_unit).
2. **Short straddle**: Exposure per unit = premium + |delta| × S. Units = −floor((max_leverage × NAV) / exposure_per_unit).

So long positions are capped by **max_invest** (fraction of NAV spent); short positions are capped by **max_leverage** (fraction of NAV as exposure). Agent_Straddles uses premium only (no delta term) for cost/exposure.

## Project Structure

```
Volatility-based_Quant_Strategy/
├── Agent_Class.py              # Unhedged straddle agent (raw VRP threshold)
├── Agent_DDH_Class.py          # Delta-hedged straddle agent (VRP z-score signal, optional hedging)
├── Build_data.ipynb            # Data collection and preprocessing notebook
├── Back_test.ipynb             # Backtesting and performance analysis notebook
├── data/                       # Auxiliary data files (e.g. futures/options references)
├── DataSet/                    # Processed option and underlying data
│   ├── underlying.csv          # Historical underlying (Close, Return, RV, imp_vol, VRP, VRP_std, VRP_mean)
│   ├── call_list.pkl          # List of call option symbols
│   ├── put_list.pkl           # List of put option symbols
│   ├── dates.pkl              # Trading dates
│   ├── date_strs.pkl          # Trading dates (string format)
│   └── O_*.csv                # Individual option contract data
└── demo/                       # Visualization outputs
```

## Key Components

### 1. Data Collection (`Build_data.ipynb`)

- Downloads historical price data for the underlying asset (AAPL) via yfinance
- Computes returns and 20-day rolling realized volatility (RV), saved to `DataSet/underlying.csv`
- Identifies trading dates (first Tuesday of each month)
- Selects at-the-money (ATM) options expiring on the second Friday of the next month
- Downloads option price data (e.g. via Polygon), calculates implied volatility (Black-Scholes)
- Adds to underlying: `imp_vol`, `VRP`, and **Bollinger-Bands style** `VRP_mean`, `VRP_std` (20-day rolling) for the VRP z-score signal
- Stores option lists and dates in pickle files; option series in `DataSet/O_*.csv`

### 2. Strategy Agents

#### `Agent_DDH` (in `Agent_DDH_Class.py`)
- Single agent used for both **delta-hedged** and **unhedged** modes via the `delta_hedge` flag.
- **Trading signal**: VRP z-score from 20-day rolling mean and std (Bollinger Bands style). Requires `VRP_std` and `VRP_mean` in `DataSet/underlying.csv`.
- **Delta-hedged mode** (`delta_hedge=True`): Initial delta hedge at entry; **rehedge** adjusts only the underlying share count (same option size) when |net_delta| > `delta_rehedge_threshold`, with PnL reflected in balance.
- **Unhedged mode** (`delta_hedge=False`): Pure straddle, no underlying position.
- Position sizing: long = `max_invest × NAV`; short = `max_leverage × NAV` (exposure cap).
- Exit: near-expiry (TTM), monthly roll (first Tuesday), or VRP close (long when VRP > upper band, short when VRP < lower band). See “How the Agent Closes a Position” above.

#### `Agent_Straddles` (in `Agent_Class.py`)
- Alternative unhedged straddle agent using **raw VRP** (no z-score); does not use `VRP_std`/`VRP_mean`.
- Position sizing: `max_invest` (long), `max_leverage` (short). Tracks Greeks; exit by TTM or optional `vrp_close_threshold`.
- Not used in the current `Back_test.ipynb` (which uses `Agent_DDH` for both strategies).

### 3. Backtesting (`Back_test.ipynb`)

- Runs both strategies over the same date range: **Delta-Hedged** (`Agent_DDH` with `delta_hedge=True`) and **Pure Straddle** (`Agent_DDH` with `delta_hedge=False`).
- Loads trade dates and option pairs from `DataSet`; for each day, selects the active option pair and calls `build_position`, `cal_value`, `should_exit`, and (for DDH) `rehedge`.
- Tracks daily NAV, Greeks, option/underlying counts, and trade events.
- Computes performance metrics: Sharpe Ratio, Annual Return, Annual Volatility, Maximum Drawdown, Total Return.
- Generates comparative NAV/returns charts and Greeks monitoring plots (saved in `demo/`).

## Features

### Position Sizing
- **NAV-based sizing**: Longs limited by `max_invest × NAV`; shorts by `max_leverage × NAV` (with premium and, for DDH, delta-based cost/exposure).
- Configurable `max_invest`, `max_leverage`, and TTM filters.

### Risk Management
- **Time-to-maturity**: Closes when TTM &lt; min_ttm/2 (near expiry).
- **Monthly roll**: On the first Tuesday of each month, the current position is closed and the agent may open the next month’s option.
- **VRP exit**: Long closes when VRP &gt; VRP_mean + k×VRP_std; short closes when VRP &lt; VRP_mean − k×VRP_std (k = `vrp_close_threshold`).
- **Delta rehedge** (DDH only): When |net_delta| &gt; `delta_rehedge_threshold`, only the underlying share count is adjusted (no option close); PnL from the hedge trade is applied to balance.

### Greeks Tracking
- **Delta**: Price sensitivity to underlying moves
- **Gamma**: Rate of change of delta
- **Vega**: Sensitivity to volatility changes
- **Theta**: Time decay

## Usage

### 1. Data Preparation

Run `Build_data.ipynb` to:
- Download underlying asset data (AAPL) and save to `DataSet/underlying.csv`
- Calculate returns and 20-day RV; then add ATM implied vol, VRP, and 20-day `VRP_mean`/`VRP_std`
- Get first-Tuesday trade dates and ATM option expiries; download option history
- Produce `call_list.pkl`, `put_list.pkl`, `dates.pkl`, `date_strs.pkl`, and `DataSet/O_*.csv`

### 2. Backtesting

Run `Back_test.ipynb` to:
- Load trade dates and option lists from `DataSet`
- Initialize **Delta-Hedged** and **Pure Straddle** agents (both `Agent_DDH` with different `delta_hedge` and `max_leverage`)
- Run the backtest loop day-by-day; generate performance metrics and comparison charts

### Configuration Parameters

The notebook uses the following (match your run for reproducibility):

#### Delta-Hedged Agent (`Agent_DDH` with hedging)
```python
agent_ddh = Agent_DDH(
    balance=5000.0,
    max_invest=0.75,
    max_leverage=0.5,
    vrp_threshold=0.75,           # VRP z-score threshold for entry
    vrp_close_threshold=0.75,     # VRP z-score for exit
    delta_rehedge_threshold=10,   # Rehedge when |net_delta| > 10
    delta_hedge=True,
)
```

#### Pure Straddle Agent (`Agent_DDH` without hedging)
```python
agent_straddle = Agent_DDH(
    balance=5000.0,
    max_invest=0.75,
    max_leverage=0,               # No short exposure cap (or set as needed)
    vrp_threshold=0.75,
    vrp_close_threshold=0.75,
    delta_hedge=False,
)
```

## Performance Metrics

The backtest generates comprehensive performance metrics:

- **Final NAV**: Ending portfolio value
- **Total Return**: Cumulative return over backtest period
- **Sharpe Ratio**: Risk-adjusted return measure
- **Annual Volatility**: Standard deviation of returns (annualized)
- **Maximum Drawdown**: Largest peak-to-trough decline

## Dependencies

```python
pandas
numpy
scipy
matplotlib
yfinance
polygon-api-client
```

## Data Requirements

- **Underlying**: Historical prices (e.g. AAPL) in `DataSet/underlying.csv` with columns: Date, Close, Return, RV, and (after Build_data) imp_vol, VRP, **VRP_std**, **VRP_mean** (20-day rolling for the z-score signal).
- **Options**: Per-contract CSVs in `DataSet/O_*.csv` with timestamp, close, k, ttm, r, imp_vol (and S where needed). Lists in `call_list.pkl`, `put_list.pkl`; trading dates in `dates.pkl`, `date_strs.pkl`.

## Notes

- **Bollinger Bands style signal**: The trading sign uses a VRP z-score: 20-day rolling mean and std of VRP in `DataSet/underlying.csv` (`VRP_mean`, `VRP_std`). Entry when |(VRP − VRP_mean) / VRP_std| &gt; `vrp_threshold`.
- **Greeks**: Black-Scholes analytical Greeks (delta, gamma, vega, theta) are used in both agents.
- **Realized volatility**: Build_data uses 20-day rolling std of returns (annualized); Agent_DDH also computes RV_30d from log returns for some logic.
- **Trading schedule**: First Tuesday of each month; options are ATM, expiring second Friday of the next month.
- **Data**: Underlying and option data are produced by `Build_data.ipynb` (yfinance, Polygon, etc.). Ensure `underlying.csv` includes `VRP_std` and `VRP_mean` for Agent_DDH.

