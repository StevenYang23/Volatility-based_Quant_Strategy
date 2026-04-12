# Volatility-based Quant Strategy

This repository implements a low-frequency volatility timing quantitative strategy. The strategy aims to capture the Volatility Risk Premium (VRP) and mean-reverting properties of implied volatility by actively trading at-the-money (ATM) ~30-day straddles. 

## Overall Strategy Idea

The core concept relies on the spread between Implied Volatility (IV) and Realized Volatility (RV). When options are "overpriced" (IV is significantly higher than RV or its historical mean), the strategy sells volatility (short straddle). When options are "underpriced" (IV is unusually low), the strategy buys volatility (long straddle). 

During the holding period, the strategy dynamically hedges its directional exposure (Delta hedging) based on a defined threshold to ensure the returns are primarily driven by Vega, Gamma, and Theta, rather than the underlying asset's price movement. The framework calculates daily mark-to-market PnL, geometric log-returns, and performs a full second-order Taylor expansion to attribute the daily PnL to individual Greeks (Delta, Gamma, Vega, Theta, Vanna, Volga, Rho, and Residual).

---

## Strategy Agents

The codebase supports multiple signal-generating agents that can be run concurrently in a backtest. Each agent inherits the same execution, PnL, and Greek attribution framework but uses a distinct entry/exit logic.

### 1. `Agent_hardThreshold`
- **Parameters:** `k` (threshold multiplier), `allow_short`, `delta_hedge`, `rehedge_threshold`
- **Signal Logic:** Uses the Z-score of the Volatility Risk Premium (VRP). It shorts the straddle when the current VRP exceeds the 20-day rolling mean plus `k * std`. It goes long when VRP falls below the rolling mean minus `k * std`. Positions are closed when the VRP reverts to the mean.

### 2. `Agent_Percentile`
- **Parameters:** `entry_low_percentile` (default: 0.20, implying top/bottom 20%), `allow_short`, `delta_hedge`, `rehedge_threshold`
- **Signal Logic:** Instead of fixed standard deviation multiples, it evaluates the current VRP Z-score against its entire historical distribution. If the Z-score crosses into the top 20% historical percentile, it triggers a short straddle. If it falls into the bottom 20% percentile, it triggers a long straddle. It exits when the signal crosses the historical median (50th percentile).

### 3. `Agent_2threshold`
- **Parameters:** `k_high` (default: 1.2), `k_low` (default: 0.8), `allow_short`, `delta_hedge`, `rehedge_threshold`
- **Signal Logic:** A regime-switching upgrade to the hard-threshold agent. It compares the 40-day VRP standard deviation against the 20-day VRP standard deviation. If the 40-day std > 20-day std (indicating a higher volatility regime), it dynamically enforces a stricter entry threshold (`k_high`). Otherwise, it uses the looser threshold (`k_low`).

### 4. `Agent_Garch`
- **Parameters:** `garch_lookback` (60), `forecast_horizon` (10), `spread_vol_lookback` (60), `band_multiplier` (0.5), `min_hold_days` (3), `rebalance_interval` (5)
- **Signal Logic:** Uses an AR(1)-GARCH(1,1) model fitted on the underlying asset's log returns to forecast annualized Realized Volatility (RV) for the next 10 days. It trades the spread between the current Implied Volatility (IV) and the GARCH-forecasted RV. If IV is significantly overpricing the forecasted RV (spread > `band_multiplier * std`), it shorts volatility. If IV is underpricing the forecasted RV, it goes long. Includes anti-whipsaw controls (minimum hold days, rebalance intervals) to reduce turnover.

---

## Strategy Statistics

The following table summarizes the key performance metrics of the four agents over the backtested period:

```text
------------------------------------------------------------------------------------------------------------------------
       Agent           Win Rate     Sharpe Ratio  Sortino Ratio  Annual Return  Annual Volatility  Max Drawdown   Calmar Ratio  Kelly's Criteria
Agent_hardThreshold     57.32%         1.7862         3.7867        157.89%           53.04%         -17.12%         9.2210          3.3678     
         Agent_Perc     50.00%         0.7118         0.8290         49.66%           56.65%         -23.21%         2.1396          1.2567     
   Agent_2threshold     56.18%         1.4366         2.1880        125.88%           56.72%         -31.59%         3.9844          2.5328     
        Agent_Garch     58.49%         1.1243         0.7963         50.47%           36.34%         -19.40%         2.6013          3.0935     
------------------------------------------------------------------------------------------------------------------------
```

---

## Visual Diagnostics

The backtest suite generates comprehensive visualizations to analyze returns, risk exposures, and PnL attribution.

### 1. Cumulative Return & Signals
![Cumulative Return](demo/Return.png)
**Explanation:** 
- **Top Panel:** Shows the cumulative geometric return (Value Ratio) of the underlying asset versus the different strategy agents over time. 
- **Middle Panel:** Displays the daily log-return rate of the agents.
- **Bottom Panel:** Illustrates the core VRP signal level along with its rolling 20-day mean and $\pm 1$ standard deviation bands, which dictate the hard-threshold trading signals.

### 2. PnL Attribution by Greek (Pie Breakdown)
![Greeks Attribution Pie](demo/Greeks_Attribution_Pie.png)
**Explanation:** 
This visual aggregates the total dollar PnL generated over the backtest period and breaks it down by its option Greek components. For a volatility timing strategy, a large portion of the PnL is expected to be driven by Vega (implied volatility changes) and Gamma/Theta trade-offs, while Delta should ideally be minimal if properly hedged. The Residual captures higher-order terms and discrete hedging friction.

### 3. Daily Greeks Attribution
![Greeks Attribution Line](demo/Greeks_Attribution_line.png)
**Explanation:** 
This plot unpacks the daily dollar attribution of the portfolio across all components (Delta, Gamma, Vega, Theta, Vanna, Volga, Rho, and Residual). It helps identify which risk factors were driving the portfolio's performance on a given day and ensures the actual exposure aligns with the strategy's market-neutral, volatility-focused intent.