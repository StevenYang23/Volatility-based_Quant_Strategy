# Volatility-based Quant Strategy

This repository implements a low-frequency volatility timing quantitative strategy. The strategy aims to capture the Volatility Risk Premium (VRP) and mean-reverting properties of implied volatility by actively trading at-the-money (ATM) ~30-day straddles. 

## Overall Strategy Idea

The core concept relies on the spread between Implied Volatility (IV) and Realized Volatility (RV). When options are "overpriced" (IV is significantly higher than RV or its historical mean), the strategy sells volatility (short straddle). When options are "underpriced" (IV is unusually low), the strategy buys volatility (long straddle). 

During the holding period, the strategy dynamically hedges its directional exposure (Delta hedging) based on a defined threshold to ensure the returns are primarily driven by Vega, Gamma, and Theta, rather than the underlying asset's price movement. The framework calculates daily mark-to-market PnL, geometric log-returns, and performs a full second-order Taylor expansion to attribute the daily PnL to individual Greeks (Delta, Gamma, Vega, Theta, Vanna, Volga, Rho, and Residual).

---

## Strategy Agents

The codebase supports multiple signal-generating agents that can be run concurrently in a backtest. Each agent inherits the same execution, PnL, and Greek attribution framework but uses a distinct entry/exit logic.

| Agent | Long Condition (Buy Volatility) | Short Condition (Sell Volatility) | Close Condition (Exit Position) |
| :--- | :--- | :--- | :--- |
| **`Agent_hardThreshold`**<br>*(k=1)* | `VRP < 20d_Mean - k * 20d_Std` | `VRP > 20d_Mean + k * 20d_Std` | `VRP` crosses `20d_Mean` |
| **`Agent_Percentile`**<br>*(low=0.20, high=0.80)* | `Z-Score(VRP)` < historical 20th percentile | `Z-Score(VRP)` > historical 80th percentile | `Z-Score(VRP)` crosses historical median |
| **`Agent_2threshold`**<br>*(k_high=1.2, k_low=0.8)* | `VRP < 20d_Mean - k_active * 20d_Std`<br>*(`k_active` is `k_high` if 40d_Std > 20d_Std, else `k_low`)* | `VRP > 20d_Mean + k_active * 20d_Std`<br>*(`k_active` is `k_high` if 40d_Std > 20d_Std, else `k_low`)* | `VRP` crosses `20d_Mean` |
| **`Agent_Garch`**<br>*(garch=60d, fcast=10d, band=0.5)* | `IV - GARCH_Forecast_RV < -band` | `IV - GARCH_Forecast_RV > +band` | Spread returns inside the no-trade band |

*Note: All agents share standard risk parameters including `allow_short`, `delta_hedge`, and `rehedge_threshold`. The `Agent_Garch` model includes additional anti-whipsaw logic (minimum hold of 3 days, rebalancing restricted to every 5 days).*

---

## Strategy Statistics

The following table summarizes the key performance metrics of the four agents over the backtested period:

```text
------------------------------------------------------------------------------------------------------------------------
       Agent           Win Rate     Sharpe Ratio  Sortino Ratio  Annual Return  Annual Volatility  Max Drawdown   Calmar Ratio  Kelly's Criteria
Agent_hardThreshold     78.05%         1.3083         1.7865        130.20%           63.73%         -27.52%         4.7308          2.0528     
         Agent_Perc     63.64%         0.7118         0.8290         49.66%           56.65%         -23.21%         2.1396          1.2567     
   Agent_2threshold     74.47%         0.7157         0.8275         55.73%           61.89%         -45.68%         1.2200          1.1564     
        Agent_Garch     72.41%         1.1243         0.7963         50.47%           36.34%         -19.40%         2.6013          3.0935     
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