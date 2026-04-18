# Volatility-based Quant Strategy

*Documentation last aligned with the repository: **April 2026**.*

This repository implements a low-frequency volatility timing quantitative strategy. The strategy aims to capture the Volatility Risk Premium (VRP) and mean-reverting properties of implied volatility by actively trading at-the-money (ATM) ~30-day straddles.

## Overall Strategy Idea

The core concept relies on the spread between Implied Volatility (IV) and Realized Volatility (RV). When options are "overpriced" (IV is significantly higher than RV or its historical mean), the strategy sells volatility (short straddle). When options are "underpriced" (IV is unusually low), the strategy buys volatility (long straddle). 

During the holding period, the strategy dynamically hedges its directional exposure (Delta hedging) based on a defined threshold to ensure the returns are primarily driven by Vega, Gamma, and Theta, rather than the underlying asset's price movement. The framework calculates daily mark-to-market PnL, geometric log-returns, and uses path-wise Greeks attribution for daily PnL decomposition.

### Annualization & Day-Count Convention

All calculations use **252 trading days per year** consistently

## Strategy Agents

The codebase supports multiple signal-generating agents that can be run concurrently in a backtest. Each agent shares the same execution, PnL, and Greek attribution framework but uses distinct entry/exit logic.

### Shared Parameters

All agents accept these parameters:

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `display_name` | Label used in plots and stats tables | agent-specific |
| `allow_short` | Whether the agent can sell straddles | `True` |
| `delta_hedge` | Whether to delta-hedge positions | `True` |
| `rehedge_threshold` | Net delta magnitude that triggers a re-hedge | `0.05` |

### Shared Execution Rules

- **Position unit**: each options trade is **1 lot = 100 options**
- **Underlying hedge**: hedge shares are always rounded to an **integer**
- **Re-hedge PnL**: realized using hedge average entry vs exit price on underlying reductions/flips

### Garch / LongTerm entry: what “Zscore” means

Let **something** be **IV − GARCH_RV** (Garch) or **IV − meanRV** (LongTerm), with **IV** = `Straddle_imp_vol`, **GARCH_RV** from the model, and **meanRV** = mean of the prior `long_term_window` daily **RV** values (today’s **RV** is not in **meanRV** for that day).

**Zscore(something)** = (**something** − **20-day rolling mean of something**) / (**20-day rolling std of something**). **`entry_threshold`** is in those dimensionless Z units.

**Entries:** long when **Zscore < −entry_threshold**; short when **Zscore > +entry_threshold** if `allow_short`. **Exits:** when **Zscore** crosses **0** (long at **≥ 0**, short at **≤ 0** in code).

### Signal Logic (summary table)

| Agent | Long (Buy Vol) | Short (Sell Vol) | Close (Exit) |
| :--- | :--- | :--- | :--- |
| **`Agent_hardThreshold`** | **Zscore(VRP) < −k** | **Zscore(VRP) > +k** if `allow_short` | **Zscore(VRP)** crosses **0** |
| **`Agent_Percentile`** | **Zscore(VRP)** at/below expanding **low** percentile of past Zscores | **Zscore(VRP)** at/above expanding **high** percentile | **Zscore(VRP)** crosses expanding **median** |
| **`Agent_Garch`** | **Z-Score(IV − GARCH_RV) < −entry_threshold** | **Zscore(IV − GARCH_RV) > +entry_threshold** if `allow_short` | Zscore **crosses 0** |
| **`Agent_LongTerm`** | **Z-Score(IV − meanRV) < −entry_threshold** | **Zscore(IV − meanRV) > +entry_threshold** if `allow_short` | Zscore **crosses 0** |

### Agent-specific parameters

| Parameter | Agents | Meaning | Default |
| :--- | :--- | :--- | :--- |
| `entry_threshold` | Garch, LongTerm | Cutoff on **Zscore(IV − benchmark)** above (rolling mean / rolling std of the difference) | Garch `1.0`; LongTerm `0.5` |
| `long_term_window` | LongTerm | Number of **past** **RV** days in **meanRV** inside **IV − meanRV** | `126` (min `5`) |

### Agent_Garch Details

1. **Model:** **GJR-GARCH(1,1)** with **Student-t** residuals (or EWMA λ = 0.94 if **`arch`** is not installed).
2. **Fit / update:** fit on **20** trailing log-return days, re-fit every **5** days; variance updated daily with asymmetry.
3. **Forecast:** forward variance path length = **business days from `Date` to `Expiry`** (fallback **30** days if dates missing); **GARCH_RV** = annualized vol from the **mean** variance along that path.
4. **Signal:** **Zscore(IV − GARCH_RV)** using **rolling mean** and **rolling std** of **(IV − GARCH_RV)** over the last **20** values, after **20** warmup differences. **`entry_threshold`** is in those Zscore units. Exit when that Zscore **crosses 0**.

### Agent_LongTerm Details

1. **Inside the difference:** **meanRV** = mean of **`long_term_window`** prior daily **RV** values; difference **IV − meanRV** (needs finite **RV** and **Straddle_imp_vol**).
2. **Signal:** **Zscore(IV − meanRV)** with the same **rolling mean** and **rolling std** (last **20** points, **20** warmup) as in the section above.
3. **Trade:** long / short vs **±entry_threshold** on that Zscore; exit when the Zscore **crosses 0**.

`Agent_Heston_Class.py` only re-exports **`Agent_Heston = Agent_LongTerm`**. Prefer **`from Agent_LongTerm_Class import Agent_LongTerm`**.

### Deterministic vs simulation

Garch and LongTerm are **closed-form / sample statistics** (no path simulation). Garch uses recursive variance; LongTerm uses a **rolling mean of past `RV`** only to build **meanRV** inside **IV − meanRV**.

---

## Data Pipeline (`Build_data.ipynb`)

Builds a single CSV per symbol at `DataSet/{symbol}.csv`:

1. Downloads daily equity data from **yfinance** (close, dividends, returns, 20-day RV)
2. Fetches the 1-month Treasury yield from **Polygon** as the risk-free rate
3. For each month, selects the ATM straddle (next-month expiry) via **Polygon** options chain
4. Retrieves daily option bars, computes Black-Scholes implied vol (Polygon IV when available, `brentq` inversion otherwise)
5. Computes straddle implied vol (bisection so BS\_straddle(sigma) = market\_price), VRP = IV - RV
6. Computes per-leg and straddle Greeks (Delta, Gamma, Vega, Theta, Rho, Vanna, Volga)

### Key Column Definitions

| Column | Description |
| :--- | :--- |
| `RV` | Realized volatility (annualized, decimal); used in **`meanRV`** for LongTerm |
| `VRP` | `Straddle_imp_vol - RV` (annualized, decimal) |
| `Straddle_Theta` | per-trading-year theta (T = trading\_days / 252) |
| `Straddle_Rho` | per +1 percentage point change in r |
| `Force_Close` | `True` on last trading day of each month |

---

## Backtest (`Back_test.ipynb`)

1. Load a dataset CSV and compute rolling VRP statistics (`VRP_20d_mean`, `VRP_20d_std`, `VRP_40d_std`)
2. Instantiate agents with desired parameters
3. Loop through each row calling `agent.trade(row)`
4. Generate performance stats and visualizations

### Backtest Snapshot

Illustrative run from `Back_test.ipynb` (metrics change with data and parameters):

- Dataset: `DataSet/QQQ.csv`
- Shared: `delta_hedge=True`, `rehedge_threshold=0.05`
- Agents configured:
  - `Agent_hardThreshold`: `k=1`
  - `Agent_Perc`: `entry_percentile=0.2`
  - `Agent_Garch`: `entry_threshold=0.5`
  - `Agent_LongTerm`: `entry_threshold=0.5`, default `long_term_window=126`

| Agent | Win Rate | Sharpe Ratio | Sortino Ratio | Annual Return | Annual Volatility | Max Drawdown | Calmar Ratio | Kelly's Criteria |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Agent_hardThreshold` | 82.50% | 0.9265 | 0.7079 | 22.14% | 21.59% | -20.95% | 1.0570 | 4.2915 |
| `Agent_Perc` | 77.42% | 1.3828 | 1.3134 | 36.58% | 22.54% | -22.18% | 1.6490 | 6.1341 |
| `Agent_Garch` | 74.55% | 1.5126 | 1.9497 | 61.95% | 31.87% | -18.27% | 3.3904 | 4.7453 |
| `Agent_LongTerm` | 83.93% | 2.0199 | 1.7615 | 54.33% | 21.48% | -18.26% | 2.9755 | 9.4035 |

## Greeks Attribution Calculation

The project uses **path-wise daily attribution** with **timestep Greeks** (Greeks at time `t` explain PnL from `t -> t+1`).

Let:
- `q_t`: option lots (`+1` long straddle lot, `-1` short straddle lot)
- `L`: lot size (`L = 100` options per lot)
- `h_t`: hedge shares held before move
- `dS = S_{t+1} - S_t`
- `dS_adj = dS + dividends`
- `dσ = σ_{t+1} - σ_t` where `σ` is `Straddle_imp_vol`
- `dr = r_{t+1} - r_t`
- `dt = busday_count(t, t+1) / 252`

Daily PnL decomposition:

- **Delta attribution**
  - `Delta_t = (q_t * L) * Delta_eff_t * dS`
  - where `Delta_eff_t = Straddle_Delta_t` normally, and on re-hedge-trigger days uses midpoint delta `0.5*(Delta_t_prev + Delta_t_curr)` to reduce residual leakage
- **Gamma attribution (hedge merged in)**
  - `Gamma_t = (q_t * L) * 0.5 * Straddle_Gamma_t * dS^2 + h_t * dS_adj`
  - Note: hedge PnL is intentionally merged into `gamma_attribute`.
- **Vega attribution**
  - `Vega_t = (q_t * L) * Straddle_Vega_t * dσ`
- **Vanna attribution**
  - `Vanna_t = (q_t * L) * Straddle_Vanna_t * dS * dσ`
- **Volga attribution**
  - `Volga_t = (q_t * L) * 0.5 * Straddle_Volga_t * dσ^2`
- **Theta attribution**
  - `Theta_t = (q_t * L) * Straddle_Theta_t * dt`
- **Rho attribution**
  - `Rho_t = (q_t * L) * Straddle_Rho_t * dr * 100`
  - (`Straddle_Rho` is stored per +1 percentage-point rate change.)
- **Residual**
  - `Residual_t = TotalPnL_t - (Delta_t + Gamma_t + Vega_t + Vanna_t + Volga_t + Theta_t + Rho_t)`

where:
- `TotalPnL_t = (q_t * L) * (StraddlePrice_{t+1} - StraddlePrice_t) + h_t * dS_adj`

This setup preserves the daily identity:
- `Delta + Gamma + Vega + Vanna + Volga + Theta + Rho + Residual = Total PnL`

---

## Visual Diagnostics

The backtest suite generates comprehensive visualizations to analyze returns, risk exposures, and PnL attribution.

### 1. Cumulative Return & Signals
![Cumulative Return](demo/Return.png)
- **Panel 1 (top):** Cumulative geometric return (value ratio) of the underlying versus each agent
- **Panel 2:** Daily log-return of each agent
- **Panel 3:** **RV** (`RV` column) vs **IV** (`Straddle_imp_vol`) over time
- **Panel 4 (bottom):** **VRP** level with rolling 20-day mean and ±1 std bands (computed in-notebook or from `VRP_20d_*` columns when present)

### 2. PnL Attribution by Greek (Pie Breakdown)
![Greeks Attribution Pie](demo/Greeks_Attribution_Pie.png)
Aggregates total dollar PnL and breaks it down by Greek component. For a volatility timing strategy, Vega and Gamma/Theta should dominate; Delta should be minimal if properly hedged.

### 3. Daily Greeks Attribution
![Greeks Attribution Line](demo/Greeks_Attribution_line.png)
Daily dollar attribution across all components (Delta, Gamma, Vega, Theta, Vanna, Volga, Rho, Residual), showing which risk factors drive performance on each day.
