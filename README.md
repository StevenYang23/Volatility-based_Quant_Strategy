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

### What “Zscore” means

Let **something** be **IV − GARCH_RV** (Garch), **IV − long-term RV** (LongTerm), or **VRP**, with **IV** = `Straddle_imp_vol`, **GARCH_RV** from the model, and **long-term RV** = mean of the prior `long_term_window` daily **RV** values (today’s **RV** is not in **long-term RV** for that day).

**Zscore(something)** = (**something** − **20-day rolling mean of something**) / (**20-day rolling std of something**).

**Entries:** long when **Zscore < −entry_threshold**; short when **Zscore > +entry_threshold** if `allow_short`. **Exits:** when **Zscore** crosses **0** (long at **≥ 0**, short at **≤ 0** in code).

As an informal analogy (the series are not literally Gaussian): you can read the cutoff like a **hypothesis test** that treats **today’s** value versus the **20-day** rolling window as if **something** were **normal** in that window, with **entry_threshold** (and **k** on **VRP** for **`Agent_hardThreshold`**) playing a role similar to choosing a **confidence level**—how far into the tail “today” must sit before you enter.

### Signal Logic (summary table)

| Agent | Long (Buy Vol) | Short (Sell Vol) | Close (Exit) |
| :--- | :--- | :--- | :--- |
| **`Agent_hardThreshold`** | **Zscore(VRP) < −k** | **Zscore(VRP) > +k** if `allow_short` | **Zscore(VRP)** crosses **0** |
| **`Agent_Percentile`** | rolling **Zscore(VRP)** at/below expanding **low** percentile of **past** rolling Zscores | rolling **Zscore(VRP)** at/above expanding **high** percentile | rolling **Zscore(VRP)** crosses expanding **median** of **past** rolling Zscores |
| **`Agent_Garch`** | **Z-Score(IV − GARCH_RV) < −entry_threshold** | **Zscore(IV − GARCH_RV) > +entry_threshold** if `allow_short` | Zscore **crosses 0** |
| **`Agent_LongTerm`** | **Z-Score(IV − long-term RV) < −entry_threshold** | **Zscore(IV − long-term RV) > +entry_threshold** if `allow_short` | Zscore **crosses 0** |

### Agent_Garch Details

1. **Model:** **GJR-GARCH(1,1)** with **Student-t** residuals.
2. **Fit / update:** fit on **20** trailing log-return days, re-fit every **5** days; variance updated daily with asymmetry.
3. **Forecast:** forward variance path length = **business days from `Date` to `Expiry`** (fallback **30** days if dates missing); **GARCH_RV** = annualized vol from the **mean** variance along that path.
4. **Signal:** **Zscore(IV − GARCH_RV)** using **rolling mean** and **rolling std** of **(IV − GARCH_RV)** over the last **20** values, after **20** warmup differences. **`entry_threshold`** is in those Zscore units. Exit when that Zscore **crosses 0**.

### Agent_LongTerm Details

1. **Inside the difference:** **long-term RV** = mean of **`long_term_window`** prior daily **RV** values; difference **IV − long-term RV** (needs finite **RV** and **Straddle_imp_vol**).
2. **Signal:** **Zscore(IV − long-term RV)** with the same **rolling mean** and **rolling std** (last **20** points, **20** warmup) as in the section above.
3. **Trade:** long / short vs **±entry_threshold** on that Zscore; exit when the Zscore **crosses 0**.


### Agent_Percentile Details

1. **Zscore(VRP):** same **20-day rolling** mean and std as **`Agent_hardThreshold`** (`VRP_20d_mean`, `VRP_20d_std` from the notebook).
2. **Trade:** after **20** stored daily rolling z-scores, enter long / short when today’s **Zscore(VRP)** is at or below the **entry_percentile** low tail / at or above the **(1 − entry_percentile)** high tail of the **history of past** rolling z-scores; exit when z crosses the **median** of that history (same expanding-percentile logic as before, but on **rolling** z).

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

Each row is **one equity session** in the monthly **ATM straddle** panel (`Build_data.ipynb` writes columns in this order). Short labels below; full construction is in that notebook.

| Column | Description |
| :--- | :--- |
| `Date` | Session date. |
| `Stock_Close` | Underlying close **S** (decimal price). |
| `Stock_Dividends` | Cash dividend that day (**0** if none). |
| `r` | Annual risk-free rate, **decimal** (e.g. 1m Treasury aligned to `Date`). |
| `RV` | Trailing **realized** vol, annualized decimal (from return history; NaN until enough days). Feeds **long-term RV** in LongTerm. |
| `K` | Strike **K** of the month’s ATM straddle. |
| `Expiry` | Listed expiry of the options (used e.g. for **Garch** horizon to expiry). |
| `Call_Close` / `Put_Close` | Call / put **close** (straddle premium = sum). |
| `Call_Sym` / `Put_Sym` | Polygon OCC-style identifiers for the legs. |
| `Call_imp_vol` / `Put_imp_vol` | Leg implied vol **σ**, decimal (chain IV or BS inversion). |
| `Straddle_imp_vol` | Single **σ** such that BS straddle price matches **Call_Close + Put_Close** (bisection). |
| `VRP` | **Straddle_imp_vol − RV** (vol risk premium, decimal). |
| `Straddle_Delta` | **Call_Delta + Put_Delta** (long one call + one put at each leg’s σ). |
| `Straddle_Gamma` | **Call_Gamma + Put_Gamma**. |
| `Straddle_Vega` | **Call_Vega + Put_Vega**. |
| `Straddle_Theta` | **Call_Theta + Put_Theta** (time decay; convention matches `Build_data.ipynb`). |
| `Straddle_Rho` | **Call_Rho + Put_Rho** (sensitivity to **r**; **Rho** columns are per **+1 percentage point** in **r**). |
| `Straddle_Vanna` | **Call_Vanna + Put_Vanna**. |
| `Straddle_Volga` | **Call_Volga + Put_Volga**. |
| `Call_Delta`, `Call_Gamma`, `Call_Vega`, `Call_Theta`, `Call_Rho`, `Call_Vanna`, `Call_Volga` | BS Greeks for the **call** at **Call_imp_vol** / **K** / **Expiry** / **r** / **S**. |
| `Put_Delta`, `Put_Gamma`, `Put_Vega`, `Put_Theta`, `Put_Rho`, `Put_Vanna`, `Put_Volga` | BS Greeks for the **put** at **Put_imp_vol** / same **K**, **Expiry**, **r**, **S**. |
| `Force_Close` | **`True`** on the **last** session kept for that month (agents flatten); **`False`** otherwise. |

---

## Backtest (`Back_test.ipynb`)

1. Load a dataset CSV and pre-compute rolling VRP statistics: **`VRP_20d_mean`** / **`VRP_20d_std`** = **20**-day rolling mean / std of **VRP** (`min_periods=20`); optional **`VRP_40d_std`**
2. Instantiate agents with desired parameters
3. Loop through each row calling `agent.trade(row)`
4. Generate performance stats and visualizations

### Backtest Snapshot

Illustrative run from `Back_test.ipynb` (metrics change with data and parameters):

- Dataset: `DataSet/SPY.csv`
- Shared: `delta_hedge=True`, `rehedge_threshold=0.05`
- Agents configured:
  - `Agent_hardThreshold`: `k=1`
  - `Agent_Perc`: `entry_percentile=0.2`
  - `Agent_Garch`: `entry_threshold=0.5`
  - `Agent_LongTerm`: `entry_threshold=0.5`, default `long_term_window=126`

**Strategy statistics** (same run):

| Agent | Win Rate | Sharpe Ratio | Sortino Ratio | Annual Return | Annual Volatility | Max Drawdown | Calmar Ratio | Kelly's Criteria |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Agent_hardThreshold` | 75.00% | 1.3248 | 1.1919 | 25.96% | 17.42% | -11.12% | 2.3339 | 7.6042 |
| `Agent_Perc` | 74.36% | 1.2669 | 1.1843 | 24.85% | 17.52% | -11.12% | 2.2345 | 7.2312 |
| `Agent_Garch` | 68.18% | 1.0768 | 1.0200 | 23.34% | 19.48% | -14.19% | 1.6446 | 5.5266 |
| `Agent_LongTerm` | 79.63% | 1.2066 | 1.3323 | 29.50% | 21.43% | -12.32% | 2.3937 | 5.6317 |

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
  - Note: hedge PnL is merged into `gamma_attribute`.
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
