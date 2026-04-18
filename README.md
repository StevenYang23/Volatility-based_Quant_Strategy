# Volatility-based Quant Strategy

*Documentation last aligned with the repository: **April 2026**.*

This repository implements a low-frequency volatility timing quantitative strategy. The strategy aims to capture the Volatility Risk Premium (VRP) and mean-reverting properties of implied volatility by actively trading at-the-money (ATM) ~30-day straddles.

## Overall Strategy Idea

The core concept relies on the spread between Implied Volatility (IV) and Realized Volatility (RV). When options are "overpriced" (IV is significantly higher than RV or its historical mean), the strategy sells volatility (short straddle). When options are "underpriced" (IV is unusually low), the strategy buys volatility (long straddle). 

During the holding period, the strategy dynamically hedges its directional exposure (Delta hedging) based on a defined threshold to ensure the returns are primarily driven by Vega, Gamma, and Theta, rather than the underlying asset's price movement. The framework calculates daily mark-to-market PnL, geometric log-returns, and uses path-wise Greeks attribution for daily PnL decomposition.

### Annualization & Day-Count Convention

All calculations use **252 trading days per year** consistently:

- **Time to maturity (T)** in `Build_data.ipynb`: `np.busday_count(date, expiry) / 252`
- **Realized volatility (RV)**: `rolling_std(returns) * sqrt(252)`
- **Black-Scholes Greeks**: computed with T in trading-year fractions, so Theta is per-trading-year
- **Theta PnL attribution** in agents: `theta * dt` where `dt = busday_count / 252`
- **Performance metrics** in `Visual.py`: annualized return, vol, Sharpe, Sortino all use 252

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

### Notation: IV, RV, and the two-step Garch / LongTerm signal

Use these symbols consistently:

| Symbol | In this codebase |
| :--- | :--- |
| **IV** | Straddle implied vol, column **`Straddle_imp_vol`** (annualized, decimal). |
| **RV** | Realized vol from the panel, column **`RV`** (annualized, decimal). |
| **GARCH_RV** | `Agent_Garch`’s **model-based** forecast of realized vol for that day (GJR-GARCH path, annualized; see agent details). |
| **meanRV** | For `Agent_LongTerm` / `Agent_Myself` **entry only**: on day `t`, **`meanRV_t`** = mean of the **prior `long_term_window`** daily **`RV`** values in history. Today’s **`RV`** is stored **after** forming **`s_t`**, so it is **not** part of **`meanRV_t`** on day `t`. |

**Step 1 — level spread `s_t` (different per agent):**

- **`Agent_Garch`:**  
  **`s_t = IV_t − GARCH_RV_t`**  
  (implied vs **model** forecast RV, same units as IV.)

- **`Agent_LongTerm`** and **`Agent_Myself` (entry):**  
  **`s_t = IV_t − meanRV_t`**  
  (implied vs **rolling sample mean of past realized vol**, not vs a fixed scalar “the” long-term RV for all time.)

**Step 2 — trading score `z_t` (same structure for both; this is *not* a one-shot z of a single number):**

Each day you have a **time series** of past spreads `s`. The code stores every `s_t` and then **re-standardizes** using only the **recent** history of **`s`** itself:

- After at least **`_MIN_SPREAD_OBS` = 20** stored spreads exist, let **`μ_t`** and **`σ_t`** be the **sample mean** and **sample standard deviation** (ddof = 1) of the **last `_Z_ROLLING_WINDOW` = 20** values of **`s`** (including today’s **`s_t`** in that window).
- **`z_t = (s_t − μ_t) / σ_t`**, with **`σ_t > 0`**.

So in words: **`z_t` is a rolling z-score of the *spread process*** — “how unusual is **today’s** level spread **relative to how this same spread has behaved over the last ~20 days**?” It is **not** “z-score **IV − GARCH_RV** against the unconditional lifetime distribution of that difference,” and **not** “z-score **IV − meanRV** against a single long-run RV distribution.” There are **two** demeanings: first **IV** vs a **benchmark** (GARCH RV or rolling mean RV), then **that spread** vs its **short rolling** mean and volatility.

**`entry_threshold`:** only applies to this **`z_t`**. It is a **dimensionless** cutoff (same units as **`z_t`**), **not** volatility points, **not** VRP, **not** dollars.

**Entries (Garch and LongTerm, when `allow_short` is True for shorts):**

- Long straddle: **`z_t < −entry_threshold`**
- Short straddle: **`z_t > +entry_threshold`**

**Exits (Garch and LongTerm):** unwind when **`z_t` crosses 0** toward flat (long exits when **`z_t ≥ 0`**, short when **`z_t ≤ 0`** in the implementation).

**`Agent_Myself`:** builds the **same `s_t` and `z_t` as LongTerm** for **opening** a **long** straddle only when **`z_t < −entry_threshold`**. Exits are **not** z-based: **per-leg** **`stop_loss_pct` / `stop_profit_pct`** as **fractions of that leg’s entry premium**; if one leg is already closed, the remaining leg exits at **breakeven** (mark back to entry).

### Signal Logic (summary table)

| Agent | Long (Buy Vol) | Short (Sell Vol) | Close (Exit) |
| :--- | :--- | :--- | :--- |
| **`Agent_hardThreshold`** … | `VRP < Mean − k·Std` | `VRP > Mean + k·Std` | `VRP` crosses `Mean` |
| **`Agent_Percentile`** … | expanding low on `Z(VRP)` | expanding high on `Z(VRP)` | `Z(VRP)` crosses expanding median |
| **`Agent_2threshold`** … | `VRP` vs `Mean ± k_active·Std` (regime `k`) | same | `VRP` crosses `Mean` |
| **`Agent_Garch`** | **`z_t < −entry_threshold`** with **`s_t = IV − GARCH_RV`** | **`z_t > +entry_threshold`** if `allow_short` | **`z_t` crosses 0** |
| **`Agent_LongTerm`** | **`z_t < −entry_threshold`** with **`s_t = IV − meanRV`** | **`z_t > +entry_threshold`** if `allow_short` | **`z_t` crosses 0** |
| **`Agent_Myself`** | same **`s_t` / `z_t` as LongTerm** → long when **`z_t < −entry_threshold`** | — (no short) | leg **stop %** / **breakeven**; **`Force_Close`** |

### Agent-specific parameters

| Parameter | Agents | Meaning | Default |
| :--- | :--- | :--- | :--- |
| `entry_threshold` | Garch, LongTerm, Myself (entry) | Cutoff on **`z_t`** (rolling z of **`s`**) for entries | Garch `1.0`; LongTerm / Myself `0.5` |
| `long_term_window` | LongTerm, Myself | Length of **past `RV`** window for **meanRV** in **`s_t = IV − meanRV`** | `126` (min `5`) |
| `stop_loss_pct`, `stop_profit_pct` | Myself only | Leg exit vs **that leg’s entry price**, as **fraction** (e.g. `0.30` = 30%) | `0.30` each |

### Agent_Garch Details

1. **Model:** **GJR-GARCH(1,1)** with **Student-t** residuals (or EWMA λ = 0.94 if **`arch`** is not installed).
2. **Fit / update:** fit on **20** trailing log-return days, re-fit every **5** days; variance updated daily with asymmetry.
3. **Forecast:** forward variance path length = **business days from `Date` to `Expiry`** (fallback **30** days if dates missing); **GARCH_RV** = annualized vol from the **mean** variance along that path.
4. **Spread:** **`s_t = IV_t − GARCH_RV_t`**. Then **`z_t`** from the **last 20** values of **`s`** (after **20** spreads accumulated). **`entry_threshold`** is in **`z_t`** units. Exit on **`z_t` crossing 0**.

### Agent_LongTerm Details

1. **Benchmark:** **`meanRV_t`** = mean of **`long_term_window`** prior daily **`RV`** values; **`s_t = IV_t − meanRV_t`** (requires finite **`RV`** and **`Straddle_imp_vol`**).
2. **Score:** same **rolling `z_t`** on the **`s`** series as in **Notation** (20 spreads minimum, window 20 for **`μ_t`**, **`σ_t`**).
3. **Trade:** long / short from **`z_t`** vs **`±entry_threshold`**; exit when **`z_t`** crosses **0**.

`Agent_Heston_Class.py` only re-exports **`Agent_Heston = Agent_LongTerm`**. Prefer **`from Agent_LongTerm_Class import Agent_LongTerm`**.

### Agent_Myself Details

1. **Entry:** identical **`s_t`** and **`z_t`** to LongTerm; **long straddle** only when **`z_t < −entry_threshold`**.
2. **Exit:** per-leg stops and breakeven rule as above.
3. **`Force_Close`:** like other agents.

### Deterministic vs simulation

Garch, LongTerm, and the **signal** side of Myself are **closed-form / sample statistics** (no path simulation). Garch uses recursive variance; LongTerm and Myself entry use **rolling mean of past `RV`** only for **`meanRV`**.

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
| `RV` | Realized volatility (annualized, decimal); used in **`meanRV`** for LongTerm / Myself |
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
  - `Agent_2threshold`: `k_high=1.4`, `k_low=0.6`
  - `Agent_Garch`: `entry_threshold=0.5`
  - `Agent_LongTerm`: `entry_threshold=0.5`, default `long_term_window=126`

| Agent | Win Rate | Sharpe Ratio | Sortino Ratio | Annual Return | Annual Volatility | Max Drawdown | Calmar Ratio | Kelly's Criteria |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Agent_hardThreshold` | 82.50% | 0.9265 | 0.7079 | 22.14% | 21.59% | -20.95% | 1.0570 | 4.2915 |
| `Agent_Perc` | 77.42% | 1.3828 | 1.3134 | 36.58% | 22.54% | -22.18% | 1.6490 | 6.1341 |
| `Agent_2threshold` | 78.95% | 1.1970 | 1.0845 | 25.12% | 18.73% | -20.95% | 1.1993 | 6.3925 |
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
