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

### Rolling z-score and `entry_threshold` (Garch, LongTerm, Agent_Myself entry)

For **`Agent_Garch`**, **`Agent_LongTerm`**, and the **entry signal** of **`Agent_Myself`**, the name **`entry_threshold` means a threshold in units of a rolling z-score**, not a raw VRP level, not a raw IV spread in volatility points, and not dollars.

Each day:

1. **Level spread** `s_t` is defined (model-specific): e.g. `IV − GARCH_RV` or `IV − long_run_mean_IV`.
2. A **rolling window** of recent `s` values (defaults: at least **20** observations of `s` before any trade; z uses the last **20** spreads) yields a mean `μ_t` and sample standard deviation `σ_t` (with `σ_t > 0`).
3. **Standardized signal**  
   `z_t = (s_t − μ_t) / σ_t`  
   So `z_t` is the usual **cross-sectional-style z-score** of *today’s* spread vs the *recent distribution of spreads*.

**Entries (Garch & LongTerm):**

- **Long vol (buy straddle):** `z_t < −entry_threshold`
- **Short vol (sell straddle):** `z_t > +entry_threshold` (if `allow_short`)

**Exits (Garch & LongTerm):** position is closed when **`z_t` crosses 0** back toward neutral (same side rule as in code: long unwinds when `z ≥ 0`, short when `z ≤ 0`).

**`Agent_Myself`:** uses the **same** `s_t`, `μ_t`, `σ_t`, and `z_t` as `Agent_LongTerm` for **opening** a long straddle only when `z_t < −entry_threshold`. It does **not** use `z` for exit; legs are closed with **per-leg stop-loss / take-profit** (and breakeven on the last leg). **`stop_loss_pct` / `stop_profit_pct` are fractions of premium (e.g. `0.30` = 30%), not z-units.**

### Signal Logic

| Agent | Long (Buy Vol) | Short (Sell Vol) | Close (Exit) |
| :--- | :--- | :--- | :--- |
| **`Agent_hardThreshold`**<br>`k=1` | `VRP < Mean - k * Std` | `VRP > Mean + k * Std` | `VRP` crosses `Mean` |
| **`Agent_Percentile`**<br>`entry_percentile=0.20` | `Z(VRP)` below expanding low percentile | `Z(VRP)` above expanding high percentile | `Z(VRP)` crosses expanding median |
| **`Agent_2threshold`**<br>`k_high`, `k_low` | `VRP < Mean - k_active * Std`<br>(`k_active = k_high` if 40d\_Std > 20d\_Std, else `k_low`) | `VRP > Mean + k_active * Std`<br>(same regime switch) | `VRP` crosses `Mean` |
| **`Agent_Garch`**<br>`entry_threshold` = z-band | `z_t < −entry_threshold` on spread `IV − GARCH_RV` | `z_t > +entry_threshold` (if `allow_short`) | `z_t` crosses `0` |
| **`Agent_LongTerm`**<br>`entry_threshold`, `long_term_window` | `z_t < −entry_threshold` on spread `IV −` long-run IV mean | `z_t > +entry_threshold` (if `allow_short`) | `z_t` crosses `0` |
| **`Agent_Myself`**<br>same z as LongTerm for **entry** | `z_t < −entry_threshold` → open long straddle | — (no short) | Per-leg **stop-loss / take-profit**; last leg at breakeven; `Force_Close` |

Agent-specific parameters (others use the **Shared Parameters** table):

| Parameter | Agents | Description | Default |
| :--- | :--- | :--- | :--- |
| `entry_threshold` | `Agent_Garch`, `Agent_LongTerm`, `Agent_Myself` | **Minimum \|z\|** on the rolling z-score (see above) to enter long (`z < −threshold`) or short (`z > +threshold`) where applicable | Garch `1.0`, LongTerm / Myself `0.5` |
| `long_term_window` | `Agent_LongTerm`, `Agent_Myself` | Trading days of **past** straddle IV used for the long-run mean (today excluded) | `126` (min `5`) |
| `stop_loss_pct`, `stop_profit_pct` | `Agent_Myself` only | Leg exit vs **that leg’s entry price** as a **fraction** (not z) | `0.30` each |

### Agent_Garch Details

The GARCH agent uses a **GJR-GARCH(1,1)** model with **Student-t** innovations, then trades relative value in implied vs forecast realized vol:

1. **Fit**: GJR-GARCH(1,1) is fit on the trailing **20** trading days of log returns, re-fit every **5** days
2. **Update**: between re-fits, the conditional variance is rolled forward with asymmetry (negative shocks receive extra gamma loading)
3. **Forecast horizon**: the forward variance path length matches **trading days from valuation `Date` to option `Expiry`** (`numpy.busday_count`). If dates are missing, a **30-day** fallback is used
4. **Annualized RV forecast**: mean variance along that path, converted to annualized vol (same scale as `Straddle_imp_vol`)
5. **Signal**: `spread = IV − GARCH_RV`. Build **`z_t`** from the last **20** spreads (after at least **20** stored spreads). **Enter** long when `z_t < −entry_threshold`, short when `z_t > +entry_threshold` (if `allow_short`). **`entry_threshold` counts in z-units.** Exit when **`z_t` crosses 0** toward flat
6. **Fallback**: when `arch` is not installed, an EWMA variance (RiskMetrics, λ = 0.94) is used instead

### Agent_LongTerm Details

`Agent_LongTerm` replaces the former Heston-calibration agent with a **simple long-horizon implied-vol benchmark** (no stochastic-vol calibration, no QuantLib):

1. **Long-run mean**: on each day, the benchmark is the **mean of the prior `long_term_window` daily values of `Straddle_imp_vol`** (today’s IV is **not** included)
2. **Signal**: `spread = IV_today − long_run_mean`
3. **Entry / exit**: same **`z_t`** construction as **Rolling z-score and `entry_threshold`**—min **20** spreads, **20**-observation z-window, **`entry_threshold` in z-units** for entries, exit when **`z_t` crosses 0**

For backward compatibility, `Agent_Heston_Class.py` re-exports **`Agent_Heston = Agent_LongTerm`** (same class). Prefer `from Agent_LongTerm_Class import Agent_LongTerm`.

### Agent_Myself Details

`Agent_Myself` combines the **LongTerm IV z-score** for **entries** with **leg-level** risk management:

1. **Entry:** identical long-run IV mean and **`z_t`** as `Agent_LongTerm`; opens a **long straddle** when `z_t < −entry_threshold` (no short side).
2. **Exit:** call and put are managed **separately** with **`stop_loss_pct`** and **`stop_profit_pct`** vs each leg’s entry premium (**fractions**, not z-scores). If one leg is already closed, the survivor uses **breakeven** (exit when price returns to that leg’s entry).
3. **`Force_Close`:** flattens options and hedge like other agents.

### Deterministic vs Simulation

- `Agent_Garch`, `Agent_LongTerm`, and the signal path of `Agent_Myself` are **deterministic** (recursive vol or rolling IV mean; z-scores from sample mean and std of recent spreads).
- They do **not** run Monte Carlo multi-path simulation in this repository.
- `Agent_Garch` uses forward variance iteration; `Agent_LongTerm` and `Agent_Myself` (entry) use a rolling historical IV mean only.

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
