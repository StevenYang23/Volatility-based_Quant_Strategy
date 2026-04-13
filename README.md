# Volatility-based Quant Strategy

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
| `rehedge_threshold` | Net delta magnitude that triggers a re-hedge | `0.5` |

### Signal Logic

| Agent | Long (Buy Vol) | Short (Sell Vol) | Close (Exit) |
| :--- | :--- | :--- | :--- |
| **`Agent_hardThreshold`**<br>`k=1` | `VRP < Mean - k * Std` | `VRP > Mean + k * Std` | `VRP` crosses `Mean` |
| **`Agent_Percentile`**<br>`entry_percentile=0.20` | `Z(VRP)` below expanding low percentile | `Z(VRP)` above expanding high percentile | `Z(VRP)` crosses expanding median |
| **`Agent_2threshold`**<br>`k_high`, `k_low` | `VRP < Mean - k_active * Std`<br>(`k_active = k_high` if 40d\_Std > 20d\_Std, else `k_low`) | `VRP > Mean + k_active * Std`<br>(same regime switch) | `VRP` crosses `Mean` |
| **`Agent_Garch`**<br>`z_entry=1.0` | z-score of (IV - GARCH\_RV) < `-z_entry` | z-score of (IV - GARCH\_RV) > `+z_entry` | z-score crosses zero |

### Agent_Garch Details

The GARCH agent uses a GARCH(1,1) model to forecast realized volatility, then trades the IV-RV spread:

1. **Fit**: GARCH(1,1) is fit on the trailing 20 trading days of log returns, re-fit every 5 days
2. **Update**: between re-fits, the conditional variance is rolled forward via the GARCH recursion: `h_{t+1} = omega + alpha * eps_t^2 + beta * h_t`
3. **Signal**: `spread = IV - sqrt(h) * sqrt(252)`. The z-score of the spread (expanding-window mean & std, min 20 observations) determines entry/exit
4. **Fallback**: when `arch` is not installed, an EWMA variance (RiskMetrics, lambda=0.94) is used instead

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

| Agent | Win Rate | Sharpe Ratio | Sortino Ratio | Annual Return | Annual Volatility | Max Drawdown | Calmar Ratio | Kelly's Criteria |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Agent_hardThreshold` | 82.50% | 1.1201 | 1.0015 | 25.33% | 20.16% | -21.25% | 1.1922 | 5.5565 |
| `Agent_Perc` | 80.65% | 1.3727 | 1.2589 | 34.43% | 21.56% | -22.38% | 1.5389 | 6.3677 |
| `Agent_2threshold` | 78.95% | 1.1875 | 1.0669 | 24.98% | 18.78% | -21.25% | 1.1756 | 6.3233 |
| `Agent_Garch` | 77.78% | 1.5885 | 1.9461 | 59.39% | 29.35% | -21.88% | 2.7149 | 5.4128 |

## Greeks Attribution Calculation

The project uses **path-wise daily attribution** with **timestep Greeks** (Greeks at time `t` explain PnL from `t -> t+1`).

Let:
- `q_t`: option position (`+1` long straddle, `-1` short straddle)
- `h_t`: hedge shares held before move
- `dS = S_{t+1} - S_t`
- `dS_adj = dS + dividends`
- `dσ = σ_{t+1} - σ_t` where `σ` is `Straddle_imp_vol`
- `dr = r_{t+1} - r_t`
- `dt = busday_count(t, t+1) / 252`

Daily PnL decomposition:

- **Delta attribution**
  - `Delta_t = q_t * Straddle_Delta_t * dS`
- **Gamma attribution (hedge merged in)**
  - `Gamma_t = q_t * 0.5 * Straddle_Gamma_t * dS^2 + h_t * dS_adj`
  - Note: hedge PnL is intentionally merged into `gamma_attribute`.
- **Vega attribution**
  - `Vega_t = q_t * Straddle_Vega_t * dσ`
- **Vanna attribution**
  - `Vanna_t = q_t * Straddle_Vanna_t * dS * dσ`
- **Volga attribution**
  - `Volga_t = q_t * 0.5 * Straddle_Volga_t * dσ^2`
- **Theta attribution**
  - `Theta_t = q_t * Straddle_Theta_t * dt`
- **Rho attribution**
  - `Rho_t = q_t * Straddle_Rho_t * dr * 100`
  - (`Straddle_Rho` is stored per +1 percentage-point rate change.)
- **Residual**
  - `Residual_t = TotalPnL_t - (Delta_t + Gamma_t + Vega_t + Vanna_t + Volga_t + Theta_t + Rho_t)`

where:
- `TotalPnL_t = q_t * (StraddlePrice_{t+1} - StraddlePrice_t) + h_t * dS_adj`

This setup preserves the daily identity:
- `Delta + Gamma + Vega + Vanna + Volga + Theta + Rho + Residual = Total PnL`

---

## Visual Diagnostics

The backtest suite generates comprehensive visualizations to analyze returns, risk exposures, and PnL attribution.

### 1. Cumulative Return & Signals
![Cumulative Return](demo/Return.png)
- **Top Panel:** Cumulative geometric return (value ratio) of the underlying asset versus agents
- **Middle Panel:** Daily log-return rate of each agent
- **Bottom Panel:** VRP signal level with rolling 20-day mean and ±1 std bands

### 2. PnL Attribution by Greek (Pie Breakdown)
![Greeks Attribution Pie](demo/Greeks_Attribution_Pie.png)
Aggregates total dollar PnL and breaks it down by Greek component. For a volatility timing strategy, Vega and Gamma/Theta should dominate; Delta should be minimal if properly hedged.

### 3. Daily Greeks Attribution
![Greeks Attribution Line](demo/Greeks_Attribution_line.png)
Daily dollar attribution across all components (Delta, Gamma, Vega, Theta, Vanna, Volga, Rho, Residual), showing which risk factors drive performance on each day.
