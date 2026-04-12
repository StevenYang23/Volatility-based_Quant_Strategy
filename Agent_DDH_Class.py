from __future__ import annotations

import re
from typing import Optional
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1_val = d1(S, K, T, r, sigma)
    d2 = d1_val - sigma * np.sqrt(T)
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1_val = d1(S, K, T, r, sigma)
    d2 = d1_val - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1_val)


def get_greeks_analytical(call_row, put_row):
    T = call_row["ttm"]

    if T <= 1e-8:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
        }

    S = call_row["S"]
    K = call_row["k"]
    r = call_row["r"]

    sigma_c = call_row["imp_vol"]
    sigma_p = put_row["imp_vol"]

    sqrtT = np.sqrt(T)

    d1_c = (np.log(S / K) + (r + 0.5 * sigma_c**2) * T) / (sigma_c * sqrtT)
    d2_c = d1_c - sigma_c * sqrtT

    d1_p = (np.log(S / K) + (r + 0.5 * sigma_p**2) * T) / (sigma_p * sqrtT)
    d2_p = d1_p - sigma_p * sqrtT

    n_c = norm.pdf(d1_c)
    n_p = norm.pdf(d1_p)

    delta_c = norm.cdf(d1_c)
    delta_p = norm.cdf(d1_p) - 1
    delta = delta_c + delta_p

    gamma_c = n_c / (S * sigma_c * sqrtT)
    gamma_p = n_p / (S * sigma_p * sqrtT)
    gamma = gamma_c + gamma_p

    vega_c = S * n_c * sqrtT
    vega_p = S * n_p * sqrtT
    vega = vega_c + vega_p

    theta_c = (-S * n_c * sigma_c) / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2_c)
    theta_p = (-S * n_p * sigma_p) / (2 * sqrtT) + r * K * np.exp(-r * T) * norm.cdf(-d2_p)
    theta = theta_c + theta_p

    dd1_dsigma_c = np.sqrt(T) - d1_c / sigma_c
    dd1_dsigma_p = np.sqrt(T) - d1_p / sigma_p

    vanna_c = norm.pdf(d1_c) * dd1_dsigma_c
    vanna_p = norm.pdf(d1_p) * dd1_dsigma_p

    vanna = vanna_c + vanna_p

    volga_c = vega_c * d1_c * d2_c / sigma_c
    volga_p = vega_p * d1_p * d2_p / sigma_p
    volga = volga_c + volga_p

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "vanna": vanna,
        "volga": volga,
    }


def _parse_occ_symbol(sym: str):
    part = sym.split(":", 1)[-1]
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", part)
    if not m:
        raise ValueError(f"Unrecognized OCC option symbol: {sym}")
    yymmdd = m.group(2)
    strike_raw = m.group(4)
    exp_s = "20" + yymmdd if len(yymmdd) == 6 else yymmdd
    expiry = pd.to_datetime(exp_s, format="%Y%m%d").date()
    k = int(strike_raw) / 1000.0
    return expiry, k


def _ttm_years(asof: pd.Timestamp, expiry) -> float:
    d = asof.date() if hasattr(asof, "date") else asof
    return max((expiry - d).days / 365.25, 1e-9)


class Agent_DDH:
    def __init__(
        self,
        panel_path="DataSet/AAPL.csv",
        panel_df=None,
        balance=1000.0,
        max_invest=0.8,
        max_leverage=0.5,
        vrp_threshold=1.0,
        vrp_close_threshold=1.0,
        delta_rehedge_threshold=1e9,
        min_ttm=4 / 252,
        max_ttm=60 / 252,
        delta_hedge=True,
    ):
        self.panel_path = panel_path
        self.balance = float(balance)
        if panel_df is not None:
            df = panel_df.copy()
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
        else:
            df = pd.read_csv(panel_path, parse_dates=["Date"])
            df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

        if "IV" not in df.columns:
            df["IV"] = (df["Call_imp_vol"] + df["Put_imp_vol"]) / 2
        if "VRP" not in df.columns:
            df["VRP"] = df["IV"] - df["RV"]
        if "VRP_mean" not in df.columns:
            df["VRP_mean"] = df["VRP"].rolling(20, min_periods=1).mean()
        if "VRP_std" not in df.columns:
            df["VRP_std"] = df["VRP"].rolling(20, min_periods=1).std()
        if "Force_Close" not in df.columns:
            df["Force_Close"] = False

        self.panel_df = df
        self.underlying_df = df[
            ["Date", "Stock_Close", "RV", "VRP", "VRP_mean", "VRP_std", "IV"]
        ].copy()
        self.underlying_df = self.underlying_df.rename(columns={"Stock_Close": "Close"})

        self.call_num = 0.0
        self.put_num = 0.0
        self.underlying_num = 0.0
        self.total_value = self.balance
        self.option_value = 0.0

        self.max_invest = max_invest
        self.max_leverage = max_leverage
        self.vrp_threshold = vrp_threshold
        self.vrp_close_threshold = vrp_close_threshold
        self.delta_hedge = delta_hedge
        self.delta_rehedge_threshold = (1e9 if not delta_hedge else delta_rehedge_threshold)
        self.min_ttm = min_ttm
        self.max_ttm = max_ttm

        self.greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
        }
        self.current_call_sym = None
        self.current_put_sym = None
        self.k = None
        self.r = None
        self.ttm = None
        self.entry_value = None
        self.entry_premium = None
        self.trade_open = False
        self._rehedged_on_date = None

    def _normalize_date(self, date):
        if date is None:
            return None
        if isinstance(date, pd.Timestamp):
            return date.normalize()
        return pd.to_datetime(date).normalize()

    def _panel_row(self, date_norm):
        hit = self.panel_df[self.panel_df["Date"] == date_norm]
        if hit.empty:
            return None
        return hit.iloc[0]

    def _synth_option_rows(self, row, S: float, date_norm):
        """Build call/put row dicts for Greeks from one panel row (same expiry/strike)."""
        call_sym = row["Call_Sym"]
        put_sym = row["Put_Sym"]
        exp_c, k_c = _parse_occ_symbol(call_sym)
        exp_p, k_p = _parse_occ_symbol(put_sym)
        if exp_c != exp_p or abs(k_c - k_p) > 1e-6:
            warnings.warn(f"Call/put expiry or strike mismatch: {call_sym} vs {put_sym}")
        ttm = _ttm_years(date_norm, exp_c)
        r = float(row["r"])
        call_row = pd.Series(
            {
                "close": float(row["Call_Close"]),
                "imp_vol": float(row["Call_imp_vol"]),
                "k": k_c,
                "r": r,
                "ttm": ttm,
                "S": S,
            }
        )
        put_row = pd.Series(
            {
                "close": float(row["Put_Close"]),
                "imp_vol": float(row["Put_imp_vol"]),
                "k": k_p,
                "r": r,
                "ttm": ttm,
                "S": S,
            }
        )
        return call_row, put_row

    def cal_value(self, date):
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df["Date"].dt.normalize() == date_norm]
        if und_row.empty:
            self.total_value = np.nan
            self.option_value = 0.0
            return np.nan
        S = float(und_row["Close"].iloc[0])

        call_price = 0.0
        put_price = 0.0
        pr = self._panel_row(date_norm)
        if self.trade_open and pr is not None:
            if pr["Call_Sym"] == self.current_call_sym and pr["Put_Sym"] == self.current_put_sym:
                if pd.notna(pr["Call_Close"]):
                    call_price = float(pr["Call_Close"])
                if pd.notna(pr["Put_Close"]):
                    put_price = float(pr["Put_Close"])

        self.option_value = self.call_num * call_price + self.put_num * put_price
        self.total_value = self.balance + self.option_value + self.underlying_num * S

        if (
            self.trade_open
            and pr is not None
            and pr["Call_Sym"] == self.current_call_sym
            and pr["Put_Sym"] == self.current_put_sym
            and pd.notna(pr["Call_Close"])
            and pd.notna(pr["Put_Close"])
            and pd.notna(pr["Call_imp_vol"])
            and pd.notna(pr["Put_imp_vol"])
        ):
            try:
                call_row, put_row = self._synth_option_rows(pr, S, date_norm)
                new_greeks = get_greeks_analytical(call_row, put_row)
                self.greeks.update(new_greeks)
                self.ttm = float(call_row["ttm"])
            except Exception:
                pass

        return self.total_value

    def close_position(self, date, reason=""):
        if not self.trade_open:
            return

        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df["Date"].dt.normalize() == date_norm]
        if und_row.empty:
            return
        S = float(und_row["Close"].iloc[0])

        call_price = 0.0
        put_price = 0.0
        pr = self._panel_row(date_norm)
        if pr is not None and pr["Call_Sym"] == self.current_call_sym and pr["Put_Sym"] == self.current_put_sym:
            if pd.notna(pr["Call_Close"]):
                call_price = float(pr["Call_Close"])
            if pd.notna(pr["Put_Close"]):
                put_price = float(pr["Put_Close"])

        opt_pnl = self.call_num * call_price + self.put_num * put_price
        self.balance += opt_pnl

        und_pnl = self.underlying_num * S
        self.balance += und_pnl

        self.call_num = 0.0
        self.put_num = 0.0
        self.underlying_num = 0.0
        self.greeks = {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
        }
        self.current_call_sym = None
        self.current_put_sym = None
        self.trade_open = False
        self.entry_value = None
        self.entry_premium = None

    def build_position(self, call_sym, put_sym, date):
        if self.trade_open:
            self.close_position(date, reason="rebalance")

        date_norm = self._normalize_date(date)
        pr = self._panel_row(date_norm)
        if pr is None:
            warnings.warn(f"No panel row for {date}")
            return
        if pr["Call_Sym"] != call_sym or pr["Put_Sym"] != put_sym:
            warnings.warn("Panel symbols do not match build_position arguments")
            return
        if pd.isna(pr["Call_Close"]) or pd.isna(pr["Put_Close"]):
            return
        if pd.isna(pr["Call_imp_vol"]) or pd.isna(pr["Put_imp_vol"]):
            return

        und_row = self.underlying_df[self.underlying_df["Date"].dt.normalize() == date_norm]
        if und_row.empty:
            warnings.warn(f"No underlying data for {date}")
            return
        S = float(und_row["Close"].iloc[0])

        try:
            call_row, put_row = self._synth_option_rows(pr, S, date_norm)
        except Exception as e:
            warnings.warn(f"Failed to parse symbols {call_sym}/{put_sym}: {e}")
            return

        ttm = float(call_row["ttm"])
        if ttm < self.min_ttm or ttm > self.max_ttm:
            return

        self.greeks = get_greeks_analytical(call_row, put_row)
        delta = self.greeks["delta"]

        IV = (float(call_row["imp_vol"]) + float(put_row["imp_vol"])) / 2
        RV = float(und_row["RV"].iloc[0])
        VRP = IV - RV
        vrp_std = float(und_row["VRP_std"].iloc[0])
        vrp_mean = float(und_row["VRP_mean"].iloc[0])
        if not np.isfinite(vrp_std) or vrp_std <= 0:
            return

        action = None
        if VRP > vrp_mean + (self.vrp_threshold * vrp_std):
            action = "short"
        elif VRP < vrp_mean - (self.vrp_threshold * vrp_std):
            action = "long"
        else:
            return

        premium_per_unit = float(call_row["close"]) + float(put_row["close"])
        if premium_per_unit <= 0:
            return
        if action == "long":
            cost_per_unit = premium_per_unit + max(0.0, -delta * S)
            if cost_per_unit <= 0:
                return
            max_units = (self.max_invest * self.total_value) / cost_per_unit
            units = int(np.floor(max_units))
        else:
            exposure_per_unit = premium_per_unit + abs(delta) * S
            if exposure_per_unit <= 0:
                return
            max_units = (self.max_leverage * self.total_value) / exposure_per_unit
            units = -int(np.floor(max_units))
        if units == 0:
            return

        net_premium = units * premium_per_unit

        self.call_num = units
        self.put_num = units
        self.underlying_num = int(round(-units * delta)) if self.delta_hedge else 0

        self.balance += -net_premium
        self.balance += -(self.underlying_num * S)

        self.entry_value = self.balance + net_premium + self.underlying_num * S
        self.entry_premium = abs(net_premium)
        self.trade_open = True
        self.current_call_sym = call_sym
        self.current_put_sym = put_sym
        self.k = call_row["k"]
        self.r = call_row["r"]
        self.ttm = ttm

    def rehedge(self, date):
        if not self.trade_open or self.delta_rehedge_threshold is None:
            return
        self.cal_value(date)
        net_delta = self.greeks["delta"] * self.call_num + self.underlying_num
        if abs(net_delta) <= self.delta_rehedge_threshold:
            return
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df["Date"].dt.normalize() == date_norm]
        if und_row.empty:
            return
        S = float(und_row["Close"].iloc[0])
        target_underlying_num = int(round(-self.greeks["delta"] * self.call_num))
        old_underlying_num = self.underlying_num
        shares_to_trade = target_underlying_num - old_underlying_num
        self.balance -= shares_to_trade * S
        self.underlying_num = target_underlying_num
        self._rehedged_on_date = date

    def should_exit(self, date):
        if not self.trade_open:
            return False, ""

        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df["Date"].dt.normalize() == date_norm]
        if und_row.empty:
            return False, ""

        if self.ttm is not None and self.ttm < self.min_ttm / 2:
            return True, "near-expiry"

        if self.vrp_close_threshold is not None:
            pr = self._panel_row(date_norm)
            if (
                pr is not None
                and pr["Call_Sym"] == self.current_call_sym
                and pr["Put_Sym"] == self.current_put_sym
                and pd.notna(pr["Call_imp_vol"])
                and pd.notna(pr["Put_imp_vol"])
            ):
                vrp_std = float(und_row["VRP_std"].iloc[0])
                vrp_mean = float(und_row["VRP_mean"].iloc[0])
                if np.isfinite(vrp_std) and vrp_std > 0:
                    IV = (float(pr["Call_imp_vol"]) + float(pr["Put_imp_vol"])) / 2
                    RV = float(und_row["RV"].iloc[0])
                    VRP = IV - RV
                    long_position = self.call_num > 0
                    if long_position:
                        if VRP > vrp_mean + (self.vrp_close_threshold * vrp_std):
                            return True, f"vrp_close_long (IV-RV={VRP:.4f})"
                    else:
                        if VRP < vrp_mean - (self.vrp_close_threshold * vrp_std):
                            return True, f"vrp_close_short (IV-RV={VRP:.4f})"

        return False, ""


def kelly_binary_from_trades(trade_returns: np.ndarray) -> dict:
    t = np.asarray(trade_returns, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return {"p_win": np.nan, "R": np.nan, "kelly_binary": np.nan, "n_trades": 0}
    wins = t[t > 0]
    losses = t[t < 0]
    p = len(wins) / len(t)
    if len(wins) == 0 or len(losses) == 0:
        return {"p_win": p, "R": np.nan, "kelly_binary": np.nan, "n_trades": len(t)}
    W = float(np.mean(wins))
    L = float(np.mean(np.abs(losses)))
    if L < 1e-12:
        return {"p_win": p, "R": np.nan, "kelly_binary": np.nan, "n_trades": len(t)}
    R = W / L
    kelly = p - (1.0 - p) / R
    return {"p_win": p, "R": R, "kelly_binary": kelly, "n_trades": len(t)}


def kelly_continuous_daily(daily_r: np.ndarray) -> float:
    x = np.asarray(daily_r, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    mu = np.mean(x)
    v = np.var(x, ddof=1)
    if v < 1e-16:
        return np.nan
    return mu / v


def run_agent_ddh_backtest(
    panel_df: pd.DataFrame,
    vrp_entry_k: float,
    vrp_close_threshold: float,
    *,
    balance: float = 100_000.0,
    max_invest: float = 0.8,
    max_leverage: float = 0.5,
    delta_rehedge_threshold: float = 0.5,
    min_ttm: float = 4 / 252,
    max_ttm: float = 60 / 252,
    delta_hedge: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Day-by-day simulation using Agent_DDH.

    vrp_entry_k: σ-multiple for entry vs VRP_mean (see Agent_DDH.build_position).
    vrp_close_threshold: σ-multiple for VRP mean-reversion exit (see Agent_DDH.should_exit).
    """
    n = len(panel_df)
    nav = np.full(n, np.nan)
    daily_ret = np.full(n, np.nan)
    gamma_a = np.full(n, np.nan)
    delta_a = np.full(n, np.nan)
    vega_a = np.full(n, np.nan)
    theta_a = np.full(n, np.nan)
    vanna_a = np.full(n, np.nan)
    volga_a = np.full(n, np.nan)
    call_nums = np.zeros(n)
    und_nums = np.zeros(n)
    events = np.array([""] * n, dtype=object)
    call_syms = np.array([""] * n, dtype=object)
    exit_syms = np.array([""] * n, dtype=object)
    rolled = np.zeros(n, dtype=bool)
    rehedge_day = np.zeros(n, dtype=bool)
    trade_returns: list[float] = []
    entry_nav: Optional[float] = None

    dth = 1e9 if not delta_hedge else delta_rehedge_threshold
    agent = Agent_DDH(
        panel_df=panel_df,
        vrp_threshold=vrp_entry_k,
        vrp_close_threshold=vrp_close_threshold,
        balance=balance,
        max_invest=max_invest,
        max_leverage=max_leverage,
        delta_rehedge_threshold=dth,
        min_ttm=min_ttm,
        max_ttm=max_ttm,
        delta_hedge=delta_hedge,
    )

    for i in range(n):
        row = panel_df.iloc[i]
        d = row["Date"]
        dn = pd.Timestamp(d).normalize()
        fc = bool(row["Force_Close"])
        pr = agent._panel_row(dn)

        agent.cal_value(d)
        was_open = agent.trade_open
        u_before = agent.underlying_num

        closed_today = False
        if was_open and fc:
            exsym = agent.current_call_sym or ""
            agent.close_position(d, "force_close")
            agent.cal_value(d)
            if entry_nav is not None and np.isfinite(entry_nav):
                trade_returns.append((agent.total_value - entry_nav) / max(abs(entry_nav), 1.0))
            entry_nav = None
            events[i] = "exit:force_close"
            exit_syms[i] = exsym
            closed_today = True
        elif was_open and pr is not None and agent.current_call_sym is not None:
            if pr["Call_Sym"] != agent.current_call_sym or pr["Put_Sym"] != agent.current_put_sym:
                exsym = agent.current_call_sym or ""
                agent.close_position(d, "roll")
                agent.cal_value(d)
                if entry_nav is not None and np.isfinite(entry_nav):
                    trade_returns.append((agent.total_value - entry_nav) / max(abs(entry_nav), 1.0))
                entry_nav = None
                events[i] = "exit:roll"
                exit_syms[i] = exsym
                rolled[i] = True
                closed_today = True

        if agent.trade_open and not closed_today:
            ex, reason = agent.should_exit(d)
            if ex:
                exsym = agent.current_call_sym or ""
                agent.close_position(d, reason)
                agent.cal_value(d)
                if entry_nav is not None and np.isfinite(entry_nav):
                    trade_returns.append((agent.total_value - entry_nav) / max(abs(entry_nav), 1.0))
                entry_nav = None
                events[i] = "exit:vrp" if "vrp" in reason.lower() else "exit:other"
                exit_syms[i] = exsym
                closed_today = True

        if agent.trade_open:
            agent.rehedge(d)
            if agent.underlying_num != u_before:
                rehedge_day[i] = True

        agent.cal_value(d)

        if not agent.trade_open and not fc and pr is not None:
            if pd.notna(pr["Call_Sym"]) and pd.notna(pr["Put_Sym"]):
                open_before = agent.trade_open
                agent.build_position(pr["Call_Sym"], pr["Put_Sym"], d)
                agent.cal_value(d)
                if agent.trade_open and not open_before:
                    entry_nav = float(agent.total_value)
                    if agent.call_num > 0:
                        events[i] = "enter_long"
                    elif agent.call_num < 0:
                        events[i] = "enter_short"
                    call_syms[i] = agent.current_call_sym or ""

        agent.cal_value(d)
        nav[i] = agent.total_value if np.isfinite(agent.total_value) else np.nan
        call_nums[i] = agent.call_num
        und_nums[i] = agent.underlying_num
        if agent.trade_open:
            gamma_a[i] = agent.greeks.get("gamma", np.nan)
            delta_a[i] = agent.greeks.get("delta", np.nan) * agent.call_num + agent.underlying_num
            vega_a[i] = agent.greeks.get("vega", np.nan)
            theta_a[i] = agent.greeks.get("theta", np.nan)
            vanna_a[i] = agent.greeks.get("vanna", np.nan)
            volga_a[i] = agent.greeks.get("volga", np.nan)
            if not events[i]:
                call_syms[i] = agent.current_call_sym or ""

        if i > 0 and np.isfinite(nav[i]) and np.isfinite(nav[i - 1]):
            scale = max(abs(nav[i - 1]), 1.0)
            daily_ret[i] = (nav[i] - nav[i - 1]) / scale

    kb = kelly_binary_from_trades(np.array(trade_returns, dtype=float))
    dr = daily_ret[np.isfinite(daily_ret)]
    p_day = float(np.mean(dr > 0)) if len(dr) else np.nan
    k_cont = kelly_continuous_daily(dr)
    dr_filled = np.where(np.isfinite(daily_ret), daily_ret, 0.0)
    equity = np.cumprod(1.0 + dr_filled)

    out = pd.DataFrame(
        {
            "Date": panel_df["Date"].values,
            "PortfolioMTM": nav,
            "NAV": nav,
            "DailyReturn": daily_ret,
            "Equity": equity,
            "HedgeShares": und_nums,
            "CallNum": call_nums,
            "NetDelta": delta_a,
            "Gamma": gamma_a,
            "Vega": vega_a,
            "Theta": theta_a,
            "Vanna": vanna_a,
            "Volga": volga_a,
            "TradeEvent": events,
            "CallSymbol": call_syms,
            "ExitCallSymbol": exit_syms,
            "Rolled": rolled,
            "RehedgeDay": rehedge_day,
        }
    )

    summary = {
        "vrp_entry_k": float(vrp_entry_k),
        "vrp_close_threshold": float(vrp_close_threshold),
        "n_trades": int(kb["n_trades"]),
        "prob_win_trade": float(kb["p_win"]) if np.isfinite(kb["p_win"]) else np.nan,
        "prob_win_daily": p_day,
        "kelly_binary": float(kb["kelly_binary"]) if np.isfinite(kb["kelly_binary"]) else np.nan,
        "kelly_continuous_daily": float(k_cont) if np.isfinite(k_cont) else np.nan,
        "win_loss_ratio_R": float(kb["R"]) if np.isfinite(kb["R"]) else np.nan,
        "mean_daily_return": float(np.mean(dr)) if len(dr) else np.nan,
        "std_daily_return": float(np.std(dr, ddof=1)) if len(dr) > 1 else np.nan,
        "final_nav": float(nav[-1]) if len(nav) and np.isfinite(nav[-1]) else np.nan,
    }
    return out, summary
