import numpy as np
import pandas as pd
from pathlib import Path

from pnl_attribution import full_reval_attribution, parse_occ_symbol, ttm_trading, _zero_attr


class Agent_2threshold:
    def __init__(
        self,
        display_name="Agent_2threshold",
        k_high=1.4,
        k_low=0.6,
        allow_short=True,
        delta_hedge=True,
        rehedge_threshold=0.05,
    ):
        self.display_name = display_name
        self.k_high = k_high
        self.k_low = k_low
        self.delta_hedge = delta_hedge
        self.rehedge_threshold = rehedge_threshold
        self.allow_short = allow_short

        self.num_options = 0      # +1 long straddle, -1 short straddle, 0 flat
        self.num_underlying = 0   # shares held for delta hedge
        self.entry_straddle_price = 0.0
        self._hedge_avg_price = 0.0
        self._realized_hedge_pnl = 0.0

        self.prev_data = None

        self.PnL = []
        self.Return = []

        self.actual_delta = []
        self.delta_attribute = []
        self.gamma_attribute = []
        self.vega_attribute = []
        self.theta_attribute = []
        self.vanna_attribute = []
        self.volga_attribute = []
        self.rho_attribute = []
        self.residual = []
        self.position_state_for_pnl = []
        self._init_trade_log()

    def _init_trade_log(self):
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / f"{self.display_name}_log.csv"
        if not self.log_path.exists():
            pd.DataFrame(columns=["Date", "Transaction", "Earned"]).to_csv(self.log_path, index=False)

    def _current_pnl_for_log(self):
        return float(self.PnL[-1]) if len(self.PnL) > 0 else 0.0

    def _log_transaction(self, data, transaction, earned=None):
        ts = pd.to_datetime(data.get("Date", pd.NaT), errors="coerce")
        date_str = ts.strftime("%Y-%m-%d") if pd.notna(ts) else ""
        row = {
            "Date": date_str,
            "Transaction": transaction,
            "Earned": float(self._current_pnl_for_log() if earned is None else earned),
        }
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", header=False, index=False)

    # ------------------------------------------------------------------
    @staticmethod
    def _get_K_expiry(data):
        """Return (K, expiry_date) from CSV columns or by parsing Call_Sym."""
        from datetime import date as _date
        K = data.get("K", None)
        exp = data.get("Expiry", None)
        if K is not None and exp is not None and pd.notna(K) and pd.notna(exp):
            if isinstance(exp, str):
                parts = exp.split("-")
                exp = _date(int(parts[0]), int(parts[1]), int(parts[2]))
            return float(K), exp
        sym = data.get("Call_Sym", None)
        if sym and pd.notna(sym):
            expiry, strike = parse_occ_symbol(str(sym))
            return strike, expiry
        return None, None

    # ------------------------------------------------------------------
    def _compute_daily_pnl(self, data):
        """Mark-to-market PnL with full-repricing Greek attribution."""

        self.position_state_for_pnl.append(self.num_options)

        if self.num_options == 0 or self.prev_data is None:
            self._append_zeros()
            return

        straddle_price = data["Call_Close"] + data["Put_Close"]
        prev_straddle = self.prev_data["Call_Close"] + self.prev_data["Put_Close"]
        dS = data["Stock_Close"] - self.prev_data["Stock_Close"]

        div = data.get("Stock_Dividends", 0.0)
        if pd.isna(div):
            div = 0.0
        dS_adj = dS + div

        daily_pnl = (self.num_options * (straddle_price - prev_straddle)
                     + self.num_underlying * dS_adj)

        yesterday_exposure = (abs(self.num_options * prev_straddle)
                              + abs(self.num_underlying * self.prev_data["Stock_Close"]))
        simple_return = daily_pnl / yesterday_exposure if yesterday_exposure > 0 else 0.0
        safe_simple_return = max(simple_return, -0.999999999)
        daily_return = np.log1p(safe_simple_return)

        self.PnL.append(daily_pnl)
        self.Return.append(daily_return)

        # --- Full-repricing attribution ---
        K_curr, exp_curr = self._get_K_expiry(data)
        K_prev, exp_prev = self._get_K_expiry(self.prev_data)
        K = K_curr or K_prev
        expiry = exp_curr or exp_prev

        if K is None or expiry is None:
            self._store_attr(_zero_attr(), dS_adj)
            return

        prev_date = pd.to_datetime(self.prev_data["Date"]).date()
        curr_date = pd.to_datetime(data["Date"]).date()
        T0 = ttm_trading(prev_date, expiry)
        T1 = ttm_trading(curr_date, expiry)

        def _f(v):
            return float(v) if pd.notna(v) and np.isfinite(v) else 0.0

        attr = full_reval_attribution(
            S0=_f(self.prev_data["Stock_Close"]),
            S1=_f(data["Stock_Close"]),
            K=K,
            T0=T0,
            T1=T1,
            r0=_f(self.prev_data["r"]),
            r1=_f(data["r"]),
            sig0=_f(self.prev_data["Straddle_imp_vol"]),
            sig1=_f(data["Straddle_imp_vol"]),
            market_straddle_change=straddle_price - prev_straddle,
        )
        self._store_attr(attr, dS_adj)

    def _store_attr(self, attr, dS_adj):
        option_delta_pnl = self.num_options * attr["delta"]
        hedge_pnl = self.num_underlying * dS_adj

        prev_straddle_delta = float(self.prev_data.get("Straddle_Delta", 0.0) or 0.0)
        option_delta_exp = self.num_options * prev_straddle_delta
        actual_delta_exp = (option_delta_exp + self.num_underlying) if self.delta_hedge else option_delta_exp

        self.actual_delta.append(actual_delta_exp)
        self.delta_attribute.append(option_delta_pnl + hedge_pnl)
        self.gamma_attribute.append(self.num_options * attr["gamma"])
        self.vega_attribute.append(self.num_options * attr["vega"])
        self.theta_attribute.append(self.num_options * attr["theta"])
        self.vanna_attribute.append(self.num_options * attr["vanna"])
        self.volga_attribute.append(self.num_options * attr["volga"])
        self.rho_attribute.append(self.num_options * attr["rho"])
        self.residual.append(self.num_options * attr["residual"])

    def _append_zeros(self):
        for lst in (self.PnL, self.Return, self.actual_delta,
                    self.delta_attribute, self.gamma_attribute,
                    self.vega_attribute, self.theta_attribute,
                    self.vanna_attribute, self.volga_attribute,
                    self.rho_attribute, self.residual):
            lst.append(0.0)

    def trade(self, data):
        self._compute_daily_pnl(data)

        if data["Force_Close"]:
            self.close_position(data)
            self.prev_data = data
            return

        vrp = data["VRP"]
        vrp_mean = data["VRP_20d_mean"]
        std_20d = data["VRP_20d_std"]
        std_40d = data["VRP_40d_std"] if "VRP_40d_std" in data else np.nan

        if pd.isna(vrp) or pd.isna(vrp_mean) or pd.isna(std_20d) or std_20d == 0:
            self.prev_data = data
            return

        # Regime-switch threshold:
        # higher-vol regime (40d std > 20d std) -> use k_high, otherwise k_low.
        k_active = self.k_high if (pd.notna(std_40d) and std_40d > std_20d) else self.k_low

        if self.num_options == 0:
            if vrp > vrp_mean + k_active * std_20d and self.allow_short:
                self.short_position(data)
            elif vrp < vrp_mean - k_active * std_20d:
                self.long_position(data)
        elif self.num_options == 1:
            if vrp > vrp_mean:
                self.close_position(data)
        elif self.num_options == -1:
            if vrp < vrp_mean:
                self.close_position(data)

        if self.num_options != 0 and self.delta_hedge:
            self.rehedge(data)

        self.prev_data = data

    def _trade_underlying(self, target_underlying, spot_price):
        curr = float(self.num_underlying)
        target = float(target_underlying)
        trade_qty = target - curr
        if abs(trade_qty) <= 1e-12:
            return 0.0

        spot = float(spot_price)
        realized = 0.0
        curr_abs = abs(curr)
        avg = float(self._hedge_avg_price)

        if curr_abs <= 1e-12:
            self._hedge_avg_price = spot if abs(target) > 1e-12 else 0.0
        elif curr * trade_qty >= 0:
            new_abs = abs(target)
            if new_abs <= 1e-12:
                self._hedge_avg_price = 0.0
            else:
                self._hedge_avg_price = (avg * curr_abs + spot * abs(trade_qty)) / new_abs
        else:
            close_qty = min(curr_abs, abs(trade_qty))
            if curr > 0:
                realized = (spot - avg) * close_qty
            else:
                realized = (avg - spot) * close_qty

            if abs(target) <= 1e-12:
                self._hedge_avg_price = 0.0
            elif curr * target > 0:
                self._hedge_avg_price = avg
            else:
                self._hedge_avg_price = spot

        self.num_underlying = target
        self._realized_hedge_pnl += realized
        return realized

    def long_position(self, data):
        self.num_options = 1
        self.entry_straddle_price = data["Call_Close"] + data["Put_Close"]
        if self.delta_hedge:
            self._trade_underlying(-data["Straddle_Delta"], data["Stock_Close"])
        self._log_transaction(data, "long")

    def short_position(self, data):
        self.num_options = -1
        self.entry_straddle_price = data["Call_Close"] + data["Put_Close"]
        if self.delta_hedge:
            self._trade_underlying(data["Straddle_Delta"], data["Stock_Close"])
        self._log_transaction(data, "short")

    def close_position(self, data=None):
        was_open = self.num_options != 0
        hedge_realized = 0.0
        if was_open and data is not None and self.delta_hedge:
            hedge_realized = self._trade_underlying(0.0, data["Stock_Close"])
        self.num_options = 0
        self.num_underlying = 0
        self.entry_straddle_price = 0.0
        if was_open and data is not None:
            self._log_transaction(data, "close", earned=hedge_realized)

    def rehedge(self, data):
        net_delta = self.num_options * data["Straddle_Delta"] + self.num_underlying
        if abs(net_delta) > self.rehedge_threshold:
            target_underlying = -self.num_options * data["Straddle_Delta"]
            hedge_realized = self._trade_underlying(target_underlying, data["Stock_Close"])
            self._log_transaction(data, "rehedge", earned=hedge_realized)

    def get_result(self):
        """
        Return a structured result payload for analysis/reporting.
        """
        return {
            "display_name": self.display_name,
            "greeks_attribute": {
                "delta": self.delta_attribute,
                "gamma": self.gamma_attribute,
                "vega": self.vega_attribute,
                "theta": self.theta_attribute,
                "vanna": self.vanna_attribute,
                "volga": self.volga_attribute,
                "rho": self.rho_attribute,
                "residual": self.residual,
            },
            "pnl": self.PnL,
            "return": self.Return,
            "actual_delta": self.actual_delta,
            "position_state_for_pnl": self.position_state_for_pnl,
        }

    def regime_attribution_summary(self, include_flat=False):
        """
        Summarize cumulative attribution by pre-trade position regime.
        pre_state = -1 (short straddle), 0 (flat), +1 (long straddle).
        """
        regime_df = pd.DataFrame(
            {
                "pre_state": self.position_state_for_pnl,
                "pnl": self.PnL,
                "delta": self.delta_attribute,
                "gamma": self.gamma_attribute,
                "vega": self.vega_attribute,
                "theta": self.theta_attribute,
                "vanna": self.vanna_attribute,
                "volga": self.volga_attribute,
                "rho": self.rho_attribute,
                "residual": self.residual,
            }
        )
        if not include_flat:
            regime_df = regime_df[regime_df["pre_state"] != 0]

        if regime_df.empty:
            return regime_df

        summary = regime_df.groupby("pre_state").agg(
            days=("pre_state", "size"),
            pnl=("pnl", "sum"),
            delta=("delta", "sum"),
            gamma=("gamma", "sum"),
            vega=("vega", "sum"),
            theta=("theta", "sum"),
            vanna=("vanna", "sum"),
            volga=("volga", "sum"),
            rho=("rho", "sum"),
            residual=("residual", "sum"),
        )
        summary.index = summary.index.map({-1: "short_straddle", 0: "flat", 1: "long_straddle"})
        return summary
