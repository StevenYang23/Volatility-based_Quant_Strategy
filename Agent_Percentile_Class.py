import numpy as np
import pandas as pd
from pathlib import Path

class Agent_Percentile:
    def __init__(
        self,
        display_name="Agent_Percentile",
        entry_percentile=0.20,
        allow_short=True,
        delta_hedge=True,
        rehedge_threshold=0.05,
    ):
        self.display_name = display_name
        self.entry_low_percentile = entry_percentile
        self.entry_high_percentile = 1.0 - entry_percentile
        self.min_vrp_history = 10
        self.delta_hedge = delta_hedge
        self.rehedge_threshold = rehedge_threshold
        self.allow_short = allow_short
        if not (0.0 < self.entry_low_percentile < 0.5):
            raise ValueError("entry_low_percentile must be in (0, 0.5).")

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
        self.vrp_history = []  # all valid VRP values the agent has seen
        self.vrp_zscore_history = []  # z-scores built from seen VRP history
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
    def _compute_daily_pnl(self, data):
        """Path-wise Greeks attribution with hedge bucket and residual."""

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

        def _f(v):
            return float(v) if pd.notna(v) and np.isfinite(v) else 0.0

        d_sigma = _f(data["Straddle_imp_vol"]) - _f(self.prev_data["Straddle_imp_vol"])
        dr = _f(data["r"]) - _f(self.prev_data["r"])
        prev_date = pd.to_datetime(self.prev_data["Date"]).date().isoformat()
        curr_date = pd.to_datetime(data["Date"]).date().isoformat()
        dt_days = np.busday_count(prev_date, curr_date)
        dt = (dt_days if dt_days > 0 else 1) / 252.0

        q = self.num_options
        h = self.num_underlying
        prev_delta = _f(self.prev_data.get("Straddle_Delta"))
        delta_pnl = q * prev_delta * dS
        gamma_pnl = 0.5 * q * _f(self.prev_data.get("Straddle_Gamma")) * (dS ** 2)
        vega_pnl = q * _f(self.prev_data.get("Straddle_Vega")) * d_sigma
        vanna_pnl = q * _f(self.prev_data.get("Straddle_Vanna")) * dS * d_sigma
        volga_pnl = 0.5 * q * _f(self.prev_data.get("Straddle_Volga")) * (d_sigma ** 2)
        theta_pnl = q * _f(self.prev_data.get("Straddle_Theta")) * dt
        rho_pnl = q * _f(self.prev_data.get("Straddle_Rho")) * dr * 100.0
        hedge_pnl = h * dS_adj

        explained = (
            delta_pnl + gamma_pnl + vega_pnl + vanna_pnl +
            volga_pnl + theta_pnl + rho_pnl + hedge_pnl
        )
        actual_delta_exp = q * prev_delta + (h if self.delta_hedge else 0.0)
        self._store_attr(
            actual_delta_exp=actual_delta_exp,
            delta_pnl=delta_pnl,
            gamma_pnl=gamma_pnl,
            vega_pnl=vega_pnl,
            theta_pnl=theta_pnl,
            vanna_pnl=vanna_pnl,
            volga_pnl=volga_pnl,
            rho_pnl=rho_pnl,
            hedge_pnl=hedge_pnl,
            residual_pnl=daily_pnl - explained,
        )

    def _store_attr(
        self,
        actual_delta_exp,
        delta_pnl,
        gamma_pnl,
        vega_pnl,
        theta_pnl,
        vanna_pnl,
        volga_pnl,
        rho_pnl,
        hedge_pnl,
        residual_pnl,
    ):
        self.actual_delta.append(actual_delta_exp)
        self.delta_attribute.append(delta_pnl)
        self.gamma_attribute.append(gamma_pnl + hedge_pnl)
        self.vega_attribute.append(vega_pnl)
        self.theta_attribute.append(theta_pnl)
        self.vanna_attribute.append(vanna_pnl)
        self.volga_attribute.append(volga_pnl)
        self.rho_attribute.append(rho_pnl)
        self.residual.append(residual_pnl)

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
            vrp_fc = data["VRP"]
            if pd.notna(vrp_fc):
                vrp_fc = float(vrp_fc)
                if len(self.vrp_history) >= self.min_vrp_history:
                    mu_fc = float(np.nanmean(self.vrp_history))
                    sd_fc = float(np.nanstd(self.vrp_history, ddof=1))
                    if np.isfinite(sd_fc) and sd_fc > 1e-12:
                        self.vrp_zscore_history.append((vrp_fc - mu_fc) / sd_fc)
                self.vrp_history.append(vrp_fc)
            self.prev_data = data
            return

        vrp = data["VRP"]
        if pd.isna(vrp):
            self.prev_data = data
            return

        vrp = float(vrp)
        if len(self.vrp_history) < self.min_vrp_history:
            self.vrp_history.append(vrp)
            self.prev_data = data
            return

        # Switch from raw VRP thresholds to percentile thresholds of ZScore(VRP),
        # while still using historical percentile logic.
        mu = float(np.nanmean(self.vrp_history))
        sd = float(np.nanstd(self.vrp_history, ddof=1))
        if not np.isfinite(sd) or sd <= 1e-12:
            self.vrp_history.append(vrp)
            self.prev_data = data
            return

        curr_z = (vrp - mu) / sd
        if len(self.vrp_zscore_history) < self.min_vrp_history:
            self.vrp_zscore_history.append(curr_z)
            self.vrp_history.append(vrp)
            self.prev_data = data
            return

        low_th = float(np.nanpercentile(self.vrp_zscore_history, self.entry_low_percentile * 100.0))
        high_th = float(np.nanpercentile(self.vrp_zscore_history, self.entry_high_percentile * 100.0))
        mid_th = float(np.nanpercentile(self.vrp_zscore_history, 50.0))

        if self.num_options == 0:
            # Top percentile => short straddle.
            if curr_z >= high_th and self.allow_short:
                self.short_position(data)
            # Bottom percentile => long straddle.
            elif curr_z <= low_th:
                self.long_position(data)
        elif self.num_options == 1:
            # Close long when signal mean-reverts toward center.
            if curr_z >= mid_th:
                self.close_position(data)
        elif self.num_options == -1:
            # Close short when signal mean-reverts toward center.
            if curr_z <= mid_th:
                self.close_position(data)

        if self.num_options != 0 and self.delta_hedge:
            self.rehedge(data)

        self.vrp_zscore_history.append(curr_z)
        self.vrp_history.append(vrp)
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
