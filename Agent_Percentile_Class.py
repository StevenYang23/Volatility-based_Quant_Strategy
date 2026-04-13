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
        rehedge_threshold=0.5,
    ):
        self.display_name = display_name
        self.entry_low_percentile = entry_percentile
        self.entry_high_percentile = 1.0 - entry_percentile
        self.min_vrp_history = 10
        self.delta_hedge = delta_hedge
        self.rehedge_threshold = rehedge_threshold
        self.allow_short = allow_short
        self.theta_time_basis = "trading"
        self.trading_days_per_year = 252
        self.calendar_days_per_year = 365.25
        self._dsigma_cap = 0.15

        if self.theta_time_basis not in ("calendar", "trading"):
            raise ValueError("theta_time_basis must be either 'calendar' or 'trading'.")
        if not (0.0 < self.entry_low_percentile < 0.5):
            raise ValueError("entry_low_percentile must be in (0, 0.5).")

        self.num_options = 0      # +1 long straddle, -1 short straddle, 0 flat
        self.num_underlying = 0   # shares held for delta hedge
        self.entry_straddle_price = 0.0

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

    @staticmethod
    def dataset_greek_sign_sanity(data: pd.DataFrame):
        """
        Dataset-level sign sanity checks for long-straddle Greeks.
        """
        out = {}
        if "Straddle_Gamma" in data.columns:
            valid = data["Straddle_Gamma"].dropna()
            out["straddle_gamma_positive_ratio"] = float((valid > 0).mean()) if len(valid) else np.nan
        if "Straddle_Theta" in data.columns:
            valid = data["Straddle_Theta"].dropna()
            out["straddle_theta_negative_ratio"] = float((valid < 0).mean()) if len(valid) else np.nan
        return out

    def _time_fraction(self, prev_date, curr_date):
        prev_ts = pd.to_datetime(prev_date)
        curr_ts = pd.to_datetime(curr_date)
        if curr_ts <= prev_ts:
            return 0.0

        if self.theta_time_basis == "trading":
            prev_d = prev_ts.date().isoformat()
            curr_d = curr_ts.date().isoformat()
            business_days = np.busday_count(prev_d, curr_d)
            if business_days == 0:
                business_days = 1
            return business_days / float(self.trading_days_per_year)

        day_count = (curr_ts - prev_ts).days
        return day_count / float(self.calendar_days_per_year)

    # ------------------------------------------------------------------
    def _compute_daily_pnl(self, data):
        """Mark-to-market PnL and second-order Greek attribution."""

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

        yesterday_exposure = (abs(self.num_options * prev_straddle) + 
                              abs(self.num_underlying * self.prev_data["Stock_Close"]))

        simple_return = daily_pnl / yesterday_exposure if yesterday_exposure > 0 else 0.0
        # Store log return for additive time aggregation.
        safe_simple_return = max(simple_return, -0.999999999)
        daily_return = np.log1p(safe_simple_return)

        self.PnL.append(daily_pnl)
        self.Return.append(daily_return)

        # --- Greek attribution (Taylor expansion with prev-day Greeks) ---
        def _s(v):
            return float(v) if pd.notna(v) and np.isfinite(v) else 0.0
            
        def _diff(curr, prev):
            if pd.notna(curr) and np.isfinite(curr) and pd.notna(prev) and np.isfinite(prev):
                return float(curr) - float(prev)
            return 0.0

        d_sigma_raw = _diff(data["Straddle_imp_vol"], self.prev_data["Straddle_imp_vol"])
        d_sigma = np.clip(d_sigma_raw, -self._dsigma_cap, self._dsigma_cap)
        dr = _diff(data["r"], self.prev_data["r"])
        dt = self._time_fraction(self.prev_data["Date"], data["Date"])

        option_delta = self.num_options * _s(self.prev_data["Straddle_Delta"])
        if self.delta_hedge:
            actual_delta_exposure = option_delta + self.num_underlying
        else:
            actual_delta_exposure = option_delta

        delta_pnl = option_delta * dS + self.num_underlying * dS_adj
        gamma_pnl = 0.5 * self.num_options * _s(self.prev_data["Straddle_Gamma"]) * dS ** 2
        vega_pnl = self.num_options * _s(self.prev_data["Straddle_Vega"]) * d_sigma
        theta_pnl = self.num_options * _s(self.prev_data["Straddle_Theta"]) * dt
        vanna_pnl = self.num_options * _s(self.prev_data["Straddle_Vanna"]) * dS * d_sigma
        volga_pnl = 0.5 * self.num_options * _s(self.prev_data["Straddle_Volga"]) * d_sigma ** 2
        rho_pnl = self.num_options * _s(self.prev_data["Straddle_Rho"]) * dr * 100

        attributed = (delta_pnl + gamma_pnl + vega_pnl + theta_pnl
                      + vanna_pnl + volga_pnl + rho_pnl)

        self.actual_delta.append(actual_delta_exposure)
        self.delta_attribute.append(delta_pnl)
        self.gamma_attribute.append(gamma_pnl)
        self.vega_attribute.append(vega_pnl)
        self.theta_attribute.append(theta_pnl)
        self.vanna_attribute.append(vanna_pnl)
        self.volga_attribute.append(volga_pnl)
        self.rho_attribute.append(rho_pnl)
        self.residual.append(daily_pnl - attributed)

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

    def long_position(self, data):
        self.num_options = 1
        self.entry_straddle_price = data["Call_Close"] + data["Put_Close"]
        if self.delta_hedge:
            self.num_underlying = -data["Straddle_Delta"]
        self._log_transaction(data, "long")

    def short_position(self, data):
        self.num_options = -1
        self.entry_straddle_price = data["Call_Close"] + data["Put_Close"]
        if self.delta_hedge:
            self.num_underlying = data["Straddle_Delta"]
        self._log_transaction(data, "short")

    def close_position(self, data=None):
        was_open = self.num_options != 0
        self.num_options = 0
        self.num_underlying = 0
        self.entry_straddle_price = 0.0
        if was_open and data is not None:
            self._log_transaction(data, "close")

    def rehedge(self, data):
        net_delta = self.num_options * data["Straddle_Delta"] + self.num_underlying
        if abs(net_delta) > self.rehedge_threshold:
            self.num_underlying = -self.num_options * data["Straddle_Delta"]
            self._log_transaction(data, "rehedge")

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
            "theta_time_basis": self.theta_time_basis,
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
