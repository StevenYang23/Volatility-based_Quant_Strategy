import numpy as np
import pandas as pd
import math
from pathlib import Path


class Agent_LongTerm:
    _MIN_SPREAD_OBS = 20
    _KDE_ROLLING_WINDOW = 20

    def __init__(
        self,
        display_name="Agent_LongTerm",
        entry_threshold=0.8,
        allow_short=True,
        delta_hedge=True,
        long_rehedge_threshold=1.5,
        short_rehedge_threshold=0.5,
        long_term_window=126,
        slippage_rate=0.003,
    ):
        self.display_name = display_name
        self.delta_hedge = delta_hedge
        # k in k * sqrt(2 * |theta| * gamma); long vs short straddle uses separate k; delta change vs last hedge.
        self.long_rehedge_threshold = float(long_rehedge_threshold)
        self.short_rehedge_threshold = float(short_rehedge_threshold)
        self.allow_short = allow_short
        self.entry_threshold = max(float(entry_threshold), 0.0)
        self.long_term_window = max(int(long_term_window), 5)
        self.option_lot_size = 100
        self.slippage_rate = max(float(slippage_rate), 0.0)

        self.num_options = 0
        self.num_underlying = 0
        self.entry_straddle_price = 0.0
        self._hedge_avg_price = 0.0
        self._realized_hedge_pnl = 0.0
        self._last_return_notional = 0.0
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
        self._net_delta_at_last_hedge = None

        self._rv_buf = []
        self._rv_buf_max = max(self.long_term_window * 4, 512)
        self._spread_history = []
        self.longterm_direction_signal = []
        self.long_term_mean_history = []
        self.iv_longterm_spread_history = []
        self._init_trade_log()

    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + math.erf(x / np.sqrt(2.0)))

    @classmethod
    def _kde_cdf_signal(cls, values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 3:
            return np.nan

        x0 = float(arr[-1])
        sample = arr[:-1]
        n = sample.size
        if n < 2:
            return np.nan

        std = float(np.std(sample, ddof=1))
        if (not np.isfinite(std)) or std < 1e-12:
            return float(2.0 * np.mean(sample <= x0) - 1.0)

        h = 1.06 * std * (n ** (-1.0 / 5.0))
        h = max(float(h), 1e-6)
        z = (x0 - sample) / h
        cdf_vals = np.array([cls._norm_cdf(float(v)) for v in z], dtype=float)
        p = float(np.mean(cdf_vals))
        p = min(max(p, 0.0), 1.0)
        return float(2.0 * p - 1.0)

    # ------------------------------------------------------------------
    # Boilerplate (logging, PnL, Greeks attribution)
    # ------------------------------------------------------------------

    def _init_trade_log(self):
        logs_dir = Path(__file__).resolve().parent / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / f"{self.display_name}_log.csv"
        if not self.log_path.exists():
            pd.DataFrame(columns=["Date", "Transaction", "Earned"]).to_csv(
                self.log_path, index=False
            )

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
    def _float_greek(v):
        x = float(v) if pd.notna(v) and np.isfinite(v) else 0.0
        return x

    def _portfolio_net_delta(self, data):
        d = self._float_greek(data.get("Straddle_Delta"))
        return self.num_options * self.option_lot_size * d + self.num_underlying

    def _rehedge_k_multiplier(self):
        return (
            self.short_rehedge_threshold
            if self.num_options < 0
            else self.long_rehedge_threshold
        )

    def _rehedge_band_width(self, data):
        k = self._rehedge_k_multiplier()
        t = abs(self._float_greek(data.get("Straddle_Theta")))
        g = max(self._float_greek(data.get("Straddle_Gamma")), 0.0)
        # Straddle_Theta is per year of T (same 252-day year as PnL dt); one trading day of decay is |Theta|/252.
        inner = (2.0 * t * g) / 252.0
        if inner <= 0.0:
            return float(k)
        return float(k) * float(np.sqrt(inner))

    def _rehedge_should_trigger(self, net_delta, data):
        band = self._rehedge_band_width(data)
        if self._net_delta_at_last_hedge is None:
            return abs(net_delta) > band
        return abs(net_delta - float(self._net_delta_at_last_hedge)) > band

    def _rehedge_update_anchor(self, data):
        self._net_delta_at_last_hedge = self._portfolio_net_delta(data)

    def _rehedge_clear_anchor(self):
        self._net_delta_at_last_hedge = None

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

        option_units = self.num_options * self.option_lot_size
        daily_pnl = option_units * (straddle_price - prev_straddle) + self.num_underlying * dS_adj

        yesterday_exposure = abs(option_units * prev_straddle) + abs(
            self.num_underlying * self.prev_data["Stock_Close"]
        )
        self._last_return_notional = float(yesterday_exposure)
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

        q = self.num_options * self.option_lot_size
        h = self.num_underlying
        prev_delta = _f(self.prev_data.get("Straddle_Delta"))
        curr_delta = _f(data.get("Straddle_Delta"))
        effective_delta = prev_delta
        if self.delta_hedge:
            end_net_delta = q * curr_delta + h
            if self._rehedge_should_trigger(end_net_delta, data):
                effective_delta = 0.5 * (prev_delta + curr_delta)
        delta_pnl = q * effective_delta * dS
        gamma_pnl = 0.5 * q * _f(self.prev_data.get("Straddle_Gamma")) * (dS**2)
        vega_pnl = q * _f(self.prev_data.get("Straddle_Vega")) * d_sigma
        vanna_pnl = q * _f(self.prev_data.get("Straddle_Vanna")) * dS * d_sigma
        volga_pnl = 0.5 * q * _f(self.prev_data.get("Straddle_Volga")) * (d_sigma**2)
        theta_pnl = q * _f(self.prev_data.get("Straddle_Theta")) * dt
        rho_pnl = q * _f(self.prev_data.get("Straddle_Rho")) * dr * 100.0
        hedge_pnl = h * dS_adj

        explained = (
            delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl + theta_pnl + rho_pnl + hedge_pnl
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
        self._last_return_notional = 0.0
        for lst in (
            self.PnL,
            self.Return,
            self.actual_delta,
            self.delta_attribute,
            self.gamma_attribute,
            self.vega_attribute,
            self.theta_attribute,
            self.vanna_attribute,
            self.volga_attribute,
            self.rho_attribute,
            self.residual,
        ):
            lst.append(0.0)

    def _book_trading_cost(self, cost, trade_notional=0.0):
        cost = float(cost)
        if (not np.isfinite(cost)) or cost <= 0.0 or len(self.PnL) == 0:
            return

        self.PnL[-1] -= cost
        if len(self.residual) > 0:
            self.residual[-1] -= cost

        if len(self.Return) > 0:
            denom = max(float(self._last_return_notional), float(trade_notional))
            if denom > 0.0:
                simple_return = self.PnL[-1] / denom
                self.Return[-1] = np.log1p(max(simple_return, -0.999999999))

    def _append_signal_nan(self, direction=0):
        self.longterm_direction_signal.append(direction)
        self.long_term_mean_history.append(np.nan)
        self.iv_longterm_spread_history.append(np.nan)

    # ------------------------------------------------------------------
    # Signal: current IV vs rolling mean of prior RV values
    # ------------------------------------------------------------------

    def trade(self, data):
        self._compute_daily_pnl(data)

        if data["Force_Close"]:
            self.close_position(data)
            self._append_signal_nan(direction=0)
            self.prev_data = data
            return

        curr_iv = data.get("Straddle_imp_vol", np.nan)
        curr_rv = data.get("RV", np.nan)
        if pd.isna(curr_iv) or not np.isfinite(curr_iv):
            self._append_signal_nan(direction=0)
            self.prev_data = data
            return

        if pd.isna(curr_rv) or not np.isfinite(curr_rv):
            self._append_signal_nan(direction=0)
            self.prev_data = data
            return

        if len(self._rv_buf) < self.long_term_window:
            self._append_signal_nan(direction=0)
            self._rv_buf.append(float(curr_rv))
            if len(self._rv_buf) > self._rv_buf_max:
                self._rv_buf = self._rv_buf[-self._rv_buf_max :]
            self.prev_data = data
            return

        window_rv = self._rv_buf[-self.long_term_window :]
        long_term_mean = float(np.mean(window_rv))
        spread = float(curr_iv) - long_term_mean

        self._spread_history.append(spread)
        self.long_term_mean_history.append(long_term_mean)
        self.iv_longterm_spread_history.append(spread)

        self._rv_buf.append(float(curr_rv))
        if len(self._rv_buf) > self._rv_buf_max:
            self._rv_buf = self._rv_buf[-self._rv_buf_max :]

        if len(self._spread_history) < self._MIN_SPREAD_OBS:
            self.longterm_direction_signal.append(0)
            self.prev_data = data
            return

        spread_arr = np.asarray(self._spread_history[-self._KDE_ROLLING_WINDOW :], dtype=float)
        if np.sum(np.isfinite(spread_arr)) < 3:
            self.longterm_direction_signal.append(0)
            self.prev_data = data
            return

        kde_signal = self._kde_cdf_signal(spread_arr)
        if not np.isfinite(kde_signal):
            self.longterm_direction_signal.append(0)
            self.prev_data = data
            return

        if kde_signal > self.entry_threshold:
            direction = -1
        elif kde_signal < -self.entry_threshold:
            direction = 1
        else:
            direction = 0

        self.longterm_direction_signal.append(direction)

        target = 0
        if direction > 0:
            target = 1
        elif direction < 0 and self.allow_short:
            target = -1

        curr_pos = self.num_options
        if curr_pos == 0:
            if target != 0:
                if target > 0:
                    self.long_position(data)
                else:
                    self.short_position(data)
        else:
            should_close = (curr_pos == 1 and kde_signal >= 0) or (
                curr_pos == -1 and kde_signal <= 0
            )
            if should_close:
                self.close_position(data)
                if target != 0 and target != curr_pos:
                    if target > 0:
                        self.long_position(data)
                    else:
                        self.short_position(data)

        if self.num_options != 0 and self.delta_hedge:
            self.rehedge(data)

        self.prev_data = data

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _trade_underlying(self, target_underlying, spot_price):
        curr = int(self.num_underlying)
        target = int(np.rint(target_underlying))
        trade_qty = target - curr
        if trade_qty == 0:
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

        trade_notional = abs(trade_qty) * spot
        slippage_cost = self.slippage_rate * trade_notional
        self._book_trading_cost(slippage_cost, trade_notional=trade_notional)
        return realized

    def long_position(self, data):
        self.num_options = 1
        straddle_price = data["Call_Close"] + data["Put_Close"]
        self.entry_straddle_price = straddle_price
        if self.delta_hedge:
            self._trade_underlying(-self.option_lot_size * data["Straddle_Delta"], data["Stock_Close"])
        option_notional = self.option_lot_size * float(straddle_price)
        self._book_trading_cost(self.slippage_rate * option_notional, trade_notional=option_notional)
        self._log_transaction(data, "long")
        if self.delta_hedge:
            self._rehedge_update_anchor(data)

    def short_position(self, data):
        self.num_options = -1
        straddle_price = data["Call_Close"] + data["Put_Close"]
        self.entry_straddle_price = straddle_price
        if self.delta_hedge:
            self._trade_underlying(self.option_lot_size * data["Straddle_Delta"], data["Stock_Close"])
        option_notional = self.option_lot_size * float(straddle_price)
        self._book_trading_cost(self.slippage_rate * option_notional, trade_notional=option_notional)
        self._log_transaction(data, "short")
        if self.delta_hedge:
            self._rehedge_update_anchor(data)

    def close_position(self, data=None):
        was_open = self.num_options != 0
        lots_to_close = abs(int(self.num_options))
        hedge_realized = 0.0
        if was_open and data is not None and self.delta_hedge:
            hedge_realized = self._trade_underlying(0.0, data["Stock_Close"])
        if was_open and data is not None:
            straddle_price = float(data["Call_Close"] + data["Put_Close"])
            option_notional = lots_to_close * self.option_lot_size * straddle_price
            self._book_trading_cost(self.slippage_rate * option_notional, trade_notional=option_notional)
        self.num_options = 0
        self.num_underlying = 0
        self.entry_straddle_price = 0.0
        if was_open:
            self._rehedge_clear_anchor()
        if was_open and data is not None:
            self._log_transaction(data, "close", earned=hedge_realized)

    def rehedge(self, data):
        net_delta = self._portfolio_net_delta(data)
        if self._rehedge_should_trigger(net_delta, data):
            target_underlying = -self.num_options * self.option_lot_size * self._float_greek(
                data.get("Straddle_Delta")
            )
            hedge_realized = self._trade_underlying(target_underlying, data["Stock_Close"])
            self._rehedge_update_anchor(data)
            self._log_transaction(data, "rehedge", earned=hedge_realized)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_result(self):
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
            "longterm_direction_signal": self.longterm_direction_signal,
            "long_term_mean_rv": self.long_term_mean_history,
            "long_term_mean_iv": self.long_term_mean_history,
            "iv_longterm_spread": self.iv_longterm_spread_history,
        }

    def regime_attribution_summary(self, include_flat=False):
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
