import numpy as np
import pandas as pd
from pathlib import Path
from warnings import catch_warnings, simplefilter

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class Agent_Vote:
    _MIN_SPREAD_OBS = 20
    _Z_ROLLING_WINDOW = 20
    _LOOKBACK = 20
    _REFIT_EVERY = 5
    _DEFAULT_FORECAST_DAYS = 30

    def __init__(
        self,
        display_name="Agent_Vote",
        entry_threshold=0.5,
        longterm_rv_weight=0.5,
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
        self.hard_threshold_k = max(float(entry_threshold), 0.0)
        self.longterm_rv_weight = float(longterm_rv_weight)
        self.long_term_window = max(int(long_term_window), 5)
        self.trading_days_per_year = 252
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
        self._longterm_spread_history = []
        self._garch_spread_history = []
        self._stock_close_buf = []
        self._omega = np.nan
        self._alpha = np.nan
        self._gamma = np.nan
        self._beta = np.nan
        self._nu = np.nan
        self._h = np.nan
        self._days_since_fit = self._REFIT_EVERY
        self.hardthreshold_direction_signal = []
        self.longterm_direction_signal = []
        self.garch_direction_signal = []
        self.overall_direction_signal = []
        self.long_term_mean_history = []
        self.iv_longterm_spread_history = []
        self.garch_rv_forecast_history = []
        self.garch_zscore_history = []
        self.longterm_zscore_history = []
        self._init_trade_log()

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

    def _append_signal_nan(self):
        self.hardthreshold_direction_signal.append(0)
        self.longterm_direction_signal.append(0)
        self.garch_direction_signal.append(0)
        self.overall_direction_signal.append(0)
        self.long_term_mean_history.append(np.nan)
        self.iv_longterm_spread_history.append(np.nan)
        self.garch_rv_forecast_history.append(np.nan)
        self.garch_zscore_history.append(np.nan)
        self.longterm_zscore_history.append(np.nan)

    def _fit_garch(self):
        px = np.asarray(self._stock_close_buf[-(self._LOOKBACK + 1) :], dtype=float)
        px = px[np.isfinite(px)]
        if px.size < self._LOOKBACK + 1:
            return False

        log_ret = np.diff(np.log(px))
        log_ret = log_ret[np.isfinite(log_ret)]
        if log_ret.size < 20:
            return False

        log_ret_pct = log_ret * 100.0

        if arch_model is None:
            var = float(np.var(log_ret_pct))
            for r in log_ret_pct:
                var = 0.94 * var + 0.06 * r * r
            self._omega = 0.0
            self._alpha = 0.06
            self._gamma = 0.0
            self._beta = 0.94
            self._nu = np.nan
            self._h = var
            self._days_since_fit = 0
            return True

        try:
            with catch_warnings():
                simplefilter("ignore")
                model = arch_model(
                    log_ret_pct,
                    mean="Zero",
                    vol="GARCH",
                    p=1,
                    o=1,
                    q=1,
                    dist="t",
                    rescale=False,
                )
                fit = model.fit(disp="off")
            self._omega = float(fit.params.get("omega", 0.0))
            self._alpha = float(fit.params.get("alpha[1]", 0.06))
            self._gamma = float(fit.params.get("gamma[1]", 0.0))
            self._beta = float(fit.params.get("beta[1]", 0.94))
            self._nu = float(fit.params.get("nu", np.nan))
            cond_vol = fit.conditional_volatility
            self._h = (
                float(cond_vol.iloc[-1]) ** 2
                if len(cond_vol) > 0
                else float(np.var(log_ret_pct))
            )
            self._days_since_fit = 0
            return True
        except Exception:
            return False

    def _update_h(self, daily_log_return):
        if np.isnan(self._omega):
            return
        eps_pct = daily_log_return * 100.0
        asym = self._gamma if eps_pct < 0 else 0.0
        self._h = self._omega + (self._alpha + asym) * eps_pct**2 + self._beta * self._h

    def _days_to_strike(self, data):
        exp_dt = pd.to_datetime(data.get("Expiry", pd.NaT), errors="coerce")
        date_dt = pd.to_datetime(data.get("Date", pd.NaT), errors="coerce")
        if pd.isna(exp_dt) or pd.isna(date_dt):
            return int(self._DEFAULT_FORECAST_DAYS)
        as_of = date_dt.date() if hasattr(date_dt, "date") else pd.Timestamp(date_dt).date()
        exp = exp_dt.date() if hasattr(exp_dt, "date") else pd.Timestamp(exp_dt).date()
        bd = int(np.busday_count(np.datetime64(as_of), np.datetime64(exp)))
        return max(bd, 1)

    def _garch_rv_annualized(self, horizon_days):
        if np.isnan(self._h) or self._h <= 0:
            return np.nan
        h_step = float(self._h)
        omega = float(self._omega) if np.isfinite(self._omega) else 0.0
        alpha = float(self._alpha) if np.isfinite(self._alpha) else 0.0
        gamma = float(self._gamma) if np.isfinite(self._gamma) else 0.0
        beta = float(self._beta) if np.isfinite(self._beta) else 0.0
        phi = alpha + beta + 0.5 * gamma
        phi = min(max(phi, 0.0), 1.2)

        horizon = int(horizon_days)
        h_path = []
        for _ in range(horizon):
            h_step = omega + phi * h_step
            if not np.isfinite(h_step) or h_step <= 0:
                return np.nan
            h_path.append(h_step)

        avg_daily_var_pct = float(np.mean(h_path))
        sigma_daily = np.sqrt(avg_daily_var_pct) / 100.0
        return sigma_daily * np.sqrt(self.trading_days_per_year)

    # ------------------------------------------------------------------
    # Signal: current IV vs rolling mean of prior RV values
    # ------------------------------------------------------------------

    def trade(self, data):
        self._compute_daily_pnl(data)

        curr_spot = data.get("Stock_Close", np.nan)
        if pd.notna(curr_spot):
            self._stock_close_buf.append(float(curr_spot))

        if len(self._stock_close_buf) >= 2:
            ret = np.log(self._stock_close_buf[-1] / self._stock_close_buf[-2])
            if np.isfinite(ret) and not np.isnan(self._omega):
                self._update_h(ret)
            self._days_since_fit += 1

        if (
            self._days_since_fit >= self._REFIT_EVERY
            and len(self._stock_close_buf) >= self._LOOKBACK + 1
        ):
            self._fit_garch()

        if data["Force_Close"]:
            self.close_position(data)
            self._append_signal_nan()
            self.prev_data = data
            return

        curr_iv = data.get("Straddle_imp_vol", np.nan)
        curr_rv = data.get("RV", np.nan)
        if pd.isna(curr_iv) or not np.isfinite(curr_iv):
            self._append_signal_nan()
            self.prev_data = data
            return

        if pd.isna(curr_rv) or not np.isfinite(curr_rv):
            self._append_signal_nan()
            self.prev_data = data
            return

        if len(self._rv_buf) < self.long_term_window:
            self._append_signal_nan()
            self._rv_buf.append(float(curr_rv))
            if len(self._rv_buf) > self._rv_buf_max:
                self._rv_buf = self._rv_buf[-self._rv_buf_max :]
            self.prev_data = data
            return

        hardthreshold_signal = 0
        vrp = data.get("VRP", np.nan)
        vrp_mean = data.get("VRP_20d_mean", np.nan)
        vrp_std = data.get("VRP_20d_std", np.nan)
        if (
            pd.notna(vrp)
            and pd.notna(vrp_mean)
            and pd.notna(vrp_std)
            and np.isfinite(vrp_std)
            and float(vrp_std) > 0
        ):
            if vrp > vrp_mean + self.hard_threshold_k * vrp_std:
                hardthreshold_signal = -1
            elif vrp < vrp_mean - self.hard_threshold_k * vrp_std:
                hardthreshold_signal = 1

        window_rv = self._rv_buf[-self.long_term_window :]
        long_term_mean = float(np.mean(window_rv))
        longterm_spread = float(curr_iv) - long_term_mean
        garch_forecast_rv = self._garch_rv_annualized(self._days_to_strike(data))
        garch_spread = (
            float(curr_iv) - float(garch_forecast_rv) if np.isfinite(garch_forecast_rv) else np.nan
        )

        self._longterm_spread_history.append(longterm_spread)
        if np.isfinite(garch_spread):
            self._garch_spread_history.append(garch_spread)
        self.long_term_mean_history.append(long_term_mean)
        self.iv_longterm_spread_history.append(longterm_spread)
        self.garch_rv_forecast_history.append(
            float(garch_forecast_rv) if np.isfinite(garch_forecast_rv) else np.nan
        )

        self._rv_buf.append(float(curr_rv))
        if len(self._rv_buf) > self._rv_buf_max:
            self._rv_buf = self._rv_buf[-self._rv_buf_max :]

        longterm_signal = 0
        longterm_z = np.nan
        if len(self._longterm_spread_history) >= self._MIN_SPREAD_OBS:
            spread_arr = np.asarray(
                self._longterm_spread_history[-self._Z_ROLLING_WINDOW :], dtype=float
            )
            spread_arr = spread_arr[np.isfinite(spread_arr)]
            if spread_arr.size >= 2:
                spread_mu = float(np.mean(spread_arr))
                spread_sigma = float(np.std(spread_arr, ddof=1))
                if spread_sigma >= 1e-12:
                    longterm_z = (longterm_spread - spread_mu) / spread_sigma
                    if longterm_z > self.entry_threshold:
                        longterm_signal = -1
                    elif longterm_z < -self.entry_threshold:
                        longterm_signal = 1

        garch_signal = 0
        garch_z = np.nan
        if len(self._garch_spread_history) >= self._MIN_SPREAD_OBS and np.isfinite(garch_spread):
            garch_arr = np.asarray(
                self._garch_spread_history[-self._Z_ROLLING_WINDOW :], dtype=float
            )
            garch_arr = garch_arr[np.isfinite(garch_arr)]
            if garch_arr.size >= 2:
                garch_mu = float(np.mean(garch_arr))
                garch_sigma = float(np.std(garch_arr, ddof=1))
                if garch_sigma >= 1e-12:
                    garch_z = (garch_spread - garch_mu) / garch_sigma
                    if garch_z > self.entry_threshold:
                        garch_signal = -1
                    elif garch_z < -self.entry_threshold:
                        garch_signal = 1

        overall_vote = hardthreshold_signal + longterm_signal + garch_signal
        if overall_vote <= -1:
            direction = -1
        elif overall_vote >= 1:
            direction = 1
        else:
            direction = 0

        self.hardthreshold_direction_signal.append(hardthreshold_signal)
        self.longterm_direction_signal.append(longterm_signal)
        self.garch_direction_signal.append(garch_signal)
        self.overall_direction_signal.append(direction)
        self.longterm_zscore_history.append(longterm_z)
        self.garch_zscore_history.append(garch_z)

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
            if target == 0:
                self.close_position(data)
            elif target != curr_pos:
                self.close_position(data)
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
            "hardthreshold_direction_signal": self.hardthreshold_direction_signal,
            "longterm_direction_signal": self.longterm_direction_signal,
            "garch_direction_signal": self.garch_direction_signal,
            "overall_direction_signal": self.overall_direction_signal,
            "long_term_mean_rv": self.long_term_mean_history,
            "long_term_mean_iv": self.long_term_mean_history,
            "iv_longterm_spread": self.iv_longterm_spread_history,
            "garch_rv_forecast_annualized": self.garch_rv_forecast_history,
            "longterm_zscore": self.longterm_zscore_history,
            "garch_zscore": self.garch_zscore_history,
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
