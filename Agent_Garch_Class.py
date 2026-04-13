import numpy as np
import pandas as pd
from pathlib import Path
from warnings import catch_warnings, simplefilter

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class Agent_Garch:
    _LOOKBACK = 20
    _REFIT_EVERY = 5
    _MIN_SPREAD_OBS = 20

    def __init__(
        self,
        display_name="Agent_Garch",
        allow_short=True,
        delta_hedge=True,
        rehedge_threshold=0.05,
        z_entry=1.0,
    ):
        self.display_name = display_name
        self.delta_hedge = delta_hedge
        self.rehedge_threshold = rehedge_threshold
        self.allow_short = allow_short
        self.z_entry = max(float(z_entry), 0.0)
        self.trading_days_per_year = 252

        self.num_options = 0
        self.num_underlying = 0
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

        self._omega = np.nan
        self._alpha = np.nan
        self._beta = np.nan
        self._h = np.nan
        self._days_since_fit = self._REFIT_EVERY
        self._stock_close_buf = []
        self._spread_history = []

        self.rv_forecast_history = []
        self.iv_rv_spread_history = []
        self.garch_direction_signal = []
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
        for lst in (
            self.PnL, self.Return, self.actual_delta,
            self.delta_attribute, self.gamma_attribute,
            self.vega_attribute, self.theta_attribute,
            self.vanna_attribute, self.volga_attribute,
            self.rho_attribute, self.residual,
        ):
            lst.append(0.0)

    # ------------------------------------------------------------------
    # GARCH engine
    # ------------------------------------------------------------------

    def _fit_garch(self):
        px = np.asarray(self._stock_close_buf[-(self._LOOKBACK + 1):], dtype=float)
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
            self._beta = 0.94
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
                    p=1, q=1,
                    dist="normal",
                    rescale=False,
                )
                fit = model.fit(disp="off")
            self._omega = float(fit.params.get("omega", 0.0))
            self._alpha = float(fit.params.get("alpha[1]", 0.06))
            self._beta = float(fit.params.get("beta[1]", 0.94))
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
        self._h = self._omega + self._alpha * eps_pct**2 + self._beta * self._h

    def _garch_rv_annualized(self):
        if np.isnan(self._h) or self._h <= 0:
            return np.nan
        sigma_daily = np.sqrt(self._h) / 100.0
        return sigma_daily * np.sqrt(self.trading_days_per_year)

    # ------------------------------------------------------------------
    # Trade logic
    # ------------------------------------------------------------------

    def trade(self, data):
        self._compute_daily_pnl(data)

        curr_spot = data.get("Stock_Close", np.nan)
        curr_iv = data.get("Straddle_imp_vol", np.nan)

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
            self.garch_direction_signal.append(0)
            self.rv_forecast_history.append(np.nan)
            self.iv_rv_spread_history.append(np.nan)
            self.prev_data = data
            return

        garch_rv = self._garch_rv_annualized()

        if pd.isna(curr_iv) or not np.isfinite(garch_rv):
            self.garch_direction_signal.append(0)
            self.rv_forecast_history.append(
                garch_rv if np.isfinite(garch_rv) else np.nan
            )
            self.iv_rv_spread_history.append(np.nan)
            self.prev_data = data
            return

        spread = float(curr_iv) - garch_rv
        self._spread_history.append(spread)
        self.rv_forecast_history.append(float(garch_rv))
        self.iv_rv_spread_history.append(spread)

        if len(self._spread_history) < self._MIN_SPREAD_OBS:
            self.garch_direction_signal.append(0)
            self.prev_data = data
            return

        spread_arr = np.asarray(self._spread_history, dtype=float)
        spread_arr = spread_arr[np.isfinite(spread_arr)]
        mu = float(np.mean(spread_arr))
        sigma = float(np.std(spread_arr, ddof=1))
        if sigma < 1e-12:
            self.garch_direction_signal.append(0)
            self.prev_data = data
            return

        z = (spread - mu) / sigma

        if z > self.z_entry:
            direction = -1
        elif z < -self.z_entry:
            direction = 1
        else:
            direction = 0

        self.garch_direction_signal.append(direction)

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
            should_close = (curr_pos == 1 and z >= 0) or (curr_pos == -1 and z <= 0)
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
            "garch_direction_signal": self.garch_direction_signal,
            "rv_forecast_annualized": self.rv_forecast_history,
            "iv_rv_spread": self.iv_rv_spread_history,
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
        summary.index = summary.index.map(
            {-1: "short_straddle", 0: "flat", 1: "long_straddle"}
        )
        return summary
