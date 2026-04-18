import numpy as np
import pandas as pd
from pathlib import Path


class Agent_Myself:
    """
    **Entry (same idea as ``Agent_LongTerm``):** rolling mean of past
    ``long_term_window`` straddle IVs (today excluded) vs today's IV; rolling
    z-score of that spread. Open a **long straddle** when ``z < -entry_threshold``
    (cheap vs long-run IV). This agent does not short vol.

    **Exit:** each leg independently — ``stop_loss_pct`` / ``stop_profit_pct``
    while both legs are on; if one leg is already closed, the remaining leg
    exits at **breakeven** (mark back to entry). ``Force_Close`` like other agents.
    """

    _MIN_SPREAD_OBS = 20
    _Z_ROLLING_WINDOW = 20

    def __init__(
        self,
        display_name="Agent_Myself",
        entry_threshold=0.5,
        long_term_window=126,
        stop_loss_pct=0.30,
        stop_profit_pct=0.30,
        delta_hedge=True,
        rehedge_threshold=0.05,
    ):
        self.display_name = display_name
        self.entry_threshold = max(float(entry_threshold), 0.0)
        self.long_term_window = max(int(long_term_window), 5)
        self.stop_loss_pct = max(float(stop_loss_pct), 0.0)
        self.stop_profit_pct = max(float(stop_profit_pct), 0.0)
        self.delta_hedge = delta_hedge
        self.rehedge_threshold = rehedge_threshold
        self.option_lot_size = 100

        self.num_options = 0
        self.num_underlying = 0
        self.entry_straddle_price = 0.0
        self._hedge_avg_price = 0.0
        self._realized_hedge_pnl = 0.0
        self.prev_data = None

        self._call_open = False
        self._put_open = False
        self._entry_call_px = np.nan
        self._entry_put_px = np.nan

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
        self.myself_position_signal = []
        self._iv_buf = []
        self._iv_buf_max = max(self.long_term_window * 4, 512)
        self._spread_history = []
        self.longterm_direction_signal = []
        self.long_term_mean_history = []
        self.iv_longterm_spread_history = []

        self._init_trade_log()

    # ------------------------------------------------------------------
    # Logging
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

    def _f(self, v):
        return float(v) if pd.notna(v) and np.isfinite(v) else 0.0

    def _qc(self):
        return self.option_lot_size if self._call_open else 0

    def _qp(self):
        return self.option_lot_size if self._put_open else 0

    def _option_book_value(self, row):
        if row is None:
            return 0.0
        return self._qc() * self._f(row.get("Call_Close")) + self._qp() * self._f(row.get("Put_Close"))

    def _target_hedge_shares(self, data):
        return -(
            self._qc() * self._f(data.get("Call_Delta"))
            + self._qp() * self._f(data.get("Put_Delta"))
        )

    def _leg_greeks(self, prev_row, prefix):
        """prefix: 'Straddle', 'Call', or 'Put' — column names in dataset."""
        return {
            "Delta": self._f(prev_row.get(f"{prefix}_Delta")),
            "Gamma": self._f(prev_row.get(f"{prefix}_Gamma")),
            "Vega": self._f(prev_row.get(f"{prefix}_Vega")),
            "Theta": self._f(prev_row.get(f"{prefix}_Theta")),
            "Rho": self._f(prev_row.get(f"{prefix}_Rho")),
            "Vanna": self._f(prev_row.get(f"{prefix}_Vanna")),
            "Volga": self._f(prev_row.get(f"{prefix}_Volga")),
        }

    def _combine_leg_greeks_from_row(self, row):
        if self._call_open and self._put_open:
            return self._leg_greeks(row, "Straddle"), self._qc()
        if self._call_open:
            return self._leg_greeks(row, "Call"), self._qc()
        if self._put_open:
            return self._leg_greeks(row, "Put"), self._qp()
        return None, 0

    def _d_sigma_legs(self, data, prev_row):
        if self._call_open and self._put_open:
            return self._f(data.get("Straddle_imp_vol")) - self._f(prev_row.get("Straddle_imp_vol"))
        if self._call_open:
            return self._f(data.get("Call_imp_vol")) - self._f(prev_row.get("Call_imp_vol"))
        if self._put_open:
            return self._f(data.get("Put_imp_vol")) - self._f(prev_row.get("Put_imp_vol"))
        return 0.0

    # ------------------------------------------------------------------
    # PnL & Greeks
    # ------------------------------------------------------------------

    def _compute_daily_pnl(self, data):
        self.position_state_for_pnl.append(self.num_options)

        if (not self._call_open and not self._put_open) or self.prev_data is None:
            self._append_zeros()
            return

        prev_row = self.prev_data
        dS = data["Stock_Close"] - prev_row["Stock_Close"]
        div = data.get("Stock_Dividends", 0.0)
        if pd.isna(div):
            div = 0.0
        dS_adj = dS + div

        prev_val = self._option_book_value(prev_row)
        curr_val = self._option_book_value(data)
        qc, qp = self._qc(), self._qp()
        daily_pnl = curr_val - prev_val + self.num_underlying * dS_adj

        yesterday_exposure = abs(prev_val) + abs(self.num_underlying * self._f(prev_row["Stock_Close"]))
        simple_return = daily_pnl / yesterday_exposure if yesterday_exposure > 0 else 0.0
        safe_simple_return = max(simple_return, -0.999999999)
        daily_return = np.log1p(safe_simple_return)

        self.PnL.append(daily_pnl)
        self.Return.append(daily_return)

        d_sigma = self._d_sigma_legs(data, prev_row)
        dr = self._f(data.get("r")) - self._f(prev_row.get("r"))
        prev_date = pd.to_datetime(prev_row["Date"]).date().isoformat()
        curr_date = pd.to_datetime(data["Date"]).date().isoformat()
        dt_days = np.busday_count(prev_date, curr_date)
        dt = (dt_days if dt_days > 0 else 1) / 252.0

        g, q = self._combine_leg_greeks_from_row(prev_row)
        if g is None or q <= 0:
            self._store_attr(
                actual_delta_exp=0.0,
                delta_pnl=0.0,
                gamma_pnl=0.0,
                vega_pnl=0.0,
                theta_pnl=0.0,
                vanna_pnl=0.0,
                volga_pnl=0.0,
                rho_pnl=0.0,
                hedge_pnl=self.num_underlying * dS_adj,
                residual_pnl=daily_pnl - self.num_underlying * dS_adj,
            )
            return

        h = self.num_underlying
        prev_delta = g["Delta"]
        curr_g, _ = self._combine_leg_greeks_from_row(data)
        if curr_g is None:
            curr_g = g
        curr_delta = curr_g["Delta"]
        effective_delta = prev_delta
        if self.delta_hedge:
            end_net_delta = q * curr_delta + h
            if abs(end_net_delta) > self.rehedge_threshold:
                effective_delta = 0.5 * (prev_delta + curr_delta)

        delta_pnl = q * effective_delta * dS
        gamma_pnl = 0.5 * q * g["Gamma"] * (dS**2)
        vega_pnl = q * g["Vega"] * d_sigma
        vanna_pnl = q * g["Vanna"] * dS * d_sigma
        volga_pnl = 0.5 * q * g["Volga"] * (d_sigma**2)
        theta_pnl = q * g["Theta"] * dt
        rho_pnl = q * g["Rho"] * dr * 100.0
        hedge_pnl = h * dS_adj

        explained = delta_pnl + gamma_pnl + vega_pnl + vanna_pnl + volga_pnl + theta_pnl + rho_pnl + hedge_pnl
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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _sync_num_options_flags(self):
        self.num_options = 1 if (self._call_open or self._put_open) else 0
        if self.num_options == 0:
            self.entry_straddle_price = 0.0

    def _open_straddle(self, data):
        self._call_open = True
        self._put_open = True
        self._entry_call_px = self._f(data["Call_Close"])
        self._entry_put_px = self._f(data["Put_Close"])
        self.entry_straddle_price = self._entry_call_px + self._entry_put_px
        self._sync_num_options_flags()
        if self.delta_hedge:
            self._trade_underlying(self._target_hedge_shares(data), data["Stock_Close"])
        self._log_transaction(data, "long straddle")

    def _close_call_leg(self, data, reason):
        if not self._call_open:
            return
        self._call_open = False
        self._entry_call_px = np.nan
        self._sync_num_options_flags()
        if self.delta_hedge:
            self._trade_underlying(self._target_hedge_shares(data), data["Stock_Close"])
        if not self._put_open:
            self._flatten_hedge_book(data)
        self._log_transaction(data, f"close call ({reason})")

    def _close_put_leg(self, data, reason):
        if not self._put_open:
            return
        self._put_open = False
        self._entry_put_px = np.nan
        self._sync_num_options_flags()
        if self.delta_hedge:
            self._trade_underlying(self._target_hedge_shares(data), data["Stock_Close"])
        if not self._call_open:
            self._flatten_hedge_book(data)
        self._log_transaction(data, f"close put ({reason})")

    def _flatten_hedge_book(self, data):
        if self.delta_hedge:
            self._trade_underlying(0.0, data["Stock_Close"])
        self.num_underlying = 0
        self._hedge_avg_price = 0.0

    def _stop_hit(self, mark, entry, loss_pct, profit_pct):
        if not np.isfinite(mark) or not np.isfinite(entry) or entry <= 0:
            return None
        r = (mark - entry) / entry
        if r <= -loss_pct:
            return "stop_loss"
        if r >= profit_pct:
            return "take_profit"
        return None

    def _stop_hit_remaining_leg(self, mark, entry, loss_pct):
        """
        For the last open leg: stop-loss is unchanged; take-profit is
        *breakeven* (mark recovers to entry / buy price), i.e. 0% profit target.
        """
        if not np.isfinite(mark) or not np.isfinite(entry) or entry <= 0:
            return None
        r = (mark - entry) / entry
        if r <= -loss_pct:
            return "stop_loss"
        if mark >= entry - 1e-12:
            return "breakeven"
        return None

    def _both_legs_on(self):
        return self._call_open and self._put_open

    def _append_longterm_nan(self):
        self.longterm_direction_signal.append(0)
        self.long_term_mean_history.append(np.nan)
        self.iv_longterm_spread_history.append(np.nan)

    def _update_longterm_signal(self, data):
        """
        Same IV buffer / spread / z-score construction as ``Agent_LongTerm``.
        Returns ``True`` if long-vol signal fires (``z < -entry_threshold``),
        ``False`` if not, ``None`` if z is not available this day.
        """
        curr_iv = data.get("Straddle_imp_vol", np.nan)
        if pd.isna(curr_iv) or not np.isfinite(curr_iv):
            self._append_longterm_nan()
            return None

        if len(self._iv_buf) < self.long_term_window:
            self._iv_buf.append(float(curr_iv))
            if len(self._iv_buf) > self._iv_buf_max:
                self._iv_buf = self._iv_buf[-self._iv_buf_max :]
            self._append_longterm_nan()
            return None

        window_iv = self._iv_buf[-self.long_term_window :]
        long_term_mean = float(np.mean(window_iv))
        spread = float(curr_iv) - long_term_mean
        self._spread_history.append(spread)
        self.long_term_mean_history.append(long_term_mean)
        self.iv_longterm_spread_history.append(spread)

        self._iv_buf.append(float(curr_iv))
        if len(self._iv_buf) > self._iv_buf_max:
            self._iv_buf = self._iv_buf[-self._iv_buf_max :]

        if len(self._spread_history) < self._MIN_SPREAD_OBS:
            self.longterm_direction_signal.append(0)
            return None

        spread_arr = np.asarray(self._spread_history[-self._Z_ROLLING_WINDOW :], dtype=float)
        spread_arr = spread_arr[np.isfinite(spread_arr)]
        if spread_arr.size < 2:
            self.longterm_direction_signal.append(0)
            return None

        mu = float(np.mean(spread_arr))
        sigma_z = float(np.std(spread_arr, ddof=1))
        if sigma_z < 1e-12:
            self.longterm_direction_signal.append(0)
            return None

        z = (spread - mu) / sigma_z
        if z > self.entry_threshold:
            direction = -1
        elif z < -self.entry_threshold:
            direction = 1
        else:
            direction = 0

        self.longterm_direction_signal.append(direction)
        return direction == 1

    def trade(self, data):
        self._compute_daily_pnl(data)

        if bool(data.get("Force_Close", False)):
            self._force_close_all(data)
            self.myself_position_signal.append(0)
            self._append_longterm_nan()
            self.prev_data = data
            return

        if self._call_open or self._put_open:
            cpx = self._f(data.get("Call_Close"))
            ppx = self._f(data.get("Put_Close"))
            if self._call_open:
                if self._both_legs_on():
                    why = self._stop_hit(
                        cpx, self._entry_call_px, self.stop_loss_pct, self.stop_profit_pct
                    )
                else:
                    why = self._stop_hit_remaining_leg(
                        cpx, self._entry_call_px, self.stop_loss_pct
                    )
                if why:
                    self._close_call_leg(data, why)
            if self._put_open:
                if self._both_legs_on():
                    why = self._stop_hit(
                        ppx, self._entry_put_px, self.stop_loss_pct, self.stop_profit_pct
                    )
                else:
                    why = self._stop_hit_remaining_leg(
                        ppx, self._entry_put_px, self.stop_loss_pct
                    )
                if why:
                    self._close_put_leg(data, why)

            if (self._call_open or self._put_open) and self.delta_hedge:
                self.rehedge(data)

        want_long = self._update_longterm_signal(data)
        if (
            not self._call_open
            and not self._put_open
            and want_long is True
        ):
            self._open_straddle(data)
            if self.delta_hedge:
                self.rehedge(data)

        self.myself_position_signal.append(1 if (self._call_open or self._put_open) else 0)
        self.prev_data = data

    def _force_close_all(self, data):
        hedge_realized = 0.0
        if self.delta_hedge:
            hedge_realized = self._trade_underlying(0.0, data["Stock_Close"])
        self._call_open = False
        self._put_open = False
        self._entry_call_px = np.nan
        self._entry_put_px = np.nan
        self.num_options = 0
        self.num_underlying = 0
        self.entry_straddle_price = 0.0
        self._hedge_avg_price = 0.0
        self._log_transaction(data, "force_close", earned=hedge_realized)

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
        return realized

    def rehedge(self, data):
        net_delta = self._qc() * self._f(data.get("Call_Delta")) + self._qp() * self._f(
            data.get("Put_Delta")
        ) + self.num_underlying
        if abs(net_delta) > self.rehedge_threshold:
            hedge_realized = self._trade_underlying(self._target_hedge_shares(data), data["Stock_Close"])
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
            "myself_position_signal": self.myself_position_signal,
            "longterm_direction_signal": self.longterm_direction_signal,
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
