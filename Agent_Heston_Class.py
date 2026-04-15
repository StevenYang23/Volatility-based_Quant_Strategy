import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.special import erf

try:
    import QuantLib as ql
except ImportError:
    ql = None


class Agent_Heston:
    _MIN_SPREAD_OBS = 20
    _Z_ROLLING_WINDOW = 20
    _INTEGRAL_UPPER = 50.0
    _INTEGRAL_POINTS = 128
    _FORECAST_HORIZON_DAYS = 30
    _FULL_CALIBRATE_EVERY = 5

    def __init__(
        self,
        display_name="Agent_Heston",
        allow_short=True,
        delta_hedge=True,
        rehedge_threshold=0.05,
        z_entry=0.5,
        z_window=20,
        full_calibrate_every=5,
        rho_tail_threshold=-0.8,
        max_full_calib_failures=3,
    ):
        self.display_name = display_name
        self.delta_hedge = delta_hedge
        self.rehedge_threshold = rehedge_threshold
        self.allow_short = allow_short
        self.z_entry = max(float(z_entry), 0.0)
        self.z_window = max(int(z_window), 5)
        self.full_calibrate_every = max(int(full_calibrate_every), 1)
        self.rho_tail_threshold = float(rho_tail_threshold)
        self.max_full_calib_failures = max(int(max_full_calib_failures), 1)
        self.trading_days_per_year = 252
        self.option_lot_size = 100

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

        # Heston calibration state
        self._last_params = None  # [v0, kappa, theta, sigma, rho]
        self._days_since_full_calib = self._FULL_CALIBRATE_EVERY
        self._consecutive_full_calib_failures = 0
        self._phi_grid = np.linspace(1e-6, self._INTEGRAL_UPPER, self._INTEGRAL_POINTS)
        self._spread_history = []
        self.heston_direction_signal = []
        self.long_run_vol_history = []  # sqrt(theta)
        self.iv_longrun_spread_history = []
        self.calibrated_params_history = []
        self.model_iv_history = []
        self.feller_status_history = []
        self.tail_risk_blocked_history = []
        self.calibration_mode_history = []
        self.full_calib_failure_streak_history = []
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

        option_units = self.num_options * self.option_lot_size
        daily_pnl = option_units * (straddle_price - prev_straddle) + self.num_underlying * dS_adj

        yesterday_exposure = abs(option_units * prev_straddle) + abs(
            self.num_underlying * self.prev_data["Stock_Close"]
        )
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
            if abs(end_net_delta) > self.rehedge_threshold:
                # Rehedge day: use midpoint delta to reduce attribution leakage to residual.
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

    def _append_signal_nan(self, direction=0):
        self.heston_direction_signal.append(direction)
        self.long_run_vol_history.append(np.nan)
        self.iv_longrun_spread_history.append(np.nan)
        self.calibrated_params_history.append(np.nan)
        self.model_iv_history.append(np.nan)
        self.feller_status_history.append(np.nan)
        self.tail_risk_blocked_history.append(False)
        self.calibration_mode_history.append("none")
        self.full_calib_failure_streak_history.append(self._consecutive_full_calib_failures)

    @staticmethod
    def _feller_ratio(kappa, theta, sigma):
        sigma2 = float(sigma) * float(sigma)
        if sigma2 <= 0:
            return np.nan
        return (2.0 * float(kappa) * float(theta)) / sigma2

    @staticmethod
    def _parse_tenor_days(col_name):
        key = str(col_name).lower().replace("-", "_")
        tenor_map = {
            "1m": 30,
            "2m": 60,
            "3m": 90,
            "6m": 180,
            "9m": 270,
            "12m": 365,
            "1y": 365,
            "2y": 730,
            "3y": 1095,
            "5y": 1825,
            "7y": 2555,
            "10y": 3650,
        }
        for token, days in tenor_map.items():
            if token in key:
                return days
        return None

    def _resolve_rate(self, data, T):
        base_r = float(data.get("r", np.nan))
        term_days = max(float(T) * self.trading_days_per_year, 1.0)
        curve = []
        for key in data.keys():
            k_lower = str(key).lower()
            if not (("r_" in k_lower) or ("rate_" in k_lower) or ("yield_" in k_lower)):
                continue
            days = self._parse_tenor_days(key)
            if days is None:
                continue
            val = float(data.get(key, np.nan))
            if np.isfinite(val):
                curve.append((days, val))
        if len(curve) < 2:
            return base_r if np.isfinite(base_r) else 0.0

        curve.sort(key=lambda x: x[0])
        tenors = np.array([x[0] for x in curve], dtype=float)
        rates = np.array([x[1] for x in curve], dtype=float)
        interp_r = float(np.interp(term_days, tenors, rates))
        if np.isfinite(interp_r):
            return interp_r
        return base_r if np.isfinite(base_r) else 0.0

    # ------------------------------------------------------------------
    # Heston engine
    # ------------------------------------------------------------------
    @staticmethod
    def _norm_cdf(x):
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))

    def _bs_call_price(self, S, K, T, r, sigma):
        if not np.isfinite(S) or not np.isfinite(K) or S <= 0 or K <= 0:
            return np.nan
        if T <= 0:
            return max(S - K, 0.0)
        sigma = max(float(sigma), 1e-8)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return float(S * self._norm_cdf(d1) - K * np.exp(-r * T) * self._norm_cdf(d2))

    def _bs_put_price(self, S, K, T, r, sigma):
        c = self._bs_call_price(S, K, T, r, sigma)
        if not np.isfinite(c):
            return np.nan
        return float(c - S + K * np.exp(-r * T))

    def _implied_vol_call(self, call_price, S, K, T, r):
        if not np.isfinite(call_price) or not np.isfinite(S) or not np.isfinite(K):
            return np.nan
        if T <= 0 or S <= 0 or K <= 0:
            return np.nan
        lower = max(0.0, S - K * np.exp(-r * T))
        upper = S
        if call_price <= lower + 1e-12:
            return 1e-8
        if call_price >= upper - 1e-12:
            return np.nan

        lo, hi = 1e-6, 5.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            c_mid = self._bs_call_price(S, K, T, r, mid)
            if not np.isfinite(c_mid):
                return np.nan
            if c_mid > call_price:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def _heston_cf(self, phi, S, v0, kappa, theta, sigma, rho, T, r, j):
        """Risk-neutral Heston characteristic function component for P1/P2."""
        phi = np.asarray(phi, dtype=np.complex128)
        i = 1j
        x = np.log(S)
        u = 0.5 if j == 1 else -0.5
        b = kappa - rho * sigma if j == 1 else kappa
        a = kappa * theta

        d = np.sqrt((rho * sigma * i * phi - b) ** 2 - sigma**2 * (2.0 * u * i * phi - phi**2))
        g = (b - rho * sigma * i * phi + d) / (b - rho * sigma * i * phi - d)

        # "Little Heston Trap" representation for better numerical stability.
        c = 1.0 / g
        exp_neg_dT = np.exp(-d * T)
        one_minus_c = 1.0 - c
        one_minus_c_exp = 1.0 - c * exp_neg_dT
        # Guard against tiny denominators in complex plane.
        one_minus_c = np.where(np.abs(one_minus_c) < 1e-14, one_minus_c + 1e-14, one_minus_c)
        one_minus_c_exp = np.where(
            np.abs(one_minus_c_exp) < 1e-14, one_minus_c_exp + 1e-14, one_minus_c_exp
        )

        C = r * i * phi * T + (a / sigma**2) * (
            (b - rho * sigma * i * phi - d) * T - 2.0 * np.log(one_minus_c_exp / one_minus_c)
        )
        D = ((b - rho * sigma * i * phi - d) / sigma**2) * ((1.0 - exp_neg_dT) / one_minus_c_exp)
        return np.exp(C + D * v0 + i * phi * x)

    def _heston_prob(self, S, K, T, r, params, j, phi_grid=None):
        v0, kappa, theta, sigma, rho = params
        lnK = np.log(K)
        phi = self._phi_grid if phi_grid is None else np.asarray(phi_grid, dtype=float)
        if phi.size < 2:
            return np.nan
        cf = self._heston_cf(phi, S, v0, kappa, theta, sigma, rho, T, r, j)
        integrand = np.real(np.exp(-1j * phi * lnK) * cf / (1j * phi))
        if not np.all(np.isfinite(integrand)):
            return np.nan
        integral_val = np.trapezoid(integrand, phi)
        return 0.5 + (1.0 / np.pi) * integral_val

    def _heston_call_price_numpy(self, S, K, T, r, params):
        if T <= 0:
            return max(S - K, 0.0)
        v0, kappa, theta, sigma, rho = params
        if min(v0, kappa, theta, sigma) <= 0 or not (-0.999 < rho < 0.999):
            return np.nan
        p1 = self._heston_prob(S, K, T, r, params, j=1)
        p2 = self._heston_prob(S, K, T, r, params, j=2)
        if not np.isfinite(p1) or not np.isfinite(p2):
            return np.nan
        call = S * p1 - K * np.exp(-r * T) * p2
        return float(call) if np.isfinite(call) else np.nan

    def _heston_call_price_quantlib(self, S, K, T, r, params):
        if ql is None:
            return np.nan
        v0, kappa, theta, sigma, rho = params
        if min(v0, kappa, theta, sigma) <= 0 or not (-0.999 < rho < 0.999):
            return np.nan
        try:
            settlement_date = ql.Date.todaysDate()
            maturity_days = max(int(np.ceil(T * 365.0)), 1)
            maturity_date = settlement_date + maturity_days
            day_count = ql.Actual365Fixed()

            spot = ql.QuoteHandle(ql.SimpleQuote(float(S)))
            risk_free_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, float(r), day_count))
            dividend_ts = ql.YieldTermStructureHandle(ql.FlatForward(settlement_date, 0.0, day_count))

            process = ql.HestonProcess(
                risk_free_ts,
                dividend_ts,
                spot,
                float(v0),
                float(kappa),
                float(theta),
                float(sigma),
                float(rho),
            )
            model = ql.HestonModel(process)
            engine = ql.AnalyticHestonEngine(model)

            payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(engine)
            price = option.NPV()
            return float(price) if np.isfinite(price) else np.nan
        except Exception:
            return np.nan

    def _heston_call_price(self, S, K, T, r, params):
        price = self._heston_call_price_quantlib(S, K, T, r, params)
        if np.isfinite(price):
            return price
        return self._heston_call_price_numpy(S, K, T, r, params)

    def _extract_contract_terms(self, data):
        S = float(data.get("Stock_Close", np.nan))
        iv = float(data.get("Straddle_imp_vol", np.nan))
        if not np.isfinite(S) or S <= 0 or not np.isfinite(iv) or iv <= 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan

        # ATM fallback if strike is unavailable.
        K = float(data.get("K", np.nan))
        if not np.isfinite(K) or K <= 0:
            K = S

        date_dt = pd.to_datetime(data.get("Date", pd.NaT), errors="coerce")
        exp_dt = pd.to_datetime(data.get("Expiry", pd.NaT), errors="coerce")
        if pd.notna(date_dt) and pd.notna(exp_dt):
            cal_days = (exp_dt.date() - date_dt.date()).days
            T_days = max(cal_days, 1)
            T = T_days / float(self.trading_days_per_year)
        else:
            T = self._FORECAST_HORIZON_DAYS / float(self.trading_days_per_year)

        r = self._resolve_rate(data, T)
        return S, K, T, r, iv

    def _build_market_targets(self, data):
        S, K, T, r, market_iv = self._extract_contract_terms(data)
        if not (np.isfinite(S) and np.isfinite(K) and np.isfinite(T) and np.isfinite(market_iv)):
            return None

        market_call = self._bs_call_price(S, K, T, r, market_iv)
        market_put = self._bs_put_price(S, K, T, r, market_iv)
        if not np.isfinite(market_call) or not np.isfinite(market_put):
            return None
        market_straddle = market_call + market_put
        return S, K, T, r, market_iv, market_straddle

    def _calibrate_v0_given_structure(self, market_inputs):
        S, K, T, r, market_iv, market_straddle = market_inputs
        if self._last_params is None:
            return None, np.nan
        _, kappa, theta, sigma, rho = np.asarray(self._last_params, dtype=float)
        x0 = float(np.clip(market_iv**2, 1e-6, 2.0))

        def objective(v0_arr):
            v0 = float(v0_arr[0])
            params = np.array([v0, kappa, theta, sigma, rho], dtype=float)
            call_model = self._heston_call_price(S, K, T, r, params)
            if not np.isfinite(call_model):
                return 1e6
            put_model = call_model - S + K * np.exp(-r * T)
            straddle_model = call_model + put_model
            price_err = ((straddle_model - market_straddle) / max(1.0, market_straddle)) ** 2
            iv_model = self._implied_vol_call(call_model, S, K, T, r)
            iv_err = (iv_model - market_iv) ** 2 if np.isfinite(iv_model) else 1.0
            # Keep v0 stable around previous state and market iv^2.
            prev_v0 = float(self._last_params[0])
            reg = 0.05 * ((v0 - prev_v0) ** 2) + 0.02 * ((v0 - market_iv**2) ** 2)
            return 10.0 * price_err + 4.0 * iv_err + reg

        try:
            res = minimize(
                objective,
                x0=np.array([x0], dtype=float),
                method="L-BFGS-B",
                bounds=[(1e-6, 2.0)],
                options={"maxiter": 60},
            )
        except Exception:
            return None, np.nan
        if not res.success or res.x is None:
            return None, np.nan
        v0_new = float(res.x[0])
        params_new = np.array([v0_new, kappa, theta, sigma, rho], dtype=float)
        call_opt = self._heston_call_price(S, K, T, r, params_new)
        model_iv = self._implied_vol_call(call_opt, S, K, T, r)
        return params_new, model_iv if np.isfinite(model_iv) else np.nan

    def _calibrate_heston(self, data):
        market_inputs = self._build_market_targets(data)
        if market_inputs is None:
            return None, np.nan
        S, K, T, r, market_iv, market_straddle = market_inputs

        iv2 = max(float(market_iv) ** 2, 1e-5)
        prior = np.array([iv2, 1.5, iv2, 0.5, -0.5], dtype=float)
        if self._last_params is None:
            x0 = prior.copy()
        else:
            x0 = np.asarray(self._last_params, dtype=float)

        bounds = [
            (1e-6, 2.0),   # v0
            (1e-3, 10.0),  # kappa
            (1e-6, 2.0),   # theta
            (1e-3, 5.0),   # sigma
            (-0.999, 0.999),  # rho
        ]

        scale = np.array([0.1, 1.0, 0.1, 0.5, 0.5], dtype=float)

        def objective(x):
            v0, kappa, theta, sigma, rho = x
            # Keep process in a reasonable region while allowing flexibility.
            feller_violation = max(0.0, sigma * sigma - 2.0 * kappa * theta)
            call_model = self._heston_call_price(S, K, T, r, x)
            if not np.isfinite(call_model):
                return 1e6
            put_model = call_model - S + K * np.exp(-r * T)
            straddle_model = call_model + put_model

            price_err = ((straddle_model - market_straddle) / max(1.0, market_straddle)) ** 2
            iv_model = self._implied_vol_call(call_model, S, K, T, r)
            iv_err = (iv_model - market_iv) ** 2 if np.isfinite(iv_model) else 1.0
            prior_pen = np.sum(((x - prior) / scale) ** 2)
            prev_pen = 0.0
            if self._last_params is not None:
                prev = np.asarray(self._last_params, dtype=float)
                prev_pen = np.sum(((x - prev) / scale) ** 2)

            return (
                10.0 * price_err
                + 4.0 * iv_err
                + 0.02 * prior_pen
                + 0.10 * prev_pen
                + 2.0 * (feller_violation**2)
            )

        try:
            res = minimize(objective, x0=x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 150})
        except Exception:
            return None, np.nan

        if not res.success or res.x is None:
            return None, np.nan
        x_opt = np.asarray(res.x, dtype=float)
        call_opt = self._heston_call_price(S, K, T, r, x_opt)
        model_iv = self._implied_vol_call(call_opt, S, K, T, r)
        if not np.all(np.isfinite(x_opt)):
            return None, np.nan
        return x_opt, model_iv if np.isfinite(model_iv) else np.nan

    # ------------------------------------------------------------------
    # Trade logic
    # ------------------------------------------------------------------
    def trade(self, data):
        self._compute_daily_pnl(data)

        if data["Force_Close"]:
            self.close_position(data)
            self._append_signal_nan(direction=0)
            self.prev_data = data
            return

        curr_iv = data.get("Straddle_imp_vol", np.nan)
        if pd.isna(curr_iv):
            self._append_signal_nan(direction=0)
            self.prev_data = data
            return

        market_inputs = self._build_market_targets(data)
        if market_inputs is None:
            self._append_signal_nan(direction=0)
            self._days_since_full_calib += 1
            self.prev_data = data
            return

        need_full_calib = (
            self._last_params is None
            or self._days_since_full_calib >= self.full_calibrate_every
        )
        calibration_mode = "full" if need_full_calib else "v0_only"
        if need_full_calib:
            params, model_iv = self._calibrate_heston(data)
            if params is not None:
                self._days_since_full_calib = 0
                self._consecutive_full_calib_failures = 0
            else:
                self._consecutive_full_calib_failures += 1
        else:
            params, model_iv = self._calibrate_v0_given_structure(market_inputs)
            if params is not None:
                self._days_since_full_calib += 1

        if params is None:
            if (
                need_full_calib
                and self._consecutive_full_calib_failures >= self.max_full_calib_failures
                and self.num_options != 0
            ):
                self.close_position(data)
            self._append_signal_nan(direction=0)
            self.calibration_mode_history[-1] = calibration_mode
            self._days_since_full_calib += 1
            self.prev_data = data
            return

        self._last_params = params
        v0, kappa, theta, sigma, rho = params
        long_run_vol = np.sqrt(max(theta, 0.0))
        spread = float(curr_iv) - long_run_vol
        feller_status = self._feller_ratio(kappa, theta, sigma)

        self._spread_history.append(spread)
        self.long_run_vol_history.append(float(long_run_vol))
        self.iv_longrun_spread_history.append(spread)
        self.feller_status_history.append(float(feller_status) if np.isfinite(feller_status) else np.nan)
        self.tail_risk_blocked_history.append(False)
        self.calibration_mode_history.append(calibration_mode)
        self.full_calib_failure_streak_history.append(self._consecutive_full_calib_failures)
        self.calibrated_params_history.append(
            {
                "v0": float(v0),
                "kappa": float(kappa),
                "theta": float(theta),
                "sigma": float(sigma),
                "rho": float(rho),
                "feller_status": float(feller_status) if np.isfinite(feller_status) else np.nan,
            }
        )
        self.model_iv_history.append(float(model_iv) if np.isfinite(model_iv) else np.nan)

        if len(self._spread_history) < self._MIN_SPREAD_OBS:
            self.heston_direction_signal.append(0)
            self.prev_data = data
            return

        spread_arr = np.asarray(self._spread_history[-self.z_window :], dtype=float)
        spread_arr = spread_arr[np.isfinite(spread_arr)]
        if spread_arr.size < 2:
            self.heston_direction_signal.append(0)
            self.prev_data = data
            return

        mu = float(np.mean(spread_arr))
        sigma_z = float(np.std(spread_arr, ddof=1))
        if sigma_z < 1e-12:
            self.heston_direction_signal.append(0)
            self.prev_data = data
            return

        z = (spread - mu) / sigma_z
        if z > self.z_entry:
            direction = -1
        elif z < -self.z_entry:
            direction = 1
        else:
            direction = 0

        tail_risk_blocked = False
        if direction < 0 and self.allow_short and float(rho) <= self.rho_tail_threshold:
            # Extreme negative skew regime: avoid opening fresh short-vol exposure.
            direction = 0
            tail_risk_blocked = True

        self.heston_direction_signal.append(direction)
        self.tail_risk_blocked_history[-1] = tail_risk_blocked

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

    def long_position(self, data):
        self.num_options = 1
        self.entry_straddle_price = data["Call_Close"] + data["Put_Close"]
        if self.delta_hedge:
            self._trade_underlying(-self.option_lot_size * data["Straddle_Delta"], data["Stock_Close"])
        self._log_transaction(data, "long")

    def short_position(self, data):
        self.num_options = -1
        self.entry_straddle_price = data["Call_Close"] + data["Put_Close"]
        if self.delta_hedge:
            self._trade_underlying(self.option_lot_size * data["Straddle_Delta"], data["Stock_Close"])
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
        net_delta = self.num_options * self.option_lot_size * data["Straddle_Delta"] + self.num_underlying
        if abs(net_delta) > self.rehedge_threshold:
            target_underlying = -self.num_options * self.option_lot_size * data["Straddle_Delta"]
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
            "heston_direction_signal": self.heston_direction_signal,
            "long_run_vol": self.long_run_vol_history,
            "iv_longrun_spread": self.iv_longrun_spread_history,
            "heston_params": self.calibrated_params_history,
            "heston_model_iv": self.model_iv_history,
            "feller_status": self.feller_status_history,
            "tail_risk_blocked": self.tail_risk_blocked_history,
            "calibration_mode": self.calibration_mode_history,
            "full_calib_failure_streak": self.full_calib_failure_streak_history,
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
