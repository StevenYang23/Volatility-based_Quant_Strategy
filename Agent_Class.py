import pandas as pd
import numpy as np
from scipy.stats import norm
import warnings

def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def get_straddle_greeks(S, K, T, r, sigma_c, sigma_p):
    """Analytical Greeks for a straddle (1 call + 1 put)."""
    if T <= 1e-8:
        return {'vega': 0.0, 'theta': 0.0, 'gamma': 0.0, 'delta': 0.0}
    
    d1_c = d1(S, K, T, r, sigma_c)
    d1_p = d1(S, K, T, r, sigma_p)
    
    n_c = norm.pdf(d1_c)
    n_p = norm.pdf(d1_p)
    
    vega_c = S * n_c * np.sqrt(T)
    vega_p = S * n_p * np.sqrt(T)
    vega = vega_c + vega_p
    
    # Theta (per year)
    d2_c = d1_c - sigma_c * np.sqrt(T)
    d2_p = d1_p - sigma_p * np.sqrt(T)
    theta_c = (-S * n_c * sigma_c) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2_c)
    theta_p = (-S * n_p * sigma_p) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2_p)
    theta = theta_c + theta_p
    
    gamma_c = n_c / (S * sigma_c * np.sqrt(T))
    gamma_p = n_p / (S * sigma_p * np.sqrt(T))
    gamma = gamma_c + gamma_p
    
    delta = norm.cdf(d1_c) + (norm.cdf(d1_p) - 1)
    
    return {'vega': vega, 'theta': theta, 'gamma': gamma, 'delta': delta}


class Agent_Straddles:
    def __init__(self,
                 balance=10000.0,
                 sizing_method='vega',        # 'vega' or 'premium'
                 vega_risk_frac=0.1,          # e.g., 10% NAV in vega
                 premium_risk_frac=0.05,      # e.g., 5% NAV per trade
                 max_units=10,                # max contracts per trade
                 vrp_z_threshold=1.0,
                 tx_cost=0.005,               # 0.5% round-trip
                 stop_loss_frac=0.20,         # exit if loss > 20% of premium paid
                 min_ttm=1/252,
                 max_ttm=45/252,
                 allow_long=True,
                 allow_short=True):
        self.balance = float(balance)
        self.underlying_df = pd.read_csv("DataSet/underlying.csv", parse_dates=['Date'])
        
        # Compute RV_30d if missing
        if 'RV_30d' not in self.underlying_df.columns:
            log_ret = np.log(self.underlying_df['Close'] / self.underlying_df['Close'].shift(1))
            rv_series = np.sqrt(252) * log_ret.rolling(30).std()
            rv_series = rv_series.bfill().ffill()
            if rv_series.isna().all():
                rv_series = pd.Series(0.2, index=self.underlying_df.index)
            self.underlying_df['RV_30d'] = rv_series
        
        # Position state
        self.call_df = None
        self.put_df = None
        self.call_num = 0.0
        self.put_num = 0.0
        self.total_value = self.balance
        self.trade_open = False
        
        # Strategy params
        self.sizing_method = sizing_method
        self.vega_risk_frac = vega_risk_frac
        self.premium_risk_frac = premium_risk_frac
        self.max_units = max_units
        self.vrp_z_threshold = vrp_z_threshold
        self.tx_cost = tx_cost
        self.stop_loss_frac = stop_loss_frac
        self.min_ttm = min_ttm
        self.max_ttm = max_ttm
        self.allow_long = allow_long
        self.allow_short = allow_short
        
        # Tracking
        self.greeks = {'vega': 0.0, 'theta': 0.0, 'gamma': 0.0, 'delta': 0.0}
        self.entry_value = None
        self.entry_premium = None
        self.k = None
        self.ttm = None

    def _normalize_date(self, date):
        if isinstance(date, pd.Timestamp):
            return date.normalize()
        return pd.to_datetime(date).normalize()

    def _apply_tx_cost(self, cash_flow):
        """Inflows ↓, outflows ↑ due to costs."""
        return cash_flow * (1 - self.tx_cost) if cash_flow > 0 else cash_flow * (1 + self.tx_cost)

    def cal_value(self, date):
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            self.total_value = np.nan
            return np.nan
        S = float(und_row['Close'].iloc[0])
        
        call_price = put_price = 0.0
        if self.call_df is not None:
            cr = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            if not cr.empty:
                call_price = float(cr.iloc[0]['close'])
        if self.put_df is not None:
            pr = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not pr.empty:
                put_price = float(pr.iloc[0]['close'])
        
        self.total_value = self.balance + self.call_num * call_price + self.put_num * put_price
        return self.total_value

    def close_position(self, date, reason=""):
        if not self.trade_open:
            return
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            return
        S = float(und_row['Close'].iloc[0])
        
        call_price = put_price = 0.0
        if self.call_df is not None:
            cr = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            if not cr.empty:
                call_price = float(cr.iloc[0]['close'])
        if self.put_df is not None:
            pr = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not pr.empty:
                put_price = float(pr.iloc[0]['close'])
        
        # Close both legs
        opt_pnl = self.call_num * call_price + self.put_num * put_price
        self.balance += self._apply_tx_cost(opt_pnl)
        
        # Reset
        self.call_num = self.put_num = 0.0
        self.call_df = self.put_df = None
        self.greeks = {'vega': 0.0, 'theta': 0.0, 'gamma': 0.0, 'delta': 0.0}
        self.trade_open = False
        self.entry_value = None
        self.entry_premium = None
        # print(f"[{date.date()}] Closed straddle ({reason}). Balance: ${self.balance:.2f}")

    def should_exit(self, date):
        if not self.trade_open:
            return False, ""
        
        # Check TTM
        if self.ttm is not None and self.ttm < self.min_ttm:
            return True, "expiry"
        
        # Stop-loss
        current_val = self.cal_value(date)
        if not np.isnan(current_val) and self.entry_value is not None:
            pnl = current_val - self.entry_value
            if pnl < -self.stop_loss_frac * self.entry_premium:
                return True, f"stop-loss ({pnl/self.entry_premium:.1%})"
        
        return False, ""

    def build_position(self, call_sym, put_sym, date):
        # Close existing
        if self.trade_open:
            self.close_position(date, "rebalance")
        
        # Load data
        try:
            self.call_df = pd.read_csv(f"DataSet/{call_sym}.csv", parse_dates=['timestamp'])
            self.put_df = pd.read_csv(f"DataSet/{put_sym}.csv", parse_dates=['timestamp'])
        except Exception as e:
            warnings.warn(f"Failed to load {call_sym}/{put_sym}: {e}")
            return
        
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            return
        S = float(und_row['Close'].iloc[0])
        RV = float(und_row['RV_30d'].iloc[0])
        
        # Get options
        cr = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
        pr = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
        if cr.empty or pr.empty:
            return
        
        call_row, put_row = cr.iloc[0], pr.iloc[0]
        K = float(call_row['k'])
        if abs(K - float(put_row['k'])) > 1e-2:
            warnings.warn("Call/put strikes differ — using call strike.")
        T = float(call_row['ttm'])
        r = float(call_row['r'])
        sigma_c = float(call_row['imp_vol'])
        sigma_p = float(put_row['imp_vol'])
        
        if T < self.min_ttm or T > self.max_ttm:
            return
        
        # Greeks
        self.greeks = get_straddle_greeks(S, K, T, r, sigma_c, sigma_p)
        
        # VRP z-score
        IV = (sigma_c + sigma_p) / 2
        VRP = IV - RV
        
        # Historical VRP std (60-day)
        vrp_hist = []
        for i in range(1, 61):
            past_date = date_norm - pd.Timedelta(days=i)
            past_und = self.underlying_df[self.underlying_df['Date'].dt.normalize() == past_date]
            if not past_und.empty:
                try:
                    past_cr = self.call_df[self.call_df['timestamp'].dt.normalize() == past_date]
                    past_pr = self.put_df[self.put_df['timestamp'].dt.normalize() == past_date]
                    if not past_cr.empty and not past_pr.empty:
                        iv_past = (float(past_cr.iloc[0]['imp_vol']) + float(past_pr.iloc[0]['imp_vol'])) / 2
                        rv_past = float(past_und['RV_30d'].iloc[0])
                        vrp_hist.append(iv_past - rv_past)
                except:
                    pass
        vrp_std = np.std(vrp_hist) if len(vrp_hist) > 10 else 0.05
        VRP_z = VRP / max(vrp_std, 0.01)
        
        # Decision
        action = None
        if abs(VRP_z) < self.vrp_z_threshold:
            return
        if VRP_z > self.vrp_z_threshold and self.allow_short:
            action = 'short'
        elif VRP_z < -self.vrp_z_threshold and self.allow_long:
            action = 'long'
        else:
            return
        
        # Sizing
        premium = float(call_row['close']) + float(put_row['close'])
        if self.sizing_method == 'vega' and abs(self.greeks['vega']) > 1e-6:
            target_vega = self.total_value * self.vega_risk_frac
            units = target_vega / self.greeks['vega']
        else:  # 'premium' or fallback
            risk_budget = self.total_value * self.premium_risk_frac
            units = risk_budget / premium
        
        units = np.sign(units) * min(abs(units), self.max_units)
        units = int(np.round(units))
        if units == 0:
            return
        
        if action == 'short':
            units = -abs(units)
        else:
            units = abs(units)
        
        # Execute
        total_premium = units * premium
        self.balance += self._apply_tx_cost(-total_premium)  # long: pay, short: receive
        self.call_num = units
        self.put_num = units
        self.trade_open = True
        self.k = K
        self.ttm = T
        self.entry_value = self.balance + total_premium  # hypothetical pre-trade NAV
        self.entry_premium = abs(total_premium)
        
        # print(f"[{date.date()}] {action.upper()} {abs(units)} straddle(s) @ K={K:.0f}, "
        #       f"IV={IV:.2%}, RV={RV:.2%}, VRP_z={VRP_z:.1f}")