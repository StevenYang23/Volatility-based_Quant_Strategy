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
                 max_invest=0.8,              # long: spend at most this fraction of NAV
                 max_leverage=0.5,             # short: borrow at most this fraction of NAV (exposure cap)
                 vrp_threshold=1.0,
                 vrp_close_threshold=None,   # close when (IV - RV) < this (None = disabled)
                 min_ttm=1/252,
                 max_ttm=45/252):
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
        self.max_invest = max_invest
        self.max_leverage = max_leverage
        self.vrp_threshold = vrp_threshold
        self.vrp_close_threshold = vrp_close_threshold
        self.min_ttm = min_ttm
        self.max_ttm = max_ttm
        
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
        
        # Calculate Greeks every timestep if position is open
        if self.trade_open and self.call_df is not None and self.put_df is not None:
            cr = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            pr = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not cr.empty and not pr.empty:
                call_row = cr.iloc[0]
                put_row = pr.iloc[0]
                K = float(call_row['k'])
                T = float(call_row['ttm'])
                r = float(call_row['r'])
                sigma_c = float(call_row['imp_vol'])
                sigma_p = float(put_row['imp_vol'])
                
                # Update Greeks
                self.greeks = get_straddle_greeks(S, K, T, r, sigma_c, sigma_p)
                self.ttm = T
        
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
        self.balance += opt_pnl
        
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
        
        # Close when (IV - RV) < vrp_close_threshold
        if self.vrp_close_threshold is not None and self.call_df is not None and self.put_df is not None:
            date_norm = self._normalize_date(date)
            und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
            cr = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            pr = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not und_row.empty and not cr.empty and not pr.empty:
                RV = float(und_row['RV_30d'].iloc[0])
                IV = (float(cr.iloc[0]['imp_vol']) + float(pr.iloc[0]['imp_vol'])) / 2
                if (IV - RV) < self.vrp_close_threshold:
                    return True, f"vrp_close (IV-RV={IV-RV:.4f})"
        
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

        # Decision
        action = None
        if abs(VRP) < self.vrp_threshold:
            return
        if VRP > self.vrp_threshold:
            action = 'short'
        elif VRP < -self.vrp_threshold:
            action = 'long'
        else:
            return
        
        # Sizing: max_invest (long) = spend at most 80% of NAV; max_leverage (short) = exposure at most 50% of NAV
        premium = float(call_row['close']) + float(put_row['close'])
        if premium <= 0:
            return
        if action == 'long':
            # Long: total spend = premium * units <= max_invest * total_value
            max_units = (self.max_invest * self.total_value) / premium
            units = int(np.floor(max_units))
        else:
            # Short: exposure = premium * |units| <= max_leverage * total_value
            max_units = (self.max_leverage * self.total_value) / premium
            units = -int(np.floor(max_units))
        if units == 0:
            return
        
        # Execute
        total_premium = units * premium
        self.balance += -total_premium  # long: pay, short: receive
        self.call_num = units
        self.put_num = units
        self.trade_open = True
        self.k = K
        self.ttm = T
        self.entry_value = self.balance + total_premium  # pre-trade NAV
        self.entry_premium = abs(total_premium)
        
        # print(f"[{date.date()}] {action.upper()} {abs(units)} straddle(s) @ K={K:.0f}, "
        #       f"IV={IV:.2%}, RV={RV:.2%}, VRP_z={VRP_z:.1f}")