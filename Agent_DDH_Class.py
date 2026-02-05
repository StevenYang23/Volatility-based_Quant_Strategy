import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

def d1(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return np.inf if S > K else -np.inf
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1_val = d1(S, K, T, r, sigma)
    d2 = d1_val - sigma * np.sqrt(T)
    return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1_val = d1(S, K, T, r, sigma)
    d2 = d1_val - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1_val)

# Analytical Greeks (preferred — accurate & fast)
def get_greeks_analytical(call_row, put_row):
    if call_row['ttm'] <= 1e-8:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
    
    S = call_row['S']
    K = call_row['k']
    r = call_row['r']
    T = call_row['ttm']
    sigma_c = call_row['imp_vol']
    sigma_p = put_row['imp_vol']
    
    # d1 for call and put (same K,T,r,S)
    d1_c = d1(S, K, T, r, sigma_c)
    d1_p = d1(S, K, T, r, sigma_p)
    
    # Delta
    delta_c = norm.cdf(d1_c)
    delta_p = norm.cdf(d1_p) - 1
    delta = delta_c + delta_p  # total straddle delta per unit
    
    # Gamma
    n_c = norm.pdf(d1_c)
    n_p = norm.pdf(d1_p)
    gamma_c = n_c / (S * sigma_c * np.sqrt(max(T, 1e-12)))
    gamma_p = n_p / (S * sigma_p * np.sqrt(max(T, 1e-12)))
    gamma = gamma_c + gamma_p
    
    # Vega (per 1.0 vol, not 1%)
    vega_c = S * n_c * np.sqrt(max(T, 1e-12))
    vega_p = S * n_p * np.sqrt(max(T, 1e-12))
    vega = vega_c + vega_p
    
    # Theta (approx, per year — convert to per day if needed)
    # Call theta
    theta_c = (-S * n_c * sigma_c) / (2 * np.sqrt(max(T, 1e-12))) - r * K * np.exp(-r * T) * norm.cdf(d1_c - sigma_c * np.sqrt(T))
    # Put theta
    theta_p = (-S * n_p * sigma_p) / (2 * np.sqrt(max(T, 1e-12))) + r * K * np.exp(-r * T) * norm.cdf(-d1_p + sigma_p * np.sqrt(T))
    theta = theta_c + theta_p  # per year; divide by 252 for daily
    
    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta
    }

class Agent_DDH:
    def __init__(self, 
                 balance=1000.0,
                 max_invest=0.8,             # long: spend at most this fraction of NAV
                 max_leverage=0.5,           # short: borrow/exposure at most this fraction of NAV
                 vrp_threshold=1.0,        # trade only if |VRP_z| > 1.0
                 vrp_close_threshold=1.0,  # close when (IV - RV) < this (None = disabled)
                 delta_rehedge_threshold=1e9,  # rehedge when |net_delta| > this (None = disabled)
                 min_ttm=1/252,              # close if TTM < 1 day
                 max_ttm=60/252,            # only consider <=60 DTE
                 delta_hedge=True):         # if False, underlying_num always 0 and delta_rehedge_threshold=1e9
        self.balance = float(balance)
        self.underlying_df = pd.read_csv("DataSet/underlying.csv", parse_dates=['Date'])
        # Ensure RV_30d exists; if not, compute it
        # if 'RV_30d' not in self.underlying_df.columns:
        self.underlying_df['log_ret'] = np.log(self.underlying_df['Close'] / self.underlying_df['Close'].shift(1))
        self.underlying_df['RV_30d'] = np.sqrt(252) * self.underlying_df['log_ret'].rolling(30).std()
        self.underlying_df['RV_30d'] = self.underlying_df['RV_30d'].bfill()
        
        self.call_df = None
        self.put_df = None
        self.call_num = 0.0
        self.put_num = 0.0
        self.underlying_num = 0.0
        self.total_value = self.balance
        
        # Strategy params
        self.max_invest = max_invest
        self.max_leverage = max_leverage
        self.vrp_threshold = vrp_threshold
        self.vrp_close_threshold = vrp_close_threshold
        self.delta_hedge = delta_hedge
        self.delta_rehedge_threshold = (1e9 if not delta_hedge else delta_rehedge_threshold)
        self.min_ttm = min_ttm
        self.max_ttm = max_ttm
        
        # State tracking
        self.greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
        self.current_call_sym = None  # for rehedge: close and reopen same options
        self.current_put_sym = None
        self.k = None
        self.r = None
        self.ttm = None
        self.entry_value = None
        self.entry_premium = None
        self.trade_open = False
        
    def _normalize_date(self, date):
        if date is None:
            return None
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
        
        call_price = 0.0
        put_price = 0.0
        if self.call_df is not None and not self.call_df.empty:
            call_row = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            if not call_row.empty:
                call_price = float(call_row.iloc[0]['close'])
        if self.put_df is not None and not self.put_df.empty:
            put_row = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not put_row.empty:
                put_price = float(put_row.iloc[0]['close'])
        
        # Portfolio value: cash + options + underlying
        self.total_value = (self.balance +
                            self.call_num * call_price +
                            self.put_num * put_price +
                            self.underlying_num * S)
        
        # Calculate Greeks every timestep if position is open
        if self.trade_open and self.call_df is not None and self.put_df is not None:
            call_row_df = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            put_row_df = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not call_row_df.empty and not put_row_df.empty:
                call_row = call_row_df.iloc[0].copy()
                put_row = put_row_df.iloc[0].copy()
                call_row['S'] = S
                put_row['S'] = S
                call_row['k'] = float(call_row['k'])
                put_row['k'] = float(put_row['k'])
                call_row['r'] = float(call_row['r'])
                put_row['r'] = call_row['r']
                call_row['ttm'] = float(call_row['ttm'])
                put_row['ttm'] = call_row['ttm']
                
                # Update Greeks
                new_greeks = get_greeks_analytical(call_row, put_row)
                self.greeks.update(new_greeks)
                self.ttm = call_row['ttm']
        
        return self.total_value

    def close_position(self, date, reason=""):
        if not self.trade_open:
            return
        
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            return
        S = float(und_row['Close'].iloc[0])
        
        # Get current prices
        call_price = 0.0
        put_price = 0.0
        if self.call_df is not None and not self.call_df.empty:
            cr = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            if not cr.empty:
                call_price = float(cr.iloc[0]['close'])
        if self.put_df is not None and not self.put_df.empty:
            pr = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not pr.empty:
                put_price = float(pr.iloc[0]['close'])
        
        # Close options
        opt_pnl = self.call_num * call_price + self.put_num * put_price
        self.balance += opt_pnl
        
        # Close underlying
        und_pnl = self.underlying_num * S
        self.balance += und_pnl
        
        # Reset
        self.call_num = 0.0
        self.put_num = 0.0
        self.underlying_num = 0.0
        self.call_df = None
        self.put_df = None
        self.greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
        self.current_call_sym = None
        self.current_put_sym = None
        self.trade_open = False
        self.entry_value = None
        self.entry_premium = None
        # print(f"[{date.date()}] Position closed ({reason}). Cash: ${self.balance:.2f}")

    def build_position(self, call_sym, put_sym, date):
        # Close existing first (safe)
        if self.trade_open:
            self.close_position(date, reason="rebalance")
        
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
            warnings.warn(f"No underlying data for {date}")
            return
        S = float(und_row['Close'].iloc[0])
        RV = float(und_row['RV_30d'].iloc[0])  # horizon-matched!
        
        # Get option data
        call_rows = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
        put_rows = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
        if call_rows.empty or put_rows.empty:
            return
        
        call_row = call_rows.iloc[0].copy()
        put_row = put_rows.iloc[0].copy()
        ttm = float(call_row['ttm'])
        
        # Filter by TTM
        if ttm < self.min_ttm or ttm > self.max_ttm:
            # print(f"TTM={ttm*252:.1f}D outside [{self.min_ttm*252:.0f}, {self.max_ttm*252:.0f}]D — skip.")
            return
        
        # Prepare rows
        for row in [call_row, put_row]:
            row['S'] = S
            row['k'] = float(row['k'])
            row['r'] = float(row['r'])
            row['ttm'] = ttm
        
        # Greeks
        self.greeks = get_greeks_analytical(call_row, put_row)
        delta = self.greeks['delta']
        
        # VRP signal: z-score of (IV - RV)
        port_IV = (call_row['imp_vol'] + put_row['imp_vol']) / 2
        VRP = port_IV - RV
        vrp_std = float(und_row['VRP_std'].iloc[0])
        vrp_mean = float(und_row['VRP_mean'].iloc[0])
        # vrp_std = 1
        # Decision
        action = None
        if ((VRP-vrp_mean)/vrp_std) > self.vrp_threshold:
            action = 'short'
        elif ((VRP-vrp_mean)/vrp_std) < -self.vrp_threshold:
            action = 'long'
        else:
            return
        
        # Sizing: max_invest (long) = spend at most 80% of NAV; max_leverage (short) = exposure at most 50% of NAV
        premium_per_unit = float(call_row['close']) + float(put_row['close'])
        if premium_per_unit <= 0:
            return
        if action == 'long':
            # Long: total spend = premium*units + max(0, underlying_cost). underlying_num = -units*delta; cost = underlying_num*S when underlying_num>0.
            # So cost per unit = premium_per_unit + max(0, -delta*S)
            cost_per_unit = premium_per_unit + max(0.0, -delta * S)
            if cost_per_unit <= 0:
                return
            max_units = (self.max_invest * self.total_value) / cost_per_unit
            units = int(np.floor(max_units))
        else:
            # Short: exposure per unit = premium + |underlying| = premium_per_unit + |delta|*S
            exposure_per_unit = premium_per_unit + abs(delta) * S
            if exposure_per_unit <= 0:
                return
            max_units = (self.max_leverage * self.total_value) / exposure_per_unit
            units = -int(np.floor(max_units))
        if units == 0:
            return
        
        # Premium paid/received (absolute)
        net_premium = units * premium_per_unit
        
        # Initial hedge
        self.call_num = units
        self.put_num = units
        self.underlying_num = int(round(-units * delta)) if self.delta_hedge else 0
        
        # Cash flow: options — long → pay, short → receive
        self.balance += -net_premium
        # Cash flow: underlying — buy → pay, short sell → receive
        self.balance += -(self.underlying_num * S)
        
        self.entry_value = self.balance + net_premium + self.underlying_num * S  # pre-trade NAV
        self.entry_premium = abs(net_premium)
        self.trade_open = True
        self.current_call_sym = call_sym
        self.current_put_sym = put_sym
        self.k = call_row['k']
        self.r = call_row['r']
        self.ttm = ttm
        

    def rehedge(self, date):
        """If |net_delta| > delta_rehedge_threshold, close and reopen the position (same options)."""
        if not self.trade_open or self.delta_rehedge_threshold is None:
            return
        self.cal_value(date)  # refresh Greeks
        net_delta = self.greeks['delta'] * self.call_num + self.underlying_num
        if abs(net_delta) <= self.delta_rehedge_threshold:
            return
        call_sym = self.current_call_sym
        put_sym = self.current_put_sym
        if call_sym is None or put_sym is None:
            return
        # self.close_position(date, reason="rehedge")
        self.build_position(call_sym, put_sym, date)

    def should_exit(self, date):
        if not self.trade_open:
            return False, ""
        
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            return False, ""
        
        # Relaxed TTM exit — only if critically low
        if self.ttm is not None and self.ttm < self.min_ttm / 2:  # e.g., <0.5 days
            return True, "near-expiry"
        
        # Close when (IV - RV) < vrp_close_threshold
        if self.vrp_close_threshold is not None and self.call_df is not None and self.put_df is not None:
            call_row_df = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
            put_row_df = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
            if not call_row_df.empty and not put_row_df.empty:
                vrp_std = float(und_row['VRP_std'].iloc[0])
                vrp_mean = float(und_row['VRP_mean'].iloc[0])
                RV = float(und_row['RV_30d'].iloc[0])
                IV = (float(call_row_df.iloc[0]['imp_vol']) + float(put_row_df.iloc[0]['imp_vol'])) / 2
                VRP = IV - RV
                if ((VRP-vrp_mean)/vrp_std) > self.vrp_close_threshold:
                    return True, f"vrp_close (IV-RV={IV-RV:.4f})"
        
        return False, ""