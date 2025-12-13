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
                 vega_risk_frac=0.1,         # e.g., 10% of NAV in vega
                 max_vega_per_trade=500,     # cap absolute vega (increased for bravery)
                 vrp_threshold=1.0,        # trade only if |VRP_z| > 1.0
                 max_leverage=1.5,           # max notional / NAV
                 tx_cost=0.005,              # 0.5% round-trip (realistic for options)
                 rehedge_freq=1,             # daily rehedge
                 delta_drift_threshold=0.1,  # rehedge if |delta error| > 10% of position
                 stop_loss_frac=0.15,        # stop if trade P&L < -15% of premium paid (loosened)
                 profit_take_frac=-1.0,      # take profit at +X% of premium (default -1 = disabled)
                 min_ttm=1/252,              # close if TTM < 1 day
                 max_ttm=60/252,             # only consider <=60 DTE
                 allow_long=True,
                 allow_short=True):           # NEW: Auto-tune for bolder trading
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
        self.vega_risk_frac = vega_risk_frac
        self.max_vega_per_trade = max_vega_per_trade
        self.vrp_threshold = vrp_threshold
        self.max_leverage = max_leverage
        self.tx_cost = tx_cost
        self.rehedge_freq = rehedge_freq
        self.delta_drift_threshold = delta_drift_threshold
        self.stop_loss_frac = stop_loss_frac
        self.profit_take_frac = profit_take_frac if profit_take_frac > 0 else -1.0  # Disable if <=0
        self.min_ttm = min_ttm
        self.max_ttm = max_ttm
        self.allow_long = allow_long
        self.allow_short = allow_short
        
        # State tracking
        self.last_hedge_date = None
        self.greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
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

    def _apply_tx_cost(self, cash_flow):
        """Apply tx cost: inflows reduced, outflows increased."""
        if cash_flow > 0:
            return cash_flow * (1 - self.tx_cost)  # receive less
        else:
            return cash_flow * (1 + self.tx_cost)  # pay more

    def rehedge(self, date):
        if not self.trade_open:
            return
        
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            return
        S = float(und_row['Close'].iloc[0])
        
        call_row_df = self.call_df[self.call_df['timestamp'].dt.normalize() == date_norm]
        put_row_df = self.put_df[self.put_df['timestamp'].dt.normalize() == date_norm]
        if call_row_df.empty or put_row_df.empty:
            return
        
        call_row = call_row_df.iloc[0].copy()
        put_row = put_row_df.iloc[0].copy()
        call_row['S'] = S
        put_row['S'] = S
        call_row['k'] = float(call_row['k'])
        put_row['k'] = float(put_row['k'])
        call_row['r'] = float(call_row['r'])
        put_row['r'] = call_row['r']
        self.ttm = float(call_row['ttm'])
        
        # Recompute Greeks
        new_greeks = get_greeks_analytical(call_row, put_row)
        self.greeks.update(new_greeks)
        
        # Target hedge: neutralize total delta
        target_underlying = -self.call_num * self.greeks['delta']
        delta_error = target_underlying - self.underlying_num
        
        # Rehedge if: time due OR delta drift too large
        time_due = (self.last_hedge_date is None) or ((date - self.last_hedge_date).days >= self.rehedge_freq)
        drift_too_large = abs(delta_error) > self.delta_drift_threshold * max(1, abs(self.call_num))
        
        if time_due or drift_too_large:
            trade_size = delta_error
            trade_value = trade_size * S
            net_cash = self._apply_tx_cost(-trade_value)  # negative: buying costs cash
            self.balance += net_cash
            self.underlying_num = target_underlying
            self.last_hedge_date = date
            # print(f"[{date.date()}] Rehedged: Δ={trade_size:+.2f}, Cost=${abs(trade_value * self.tx_cost):.2f}")

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
        self.balance += self._apply_tx_cost(opt_pnl)
        
        # Close underlying
        und_pnl = self.underlying_num * S
        self.balance += self._apply_tx_cost(und_pnl)
        
        # Reset
        self.call_num = 0.0
        self.put_num = 0.0
        self.underlying_num = 0.0
        self.call_df = None
        self.put_df = None
        self.greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}
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
            warnings.warn(f"No option data for {date}")
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
        if abs(self.greeks['vega']) < 1e-6:
            return
        
        # VRP signal: z-score of (IV - RV)
        port_IV = (call_row['imp_vol'] + put_row['imp_vol']) / 2
        VRP = port_IV - RV
        
        # Decision
        action = None
        if abs(VRP) < self.vrp_threshold:
            return  # no signal
        if VRP > self.vrp_threshold and self.allow_short:
            action = 'short'
        elif VRP < -self.vrp_threshold and self.allow_long:
            action = 'long'
        else:
            return
        
        # Vega-based sizing
        target_vega = self.total_value * self.vega_risk_frac
        units = target_vega / self.greeks['vega']
        units = np.sign(units) * min(abs(units), self.max_vega_per_trade / abs(self.greeks['vega']))
        units = np.clip(units, -10, 10)  # hard cap
        units = np.round(units)  # round to whole contracts
        if abs(units) < 1:
            return
        
        if action == 'short':
            units = -abs(units)
        else:
            units = abs(units)
        
        # Premium paid/received (absolute)
        premium_per_unit = call_row['close'] + put_row['close']
        net_premium = units * premium_per_unit
        
        # Initial hedge
        self.call_num = units
        self.put_num = units
        self.underlying_num = -units * self.greeks['delta']  # delta-neutral init
        
        # Cash flow: long → pay (negative), short → receive (positive)
        self.balance += self._apply_tx_cost(-net_premium)
        
        # Track for stop-loss
        self.entry_value = self.balance + net_premium  # pre-trade value
        self.entry_premium = abs(net_premium)
        self.trade_open = True
        self.last_hedge_date = date
        self.k = call_row['k']
        self.r = call_row['r']
        self.ttm = ttm
        

    def should_exit(self, date):
        if not self.trade_open:
            return False, ""
        
        date_norm = self._normalize_date(date)
        und_row = self.underlying_df[self.underlying_df['Date'].dt.normalize() == date_norm]
        if und_row.empty:
            return False, ""
        
        # UPDATED: Relaxed TTM exit — only if critically low
        if self.ttm is not None and self.ttm < self.min_ttm / 2:  # e.g., <0.5 days
            return True, "near-expiry"
        
        # UPDATED: Check P&L stop-loss (looser default)
        current_val = self.cal_value(date)
        if not np.isnan(current_val) and self.entry_value is not None:
            pnl = current_val - self.entry_value
            pnl_frac = pnl / self.entry_premium
            if pnl_frac < -self.stop_loss_frac:  # e.g., -15%
                return True, f"stop-loss ({pnl_frac:.1%})"
            
            # NEW: Optional profit-take (disabled if profit_take_frac <=0)
            if self.profit_take_frac > 0 and pnl_frac > self.profit_take_frac:
                return True, f"profit-take ({pnl_frac:.1%})"
        
        return False, ""