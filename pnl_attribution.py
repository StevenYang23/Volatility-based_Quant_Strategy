"""Full-repricing PnL attribution for ATM straddle positions.

Instead of a second-order Taylor expansion (which leaves a large residual),
this module uses sequential bump-and-reprice through the Black-Scholes model.

Given consecutive-day inputs (S, K, T, r, σ) the straddle price change is
decomposed into exact additive buckets via telescoping:

    p_base     = BS(S0, K, T0, r0, σ0)       # yesterday's model price
    p_S        = BS(S1, K, T0, r0, σ0)       # bump S
    p_σ        = BS(S0, K, T0, r0, σ1)       # bump σ  (at old S)
    p_Sσ       = BS(S1, K, T0, r0, σ1)       # bump S and σ
    p_SσT      = BS(S1, K, T1, r0, σ1)       # bump S, σ, T
    p_all      = BS(S1, K, T1, r1, σ1)       # today's model price

    underlying  = p_S   - p_base              # all S  effects
    pure_vol    = p_σ   - p_base              # all σ  effects at old S
    cross_Sσ    = p_Sσ  - p_S - p_σ + p_base # interaction  S × σ
    time_decay  = p_SσT - p_Sσ               # time decay
    rate_effect = p_all  - p_SσT              # rate effect

The sum telescopes to  p_all − p_base  exactly.

Sub-decomposition uses model Greeks at the base point to split
each bucket into first-order and higher-order terms:

    delta  = straddle_delta · dS          (first-order  S)
    gamma  = underlying − delta           (convexity in S, exact)
    vega   = straddle_vega  · dσ          (first-order  σ)
    volga  = pure_vol − vega              (convexity in σ, exact)
    vanna  = cross_Sσ                     (exact cross-effect)
    theta  = time_decay
    rho    = rate_effect
    resid  = market Δ(straddle) − model Δ(straddle)
"""

from __future__ import annotations

import math
import re
from datetime import date as _date
from typing import Dict, Optional, Tuple

import numpy as np

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_SQRT_2 = math.sqrt(2.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def bs_straddle(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """BS straddle price (call + put) at a single implied vol."""
    if T <= 1e-12:
        return abs(S - K)
    sigma = max(sigma, 1e-8)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    df = math.exp(-r * T)
    call = S * Nd1 - K * df * Nd2
    put = K * df * (1.0 - Nd2) - S * (1.0 - Nd1)
    return call + put


def _straddle_delta_vega(
    S: float, K: float, T: float, r: float, sigma: float
) -> Tuple[float, float]:
    """Straddle delta and vega at a single implied vol (for sub-decomposition)."""
    if T <= 1e-12:
        return 0.0, 0.0
    sigma = max(sigma, 1e-8)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    delta = 2.0 * _norm_cdf(d1) - 1.0
    vega = 2.0 * S * _norm_pdf(d1) * sqrtT
    return delta, vega


def parse_occ_symbol(sym: str) -> Tuple[Optional[_date], Optional[float]]:
    """Parse OCC option symbol → (expiry, strike).  Returns (None, None) on failure."""
    part = sym.split(":")[1] if ":" in sym else sym
    m = re.search(r"^[A-Z]+(\d{6})([CP])(\d{8})$", part)
    if not m:
        return None, None
    raw = m.group(1)
    exp_str = ("20" + raw) if len(raw) == 6 else raw
    y, mo, d = int(exp_str[:4]), int(exp_str[4:6]), int(exp_str[6:8])
    return _date(y, mo, d), int(m.group(3)) / 1000.0


def ttm_trading(as_of: _date, expiry: _date) -> float:
    """Trading-day fraction of year (busday_count / 252)."""
    bd = int(np.busday_count(np.datetime64(as_of), np.datetime64(expiry)))
    return max(bd / 252.0, 1e-12)


def full_reval_attribution(
    S0: float,
    S1: float,
    K: float,
    T0: float,
    T1: float,
    r0: float,
    r1: float,
    sig0: float,
    sig1: float,
    market_straddle_change: float,
) -> Dict[str, float]:
    """Return per-straddle attribution dict with near-zero residual.

    All values are for **one long straddle**; the caller multiplies by
    ``num_options`` and adds the hedge PnL to the delta bucket.
    """
    bad = any(
        not math.isfinite(v)
        for v in (S0, S1, K, T0, T1, r0, r1, sig0, sig1, market_straddle_change)
    )
    if bad or K <= 0 or S0 <= 0 or S1 <= 0:
        return _zero_attr()

    p_base = bs_straddle(S0, K, T0, r0, sig0)
    p_S = bs_straddle(S1, K, T0, r0, sig0)
    p_sig = bs_straddle(S0, K, T0, r0, sig1)
    p_S_sig = bs_straddle(S1, K, T0, r0, sig1)
    p_S_sig_T = bs_straddle(S1, K, T1, r0, sig1)
    p_all = bs_straddle(S1, K, T1, r1, sig1)

    underlying_effect = p_S - p_base
    pure_vol_effect = p_sig - p_base
    cross_S_sig = p_S_sig - p_S - p_sig + p_base
    theta_effect = p_S_sig_T - p_S_sig
    rho_effect = p_all - p_S_sig_T

    delta_lin, vega_lin = _straddle_delta_vega(S0, K, T0, r0, sig0)
    dS = S1 - S0
    d_sig = sig1 - sig0

    delta_pnl = delta_lin * dS
    gamma_pnl = underlying_effect - delta_pnl
    vega_pnl = vega_lin * d_sig
    volga_pnl = pure_vol_effect - vega_pnl
    vanna_pnl = cross_S_sig

    model_change = p_all - p_base
    residual = market_straddle_change - model_change

    return {
        "delta": delta_pnl,
        "gamma": gamma_pnl,
        "vega": vega_pnl,
        "volga": volga_pnl,
        "vanna": vanna_pnl,
        "theta": theta_effect,
        "rho": rho_effect,
        "residual": residual,
    }


def _zero_attr() -> Dict[str, float]:
    return {k: 0.0 for k in ("delta", "gamma", "vega", "volga", "vanna", "theta", "rho", "residual")}
