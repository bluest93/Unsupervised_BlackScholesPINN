import numpy as np
from scipy.stats import norm
import torch
import QuantLib as ql

def black_scholes_call_solution(S, K, T, r, sigma):
    S = np.array(S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put_solution(S, K, T, r, sigma):
    call = black_scholes_call_solution(S, K, T, r, sigma)
    return call - S + K * np.exp(-r * T)

def prepare_dataset(config, time=0):
    S_eval = torch.linspace(config["min_S"], config["max_S"], 200).view(-1, 1)
    t_eval = torch.ones_like(S_eval)*time
    
    return S_eval, t_eval

def american_put_ql(S, K, r, sigma, T, bias=0, noise_variance=0, q=0.0):
    calendar  = ql.NullCalendar()
    day_count = ql.Actual365Fixed()
    today     = ql.Date(1, 1, 2020)
    maturity  = calendar.advance(today, int(T * 365), ql.Days)
    ql.Settings.instance().evaluationDate = today

    spot   = ql.SimpleQuote(float(S))
    r_ts   = ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(float(r))), day_count)
    q_ts   = ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(float(q))), day_count)
    vol_ts = ql.BlackConstantVol(today, calendar, ql.QuoteHandle(ql.SimpleQuote(float(sigma))), day_count)

    process = ql.BlackScholesMertonProcess(
        ql.QuoteHandle(spot),
        ql.YieldTermStructureHandle(q_ts),
        ql.YieldTermStructureHandle(r_ts),
        ql.BlackVolTermStructureHandle(vol_ts)
    )

    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
    exercise = ql.AmericanExercise(today, maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(ql.FdBlackScholesVanillaEngine(process, 200, 200))
    price = option.NPV()

    if noise_variance > 0:
        noise = np.random.normal(bias, noise_variance)
        price = price + noise 

    return float(price)
