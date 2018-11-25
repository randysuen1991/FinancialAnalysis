import numpy as np
import scipy.stats as si


class BlackScholes:

    @staticmethod
    def call_option_price(s, k, t, r, sigma):
        # s: spot price
        # k: strike price
        # t: time to maturity
        # r: interest rate
        # sigma: volatility of underlying asset
        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return s * si.norm.cdf(d1, 0.0, 1.0) - k * np.exp(-r * t) * si.norm.cdf(d2, 0.0, 1.0)

    @staticmethod
    def put_option_price(s, k, t, r, sigma):
        # s: spot price
        # k: strike price
        # t: time to maturity
        # r: interest rate
        # sigma: volatility of underlying asset
        d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)
        return k * np.exp(-r * t) * si.norm.cdf(-d2, 0.0, 1.0) - s * si.norm.cdf(-d1, 0.0, 1.0)



