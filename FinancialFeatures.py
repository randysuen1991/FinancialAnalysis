import LimitOrderBook as LOB


class FinancialFeatures:

    @staticmethod
    def OBP(OBlist, depth):
        bid_pressure = 0
        ask_pressure = 0
        for OB in OBlist:
            bid_keys = OB.buy_limits.keys()
            ask_keys = OB.sell_limits.keys()
            for d in range(depth):
                bid_price = bid_keys[d]
                ask_price = ask_keys[d]
                bid_pressure += OB.buy_limits[bid_price].size
                ask_pressure += OB.sell_limits[ask_price].size

