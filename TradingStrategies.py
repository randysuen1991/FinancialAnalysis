import numpy as np
from DataAnalysis import DataAnalysis as DA
import matplotlib.pyplot as plt


class PairTrading:
    def __init__(self, regressor):
        self.regressor = regressor()
        self.residual = None
        self.result = None
        self.maximum_position = 1

    def fit(self, series1, series2):
        self.regressor.fit(x_train=series1.reshape(-1, 1), y_train=series2.reshape(-1, 1))
        prediction = self.regressor.predict(series1.reshape(-1, 1))
        self.residual = (prediction - series2.reshape(-1, 1)).ravel()
        self.result = DA.TimeSeriesAnalysis.adfuller(self.residual)
        return self.result

    # return the days in which we should go in and we should go out.
    def simulate(self, in_threshold, out_threshold, series=None, plot=False):
        if series is None:
            series = self.residual

        last_price = None
        position = 0
        earn = 0
        for num in series:
            if abs(num) > in_threshold and abs(position) < self.maximum_position:
                if position != 0:
                    if num > in_threshold and position < 0:
                        earn += (num-last_price)
                        position = 1
                        last_price = num
                        print('offset and long', num, earn)
                    elif num < in_threshold and position > 0:
                        earn += (last_price-num)
                        position = -1
                        last_price = num
                        print('offset and short', num, earn)
                else:
                    if num > 0:
                        position += 1
                        last_price = num
                        print('long', num)
                    else:
                        position -= 1
                        last_price = num
                        print('short', num, earn)
            elif abs(num) < out_threshold and position != 0:
                position = 0
                earn += abs(last_price - num)
                last_price = 0
                print('offset', num, earn)
        if plot:
            plt.plot(series)
            plt.show()

        return position, earn

        # in_set = set()
        # for _in in in_threshold:
        #     in_days = np.where(np.abs(series) > _in)[0]
        #     print(in_days)
        #     for day in in_days:
        #         in_set.add(day)
        #
        # out_set = set()
        # for _out in out_threshold:
        #     out_days = np.where(np.abs(series) < _out)[0]
        #     # print(out_days)
        #     for day in out_days:
        #         out_set.add(day)
        #
        # return in_set, out_set

