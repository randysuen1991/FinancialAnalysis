import numpy as np
from DataAnalysis import DataAnalysis as DA
import matplotlib.pyplot as plt

class PairTrading:
    def __init__(self, regressor):
        self.regressor = regressor()
        self.residual = None
        self.result = None

    def fit(self, series1, series2):
        self.regressor.fit(x_train=series1.reshape(-1, 1), y_train=series2.reshape(-1, 1))
        prediction = self.regressor.predict(series1.reshape(-1, 1))
        self.residual = (prediction - series2.reshape(-1, 1)).ravel()
        self.result = DA.TimeSeriesAnalysis.adfuller(self.residual)
        return self.result

    # return the days in which we should go in and we should go out.
    def simulate(self, in_threshold, out_threshold, series=None):
        if series is None:
            series = self.residual
        plt.plot(series)
        plt.show()
        in_set = set()
        for _in in in_threshold:
            in_days = np.where(np.abs(series) > _in)[0]
            print(in_days)
            for day in in_days:
                in_set.add(day)

        out_set = set()
        for _out in out_threshold:
            out_days = np.where(np.abs(series) < _out)[0]
            # print(out_days)
            for day in out_days:
                out_set.add(day)

        return in_set, out_set
