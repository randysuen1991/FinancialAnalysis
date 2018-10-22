import numpy as np
from DataAnalysis import DataAnalysis as DA


class PairTrading:
    def __init__(self, regressor):
        self.regressor = regressor()
        self.residual = None

    def fit(self, series1, series2):
        self.regressor.fit(x_train=series1, y_train=series2)
        prediction = self.regressor.predict(series1)
        self.residual = (series2-prediction).ravel()
        result = DA.TimeSeriesAnalysis.adfuller(self.residual)
        return result

    def simulate(self, in_threshold, out_threshold, series=None):
        if series is None:
            series = self.residual

        in_set = set()
        for _in in in_threshold:
            in_days = np.where(np.abs(series) > _in)[0]
            in_set.add(in_days)

        out_set = set()
        for _out in out_threshold:
            out_days = np.where(np.abs(series) < _out)[0]
            out_set.add(out_days)

        return in_set, out_set
