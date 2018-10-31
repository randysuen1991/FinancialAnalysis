import numpy as np
from DataAnalysis import DataAnalysis as DA
import matplotlib.pyplot as plt


class PairTrading:

    def __init__(self, regressor):
        self.regressor = regressor()
        self.residual = None
        self.result = None
        self.maximum_position = 1
        self.dates = None

    def fit(self, series1, series2, dates):
        self.dates = dates
        self.regressor.fit(x_train=series1.reshape(-1, 1), y_train=series2.reshape(-1, 1))
        prediction = self.regressor.predict(series1.reshape(-1, 1))
        self.residual = (prediction - series2.reshape(-1, 1)).ravel()
        self.result = DA.TimeSeriesAnalysis.adfuller(self.residual)
        return self.result

    def transform(self, series1, series2, dates):
        self.dates = dates
        prediction = self.regressor.predict(series1.reshape(-1, 1))
        residual = (prediction - series2.reshape(-1, 1)).ravel()
        return residual, DA.TimeSeriesAnalysis.adfuller(residual)

    # return the days in which we should go in and we should go out.
    def simulate(self, in_threshold, out_threshold, series=None, plot=False):
        if series is None:
            series = self.residual

        std = np.std(series)
        mean = np.mean(series)

        series_std = series - mean
        series_std /= std

        days = list()
        actions = list()

        last_price = None
        position = 0
        earn = 0
        earns = list()
        residuals = list()
        for i, num in enumerate(series_std):
            if abs(num) > in_threshold and abs(position) < self.maximum_position:
                if position != 0:
                    if num > in_threshold and position < 0:
                        earn += (num-last_price)
                        position = 1
                        last_price = num
                        actions.append('o&s')
                        days.append(i)
                        earns.append(earn)
                        residuals.append(series[i])
                    elif num < in_threshold and position > 0:
                        earn += (last_price-num)
                        position = -1
                        last_price = num
                        actions.append('o&l')
                        days.append(i)
                        earns.append(earn)
                        residuals.append(series[i])
                else:
                    days.append(i)
                    if num > 0:
                        position += 1
                        last_price = num
                        actions.append('s')
                        residuals.append(series[i])
                        earns.append('')
                    else:
                        position -= 1
                        last_price = num
                        actions.append('l')
                        earns.append('')
                        residuals.append(series[i])
            elif abs(num) < out_threshold and position != 0:
                position = 0
                earn += abs(last_price - num)
                days.append(i)
                actions.append('o')
                residuals.append(series[i])
                last_price = 0
                earns.append(earn)
        if plot:
            plt.plot(series)
            plt.show()

        return position, earns, list(self.dates[days]), actions, residuals


class CumulativeVolumeTrading:

    def __init__(self, stocks, mean_dict, std_dict):
        self.stocks = stocks
        self.mean_dict = mean_dict
        self.std_dict = std_dict

    def start(self, df, n_std, threshold):
        long_count = 0
        short_count = 0

        for stock in self.stocks:
            cum_ask = df.loc[stock, 'cum_ask']
            cum_bid = df.loc[stock, 'cum_bid']
            n = ((cum_ask - cum_bid) - self.mean_dict[stock]) / self.std_dict[stock]
            if n > n_std:
                long_count += 1
            elif n < -n_std:
                short_count += 1

        print('Number of long:', long_count)
        print('Number of short:', short_count)

        if long_count - short_count >= threshold:
            print('Long!')
        elif long_count - short_count <= -threshold:
            print('Short!')