import numpy as np
from DataAnalysis import DataAnalysis as DA
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import sys


def save_file(df, filename):
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, 'data', filename)
    df.to_excel(data_path)


class PairTrading:

    def __init__(self, regressor, window_size=10):
        self.regressor = regressor()
        self.window_size = 10
        self.residual = None
        self.result = None
        self.maximum_position = 1
        self.dates = None
        self.rolling_std = None
        self.rolling_mean = None

    def fit(self, series1, series2, dates):
        self.dates = dates
        self.regressor.fit(x_train=series1.reshape(-1, 1), y_train=series2.reshape(-1, 1))
        prediction = self.regressor.predict(series1.reshape(-1, 1))
        residual = (prediction - series2.reshape(-1, 1)).ravel()
        rolling_mean = pd.Series(residual).rolling(self.window_size).mean()
        self.rolling_mean = rolling_mean.to_frame()
        self.rolling_mean.index = dates
        rolling_std = pd.Series(residual).rolling(self.window_size).std()
        self.rolling_std = rolling_std.to_frame()
        self.rolling_std.index = dates
        self.residual = pd.DataFrame(residual, index=self.dates)
        self.result = DA.TimeSeriesAnalysis.adfuller(residual)
        return self.result

    def transform(self, series1, series2, dates):
        self.dates = dates
        prediction = self.regressor.predict(series1.reshape(-1, 1))
        residual = (prediction - series2.reshape(-1, 1)).ravel()
        return residual, DA.TimeSeriesAnalysis.adfuller(residual)

    # return the days in which we should go in and we should go out.
    # the series should be a dataframe whose index should be the date.
    def simulate(self, in_threshold, out_threshold, series=None, plot=False):
        if series is None:
            series = self.residual[self.window_size-1:]
            series_std = series - self.rolling_mean.iloc[self.window_size-1:, :]
            series_std /= self.rolling_std.iloc[self.window_size-1:, :]
            series_std = series_std.values
            series_std = series_std.ravel()
            series_std = np.around(series_std, 5)
            series = series.values
        else:
            index = series.index
            if index[0] in self.dates[0: self.window_size]:
                raise ValueError('Please try starting later days')
            else:
                pass
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

        return position, earns, list(self.dates[days]), actions, residuals, series_std[days]


class CumulativeVolumeTrading:

    def __init__(self, stocks, mean_std_dict, start_time, end_time):
        self.stocks = stocks
        self.mean_std_dict = mean_std_dict
        self.start_time = start_time
        self.end_time = end_time
        str_name = 'vol_diff_'
        for time in self.start_time + self.end_time:
            str_name += time
        self.str_name = str_name
        self.diff_series = list()

    def trade(self, df, n_std, threshold):
        long_count = 0
        short_count = 0
        long_list = list()
        short_list = list()
        for stock in self.stocks:
            vol_diff = (df.loc[stock, self.str_name] - self.mean_std_dict[int(stock)][0]) / \
                        self.mean_std_dict[int(stock)][1]
            if vol_diff > n_std:
                long_count += 1
                long_list.append((stock, round(vol_diff, 2)))
            elif vol_diff < -n_std:
                short_count += 1
                short_list.append((stock, round(vol_diff, 2)))

        print('Number of longs:', long_count)
        print('Stocks of the longs:', long_list)
        print('Number of shorts:', short_count)
        print('Stocks of the shorts:', short_list)
        print('Differential:', long_count - short_count)

        self.diff_series.append(long_count - short_count)

        if long_count - short_count >= threshold:
            # print('Long!')
            pass
        elif long_count - short_count <= -threshold:
            # print('Short!')
            pass


class AdditiveTrading:

    def __init__(self, stocks, features, models):
        self.stocks = stocks
        self.features = features
        self.models = models
        self.results_num = list()
        self.results_dec = list()

    def trade(self, df):
        for iteration, stock in enumerate(self.stocks):
            stock = str(stock)
            features = df.loc[stock, self.features].values
            result_num = self.models[iteration].predict(features)[0]
            self.results_num.append(result_num)
            if result_num >= 0:
                self.results_dec.append(1)
            else:
                self.results_dec.append(0)

        average = np.mean(self.results_num)
        decision = stats.mode(self.results_dec)[0][0]
