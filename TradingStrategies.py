import numpy as np
from DataAnalysis import DataAnalysis as DA
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
import sys
import warnings
from ClassifierAndRegressor.ParametricModel import PRegressor as PR



def save_file(df, filename):
    this_dir, _ = os.path.split(__file__)
    data_path = os.path.join(this_dir, 'data', filename)
    df.to_excel(data_path)


class PairTrading:

    def __init__(self, regressor, reg_window_size=20, mean_std_window_size=10):
        self.regressor = regressor()
        self.mean_std_window_size = mean_std_window_size
        self.reg_window_size = reg_window_size
        self.residual = None
        self.result = None
        self.dates = None
        self.maximum_position = 1
        # rolling_alpha, rolling_beta, rolling_std, and rolling_mean are dataframes.
        self.rolling_alpha = None
        self.rolling_beta = None
        self.rolling_std = None
        self.rolling_mean = None
        self.model = PR.ExtendedPandasRollingOLS(window_size=reg_window_size)

    # series1 and series2 should be two dataframes, index being the dates.
    def fit(self, series1, series2):
        self.dates = series1.index
        _series1 = series1.iloc[:, 0].to_frame()
        _series2 = series2.iloc[:, 0].to_frame()
        self.model.fit(x_train=_series1, y_train=_series2)
        self.rolling_alpha = self.model.regressor.alpha
        self.rolling_beta = self.model.regressor.beta
        print(type(_series2), _series2.shape)
        prediction = self.model.predict(_series1)

        residual = (prediction - _series2.reshape(-1, 1)).ravel()
        rolling_mean = pd.Series(residual).rolling(self.mean_std_window_size).mean().to_frame()
        self.rolling_mean = rolling_mean.iloc[self.mean_std_window_size:, :]
        # self.rolling_mean.index = self.dates
        rolling_std = pd.Series(residual).rolling(self.mean_std_window_size).std().to_frame()
        self.rolling_std = rolling_std.iloc[self.mean_std_window_size:, :]
        # self.rolling_std.index = self.dates
        self.residual = pd.DataFrame(residual, index=self.dates)
        self.result = DA.TimeSeriesAnalysis.adfuller(residual)
        return self.result

    # return the days in which we should go in and we should go out.
    # the series should be a dataframe whose index should be the date.
    def simulate(self, in_threshold, out_threshold, plot=False):

        days = list()
        actions = list()

        last_price = None
        position = 0
        earn = 0
        earns = list()
        residuals = list()
        for i, num in enumerate(self.residual):
            if abs(num) > in_threshold and abs(position) < self.maximum_position:
                if position != 0:
                    if num > in_threshold and position < 0:
                        earn += (num-last_price)
                        position = 1
                        last_price = num
                        actions.append('o&s')
                        days.append(i)
                        earns.append(earn)
                        residuals.append(self.residual[i])
                    elif num < in_threshold and position > 0:
                        earn += (last_price-num)
                        position = -1
                        last_price = num
                        actions.append('o&l')
                        days.append(i)
                        earns.append(earn)
                        residuals.append(self.residual[i])
                else:
                    days.append(i)
                    if num > 0:
                        position += 1
                        last_price = num
                        actions.append('s')
                        residuals.append(self.residual[i])
                        earns.append('')
                    else:
                        position -= 1
                        last_price = num
                        actions.append('l')
                        earns.append('')
                        residuals.append(self.residual[i])
            elif abs(num) < out_threshold and position != 0:
                position = 0
                earn += abs(last_price - num)
                days.append(i)
                actions.append('o')
                residuals.append(self.residual[i])
                last_price = 0
                earns.append(earn)
        if plot:
            plt.plot(self.residual)
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
