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

    def __init__(self, reg_window_size=20, mean_std_window_size=10, rolling_reg=True, rolling_mean_std=True):
        self.mean_std_window_size = mean_std_window_size
        self.reg_window_size = reg_window_size
        self.residual = None
        self.result = None
        self.dates = None
        self.maximum_position = 1

        # rolling_alpha, rolling_beta, rolling_std, and rolling_mean are dataframes.
        self.rolling_reg = rolling_reg
        self.rolling_mean_std = rolling_mean_std

        if rolling_reg:
            self.rolling_alpha = None
            self.rolling_beta = None
            self.rolling_rsq = None
        if rolling_mean_std:
            self.rolling_std = None
            self.rolling_mean = None

        self.raw_residual = None
        self.regressor = PR.ExtendedPandasRollingOLS(window_size=reg_window_size)

    # series1 and series2 should be two dataframes, index being the dates.
    def fit(self, series1, series2):
        self.regressor.fit(x_train=series1, y_train=series2)
        if self.rolling_reg:
            rolling_alpha = self.regressor.regressor.alpha.values[0:-1]
            dates = self.regressor.regressor.alpha.index[1:]
            self.rolling_alpha = pd.Series(data=rolling_alpha, index=dates)
            rolling_beta = self.regressor.regressor.beta.values[0:-1]
            self.rolling_beta = pd.Series(data=rolling_beta.ravel(), index=dates)
            rolling_rsq = self.regressor.regressor.rsq.values[0:-1]
            self.rolling_rsq = pd.Series(data=rolling_rsq, index=dates)

            # drop the first self.reg_window_size returns.
            series1 = series1.drop(series1.index[:self.reg_window_size], axis=0)
            series2 = series2.drop(series2.index[:self.reg_window_size], axis=0)
            prediction = self.regressor.predict(series1)
            residual = (prediction - series2.values.ravel())
        else:
            prediction = self.regressor.predict(series1)
            residual = (prediction - series2.values.ravel())

        self.raw_residual = pd.DataFrame(data=residual, index=series1.index)

        if self.rolling_mean_std:

            self.rolling_rsq = self.rolling_rsq.iloc[self.mean_std_window_size:]
            self.rolling_alpha = self.rolling_alpha.iloc[self.mean_std_window_size:]
            self.rolling_beta = self.rolling_beta.iloc[self.mean_std_window_size:]

            rolling_mean = pd.Series(residual).rolling(self.mean_std_window_size).mean().to_frame()
            rolling_mean.index = series1.index
            index = rolling_mean.index[self.mean_std_window_size:]
            self.rolling_mean = rolling_mean.iloc[self.mean_std_window_size-1:-1, :]
            self.rolling_mean.index = index
            rolling_std = pd.Series(residual).rolling(self.mean_std_window_size).std().to_frame()
            rolling_std.index = series1.index
            self.rolling_std = rolling_std.iloc[self.mean_std_window_size-1:-1, :]
            self.rolling_std.index = index
            residual = pd.DataFrame(residual[self.mean_std_window_size:], index=self.rolling_std.index)
            residual -= self.rolling_mean
            residual /= self.rolling_std
        else:
            residual -= np.mean(residual)
            residual /= np.std(residual)

        self.residual = residual
        self.result = DA.TimeSeriesAnalysis.adfuller(residual.values.ravel())

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
        count = 0
        beta = list()
        for i, num in zip(self.residual.index, self.residual.values):
            num = num[0]
            if abs(num) > in_threshold and abs(position) < self.maximum_position:
                beta.append(self.rolling_beta.iloc[count])
                days.append(i)
                if position != 0:
                    if num > in_threshold and position < 0:
                        earn += (num-last_price)
                        position = 1
                        last_price = num
                        actions.append('o&s')
                        earns.append(earn)
                        residuals.append(num)
                    elif num < in_threshold and position > 0:
                        earn += (last_price-num)
                        position = -1
                        last_price = num
                        actions.append('o&l')
                        earns.append(earn)
                        residuals.append(num)
                else:
                    if num > 0:
                        position += 1
                        last_price = num
                        actions.append('s')
                        residuals.append(num)
                        earns.append('')
                    else:
                        position -= 1
                        last_price = num
                        actions.append('l')
                        earns.append('')
                        residuals.append(num)
            elif abs(num) < out_threshold and position != 0:
                beta.append(self.rolling_beta.iloc[count])
                position = 0
                earn += abs(last_price - num)
                days.append(i)
                actions.append('o')
                residuals.append(num)
                last_price = 0
                earns.append(earn)

            elif in_threshold > abs(num) > out_threshold and position != 0:
                if num < 0 and position < 0:
                    beta.append(self.rolling_beta.iloc[count])
                    days.append(i)
                    position = 0
                    actions.append('o')
                    earn += abs(last_price - num)
                    residuals.append(num)
                    last_price = 0
                    earns.append(earn)

                elif num > 0 and position > 0:
                    beta.append(self.rolling_beta.iloc[count])
                    days.append(i)
                    position = 0
                    actions.append('o')
                    earn += abs(last_price - num)
                    residuals.append(num)
                    last_price = 0
                    earns.append(earn)

            count += 1

        if plot:
            plt.plot(self.residual)
            plt.show()
        return position, earns, days, actions, residuals, beta


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
