import numpy as np
import matplotlib.pyplot as plt


class FinAnalysis:

    @staticmethod
    def PricesActions(prices, actions, low=None, high=None):
        assert len(prices) == len(actions)
        if low is None:
            low = 0
        if high is None:
            high = len(prices) 
        
        prices = prices[low:high]
        actions = actions[low:high]
        plt.clf()

        prices_plot = plt.plot(prices, linewidth=2, c='b')
        
        buy_indices = []
        sell_indices = []
        buy_prices_indices = []
        sell_prices_indices = []

        for action, price, index in zip(actions, prices, range(len(prices))):
            if action == 1:
                buy_indices.append(index)
                buy_prices_indices.append(price)
            if action == -1:
                sell_indices.append(index)
                sell_prices_indices.append(price)

        buy_scatter = plt.scatter(buy_indices, buy_prices_indices, c='r')
        sell_scatter = plt.scatter(sell_indices, sell_prices_indices, c='g')

        names = ['prices', 'buy', 'sell']
        
        plt.legend(handles=[prices_plot[0], buy_scatter, sell_scatter], labels=names, loc='best')

    @staticmethod
    def AccumulatedProfit(prices, actions, low=0, high=None):
        assert len(prices) == len(actions)
        if low is None:
            low = 0
        if high is None:
            high = len(prices) - 1

        prices = prices[low:high]
        actions = actions[low:high]

        buys_size = 0
        sells_size = 0
        balance = 0
        position = 0

        balance_list = list()
        asset_list = list()
        position_list = list()

        for action, price in zip(actions, prices):
            if action == 1:
                balance -= price
                position += 1
                buys_size += 1
            elif action == 0:
                pass
            elif action == -1:
                balance += price
                position -= 1
                sells_size += 1
            asset = position * price + balance
            asset_list.append(asset)
            balance_list.append(balance)
            position_list.append(position)
        plt.clf()
        try:
            plt.plot(asset_list)
            plt.ylabel('Asset')
            plt.xlabel('days')
            print('Number of buys:', buys_size)
            print('Number of sells:', sells_size)
            print('Largest negative open interests:', np.min(position_list))
            print('Largest positive open interests:', np.max(position_list))
            print('Last position:', position_list[-1])
            print('Asset:', asset_list[-1])
            print('Last price', prices[-1])
        except IndexError:
            print('No transactions happened.')

        try:
            return asset_list[-1], np.min(position_list)
        except IndexError:
            return 0

    @staticmethod
    def OneLotAccumulatedProfit(prices, actions, low=None, high=None):
        assert len(prices) == len(actions)
        if low is None:
            low = 0
        if high is None:
            high = len(prices) - 1
        
        prices = prices[low:high]
        actions = actions[low:high]

        num_trades = 0
        balance = 0
        position = 0
        asset_list = list()
        
        last_action = actions[0]
        for action, price in zip(actions, prices):
            if last_action != action and action != 0:
                num_trades += 1
                if action == 1:
                    balance -= price
                    position += 1
                    
                elif action == -1:
                    balance += price
                    position -= 1
                
                last_action = action
                
                asset = position * price + balance
            
                asset_list.append(asset)

        try:
            plt.clf()
        except:
            pass

        try:
            plt.plot(asset_list)
            plt.ylabel('Asset')
            plt.xlabel('days')
            print('Number of trades:', num_trades)
            print('Asset:', asset_list[-1])
            print('Last price', prices[-1])
        except:
            print('No transactions happened.')
        
        try:
            return asset_list[-1]
        except:
            return 0
