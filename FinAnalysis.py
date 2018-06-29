import numpy as np
import matplotlib.pyplot as plt



class FinAnalysis():
    
    def PricesActions(prices,actions,action_type=1,low=None,high=None):
        assert action_type in [1,0,-1]
        assert len(prices) == len(actions)
        if low == None :
            low = 0
        if high == None :
            high = len(prices) 
        
        prices = prices[low:high]
        actions = actions[low:high]
        
        
        
        try : 
            plt.clf()
        except :
            pass
        
        
        
        prices_plot = plt.plot(prices,linewidth=3,c='b')
        
        indices = []
        prices_indices = []
        
        for action, price, index in zip(actions, prices, range(len(prices))):
            if action == action_type :
                indices.append(index)
                prices_indices.append(price)
        
        actions_scatter = plt.scatter(indices,prices_indices,c='r')
        
        
        if action_type == 1 :
            names = ['prices','buy']
        elif action_type == 0 :
            names = ['prices','hold']
        else :
            names = ['prices','sell']
        
        plt.legend(handles=[prices_plot[0],actions_scatter],labels=names,loc='best')
    
    
    def AccumulatedProfit(prices,actions,low=None,high=None):
        assert len(prices) == len(actions)
        if low == None :
            low = 0
        if high == None :
            high = len(prices) 
        
        prices = prices[low:high]
        actions = actions[low:high]
        
        
        
        buys_size = 0
        sells_size = 0
        balance = 0
        position = 0
        asset = 0
        
        
        asset_list = list()
        
        for action, price in zip(actions,prices):
            
            if action == 1 :
                balance -= price
                position += 1
                buys_size += 1
            elif action == 0 :
                pass
            elif action == -1 :
                balance += price
                position -= 1
                sells_size += 1
            asset = position * price + balance
            asset_list.append(asset)
        
        
        
        try :
            plt.clf()
        except :
            pass
        
        
        try : 
            plt.plot(asset_list)
            plt.ylabel('Asset')
            plt.xlabel('days')
            print('Number of buys:',buys_size)
            print('Number of sells:',sells_size)
            print('Asset:',asset_list[-1])
            print('Last price',prices[-1]) 
        
        except :
            print('No transactions happened.')
            
            
            
        try : 
            return asset_list[-1]
        except :
            return 0
        
    def OneLotAccumulatedProfit(prices,actions,low=None,high=None):
        assert len(prices) == len(actions)
        if low == None :
            low = 0
        if high == None :
            high = len(prices) 
        
        prices = prices[low:high]
        actions = actions[low:high]
        
        
        num_trades = 0
        balance = 0
        position = 0
        asset = 0
        
        
        asset_list = list()
        
        last_action = actions[0]
        for action, price in zip(actions,prices):
            if last_action != action and action != 0:
                num_trades += 1
                if action == 1 :
                    balance -= price
                    position += 1
                    
                elif action == -1 :
                    balance += price
                    position -= 1
                
                last_action = action
                
                asset = position * price + balance
            
                asset_list.append(asset)
            
        
        
        
        try :
            plt.clf()
        except :
            pass
        
        
        try :
            plt.plot(asset_list)
            plt.ylabel('Asset')
            plt.xlabel('days')
            print('Number of trades:',num_trades)
            print('Asset:',asset_list[-1])
            print('Last price',prices[-1])
        except :
            print('No transactions happened.')
        
        try : 
            return asset_list[-1]
        except :
            return 0