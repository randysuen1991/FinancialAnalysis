import numpy as np
import matplotlib.pyplot as plt



class FinAnalysis():
    def AccumulatedProfit(prices,actions,low=None,high=None):
        if low == None :
            low = 0
        if high == None :
            high = len(prices) - 1
        
        prices = prices[low:high]
        actions = actions[low:high]
        
        
        
        balance = 0
        position = 0
        asset = 0
        
        
        asset_list = list()
        
        for action, price in zip(actions,prices):
            
            if action == 1 :
                balance -= price
                position += 1
            elif action == 0 :
                pass
            elif action == -1 :
                balance += price
                position -= 1
            asset = position * price + balance
            asset_list.append(asset)
        
        
        
        try :
            plt.clf()
        except :
            pass
        
        
        
        plt.plot(asset_list)
        plt.ylabel('Asset')
        plt.xlabel('days')
        print('Position:',position)
        print('Balance:',balance)
        print('Last price',prices[-1])  
        
    def OneLotAccumulatedProfit(prices,actions,low=None,high=None):
        if low == None :
            low = 0
        if high == None :
            high = len(prices) - 1
        
        prices = prices[low:high]
        actions = actions[low:high]
        
        
        
        balance = 0
        position = 0
        asset = 0
        
        
        asset_list = list()
        
        last_action = actions[0]
        for action, price in zip(actions,prices):
            if last_action != action and action != 0:
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
        
        
        
        plt.plot(asset_list)
        plt.ylabel('Asset')
        plt.xlabel('days')
        print('Position:',position)
        print('Balance:',balance)
        print('Last price',prices[-1])