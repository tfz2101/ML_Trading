"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch
"""

import datetime as dt
import QLearner as ql
import pandas as pd
import util as ut
import indicators
import numpy as np
import random as rand
from util import get_data

class StrategyLearner(object):

    # constructor
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.positionDict = {0:-500,1:0,2:500}

    #helper functions
    def getIndicatorStates(self, indicator,bins):
        inds = indicator.values
        out = np.digitize(inds,bins)
        return out

    def getStateFromDirects(self, directs):
        out = 1000+ directs[0] * 1000 + directs[1] * 100 + directs[2] * 10 + directs[3]
        return out

    def getDirectsFromState(self, state):
        strState = str(state)
        out = []
        for i in range(0,len(strState)):
            out.append(int(strState[i]))
        out[0]=out[0]-1
        return out

    def computeOneDayPnl(self, prices, day0,day1,position):
        assert isinstance(prices,pd.Series),"Prices has to be a Series, not DataFrame"
        shares = self.positionDict[position]
        diff = prices.iloc[day1] - prices.iloc[day0]
        pnl = shares * diff
        return pnl

    # this method should create a QLearner, and train it for trading
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2008,1,1), \
        ed=dt.datetime(2009,1,1), \
        sv = 10000): 

        datesIndex = pd.date_range(sd,ed,freq='1D').tolist()
        symbols = [symbol]
        Stock_Data = get_data(symbols,datesIndex,addSPY=False)
        Stock_Data= Stock_Data.dropna()

        MACD = indicators.getMACDValues(Stock_Data,50,15,50)
        RSI = indicators.getRSIValues(Stock_Data,30)
        Boll = indicators.getBollingerValues(Stock_Data,30)
        indicator = pd.concat([MACD,RSI,Boll,Stock_Data], axis = 1, join='inner')
        indicator = indicator.dropna()
        indicator = indicator.reset_index(drop=True)
        print('Indicator',indicator)
        self.MACD_Bins = [0.5,0.75,1.5,2.25,3.0]
        self.RSI_Bins = [10.0,30.0,50.0,70.0,90.0]
        self. Boll_Bins = [0.5,0.75,1.5,2.25,3.0]
        MACD_S = self.getIndicatorStates(indicator.iloc[:,0],self.MACD_Bins)
        RSI_S = self.getIndicatorStates(indicator.iloc[:,1],self.RSI_Bins)
        Boll_S = self.getIndicatorStates(indicator.iloc[:,2],self.Boll_Bins)
        #print('MACD_S',MACD_S)
        #print('RSI_S',RSI_S)
        #print('Boll_S',Boll_S)

        num_states = 6553
        num_actions = 3
        alpha = 0.2
        gamma = 0.9
        rar = 0.5
        radr = 0.99

        self.learner = ql.QLearner(num_states = num_states,num_actions=num_actions,alpha=alpha,gamma=gamma,rar=rar,radr=radr)
        learner = self.learner
        initial_s = [MACD_S[0],RSI_S[0],Boll_S[0],1]
        initial_s = self.getStateFromDirects(initial_s)
        print('initial_s',initial_s)

        # add your code to do learning here
        rand.seed(5)
        for iteration in range(0,1):
            action = self.learner.querysetstate(initial_s) #set the state and get first action
            for i in range(1,indicator.shape[0]):

                #move to new location according to action and then get a new action
                s_prime = self.getStateFromDirects([MACD_S[i],RSI_S[i],Boll_S[i],action])
                pos = self.getDirectsFromState(learner.s)[3]
                r = self.computeOneDayPnl(indicator.iloc[:,3],i-1,i,pos)
                action = learner.query(s_prime,r)

        '''
        # example use with new colname 
        volume_all = ut.get_data(syms, dates, colname = "Volume")  # automatically adds SPY
        volume = volume_all[syms]  # only portfolio symbols
        volume_SPY = volume_all['SPY']  # only SPY, for comparison later
        if self.verbose: print volume
        '''

    # this method should use the existing policy and test it against new data
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):

        # here we build a fake set of trades
        # your code should return the same sort of data

        datesIndex = pd.date_range(sd,ed,freq='1D').tolist()
        symbols = [symbol]
        Stock_Data = get_data(symbols,datesIndex,addSPY=False)
        Stock_Data= Stock_Data.dropna()

        qTab = self.learner.qTab

        MACD = indicators.getMACDValues(Stock_Data,50,15,50)
        RSI = indicators.getRSIValues(Stock_Data,30)
        Boll = indicators.getBollingerValues(Stock_Data,30)
        indicator = pd.concat([MACD,RSI,Boll,Stock_Data], axis = 1, join='inner')
        indicator = indicator.dropna()
        indicator = indicator.reset_index(drop=True)
        print('Indicator',indicator)
        self.MACD_Bins = [0.5,0.75,1.5,2.25,3.0]
        self.RSI_Bins = [10.0,30.0,50.0,70.0,90.0]
        self. Boll_Bins = [0.5,0.75,1.5,2.25,3.0]
        MACD_S = self.getIndicatorStates(indicator.iloc[:,0],self.MACD_Bins)
        RSI_S = self.getIndicatorStates(indicator.iloc[:,1],self.RSI_Bins)
        Boll_S = self.getIndicatorStates(indicator.iloc[:,2],self.Boll_Bins)
        print('MACD_S',MACD_S)
        print('RSI_S',RSI_S)
        print('Boll_S',Boll_S)

        num_states = 6553
        num_actions = 3
        alpha = 0.2
        gamma = 0.9
        rar = 0.5
        radr = 0.99

        trades =  indicator.iloc[:,3].copy()
        trades.values[:] = 0 # set them all to nothing

        for i in range(0,trades.shape[0]):


        '''
        trades.values[8,:] = 1000 # add a BUY at the 9th date
        if self.verbose: print type(trades) # it better be a DataFrame!
        if self.verbose: print trades
        if self.verbose: print prices_all
        '''
        return trades

if __name__=="__main__":
    print "One does not simply think up a strategy"


strat = StrategyLearner()
strat.addEvidence()
strat.testPolicy()