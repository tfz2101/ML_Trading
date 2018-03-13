
import pandas as pd
import numpy as np
import datetime as dt
import time as time
from util import get_data, plot_data
import matplotlib.pyplot as plt

start_date = dt.datetime(2006,1,3)
end_date = dt.datetime(2009,12,31)
datesIndex = pd.date_range(start_date,end_date,freq='1D').tolist()
symbols = ['IBM']
IBM_Data = get_data(symbols,datesIndex,addSPY=False)
IBM_Data= IBM_Data.dropna()


def getIndicatorPlot(indicator,title,xlabel,ylabel):
    ax = indicator.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def getMACDValues(data,slowFreq, fastFreq, stdFreq, plot = False):
    slow_MA = pd.rolling_mean(data,slowFreq)
    fast_MA = pd.rolling_mean(data,fastFreq)
    diff = fast_MA - slow_MA
    diff_std = pd.rolling_std(arg=diff,window=stdFreq)

    MACD = diff/diff_std

    '''
    df = pd.DataFrame()d
    df['fast MA'] = fast_MA['IBM']
    df['slow MA'] = slow_MA['IBM']
    df['Difference'] =  diff['IBM']
    print(df)
    df.plot()
    plt.show()
    '''
    return MACD

def getRSIValues(data,freq):
    diffs = data.diff(1)
    def RSI(data):
        #print('raw row',data)
        avg_up = data[data>=0].mean()
        #print('avg up',avg_up)
        avg_down = abs(data[data<0].mean())
        #print("avg_down",avg_down)

        rsi = avg_up/avg_down
        if avg_up/avg_up != 1: rsi = 0
        elif avg_down/avg_down != 1: rsi = 99
        #print('rsi',rsi)
        return 100-100/(1+rsi)

    out = pd.rolling_apply(arg=diffs,window=freq,func=RSI,min_periods=freq)
    '''
    out = out.rename(columns={'IBM':'RSI'})
    getIndicatorPlot(out,'RSI','Date','RSI')
    '''
    return out

def getBollingerValues(data, freq, plot = False):
    MA = pd.rolling_mean(data,freq)
    std = pd.rolling_std(arg=data,window=freq)
    dev = data - MA
    dev_std = dev / std
    '''
    upper = pd.DataFrame(index=data.index.values,columns=['Upper_Band'])
    upper['Upper_Band']=MA[:]+std[:]
    lower = pd.DataFrame(index=data.index.values,columns=['Lower_Band'])
    lower = upper['Upper_Band']=MA[:]-std[:]
    '''

    '''
    dev_std = dev_std.rename(columns={'IBM':'STD'})
    getIndicatorPlot(dev_std,'STD from SMA','Date','STD')
    '''

    return dev_std



#getMACDValues(IBM_Data,50,15,50,True)
#getRSIValues(IBM_Data,30)
getBollingerValues(IBM_Data,30)
