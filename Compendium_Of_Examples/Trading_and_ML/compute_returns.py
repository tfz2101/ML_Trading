"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def normalize_data(df):
    return df/df.ix[0,:]

def assess_portfolio(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0, \
    gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = prices_SPY # add code here to compute daily portfolio values
    prices.fillna(method='ffill')
    sv_port = float(sv)/np.dot(prices.iloc[0,:].values,np.array(allocs).transpose())
    prices = normalize_data(prices)
    prices_SPY =  normalize_data(prices_SPY)
    print('prices')
    print(prices)
    mv = prices.copy()
    for i in range(0,prices.shape[0]):
        #mv.iloc[i,:] = [x*sv_port for x in allocs]
        mv.iloc[i,:] = [x for x in allocs]
    mv = pd.DataFrame(mv.values.transpose(),index = mv.columns.values, columns=mv.index.values)

    temp = np.dot(prices.values,mv.values)
    out = pd.DataFrame(temp)
    daily_mv = pd.DataFrame(out.iloc[:,1].values,index = prices.index.values,columns=['MV'])



    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    cr = (daily_mv.iloc[daily_mv.shape[0]-1,0]-daily_mv.iloc[0,0])/daily_mv.iloc[0,0]
    adr = (daily_mv.pct_change(periods=(252/sf),axis=0)).iloc[:,0].mean(axis=0, skipna=True)
    daily_ret = (daily_mv.pct_change(periods=1,axis=0))

    sddr = daily_ret.std(skipna=True).iloc[0]
    avg_daily_ret = (daily_mv.pct_change(periods=1,axis=0)).iloc[:,0].mean(axis=0,skipna=True)
    sr = (avg_daily_ret - rfr)/sddr * 252**(0.5)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([daily_mv, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp)


    # Add code here to properly compute end value
    ev = daily_mv.iloc[daily_mv.shape[0]-1,0]

    return cr, adr, sddr, sr, ev

def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,\
        syms = symbols, \
        allocs = allocations,\
        sv = start_val, rfr=risk_free_rate, sf = sample_freq, \
        gen_plot = True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
