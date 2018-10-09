import pandas as pd
import numpy as np
import datetime as dt
import time as time
import os

import arch.unitroot as UnitRoot
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf, adfuller
from operator import itemgetter
from sklearn import linear_model as LM
from sklearn.decomposition import PCA
from Stat_Fcns import acf_fcn_highestlag,acf_fcn_highestlag_P_Val
from openpyxl import load_workbook


def getTripleBarrier(data):
    #@FORMAT: data = df(prices,  index=dates)
    pass

def getNextExecutionLevels(data):
    #@FORMAT: data = df(prices,... index=dates)
    data_out = data.copy()
    for i in range(0, data.shape[0]):
        buy_found = False
        sell_found = False
        for ii in range(i, data.shape[0]):
            if buy_found and sell_found:
                break
            if data.iloc[ii+1,0] < data.iloc[ii,0]:
                data_out.ix[i, 'next_buy'] = data.iloc[ii,0]
                buy_found = True
            if data.iloc[ii+1,0] > data.iloc[ii,0]:
                data_out.ix[i, 'next_sell'] = data.iloc[ii,0]
                sell_found = True
    #@RETURN: data = df(prices, ..., next_buy, next_sell, index=dates)
    return data_out

#Applies 'fcn' to every block. It doesn't roll for every datapoint
def rolling_block_data_fcn(data,fcn,gap=5,*args,**kwargs):
    #@FORMAT: data = df(data,index=dates)
    dates = data.index.values
    values = data.values
    out = []
    out.append([dates[0],0])
    for i in range(0,values.shape[0],gap):
        block_values = values[i:i+gap]
        stat = fcn(block_values,**kwargs)
        out.append([dates[i],stat])
    return out

#Applies 'fcn' to every datapoint for a given lookback
def rolling_data_fcn(data,fcn,gap=5,*args,**kwargs):
    #@FORMAT: data = df(data,index=dates)
    dates = data.index.values
    values = data.values
    out = []
    out.append([dates[0],0])
    for i in range(0,values.shape[0],1):
        block_values = values[i:i+gap]
        stat = fcn(block_values,**kwargs)
        out.append([dates[i],stat])
    return out

def rolling_data_fcn2(data,fcn,gap=5,*args,**kwargs):
    #@FORMAT: data = np.array
    out = np.empty([data.shape[0],1])
    out[:] = np.nan
    for i in range(0,data.shape[0],1):
        block_values = data[i:i+gap]
        stat = fcn(block_values,**kwargs)
        out.append([stat])
    return out




#Calculate correlation when two signals only have for certain dates, this calculates the correlation when the two sigals do overlap.
def calcSignalCorrelation(data):
    #@FORMAT: data = df(signal1_hr,signal2_hr,index=dates)
    newData = data.dropna()
    corr = data.corr()
    return corr, 1.0*newData.shape[0]/data.shape[0]


#Applies each 'fcn' to a rolling block of data
def getRollingTraits(data,fcn_list,gap=5,*args,**kwargs):
    #@FORMAT: data = df(price,index=dates) - Only Price Column
    #@FYI: Each 'fcn' needs to be able to take 1D numpy array and outputs one value

    values = data.values
    traits_data = pd.DataFrame(index=data.index.values, columns=range(0, len(fcn_list)))

    for i in range(gap,traits_data.shape[0],1):
        block_values = values[(i-gap):i,:]
        stats = []
        for fcn in fcn_list:
            stats.append(fcn(block_values))
        traits_data.loc[traits_data.index.values[i]] = stats

    #@RETURNS: traits_df = df(traits,index=dates)
    return traits_data




#Makes a new excel sheet and writes to it
def write_new(datadf, path, tab="Sheet1"):
    WRITE_PATH = path
    writer = pd.ExcelWriter(WRITE_PATH, engine='xlsxwriter')
    datadf.to_excel(writer, sheet_name=tab)
    writer.save()

#Takes existing excel file and writes to specified sheet
def write(datadf, path, tab="Sheet1"):
    WRITE_PATH = path
    book = load_workbook(path)
    writer = pd.ExcelWriter(WRITE_PATH, engine='openpyxl')
    writer.book = book
    datadf.to_excel(writer, sheet_name=tab)
    writer.save()

