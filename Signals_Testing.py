import pandas as pd
import numpy as np
import datetime as dt
import time as time
import os

from pykalman import KalmanFilter
import arch.unitroot as UnitRoot
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf, adfuller
from operator import itemgetter
from sklearn import linear_model as LM
from sklearn.decomposition import PCA
from Stat_Fcns import dickeyfuller_fcn,acf_fcn_highestlag,acf_fcn_highestlag_P_Val


#Computes hit ratio for rolling n trades
def getLastNHitRatio(data, n, hitInd, acceptableValues=[0,1]):
    #@FORMAT: df[dates, hitOrNots]
    out = []
    temp = []
    count = 0
    for i in range(0,data.shape[0]):
        if data.iloc[i,hitInd] in acceptableValues:
            count = count +1
            temp.append(data.values[i,:])
        if count >= n:
            hr = np.average(temp,axis=0)[hitInd]
            out.append([data.values[i,0],hr])
            count = 0
            temp = []

    out = pd.DataFrame(out)
    #RETURNS: df[date, hit ratio for rolling n trades]
    return out

#Computes hit ratio for rolling n trades, computes s.t. that it finds n valid trades first
def getLastNHitRatioEveryLine(data, n, hitInd, acceptableValues=[0,1]):
    #@FORMAT: df[dates, hitOrNots]
    out = data.values
    avgs = np.empty((data.shape[0],1))
    avgs[:] = np.nan
    for i in range(0,data.shape[0]):
        count = 0
        temp = []
        j = i
        while j < data.shape[0]:
            if out[j,hitInd] in acceptableValues:
                count = count +1
                temp.append(out[j,:])
            if count >= n:
                hr = np.average(temp,axis=0)[hitInd]
                avgs[i]=hr
                break
            j = j +1

    print(avgs)
    out =  data.copy()
    out["Last Trades %"] = avgs
    print("out",out)
    #RETURNS: df[date, hit ratio for rolling n trades]
    return out

#Computes hit ratio for rolling blocks of n lines. Does NOT compute for every line on a rolling basis.
def getNBlockHitRatio(data, gap):
    #@FORMAT: series[hitOrNots,index=dates]
    out = []
    for i in range(0,data.shape[0],gap):
        hr = data.iloc[i:(i+gap)].mean(skipna=True)*1.0
        out.append([data.index.values[i],hr])

    out = pd.DataFrame(out)
    #RETURNS: df[date, hit ratio for rolling n trades]
    return out


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



#Returns rolling stats for dickey fuller test, p val for the best fit lag for ACF, realized volatility,
def getDataTraits(data,gap):
    #@FORMAT: data = df(data,index=dates)
    kwargs ={"maxlag":1}
    rolling_df_data = rolling_block_data_fcn(data,dickeyfuller_fcn,gap=gap,**kwargs)
    rolling_df_data = pd.DataFrame(rolling_df_data)

    kwargs ={"lags":1}
    rolling_acf_data = rolling_block_data_fcn(data,acf_fcn_highestlag,gap=gap,**kwargs)
    rolling_acf_data= pd.DataFrame(rolling_acf_data)


    rolling_rl_data = rolling_block_data_fcn(data,rl_fcn,gap=gap)
    rolling_rl_data = pd.DataFrame(rolling_rl_data)

    output = pd.DataFrame(rolling_df_data.iloc[:,1].tolist(),columns=['Dickey Fuller'],index=rolling_df_data.iloc[:,0])
    output['Autocorrelation'] = rolling_acf_data.iloc[:,1].tolist()
    output['RL'] = rolling_rl_data.iloc[:,1].tolist()
    #@RETURNS: df(df_value, [acf_corr, lag_number], rl_data],index=dates]
    return output

def getDataTraitsOnlyPValue(data,gap):
    #Like getDataTraitsOnly but returns only the P Values for the ACF analysis, excludes the best fit lag number
    #@FORMAT: data = df(data,index=dates)
    kwargs ={"maxlag":1}
    rolling_df_data = rolling_block_data_fcn(data,dickeyfuller_fcn,gap=gap,**kwargs)
    rolling_df_data = pd.DataFrame(rolling_df_data)

    kwargs ={"lags":1}
    rolling_acf_data = rolling_block_data_fcn(data,acf_fcn_highestlag_P_Val,gap=gap,**kwargs)
    rolling_acf_data= pd.DataFrame(rolling_acf_data)


    rolling_rl_data = rolling_block_data_fcn(data,rl_fcn,gap=gap)
    rolling_rl_data = pd.DataFrame(rolling_rl_data)

    output = pd.DataFrame(rolling_df_data.iloc[:,1].tolist(),columns=['Dickey Fuller'],index=rolling_df_data.iloc[:,0])
    output['Autocorrelation'] = rolling_acf_data.iloc[:,1].tolist()
    output['RL'] = rolling_rl_data.iloc[:,1].tolist()
    #@RETURNS: df(df_value, [acf_corr, lag_number], rl_data],index=dates]
    return output

#Calculate correlation when two signals only have for certain dates, this calculates the correlation when the two sigals do overlap.
def calcSignalCorrelation(data):
    #@FORMAT: data = df(signal1_hr,signal2_hr,index=dates)
    newData = data.dropna()
    corr = data.corr()
    return corr, 1.0*newData.shape[0]/data.shape[0]



def write(datadf, path, tab="Sheet1"):
    WRITE_PATH = path
    writer = pd.ExcelWriter(WRITE_PATH, engine='xlsxwriter')
    datadf.to_excel(writer, sheet_name=tab)
    writer.save()





DATA_PATH = "L:\Trade_Data.xlsx"
TAB_NAME = "CMOs"
WRITE_PATH = "L:\Trade_Output3.xlsx"
TAB = 'Output'

'''
file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)
data = data.dropna(axis=0)
pcatool = PCAAnalysis(data)
'''

'''
pcatool.getPCA(3)
lincomp = pcatool.createLinearComponent()
X = lincomp.transpose()
print(X)
Y = data.values.transpose()[0]
print(Y)
Y_Pred = pcatool.createLM(X,Y,X)
print(Y_Pred)
'''
