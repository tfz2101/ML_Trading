from nolds import hurst_rs, dfa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from Signals_Testing import rolling_block_data_fcn,rolling_data_fcn, write
from ML_functions import getBlendedSignal
from ML_functions import crossValidate
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

'''
DATA_PATH = "Trading_Input.xlsx"
TAB_NAME = "momentum_data"
writer = pd.ExcelWriter('output.xlsx')



file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)
data = data.dropna(axis=0)

hurst_data = data['Change']
hs = rolling_data_fcn(hurst_data,hurst_rs,gap=30)



hs = pd.DataFrame(hs)

ks = rolling_data_fcn(hurst_data,kurtosis,gap=30)
ks = pd.DataFrame(ks)
hs['Kurtosis'] = ks.iloc[:,1]

#hs.to_excel(writer,'hurst')

#writer.save()



DATA_PATH = "Trading_Input.xlsx"
TAB_NAME = "pnl"
file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)


#data = data.dropna(axis=0)
data = data['Pnl']

def sharpe_ratio(pnl):
    pnl = pd.DataFrame(pnl)
    sharpe =  pnl.mean(skipna=True)/pnl.std(skipna=True)
    return sharpe

pnl = rolling_block_data_fcn(data,sharpe_ratio,gap=30)
pnl = pd.DataFrame(pnl)


'''

DATA_PATH = "Trading_Input.xlsx"
TAB_NAME = "Sheet1"
file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)

data = data.dropna()

ml_out = getBlendedSignal(data, RandomForestRegressor, gap=500)

ml_out = pd.DataFrame(ml_out)

#write(ml_out,'ml_output.xlsx','rf')

X = data.drop(['tenyear','change'])
Y = data['change']
score = crossValidate(X,Y,RandomForestRegressor)