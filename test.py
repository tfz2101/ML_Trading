from nolds import hurst_rs, dfa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from Signals_Testing import rolling_block_data_fcn,rolling_data_fcn, write
from ML_functions import getBlendedSignal,crossValidate, rollingMultivariateML, featureImportance, getPredictionandCrossValidate, MDI,normalizeDF,getBlendedSignalKeepColumns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing


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


DATA_PATH = "Trading_Input_GDAX.xlsx"
TAB_NAME = "ml_input_3"
file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)


'''
features = ['KURTOSIS_30','SKEW','volume_zscore','volume_signal','signal','interval_range_pct_px','num_tics_z_score']

for feature in features:
    cur_data = data.loc[:, ['Y_5', 'KURTOSIS_30', 'SKEW', 'volume_zscore', 'volume_signal', 'signal', 'interval_range_pct_px','num_tics_z_score']]
    cur_data = cur_data.dropna()

    X = cur_data.drop('Y_5',axis=1)

    X = cur_data.drop(feature,axis=1)

    Y = cur_data['Y_5']


    X_norm = normalizeDF(X)
    cur_data = pd.concat([Y,X_norm],axis=1)


    ml_out = getBlendedSignal(cur_data, RandomForestRegressor, gap=150)
    ml_out = pd.DataFrame(ml_out)

    OUT_FILE =  'ml_output_' + feature +'.xlsx'
    write(ml_out,OUT_FILE,'rf')
'''

cur_data = data.loc[:,['Y_5', 'LAST_PRICE', 'KURTOSIS_30', 'SKEW', 'volume_zscore', 'volume_signal', 'signal', 'interval_range_pct_px','VWAP_Diff_Zscore']]
cur_data = cur_data.dropna()
X = cur_data.drop(['Y_5','LAST_PRICE'], axis=1)
Y = cur_data['Y_5']
px_col = cur_data['LAST_PRICE']
X_norm = normalizeDF(X)
cur_data = pd.concat([Y, X_norm, px_col], axis=1)

ml_out = getBlendedSignalKeepColumns(cur_data,'LAST_PRICE',RandomForestRegressor, gap=25)
ml_out = pd.DataFrame(ml_out)
write(ml_out, 'ml_output_GDAX_25_lookback.xlsx', 'rf')







X_df = X
Y_df = Y

X = X.values
Y = Y.values

model_kwargs = {'n_estimators':2}
#score = crossValidate(X,Y,0.7,RandomForestClassifier,**model_kwargs)

kwargs = {'trainSplit':0.7, 'model_fcn': RandomForestClassifier, 'model_kwargs': model_kwargs}

#scores = rollingMultivariateML(dataML,100,crossValidate, **kwargs)

#FIs = featureImportance(X,Y,0.7,RandomForestClassifier,**model_kwargs)

#cv_pred = rollingMultivariateML(dataML,100,getPredictionandCrossValidate, **kwargs)

#mdi = MDI(X_df, Y_df, 0.7, RandomForestClassifier, **model_kwargs)

