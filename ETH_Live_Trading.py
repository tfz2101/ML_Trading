from nolds import hurst_rs, dfa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from Signals_Testing import rolling_block_data_fcn,rolling_data_fcn, write
from ML_functions import getBlendedSignal,crossValidate, rollingMultivariateML, featureImportance, getPredictionandCrossValidate, MDI,normalizeDF
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import preprocessing



DATA_PATH = "Trading_Sheet_Live.xlsx"
TAB_NAME = "ml_input"
file  = pd.ExcelFile(DATA_PATH)
orig_data = file.parse(TAB_NAME)

raw_cur_data = orig_data.loc[:,['Y_5','KURTOSIS_30','SKEW','volume_zscore','volume_signal','signal','interval_range_pct_px','num_tics_z_score','VWAP_Diff_Zscore']]
cur_data = raw_cur_data.dropna()

X = cur_data.drop('Y_5',axis=1)
Y = cur_data['Y_5']

#NOTE THE NORMALIZATION MIGHT BE THROWN OFF BY THE FACT THAT YOU'RE USING A SHORTER DATASET!!!!!!
X_norm = normalizeDF(X)
cur_data = pd.concat([Y,X_norm],axis=1)

#ml_out = getBlendedSignal(cur_data, RandomForestRegressor, gap=150)
#ml_out = pd.DataFrame(ml_out)


#GET UPDATED SIGNAL
gap = 150
ml_model = RandomForestRegressor

data = cur_data
dates = data.index.values
Y = data.iloc[:, 0].values
X = data.drop(data.columns[[0]], axis=1).values
endInd = X.shape[0]
out = []

X_ = X[(endInd - gap):endInd]
Y_ = Y[(endInd - gap):endInd]

X = raw_cur_data.drop('Y_5',axis=1).values
X_test = X[X.shape[0]-1]
X_test = X_test.reshape(1, -1)

print("X_TEST",X_test)

# model = ml_model(**kwargs)
model = ml_model()
model.fit(X_, Y_)

pred = model.predict(X_test)
out.append([raw_cur_data.index[raw_cur_data.shape[0]-1],orig_data.ix[raw_cur_data.shape[0]-1,'LAST_PRICE'],pred[0]])

print(out)
#write(ml_out,'ml_output3.xlsx','rf')







