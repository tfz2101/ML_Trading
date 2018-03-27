from nolds import hurst_rs, dfa
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from Signals_Testing import rolling_block_data_fcn,rolling_data_fcn, write
from ML_functions import getBlendedSignal,crossValidate, rollingMultivariateML, featureImportance, getPredictionandCrossValidate, MDI
from sklearn.ensemble import RandomForestClassifier
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
DATA_PATH = "Trading_Input.xlsx"
TAB_NAME = "ETH_5MIN"
file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)



#ml_out = getBlendedSignal(data, RandomForestRegressor, gap=500)

#ml_out = pd.DataFrame(ml_out)

#write(ml_out,'ml_output.xlsx','rf')




#X = data.drop(['tenyear','upordown','change'],axis=1)
#Y = data['upordown']

cur_data = data.loc[:,['KURTOSIS_30','SKEW','CUR_OVER_MA10','Y_1_CLASSIFICATION']]
cur_data = cur_data.dropna()


X = cur_data.drop('Y_1_CLASSIFICATION',axis=1)
print(X)

Y = cur_data['Y_1_CLASSIFICATION']
print(Y)
X_df = X
Y_df = Y

X = X.values
Y = Y.values



#X = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
#Y = np.array([100,101,102,103])

#dataML = data.drop(['tenyear','change'],axis=1)

model_kwargs = {'n_estimators':2}
#score = crossValidate(X,Y,0.7,RandomForestClassifier,**model_kwargs)

kwargs = {'trainSplit':0.7, 'model_fcn': RandomForestClassifier, 'model_kwargs': model_kwargs}
#scores = rollingMultivariateML(dataML,100,crossValidate, **kwargs)

FIs = featureImportance(X,Y,0.7,RandomForestClassifier,**model_kwargs)
print(FIs)


#cv_pred = rollingMultivariateML(dataML,100,getPredictionandCrossValidate, **kwargs)
#write(pd.DataFrame(cv_pred),"output_2.xlsx","pred_and_backtest")

#mdi = MDI(X_df, Y_df, 0.7, RandomForestClassifier, **model_kwargs)

