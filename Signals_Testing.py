import pandas as pd
import numpy as np
import datetime as dt
import time as time
import os

from pykalman import KalmanFilter
import arch.unitroot as UnitRoot
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf,pacf, adfuller
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from operator import itemgetter
from sklearn import linear_model as LM
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA

#Kalman Filter
'''
tau = 0.1
kf = KalmanFilter(n_dim_obs=1,n_dim_state=2,
                  initial_state_mean=starting_point,
                  initial_state_covariance=np.eye(2),
                  transition_matrices=[[1,tau],[0,1]],
                  observation_matrices=[[1,0]],
                  observation_covariance=3,
                  transition_covariance=np.zeros((2,2)),
                  transition_offsets=[0,0])


np_data = tst_data["10y Close"].values

state_means, state_covs = kf.filter(np_data)
#tst_data['kf_predict']=pd.Series(state_means[:,0])
#print(state_means.shape)


times = np.arange(tst_data.shape[0])
plt.plot(times, state_means[:,0])
plt.plot(times, tst_data["10y Close"])

#plt.show()


tst_data['KF_Value']=state_means[:,0]
print("tst data",tst_data)

WRITE_PATH = "L:\Trade_Output.xlsx"
writer = pd.ExcelWriter(WRITE_PATH, engine='xlsxwriter')
tst_data.to_excel(writer, sheet_name='Data')
writer.save()
'''

#Computes hit ratio for rolling n trades
def getLastNHitRatio(data, n, hitInd, acceptableValues):
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
def getLastNHitRatioEveryLine(data, n, hitInd, acceptableValues):
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

#Autocorrelation analysis


acceptableValues = [0,1]



#Calcs p values and correlations for all lags
def acf_fcn(data,lags=2,alpha=.05):
    #@FORMAT: data = np(values)
    try:
        acfvalues, confint,qstat,pvalues = acf(data,nlags=lags,qstat=True,alpha=alpha)
        return [acfvalues,pvalues]
    except:
        return [np.nan]

#Calcs only correlations for all lags
def acf_fcn_only_cor(data,lags=2,alpha=.05):
    #@FORMAT: data = np(values)
    result = acf_fcn(data,lags, alpha)
    return result[0]


#Calcs only correlations, get only the ith lags
def acf_fcn_ith_cor(data,ith=2,lags=4,alpha=.05):
    #@FORMAT: data = np(values)
    try:
        result = acf_fcn_only_cor(data,lags, alpha)
        return result[ith]
    except:
        return [np.nan]


#Calcs the p value for for the best fit lag
def acf_fcn_highestlag(data,lags,alpha=.05):
    #@FORMAT: data = np(values)
    acfarr = acf_fcn(data,lags,alpha=alpha)
    lagNum = range(1,acfarr[1].shape[0]+1)
    lagP = np.array(acfarr[1])
    ordered_arr = np.column_stack((lagNum,lagP))
    #print('unordered',ordered_arr)
    ordered_arr.sort(axis=-1)
    return ordered_arr[0]

#Calcs the p value for for the best fit lag, include only P Value
def acf_fcn_highestlag_P_Val(data,lags,alpha=.05):
    #@FORMAT: data = np(values)
    acfarr = acf_fcn(data,lags,alpha=alpha)
    lagNum = range(1,acfarr[1].shape[0]+1)
    lagP = np.array(acfarr[1])
    ordered_arr = np.column_stack((lagNum,lagP))
    ordered_arr.sort(axis=-1)
    ordered_arr = pd.DataFrame(ordered_arr[0]).iloc[:,0].values.tolist()
    return ordered_arr[0]


#Calcs p value for DF test
def dickeyfuller_fcn(data,maxlag):
    #@FORMAT: data = np(values)
    try:
        df_fcn = adfuller(data,maxlag)
        return df_fcn[1]
    except:
        return np.nan

#Calcs p value for Phillips-Perron test for Stationarity
def pp_test_fcn(data,maxlag):
    #@FORMAT: data = np(values)
    try:
        pp_fcn = UnitRoot.PhillipsPerron(data,maxlag)
        return pp_fcn.pvalue
    except:
        return np.nan



#Calcs realized volatility
def rl_fcn(data):
    #@FORMAT: data = np(values)
    rl = np.std(data)
    return rl

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

class PCAAnalysis():
    def __init__(self, data):
        self.data = data
        self.X = self.data.values

    def getPCA(self,n_components):
        #@FORMAT: data = df(data,index=dates)
        data = self.data
        X = data.values
        pca = PCA(n_components)
        pca.fit(X)
        newdata = pca.fit_transform(X)
        newdata = pd.DataFrame(newdata)
        self.pca_components = pca.components_

        self.pca_explained_var = pca.explained_variance_ratio_
        print(pca.explained_variance_ratio_)

        #covar_o = np.cov(np.transpose(X.values))
        #eigval_o, eigvec_o = np.linalg.eig(covar_o)
        #print(eigvec_o)
        #print(eigval_o)
        #covar = np.cov(np.transpose(newdata.values))
        #eigval, eigvec = np.linalg.eig(covar)
        #print(eigvec)
        #print(eigval)


    def createLinearComponent(self):
        #Need to run getPCA() first
        a = self.X
        b = np.transpose(self.pca_components)
        c = np.dot(a,b)

        #Returns the linear component based on the loadings of each component generated by getPCA
        self.linearComponents = c
        print(c)
        return c

    #Creates a LM model and returns the prediction vector for the X_Pred
    def createLM(self,X,Y,X_Pred):
        lm = LM.LinearRegression(fit_intercept=True,copy_X=True)
        lm.fit(X,Y)
        Y_Pred = lm.predict(X_Pred)
        return Y_Pred



def write(datadf, path, tab="Sheet1"):
    WRITE_PATH = path
    writer = pd.ExcelWriter(WRITE_PATH, engine='xlsxwriter')
    datadf.to_excel(writer, sheet_name=tab)
    writer.save()






#Kmeans - returns cluster number and silhouette score
def kmeans_s_scores(data):
    #@FORMAT: data = df(data,index=dates)
    scs = []
    X = data.values

    for i in range(2,10):
        knn = KMeans(n_clusters=i).fit(X)
        labels = knn.labels_
        sc = silhouette_score(X, labels)
        scs.append([i,sc])
    #RETURNS: list[cluster number, silhouette_score]
    return scs

#Returns the silhouette_score for the best cluster
def kmeans_best_fit_cluster(data):
    #@FORMAT: data = df(data,index=dates)
    scores = kmeans_s_scores(data)
    scs = sorted(scores,key=itemgetter(1),reverse=True)
    return scs[0]

def kmeans_best_fit_cluster_labels(data):
    #@FORMAT: data = df(data,index=dates)
    scs = kmeans_best_fit_cluster(data)
    cluster = scs[0]
    X = data.values
    labels = KMeans(n_clusters=cluster).fit(X).labels_


    #scs = pd.DataFrame(scs,columns=['K','Score'])
    #RETURNS: np[labels for each row of DATA]
    return labels




#Rolling ML Methods

def getSKLearnModel(Y,X,model,**kwargs):
    model = model(**kwargs)
    model.fit(X,Y)
    return model

def getSKLearnModelPredictions(model, X_test):
    Y_test = model.predict(X_test)
    return Y_test

def getCrossValScore(model,Y,X,folds=5):
    scores = cross_val_score(model,X,Y,cv=folds)
    return scores


def getBlendedSignal(data,ml_model, gap=60):
    #@FORMAT: data = df(Y,X1,X2...,index=dates), dates goes from latest to earliest
    dates = data.index.values
    Y = data.iloc[:,0].values
    X = data.drop(data.columns[[0]],axis=1).values
    out = []

    for i in range(X.shape[0]-gap,0,-1):
        X_ = X[(i+1):(i+gap)]
        Y_ = Y[(i+1):(i+gap)]
        X_test = X[i]
        X_test = X_test.reshape(1,-1)
        Y_test = Y[i]

        model = getSKLearnModel(Y_,X_, ml_model)
        pred = getSKLearnModelPredictions(model,X_test)
        out.append([dates[i],Y_test,pred[0]])

    #@RETURNS: [date, Y, Y_pred]
    return out


DATA_PATH = "L:\Trade_Data.xlsx"
TAB_NAME = "CMOs"
WRITE_PATH = "L:\Trade_Output3.xlsx"
TAB = 'Output'


file  = pd.ExcelFile(DATA_PATH)
data = file.parse(TAB_NAME)
data = data.dropna(axis=0)
pcatool = PCAAnalysis(data)


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
