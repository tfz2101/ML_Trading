import pandas as pd
import numpy as np
import datetime as dt
import time as time
import os

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from operator import itemgetter
from sklearn import linear_model as LM
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


#PREPROCESSING
#-----------------------------------------------------------------------------------------------------------
def normalizeDF(df):
    columns =  df.columns.values
    index = df.index.values
    scaler = StandardScaler(copy=False,with_mean=True,with_std=True)
    scaler.fit(df)

    df =  scaler.transform(df)
    df = pd.DataFrame(df,columns=columns,index=index)
    return df



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
#---------------------------------------------------------------------------------------
#Fits a ML model on a rolling basis for a given lookback and makes an prediction based on it for time T
def getBlendedSignal(data,ml_model, gap=60,Y_index=1, **kwargs):
    #@FORMAT: data = df(Y,X1,X2...,index=dates), dates goes from earliest to latest
    #Kwaargs contains the parameters for the ML function

    #If Y_index is blank, assume Y is in the first column
    Y = data.iloc[:, 0].values

    dates = data.index.values

    X = data.drop(data.columns[[0]],axis=1).values
    out = []

    for i in range(gap,X.shape[0],1):
        X_ = X[(i-gap):i]
        Y_ = Y[(i-gap):i]
        X_test = X[i]
        X_test = X_test.reshape(1,-1)
        Y_test = Y[i]

        model = ml_model(**kwargs)
        model.fit(X_, Y_)

        pred = model.predict(X_test)
        out.append([dates[i],Y_test,pred[0]])

    #@RETURNS: [date, Y, Y_pred]
    return out


#Main ML rollthrough time method
def rollingMultivariateML(data, gap, fcn, **kwargs):
    #@FORMAT: data = df(Y,X1,X2...,index=dates), dates goes from earliest to latest

    dates = data.index.values
    Y = data.iloc[:,0].values
    X = data.drop(data.columns[[0]],axis=1).values
    out = []

    for i in range(0,X.shape[0]-gap,1):
        X_ = X[i:(i+gap)]
        Y_ = Y[i:(i+gap)]

        fcn_out = fcn(X_,Y_,**kwargs)
        #@RETURNS: list

        line = [dates[i]] + fcn_out

        out.append([line])

    #@RETURNS: list[date, fcn_output]
    return out


def trainTestSplit(X, Y, trainSplit):
    #@FORMAT: X = array, Y = array

    trainSplit = 1- trainSplit


    splitInd = int(X.shape[0] * trainSplit)
    try:
        x_test = X[0:splitInd,:]
        x_train = X[splitInd:X.shape[0],:]
    except:
        x_test = X[0:splitInd]
        x_test = x_test.reshape(-1,1)
        x_train = X[splitInd:len(X)]
        x_train = x_train.reshape(-1,1)


    y_test = Y[0:splitInd]
    y_train = Y[splitInd:len(Y)]
    return x_train, x_test, y_train, y_test


#Assumes the data goes from old to new (descending order). Splits the data and takes the data that is the NEWEST and uses it as the training set.
#The model makes predictions on the test set and gives accuracy score.
#Labels must be classification, not regression for the accuracy_score to work.
def crossValidate(X, Y, trainSplit, model_fcn, **model_kwargs):
    #@FORMAT: X = array, Y = array

    x_train, x_test, y_train, y_test = trainTestSplit(X,Y, trainSplit)

    try:
        model = model_fcn(**model_kwargs).fit(x_train, y_train)

    except:
        model = model_fcn().fit(x_train, y_train)

    y_pred = model.predict(x_test)
    out = accuracy_score(y_pred, y_test)

    #@RETURN: list
    return [out]




#Combines the crossValidates function as well as gives a prediction for the last X row. Answers the question - if a ML
#function has been pretty accurate in OOS backtest, does it a better predictor for this current row?
def getPredictionandCrossValidate(X, Y, trainSplit, model_fcn, **model_kwargs):
    #@FORMAT: X = array, Y = array
    X_train = X[0:(X.shape[0]-1),:]
    X_target = X[X.shape[0]-1,:]
    X_target = X_target.reshape(1,-1)
    Y_train = Y[0:(len(Y) - 1)]
    Y_target = Y[len(Y) - 1]

    try:
        model = model_fcn(**model_kwargs).fit(X_train, Y_train)
    except:
        model_kwargs = model_kwargs['model_kwargs']
        model = model_fcn(**model_kwargs).fit(X_train, Y_train)

    pred = model.predict(X_target)

    past_accuracy = crossValidate(X_train, Y_train, trainSplit, model_fcn, **model_kwargs)

    out =  past_accuracy + [pred[0]] + [Y_target]

    #@RETURN: list[past_accuracy, prediction, actual_Y]
    return out

#Iterates through the features one by one and gives the accuracy of each feature
def MDI(X, Y, trainSplit, model_fcn, **model_kwargs):
    #@FORMAT: X = df, Y = df
    features = X.columns.values.tolist()
    Y = Y.values
    for feature in features:
        X_ = X[feature].values
        score = crossValidate(X_,Y,trainSplit, model_fcn, **model_kwargs)
        print(feature,score)

    