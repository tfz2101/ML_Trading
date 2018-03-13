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
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA



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

#Fits a ML model on a rolling basis for a given lookback and makes an prediction based on it for time T
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



def rollingMultivariateML(data, gap, fcn, **kwargs):
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


        #@FORMAT: fcn takes in
        fcn_out = fcn(**kwargs)
        # fcn must return a list of the data meant to be stored

        line = [dates[i]].extend(fcn_out)

        out.append([line])

    #@RETURNS: df
    return out
