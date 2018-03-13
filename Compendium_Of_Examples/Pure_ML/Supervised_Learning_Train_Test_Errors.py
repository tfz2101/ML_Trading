from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN
import seaborn as sns
import matplotlib.pyplot as plt
import time as time
from scipy.stats.mstats import normaltest


bmi = pd.ExcelFile("BMI2.xlsx")
data = bmi.parse("raw_data")
data = data[data['10y']!=0]
#print(data)

N_FOLDS =  10


def getSVCLinear(data, target, kern='linear'):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = svm.SVC(kernel=kern)
    model.fit(X,Y)
    #output =  {'results':cv_results, 'mean':cv_results.mean(),'std':cv_results.std()}
    return model

def getSVCRbf(data, target, kern='rbf'):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = svm.SVC(kernel=kern)
    model.fit(X,Y)
    #output =  {'results':cv_results, 'mean':cv_results.mean(),'std':cv_results.std()}
    return model

#Because of Greedy algorithm, favors short trees over complex tall trees.
#Overfitting - if there exists another hypothesis that performs worse in training but better throught out the dataset as a whole and beyond.  You will see divigence of accuracy as the tree grows in length.
def getDTC(data, target, depth):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = DTC(max_depth=depth)


def getBoostingTree(data,target):
    Y = data[target]
    X = data.drop(target,axis=1)
    #aggressive pruning!!
    model = GB(max_depth=1)
    model.fit(X,Y)
    return model

def getRF(data,target):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = RF()
    model.fit(X,Y)
    return model

#Good for a complex problem with a collection of simpler solutions.
#Nearly all computations happen during classfication time, not on the training - could increase runtime.
#Curse of dimensionality - if a lot of attributes are non-useful, it will throw off the algorithm which use ALL attributes to determine 'closeness'

def getKNN(data,target, N):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = KNN(n_neighbors=N)
    model.fit(X,Y)
    #cv = cross_val_score(model,X,Y,cv=N_FOLDS)
    return model



'''
knnout = getKNNiterate(data,'10y',20,10)
print(knnout)
sns.lmplot('n','accuracy',knnout, fit_reg=False)
plt.show()
'''


def trainTestErrorRates(data, target, GETFUN, N_FOLDS,extraArg='None',extraInput = 0):
    kf = KFold(data.shape[0],n_folds=N_FOLDS,shuffle=True)
    output = []
    for train, test in kf:
        if extraArg != 'None':
            model = GETFUN(data.iloc[train,:], target, extraInput)
        else:
            model = GETFUN(data.iloc[train,:], target)

        trainset = data.iloc[train,:].drop(target,axis=1)
        pred_train = model.predict(trainset)
        train_y = data.ix[train,target]
        train_score = accuracy_score(train_y,pred_train)

        testset = data.iloc[test,:].drop(target,axis=1)
        pred = model.predict(testset)
        y_true = data.ix[test,target]
        score = accuracy_score(y_true,pred)

        output.append([train_score,score])
    output = pd.DataFrame(output, columns=['Train_Accuracy','Test_Accuracy'])
    #print([output.mean(),output.std()])
    return output

#knn = trainTestErrorRates(data, '10y',getKNN, 10, 'K', 8)
#print(knn)

def cvLoop(data, target, GETFUN, min_cv,max_cv,extraArg='None',extraInput = 0,):
    output1 = []

    for i in range(min_cv,max_cv+1):
        start = time.time()
        n_test = trainTestErrorRates(data, target, GETFUN, i, extraArg, extraInput)
        avgs = n_test.mean()
        sd = n_test.std()
        end = time.time()
        run=end - start
        output1.append([avgs.iloc[0],avgs.iloc[1], sd.iloc[0], sd.iloc[1],run])

    output1=pd.DataFrame(output1, index=range(min_cv,max_cv+1),columns=['Train_Accuracy','Test_Accuracy','Train_STD','Test_STD','Time'])
    return output1

#loop1 = cvLoop(data, '10y', getKNN, 10,30, 'K', 3)
#print(loop1)

#Loop thru different values of K
koutput = []
for k in range(1,20):
    loop2 = cvLoop(data, '10y', getKNN, 10,30, 'K', k)
    koutput.append(loop2.mean().tolist())
    #print(k)
    #print(loop2.mean())
    #print('--------------')
koutput = pd.DataFrame(koutput)
print(koutput)

#scores1 =  trainTestErrorRates(data, '10y', getBoostingTree,N_FOLDS)
#scores1 =  trainTestErrorRates(data, '10y', getRF,N_FOLDS)
#scores2 =  trainTestErrorRates(data, '10y', getSVCLinear,N_FOLDS)
#scores3 =  trainTestErrorRates(data, '10y', getSVCRbf,N_FOLDS)

#print(data.iloc[:,0])
#norm = normaltest(data.iloc[:,1:5])
#print(norm.pvalue)

#print(data.describe())
