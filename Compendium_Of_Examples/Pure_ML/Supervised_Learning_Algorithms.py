from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier as GB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier as DTC
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier as KNN

import matplotlib.pyplot as plt
import time as time
from scipy.stats.mstats import normaltest
from sklearn.cross_validation import train_test_split
import pickle as pk
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix


bmi = pd.ExcelFile("L:\GA_ML\Abalone.xlsx")
data = bmi.parse("data_norm")

#train, test  = train_test_split(data, test_size=0.30, random_state=42)

'''
train  = pd.ExcelFile("L:\GA_ML\set_training_abolone.xlsx")
train = train.parse('main')
test  = pd.ExcelFile("L:\GA_ML\set_test_abalone.xlsx")
test = test.parse('test_abalone')
'''


train  = pd.ExcelFile("L:\GA_ML\s_aba_train_normal.xlsx")
train = train.parse('s_aba_train_normal')
test  = pd.ExcelFile("L:\GA_ML\s_aba_test_normal.xlsx")
test = test.parse('s_aba_test_normal')


N_FOLDS =  10
targetName = 'rings'

Y = train[targetName]
X = train.drop(targetName,axis=1)


def getAccuracy(testdata, model, target):
    testset = testdata.drop(target,axis=1)
    pred_test = model.predict(testset)
    test_y = testdata[target]
    test_score = accuracy_score(test_y,pred_test)
    return test_score

def getGrid(data, target, model, gridCV):
    Y = data[target]
    X = data.drop(target,axis=1)
    clf = GridSearchCV(model, gridCV)
    clf.fit(X,Y)
    return clf

def getSVCLinear(data, target, kern='linear'):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = svm.SVC(kernel=kern)
    model.fit(X,Y)
    return model


def getSVCRbf(data, target, kern='rbf'):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = svm.SVC(kernel=kern)
    model.fit(X,Y)
    return model



#Because of Greedy algorithm, favors short trees over complex tall trees.
#Overfitting - if there exists another hypothesis that performs worse in training but better throught out the dataset as a whole and beyond.  You will see divigence of accuracy as the tree grows in length.
def getDTC(data, target, depth):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = DTC(max_depth=depth)
    return model


def getBoostingTree(data,target):
    Y = data[target]
    X = data.drop(target,axis=1)
    #aggressive pruning!!
    model = GB()
    model.fit(X,Y)
    return model

#Good for a complex problem with a collection of simpler solutions.
#Nearly all computations happen during classfication time, not on the training - could increase runtime.
#Curse of dimensionality - if a lot of attributes are non-useful, it will throw off the algorithm which use ALL attributes to determine 'closeness'

def getKNN(data,target, N=27):
    Y = data[target]
    X = data.drop(target,axis=1)
    model = KNN(n_neighbors=N)
    model.fit(X,Y)
    return model

def loopKNNonN(data, target, max_n, N_FOLDS):
    output = []
    for i in range(1,max_n+1):
        cur_model = getKNN(data, target,i)
        X = data.drop(target,axis=1)
        Y = data[target]
        cur_train_score = cross_val_score(cur_model,X, Y, cv=N_FOLDS)
        out = [cur_train_score.mean(),cur_train_score.std()]
        output.append(out)
    output = pd.DataFrame(output)
    return output
#loopN = loopKNNonN(train, targetName, 40, N_FOLDS)
#print(loopN)


cur_model = getKNN(train, targetName, N=27) #CHANGE
cur_train_score = cross_val_score(cur_model,X, Y, cv=N_FOLDS)
print('Default Model Results')
print([cur_train_score.mean(),cur_train_score.std()])


params_svc_lin = [{'kernel':['linear'],'C':[1,10,50,100]}]
params_svc_rbf =  [{'kernel':['rbf'],'C':[1,10,50,100],'gamma':[0.001,.005,'auto',0.2,0.5,1,5]}]
params_boosting =  [{'kernel':['rbf'],'C':[1,10,50,100]}]
params_knn =  [{'kernel':['rbf'],'C':[1,10,50,100]}]

'''
tmp_model = svm.SVC(kernel='linear') #CHANGE
grid =  getGrid(train, targetName,tmp_model, params_svc_lin) #CHANGE
cur_model=grid
print('Grid Search Results')
print(grid.best_params_)
print(grid.best_score_)
'''

'''
output2 = []
for i in [1,2,3]:
    grid = KNN(weights='distance',n_neighbors=5,p=i) #CHANGE
    cur_model = grid.fit(X,Y)
    cur_train_score = cross_val_score(cur_model,X, Y, cv=N_FOLDS)
    out = [i,cur_train_score.mean(),cur_train_score.std()]
    print(i)
    print(out)
    output2.append(out)
output2 = pd.DataFrame(output2)
print(output2)
'''


tst_acc = getAccuracy(test, cur_model, targetName)
print('Test Accuracy')
print(tst_acc)



'''
#Produces the train/test vs sample size curve
train_tst__crv = []
for i in range(20,train.shape[0],50):
  temp_data = train.iloc[0:i,:]
  model =  getKNN(temp_data, targetName) #CHANGE
  train_acc = cross_val_score(model, temp_data.drop(targetName, axis=1),temp_data[targetName],cv=N_FOLDS)
  test_acc = getAccuracy(test, model, targetName)
  train_tst__crv.append([i,train_acc.mean(),test_acc])
train_tst__crv = pd.DataFrame(train_tst__crv,columns=['Size','Train_Accuracy','Test_Accuracy'])

pk.dump(train_tst__crv, open('knn.p','wb')) #CHANGE
print(train_tst__crv)
writer = pd.ExcelWriter('L:\GA_ML\data_dump.xlsx', engine='xlsxwriter')
train_tst__crv.to_excel(writer, sheet_name='main')
writer.save()
'''

'''
crv_load = pk.load(open('train_tst_crv.p','rb'))
print(crv_load)
#sns.lmplot(crv)
'''

'''
knnout = getKNNiterate(data,targetName,20,10)
print(knnout)
sns.lmplot('n','accuracy',knnout, fit_reg=False)
plt.show()
'''



#norm = normaltest(data.iloc[:,1:5])
#print(norm.pvalue)




treeResult  = pd.ExcelFile("L:\GA_ML\s_aba_tree_result.xlsx")
treeResult = treeResult.parse('aba_tree_result')

nnResult  = pd.ExcelFile("L:\GA_ML\s_aba_nn_result.xlsx")
nnResult = nnResult.parse('aba_nn_result')

'''
cmat_nn = confusion_matrix(nnResult[targetName],nnResult['pred'])
cmat_nn = pd.DataFrame(cmat_nn)
print(cmat_nn)
'''

model = svm.SVC(kernel='rbf',C=100, gamma=1)
model.fit(X,Y)
pred=model.predict(test.drop(targetName,axis=1))
cmat_boost = confusion_matrix(test[targetName],pred)
cmat_boost = pd.DataFrame(cmat_boost)
#print(cmat_boost)

model = KNN(n_neighbors=18,weights='distance',p=2)
model.fit(X,Y)
pred=model.predict(test.drop(targetName,axis=1))
cmat_boost = confusion_matrix(test[targetName],pred)
cmat_boost = pd.DataFrame(cmat_boost)
#print(cmat_boost)

