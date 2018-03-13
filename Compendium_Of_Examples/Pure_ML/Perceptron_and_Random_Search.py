import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time
import math
import random


class perceptron:

    def __init__(self,w,x,w0=0):
        self.w = w
        self.x = x
        self.w0 = w0

    def getOutput(self):
        w = np.array(self.w)
        x = np.array(self.x)
        out = sum(x * w) + self.w0
        out = 1/(1+math.exp(-out))
        return out


class h1nnet:
    def __init__(self,x,y,w1,w2,w3,oo):
        self.x = x
        self.y = y
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.oo=oo
        self.h1 = []
        self.h2 = []
        self.o1 = []
    def calcNet(self):
        self.h1 = perceptron(self.w1,self.x,self.oo[0])
        self.h2 = perceptron(self.w2,self.x,self.oo[1])
        x = np.array([self.h1.getOutput(),self.h2.getOutput()])
        w = np.array(self.w3)
        out = sum(w*x) + self.oo[2]
        return out

    def getOutputW(self,weights):
        self.inputW(weights)
        output = self.calcNet()
        return output

    def inputW(self,weights):
        self.w1 = weights[0:(len(self.w1))]
        self.w2 = weights[len(self.w1):(len(self.w1)+len(self.w2))]
        self.w3 = weights[(len(self.w1)+len(self.w2)):(len(self.w1)+len(self.w2)+len(self.w3))]
        self.oo = weights[(len(weights)-3):(len(weights))]

    def getError(self,weights):
        o = self.getOutputW(weights)
        return -abs(self.y - o)




def random_hill_climbing(input, index, step, func):
    output = input[:]
    cur_fit = func(input)
    temp = input[:]
    temp[index] = input[index]+step
    if func(temp)>cur_fit:
        output=temp
    temp2 = input[:]
    temp2[index] = input[index]-step
    if func(temp2)>cur_fit:
        output=temp2
    return output

def sim_anneal(input, index, step,T,func):
    output = input[:]
    cur_fit = func(input)
    temp = output[:]
    temp[index] = output[index]+step
    if func(temp)>=cur_fit:
        output=temp
        cur_fit=func(temp)
        #print("boom1")
    elif func(temp)<cur_fit:
        #print("no boom1")
        p = math.exp((func(temp)-cur_fit)/T)
        #print(p)
        r = np.random.random_sample()
        #print(r)
        if(r<p):
            output=temp
            cur_fit=func(temp)
    temp2 = output[:]
    temp2[index] = output[index]-step

    #print(output)

    if func(temp2)>=cur_fit:
        output=temp2
        cur_fit=func(temp2)
        #print('boom2')
    elif func(temp2)<cur_fit:
        #print('no boom2')
        p = math.exp((func(temp2)-cur_fit)/T)
        #print(p)
        r = np.random.random_sample()
        #print(r)
        if(r<p):
            output=temp2
            cur_fit=func(temp2)
    return output

class GA:
    def __init__(self, pop, fitness_function, samples=1000, percentile=0.5):

        self.pop = pop
        self.fitness_function = fitness_function
        self.percentile = percentile
        self.sorted_samples = self.calculate_fitness()
        self.fit_samples= self.get_percentile()


    def calculate_fitness(self):
        sorted_samples = sorted(self.pop,key=self.fitness_function,reverse=True)
        return np.array(sorted_samples).tolist()

    def get_percentile(self):
        index = int(len(self.sorted_samples) * self.percentile)
        fit_samples = self.sorted_samples[:index]
        return fit_samples[:index]

    def crossover(self,parent1,parent2):
        child1 = parent1[0:int(len(parent1)/2)]+parent2[int(len(parent2)/2):(len(parent2))]
        #child2 = parent2[0:int(len(parent2)/2)]+parent1[int(len(parent1)/2):(len(parent1))]
        return child1  #child2
    def mutate(self, parent):
        r = random.randrange(0,len(parent),1)
        u = random.uniform(0,1)
        child = parent[:]
        if u >0.5:
            child[r] = parent[r] + 1
        else:
            child[r] = parent[r] - 1
        return child
    def getNextPopCO(self):
        createNum = (len(self.pop)-len(self.fit_samples))
        nextPop = self.fit_samples[:]
        for i in range(0,createNum,1):
            parent1 = random.randrange(0,len(self.fit_samples),1)
            parent2 = random.randrange(0,len(self.fit_samples),1)
            child = self.crossover(self.fit_samples[parent1],self.fit_samples[parent2])
            nextPop.append(child)
        return nextPop
    def getNextPopMu(self):
        createNum = (len(self.pop)-len(self.fit_samples))
        nextPop = self.fit_samples[:]
        for i in range(0,createNum,1):
            parent = random.randrange(0,len(self.fit_samples),1)
            child = self.mutate(self.fit_samples[parent])
            nextPop.append(child)
        return nextPop


def getRandomSample(domain,intOnly=False):
    if intOnly == False:
        output =  [random.uniform(domain[i][0], domain[i][1]) for i in range(len(domain))]
    else:
        output =  [random.randint(domain[i][0], domain[i][1]) for i in range(len(domain))]

    return output

def getRandomPop(domain,sample,intOnly=False):
    return [getRandomSample(domain,intOnly) for i in range(sample)]


train  = pd.ExcelFile("L:\GA_ML\HW_2\s_wine_train_normal.xls")
train = train.parse('s_wine_train_normal')
test  = pd.ExcelFile("L:\GA_ML\HW_2\s_wine_test_normal.xls")
test =test.parse('s_wine_test_normal')

X = train.drop('quality',axis=1)
Y = train['quality']
X = X.values.tolist()
Y = Y.values.tolist()
w1 = [0]*(len(X[0]))
w2 = [0]*(len(X[0]))
w3 = [0,0]
oo= [0,0,0]
weights = w1 + w2 + w3 + oo

#Randomized Hill Climbing

#Train
acc= []
'''
domain = [(-5,5)]*11
w1 = getRandomSample(domain, False)
w2 = getRandomSample(domain, False)
w3 = [random.uniform(-5,5),random.uniform(-5,5)]
oo = [random.uniform(-5,5),random.uniform(-5,5)]
print(w1)
print(w2)
print(w3)
print(oo)
weights = w1 + w2 + w3 + oo
'''
for l in range(1,100):
    for c in range(0,train.shape[0]):
        x = X[c]
        y = Y[c]
        nnet = h1nnet(x,y,w1,w2,w3,oo)
        nnet.inputW(weights)
        output = weights[:]
        cur_err = nnet.getError(output)
        for i in range(0,len(output)):
            output = random_hill_climbing(output,i,.05,nnet.getError)
            nnet.inputW(output)
        weights = output

    #Test
    X_t = test.drop('quality',axis=1)
    Y_t = test['quality']
    X_t = X_t.values.tolist()
    Y_t = Y_t.values.tolist()
    errs = []
    for c in range(0,test.shape[0]):
        x = X_t[c]
        y = Y_t[c]
        nnet1 = h1nnet(x,y,w1,w2,w3,oo)
        nnet1.inputW(weights)
        if abs(round(nnet1.getError(weights),0)) < 0.001:
            errs = errs + [1]
        else:
            errs = errs + [0]

    accuracy = sum(errs)/test.shape[0]
    print(accuracy)
    acc.append([l,accuracy])
acc = pd.DataFrame(acc)
print(acc)
writer = pd.ExcelWriter('L:\GA_ML\HW_2\data_dump.xlsx', engine='xlsxwriter')
acc.to_excel(writer, sheet_name='main')
writer.save()


#Simulated Annealing
'''
#Train
acc = []
T = 3
weights = w1 + w2 + w3 + oo
for l in range(1,100):
    for c in range(0,train.shape[0]):
        x = X[c]
        y = Y[c]
        nnet = h1nnet(x,y,w1,w2,w3,oo)
        nnet.inputW(weights)
        cur_err = nnet.getError(weights)
        output = weights[:]
        for i in range(0,len(output)):
            output = sim_anneal(output,i,.05,T,nnet.getError)
            nnet.inputW(output)
        weights = output

    #Test
    X_t = test.drop('quality',axis=1)
    Y_t = test['quality']
    X_t = X_t.values.tolist()
    Y_t = Y_t.values.tolist()
    errs = []
    for c in range(0,test.shape[0]):
        x = X_t[c]
        y = Y_t[c]
        nnet1 = h1nnet(x,y,w1,w2,w3,oo)
        nnet1.inputW(weights)
        if abs(round(nnet1.getError(weights),0)) < 0.001:
            errs = errs + [1]
        else:
            errs = errs + [0]

    accuracy = sum(errs)/test.shape[0]
    print(accuracy)
    acc.append([l,accuracy])
    T = T * 0.9
acc = pd.DataFrame(acc)
print(acc)
writer = pd.ExcelWriter('L:\GA_ML\HW_2\data_dump.xlsx', engine='xlsxwriter')
acc.to_excel(writer, sheet_name='main')
writer.save()
'''

#Genetic Algorithms
'''
#Train
acc = []
popDomain = domain = [(-5,5)]*26
pop = getRandomPop(popDomain,1000,False)
print(pop)

for l in range(1,100):
    for c in range(0,train.shape[0]):
        x = X[c]
        y = Y[c]
        nnet = h1nnet(x,y,w1,w2,w3,oo)
        ga = GA(pop,nnet.getError,samples = 500, percentile=0.5)
        bestWeight = ga.calculate_fitness()[0]
        nnet.inputW(bestWeight)
        pop = ga.getNextPopMu()

    #Test
    X_t = test.drop('quality',axis=1)
    Y_t = test['quality']
    X_t = X_t.values.tolist()
    Y_t = Y_t.values.tolist()
    errs = []
    for c in range(0,test.shape[0]):
        x = X_t[c]
        y = Y_t[c]
        nnet1 = h1nnet(x,y,w1,w2,w3,oo)
        nnet1.inputW(bestWeight)
        if abs(round(nnet1.getError(bestWeight),0)) < 0.001:
            errs = errs + [1]
        else:
            errs = errs + [0]
    accuracy = sum(errs)/test.shape[0]
    print(accuracy)
    acc.append([l,accuracy])
acc = pd.DataFrame(acc)
print(acc)
writer = pd.ExcelWriter('L:\GA_ML\HW_2\data_dump.xlsx', engine='xlsxwriter')
acc.to_excel(writer, sheet_name='main')
writer.save()
'''


