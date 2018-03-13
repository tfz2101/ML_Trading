import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time
import math
import random


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

#KNAPSACK PROBLEM--------------------------------------------------------------------------------------------



#Knapsack Problem
class ksProb:
    def __init__(self,w,v,W):
        self.w = w
        self.v = v
        self.W = W
    def getFitness(self,combo):
        combo = np.array(combo)
        v = np.array(self.v)
        output = sum(combo*v)
        return output
    def getWeight(self, combo):
        combo = np.array(combo)
        w = np.array(self.w)
        output = sum(combo*w)
        return output
    def getRestrictedFitness(self, combo):
        output = self.getFitness(combo)
        weight = self.getWeight(combo)
        if weight > self.W:
            output = 0
        if sum(x < 0 for x in combo)>0:
            output = 0
        return output

w_ks = [1,3,2]
v_ks = [2,4,1]
W_ks = 13
combo = [0,0,0]

ks = ksProb(w_ks, v_ks, W_ks)


#Genetic Algorithms
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
times = []
errs = []
GO = 26
domain = [(0,13),(0,4),(0,6)]

samples =300
for l in range(1,100):
    counter = 0
    pop = getRandomPop(domain,samples,True)
    cur_fit=0
    s1 = time.time()
    while cur_fit != GO:
        ga = GA(pop,ks.getRestrictedFitness,samples, percentile=0.5)
        counter = counter + samples
        best = ga.calculate_fitness()[0]
        cur_fit= ks.getRestrictedFitness(best)
        pop = ga.getNextPopMu()
    s2 = time.time()
    times.append((s2-s1))
    countIter.append(counter)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
'''



#Randomized Hill Climbing
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
GO = 26
times = []
for i in range(0,1000):
    counter = 0
    cur_fit = 0
    while cur_fit < GO:
        output_ks = [random.randrange(0,14,1),random.randrange(0,5,1),random.randrange(0,7,1)]
        cur_fit = ks.getRestrictedFitness(output_ks)
        peak = 1
        s = time.time()
        while peak == 1: #Do one complete hill climb
            #a = time.time()
            last_fit = cur_fit
            for i in range(0,len(output_ks)):
                it = random_hill_climbing(output_ks, i, 1, ks.getRestrictedFitness)
                counter = counter +1
                if ks.getRestrictedFitness(it)>cur_fit:
                    output_ks=it
                    cur_fit=ks.getRestrictedFitness(it)
                #errs.append([countIter,output_ks,cur_fit])
            if cur_fit == last_fit:
                peak = 0
                errs.append([countIter,output_ks,cur_fit])
        e = time.time()
        times.append((e-s))
    countIter.append(counter)
#print(countIter)
#print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))

def filterForOptima(lst):
    filtered = []
    for i in range(0,len(lst)):
        if lst[i][2]>0:
           if lst[i][1] not in filtered:
                filtered.append(lst[i][1])
    return filtered

localOpt = filterForOptima(errs)
print(len(localOpt))
'''


#Simulated Annealing
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
times = []
errs = []
GO = 26
for i in range(0,1000):
    counter = 0
    cur_fit = 0
    while cur_fit != GO:
        output_ks = [random.randrange(0,14,1),random.randrange(0,5,1),random.randrange(0,7,1)]
        cur_fit = ks.getRestrictedFitness(output_ks)
        T = 5
        #a = time.time()
        s = time.time()
        while T > 0.0001: #One full T schedule
            for i in range(0,len(output_ks)):
                output_ks = sim_anneal(output_ks, i, 1,T, ks.getRestrictedFitness)
                cur_fit=ks.getRestrictedFitness(output_ks)
                counter = counter +1
                #errs.append([countIter,output_ks,cur_fit])
            T = T * 0.9
        e = time.time()
        times.append((e-s))
    countIter.append(counter)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
'''

#COUNT ONES PROBLEM------------------------------------------------------------------------------------------------------------------------

class countOnes:

    def getRestrictedFitness(self, combo):
        output =  sum(x == 1 for x in combo)
        if sum(x < 0 for x in combo)>0:
            output = 0
        return output
ks = countOnes()
N = 9
domain = [(0,1)]*N
combo = getRandomSample(domain,True)


#Genetic Algorithms
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
GO = N
domain = [(0,1)]*N
times = []
samples = 300
for l in range(1,1000):
    counter = 0
    pop = getRandomPop(domain,samples,True)
    cur_fit=0
    s1 = time.time()
    while cur_fit != GO:
        ga = GA(pop,ks.getRestrictedFitness,samples, percentile=0.8)
        counter = counter + samples
        best = ga.calculate_fitness()[0]
        cur_fit= ks.getRestrictedFitness(best)
        #print(ga.fit_samples)
        #print(ga.crossover(pop[0],pop[1]))
        pop = ga.getNextPopMu()
    s2 = time.time()
    countIter.append(counter)
    times.append((s2-s1))
print(best)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
'''

#Randomized Hill Climbing
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
GO = N
times = []
for i in range(0,1000):
    counter = 0
    cur_fit = 0
    s1 = time.time()
    while cur_fit < GO:
        output_ks = getRandomSample(domain,True)
        cur_fit = ks.getRestrictedFitness(output_ks)
        peak = 1
        while peak == 1: #Do one complete hill climb
            #a = time.time()
            last_fit = cur_fit
            for i in range(0,len(output_ks)):
                it = random_hill_climbing(output_ks, i, 1, ks.getRestrictedFitness)
                counter = counter +1
                if ks.getRestrictedFitness(it)>cur_fit:
                    output_ks=it
                    cur_fit=ks.getRestrictedFitness(it)
                #errs.append([countIter,output_ks,cur_fit])
            if cur_fit == last_fit:
                peak = 0
                errs.append([countIter,output_ks,cur_fit])
    s2 = time.time()
    countIter.append(counter)
    times.append((s2-s1))
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
#print(errs)


def filterForOptima(lst):
    filtered = []
    for i in range(0,len(lst)):
        if lst[i][2]>0:
           if lst[i][1] not in filtered:
                filtered.append(lst[i][1])
    return filtered

localOpt = filterForOptima(errs)
print(len(localOpt))
print(localOpt)
'''

#Simulated Annealing
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
GO = N
times = []
for i in range(0,1000):
    counter = 0
    cur_fit = 0
    s1 = time.time()
    while cur_fit != GO:
        output_ks = getRandomSample(domain,True)
        cur_fit = ks.getRestrictedFitness(output_ks)
        T = .1
        #a = time.time()
        while T > 0.0001: #One full T schedule
            for i in range(0,len(output_ks)):
                output_ks = sim_anneal(output_ks, i, 1,T, ks.getRestrictedFitness)
                cur_fit=ks.getRestrictedFitness(output_ks)
                counter = counter +1
                #errs.append([countIter,output_ks,cur_fit])
            T = T * 0.1
        #print(output_ks)
        #print(cur_fit)
    #print(counter)
    s2 = time.time()
    times.append((s2-s1))
    countIter.append(counter)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
#print(errs)
'''





#4 PEAKS-------------------------------------------------------------------------------------------------

class fourPeaks:
    def __init__(self,T):
        self.T = T
        self.head = 0
        self.tail = 0

    def getRestrictedFitness(self,combo):
        i = 0
        while (i < len(combo)) and (combo[i] == 1):
            i =i +1
        head = i
        i = len(combo) - 1
        while (i >= 0) and (combo[i] == 0):
            i =i-1
        tail = len(combo) - 1 - i
        r = 0
        self.head = head
        self.tail = tail
        if (head > self.T) and (tail > self.T):
            r = len(combo)
        return (max(tail, head) + r)

N = 10
t = 2
ks = fourPeaks(t)
domain = [(0,1)]*N
combo = getRandomSample(domain,True)
GO = N + N -(t+1)


#Genetic Algorithms
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
times = []
samples = 1000
for l in range(1,1000):
    counter = 0
    pop = getRandomPop(domain,samples,True)
    cur_fit=0
    s1 = time.time()
    while cur_fit != GO:
        ga = GA(pop,ks.getRestrictedFitness,samples, percentile=0.8)
        counter = counter + samples
        best = ga.calculate_fitness()[0]
        cur_fit= ks.getRestrictedFitness(best)
        #print(best)
        #print(cur_fit)
        #print(ga.fit_samples)
        #print(ga.crossover(pop[0],pop[1]))
        pop = ga.getNextPopMu()
    countIter.append(counter)
    s2 = time.time()
    times.append(((s2-s1)))
print(best)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
'''

#Randomized Hill Climbing
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
times = []
for i in range(0,100):
    counter = 0
    cur_fit = 0
    s1 = time.time()
    while cur_fit < GO:
        output_ks = getRandomSample(domain,True)
        cur_fit = ks.getRestrictedFitness(output_ks)
        peak = 1
        while peak == 1: #Do one complete hill climb
            #a = time.time()
            last_fit = cur_fit
            for i in range(0,len(output_ks)):
                it = random_hill_climbing(output_ks, i, 1, ks.getRestrictedFitness)
                counter = counter +1
                if ks.getRestrictedFitness(it)>cur_fit:
                    output_ks=it
                    cur_fit=ks.getRestrictedFitness(it)
                #errs.append([countIter,output_ks,cur_fit])
            if cur_fit == last_fit:
                peak = 0
                errs.append([countIter,output_ks,cur_fit])
    s2 = time.time()
    times.append((s2-s1))
    countIter.append(counter)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
#print(errs)


def filterForOptima(lst):
    filtered = []
    for i in range(0,len(lst)):
        if lst[i][2]>0:
           if lst[i][1] not in filtered:
                filtered.append(lst[i][1])
    return filtered

localOpt = filterForOptima(errs)
print('local optimas')
print(localOpt)
print(len(localOpt))
'''

#Simulated Annealing
'''
output_ks = combo[:]
cur_fit = ks.getRestrictedFitness(combo)
countIter = []
errs = []
times = []
for i in range(0,1000):
    counter = 0
    cur_fit = 0
    s1 = time.time()
    while cur_fit != GO:
        output_ks = getRandomSample(domain,True)
        cur_fit = ks.getRestrictedFitness(output_ks)
        T = 5
        s1 = time.time()
        while T > 0.0001: #One full T schedule
            for i in range(0,len(output_ks)):
                output_ks = sim_anneal(output_ks, i, 1,T, ks.getRestrictedFitness)
                cur_fit=ks.getRestrictedFitness(output_ks)
                counter = counter +1
                #errs.append([countIter,output_ks,cur_fit])
            T = T * 0.7

    s2 = time.time()
    times.append((s2-s1))
    countIter.append(counter)
print(countIter)
print(sum(countIter)/len(countIter))
print(times)
print(sum(times)/len(times))
#print(errs)
'''