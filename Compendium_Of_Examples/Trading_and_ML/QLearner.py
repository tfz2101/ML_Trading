"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand
import time

class QLearner(object):

    def convertActionToState(self,s_old,a):
        # update the test location
        s_new = 0
        if a == 0: #north
            s_new = s_old - 10
        elif a == 1: #east
            s_new = s_old + 1
        elif a == 2: #south
            s_new = s_old + 10
        elif a == 3: #west
            s_new = s_old - 1

        if s_new >= self.num_states or s_new <0:
            s_new = s_old

        return s_new

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.num_states = num_states
        self.verbose = verbose
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.s = 0
        self.a = 0
        self.dyna =  dyna
        self.rar = rar
        self.radr = radr

        #self.qTab = np.random.randint(-100,100,size=(self.num_states,self.num_actions))/100.00
        self.qTab = np.zeros(shape=(self.num_states,self.num_actions),dtype=np.float64)


        self.tMat = np.zeros(shape=(self.num_states,self.num_actions,self.num_states))
        #self.tMat.fill(np.nan)
        self.tMat.fill(0)
        self.tMat.tolist()

        self.rMat = np.zeros(shape=(self.num_states,self.num_actions))
        self.rMat.fill(-1.0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action  = self.qTab[s,:].argmax()

        isRandAct = np.random.choice([True,False],p=[self.rar,1-self.rar])
        if isRandAct:
            action = np.random.choice(range(0,self.num_actions))
            #print(action)
        self.rar = self.rar * self.radr


        if self.verbose: print "s =", s,"a =",action

        self.a = action

        return action



    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        qTab = self.qTab
        tMat = self.tMat
        rMat = self.rMat


        qMaxT = qTab[s_prime,:].max()
        qTab[self.s,self.a] = (1-self.alpha)*qTab[self.s,self.a] + self.alpha * (r + self.gamma * qMaxT)
        action = qTab[s_prime,:].argmax()


        if tMat[self.s][self.a][s_prime] == np.nan:
           tMat[self.s][self.a][s_prime] = 1

        else:
           tMat[self.s][self.a][s_prime] =+ 1

        tMat[self.s,self.a,s_prime] =+1
        rMat[self.s,self.a] = (1-self.alpha) * rMat[self.s,self.a] + self.alpha * r

        #dyna_start = time.time()
        if self.dyna != 0:
            temp_s_arr = np.random.choice(range(0,self.num_states),size=self.dyna,replace=True)
            temp_a_arr = np.random.choice(range(0,self.num_actions),size=self.dyna,replace=True)
            temp_p_mat = np.array(tMat)
            temp_p_mat = temp_p_mat.astype(dtype=np.float64)
            temp_p_mat = temp_p_mat/temp_p_mat.sum(axis=2,keepdims=True)
            temp_p_mat = np.nan_to_num(temp_p_mat)
            #print(temp_p_mat)


            for i in range(0, self.dyna):
                temp_s = temp_s_arr[i]
                temp_a = temp_a_arr[i]
                allErr = sum(tMat[temp_s][temp_a])
                if allErr==0:
                    continue
                    #randA = np.random.choice(range(0,self.num_actions))
                    #temp_s_p = self.convertActionToState(temp_s,randA)
                #temp_p = tMat[temp_s][temp_a]/sum(tMat[temp_s][temp_a])
                temp_p = temp_p_mat[temp_s,temp_a,:]

                #print('temp p', temp_p)
                temp_s_p = np.random.choice(range(0,self.num_states),p=temp_p)
                #print('temp s_p', temp_s_p)

                qMax_Temp = qTab[temp_s_p,:].max()
                qTab[temp_s,temp_a] = (1-self.alpha)*qTab[temp_s,temp_a] + self.alpha * (rMat[temp_s,temp_a] + self.gamma * qMax_Temp)

            '''
            print('temp s', temp_s)
            print('temp a', temp_a)
            print('temp p', temp_p)
            print('temp s_p', temp_s_p)
            print('qtab row',qTab[temp_s_p,:])
            print('qtab max',qTab[temp_s_p,:].max())
            '''


        isRandAct = np.random.choice([True,False],p=[self.rar,1-self.rar])
        if isRandAct:
            action = np.random.choice(range(0,self.num_actions))
        self.rar = self.rar * self.radr

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        self.s = s_prime
        self.a = action

        return action


if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"



learner = QLearner(num_states=100,\
    num_actions = 4, \
    alpha = 0.2, \
    gamma = 0.9, \
    rar = 0.98, \
    radr = 0.999, \
    dyna = 0, \
    verbose=False) #initialize the learner

action = learner.querysetstate(0)
