"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import pandas as pd
import RTLearner as rt

class BagLearner(object):
    def __init__(self,learner,bags,kwargs={'leaf_size':1},boost=False,verbose=False):
        self.learner = learner
        self.bags = bags
        self.kwargs = kwargs

    def getBag(self,X,Y, size):
        indices = np.random.choice(range(0,len(Y)),size, replace=False)
        X = pd.DataFrame(X)
        Y = pd.DataFrame(Y)

        bX = X.iloc[indices.tolist(),:].as_matrix()
        bX = pd.DataFrame(bX)
        bY = Y.iloc[indices.tolist(),:].as_matrix()
        bY = pd.DataFrame(bY)
        return bX, bY

    def countVote(self,votes):
        votes_set = set(votes)
        votes_set = list(votes_set)
        count = []
        for i in range(0,len(votes_set)):
            count.append(votes.count(votes_set[i]))

        tally =  max(count)
        return votes_set[count.index(tally)]

    def addEvidence(self, X, Y):
        SIZE = int((len(X)*0.5))
        trees = []
        for i in range(0,self.bags):
            bX,bY = self.getBag(X,Y,SIZE)
            dt = self.learner(**self.kwargs)
            dt.addEvidence(bX,bY)
            trees.append(dt)
        self.trees = trees

    def query(self,points):
        out = []
        for point in points:
            vote = []
            for tree in self.trees:
                vote.append(tree.query([point])[0])
            res = self.countVote(vote)
            out.append(res)
        return np.array(out)

if __name__=="__main__":
    print "the secret clue is 'zzyzx'"


