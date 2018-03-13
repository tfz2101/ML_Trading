"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
"""

import numpy as np
import pandas as pd

class PERT(object):
    def __init__(self):
        pass

    def split(self,data,start, end, candidates):

        colSplit = np.random.choice(candidates)
        elements = np.random.choice(range(start,end+1),2, replace=False)


        mean = (data[elements[0],colSplit]+data[elements[1],colSplit])/2
        newCol = np.empty([len(data),1])
        for i in range(start, (end+1)):
            if data[i,colSplit] >= mean:
                newCol[i]=1
            else:
                newCol[i]=0

        data =  np.column_stack((data,newCol))
        data = data.tolist()
        data =  sorted(data,key=lambda x: x[len(data[0])-1],reverse=True)
        data= np.array(data)
        splitInd = np.sum(data[:,(len(data[0])-1)])
        data = data[:,0:(len(data[0])-1)]
        split1 = data[0:splitInd,:]
        split2 = data[splitInd:len(data),:]
        splitInfo = [colSplit,mean]
        return split1, split2, splitInfo


class Node(object):
    def __init__(self, data):
        self.data = data
        #self.history = pd.DataFrame(columns=['column','mean','upOrBottom'])
        self.history = []

    def update_hist(self, value):
        #value1 =  pd.DataFrame([value],columns=['column','mean','upOrBottom'])
        #out  = pd.concat([self.history,value1],axis=0,ignore_index=True)
        self.history.append(value)


    def copy_hist(self,hist):
        self.history = hist[:]

    def assign_y(self,Y):
        self.Y = Y


class RTLearner(object):

    def __init__(self, leaf_size=1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def getPDData(self):
        data = pd.read_csv("C:\Users\Frank Zhi\Desktop\Python Work\MLT\ML4T_2016Fall-master\ML4T_2016Fall-master\mc3_p1\Data\winequality-white.csv",sep=',')
        data =  pd.DataFrame(data)

        return data

    def isLeaf(self,Node, Y):
        out  =np.unique(Node.data[:,Y])
        if len(out) ==1:
            res =  True
        else:
            res = False
        return res

    def addEvidence(self,dataX,dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """

        data  = np.column_stack((dataX, dataY))
        dt = PERT()
        self.columns = np.array(range(0,len(data[0])))

        count = 0
        dict = {}
        for a in self.columns:
            dict[a]=count
            count = count+1
        self.dict = dict

        #Y = data.columns.values
        #Y = Y[len(Y)-1]
        Y = len(data[0])-1

        candidates =  np.delete(self.columns,len(self.columns)-1)

        n = Node(data)
        dataObj = [n]
        leaves = []


        while len(dataObj)>0:
            newdataobj = []
            for obj in dataObj:

                if self.isLeaf(obj,Y) ==True:
                    obj.assign_y(obj.data[0,Y])
                    leaves.append(obj)
                    dataObj.remove(obj)
                    continue

                df1, df2, node = dt.split(obj.data,0,len(obj.data)-1,candidates)
                if (len(df1))== 0 or (len(df2)==0):
                    newdataobj.append(obj)
                    continue
                up = Node(df1)
                up.copy_hist(obj.history)
                up_hist = [node[0],node[1],1]
                up.update_hist(up_hist)

                if len(df1) < self.leaf_size:
                    up.assign_y(up.data[:,Y].mean())
                    leaves.append(up)
                elif self.isLeaf(up,Y):
                    up.assign_y(up.data[0,Y])
                    leaves.append(up)
                else:
                    newdataobj.append(up)
                down = Node(df2)
                down.copy_hist(obj.history)
                down_hist = [node[0],node[1],0]
                down.update_hist(down_hist)
                if len(df2) < self.leaf_size:
                    down.assign_y(down.data[:,Y].mean())
                    leaves.append(down)
                elif self.isLeaf(down,Y):
                    down.assign_y(down.data[0,Y])
                    leaves.append(down)
                else:
                    newdataobj.append(down)

            dataObj =  newdataobj

        self.model = leaves
        #for l in leaves:
        #    l.history = np.array(l.history)

    def loopThruRTHistory(self,leaf,point):
        dict = self.dict
        out = False
        if point[dict[leaf[0]]]>=leaf[1]:
            ans = 1
        else:
            ans = 0
        if (ans == leaf[2]):
            out = True
        else:
            out = False
        return out


    def loopThruRTLearner(self, point):
        out = 0
        dict = self.dict
        if len(self.model) ==1:
                out =  self.model[0].Y
        else:
            for a in self.model:
                hist = a.history
                breaker = False
                #loop = np.apply_along_axis(self.loopThruRTHistory,1,hist,point)
                #if all(loop):
                #    out =  a.Y
                #    break

                for i in range(0,len(hist)):
                    if point[dict[hist[i][0]]]>=hist[i][1]:
                        ans = 1
                    else:
                        ans = 0
                    if (ans == hist[i][2]) and (i == (len(hist)-1)):
                        out = a.Y
                        breaker = True
                        break
                    if ans != hist[i][2]:
                        break

                if breaker==True:
                    break


        return out

    def query(self,points):
        """
        @summary: Estimate a set of test points given the model we built.
        @param points: should be a numpy array with each row corresponding to a specific query.
        @returns the estimated values according to the saved model.
        """


        preds = np.apply_along_axis(self.loopThruRTLearner,1,points)

        '''
        preds = []
        for point in points:

            if len(self.model) ==1:
                preds.append(self.model[0].Y)
            else:
                for a in self.model:
                    hist = a.history
                    breaker = False
                    for i in range(0,len(hist)):
                        if point[dict[hist[i,0]]]>=hist[i,1]:
                            ans = 1
                        else:
                            ans = 0
                        if (ans == hist[i,2]) and (i == (len(hist)-1)):
                            preds.append(a.Y)
                            breaker = True
                            break
                        if ans != hist[i,2]:
                            break

                    if breaker==True:
                        break
            '''

        return preds



if __name__=="__main__":
    print "the secret clue is 'zzyzx'"


