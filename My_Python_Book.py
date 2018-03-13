import numpy as np
import pandas as pd

'''Recursion'''
def countDown(n):
    if n<= 0:
        print('whoa\n')
    else:
        print(n)
        countDown(n-1)

countDown(10)

'''USER PROMPT'''
#User prompt
#name = raw_input("What is your name?\n")
#print(name)



'''STRINGS'''
#Characters in Strings
fruit = 'banana'
letter = fruit[2]
for char in fruit:
    print(char)

findLetter =  fruit.find('na')
#print(findLetter)

if 'na' in fruit:
    print('you found a letter')



#String into a list
s = 'spam'
t = list(s)
#print(t)

#The same word is stored in the same String object
a = 'word'
b = 'word'
if a is b:
    print('a and b points to the same object')


'''LISTS'''
class ListExamples():
    def __init__(self):
        self.a = [1, 2, 3]
        self.b = [4, 5, 6]

    def combineLists(self):
        self.a.extend(b)
        print(self.a)

    def removeElement(self):
        x = self.a.pop(0)
        print(x)

del t[0]
#print(t)

t.remove('c')
#print(t)

#Sort
b = np.array([[1,2,3], [4,5,6], [7,8,9]])
b = b.tolist()
b = sorted(b,key= lambda x: x[1],reverse=True)
print(b)

#Can only use the IN operator on a list, not a Series
x=['aapl','ibm','dell']
x = pd.DataFrame(x,columns=['a'])
print(x)
if 'dell' in x['a'].tolist():
    print('your right')


#List Comps
name = ['frank','dave']
age = [12,33]
y = [(x,y) for x in name for y in age]
print(y)

#List Comps #2
a = [1,3,4,5]
comp = ['WORD' if x == 1 else x for x in a]


#Assigning to slices
l = [1,2,3,4,5,6]
l[0:4]=[1] #object assigned to slice must be iterable
print('slice assignment')
print(l)

#Splitting nested lists into two separate lists
tst =  [[1,2],[2,4]]
lst1, lst2 = map(list,zip(*tst))
print('lst1',lst1)
print('lst2',lst2)


'''DATETIME Class/TIME Class'''
#convert string to time object
date = '2001-12-01'
import datetime as datetime
date = datetime.datetime.strptime(date,'%Y-%m-%d')
#print(date)


#Create iterable dateTime object
start = datetime.datetime(2016,12,1)
end = datetime.datetime(2016,12,30)
date_ranges = pd.date_range(start,end)
#print('DATE RANGES')
#print(date_ranges)

#Convert DataIndex of a DataFrame to Date Time format
dateIndex = pd.date_range(datetime.datetime(12,1,2009),datetime.datetime(12,1,2012),freq='1D')
tst = pd.DataFrame(index = dateIndex)
day = tst.index[0]
print('date type for panda dateIndex',type(day))
reformatDT = pd.Timestamp(day).to_pydatetime()
print('data type should be datetime now',type(reformatDT))

#Substract two TimeStamps
start = pd.Timestamp(datetime.datetime(2009,1,1))
end = pd.Timestamp(datetime.datetime(2012,1,1))
diff = end - start
print(diff.days)



'''PANDAS'''
#Add a new column to DataFrame
a = [[1,2,3],[4,5,6]]
p = pd.DataFrame(a)
p['newcol'] = pd.Series(np.repeat(0,p.shape[0]))

#Drop rows with NANA
a = [[1,2,3],[4,5,6]]
p = pd.DataFrame(a)
p.dropna(axis=0)

#List into DataFrame as a Row
a = [1,3,4,5]
p = pd.DataFrame([a])
#print(p)

#Append into DataFrame
row = [4,5,5,5]
p = p.append([row],ignore_index=True)
print(p)

#Merge two DataFrames
a = [1,3,4,5]
a = pd.DataFrame([a],index=[0])
b = [4,3,7,5]
b = pd.DataFrame([a],index=[10])

#Rename Columns
test = pd.DataFrame(columns=['A','B'])
test = test.rename(columns={'A':1,'B':2})
vals = pd.concat([a,b],axis=0,join='outer',ignore_index=True)
print(vals)

#Find the index of value via boolean mask
df = pd.DataFrame([1,2,1],columns=['A'])
idx = df.ix[df['A']==1,'A']
print('index')
print(idx)

#Reset Index
df1 = df.copy()
df1 = df.reset_index(drop=True)


#Rolling


#GroupBy
data = pd.DataFrame({'nums':[1,3,5,2,6],'num2':[2,4,2,5,7],'type':['odd','odd','odd','even','even'],'level':[1,2,2,3,1]})
grouped =  data['nums'].groupby([data['type'],data['level']])
#if data and map in same dataframe
grouped = data.groupby(['type','level'])
#print(grouped.mean())
#print(grouped.size())


for type, level in data.groupby('type'):
    #print(type)
    #print(level)
    pass

#Groupby into Dictionaries
pieces = dict(list(data.groupby('type')))
print(pieces['odd'])
print('marker')
x = list(data.groupby('type'))[0]
print(x)

#Groupby using functions
def rounder(x):
    import math as math
    return math.ceil(x)
grouped = data['nums'].groupby(rounder)
print('functions')
print(grouped.mean())

#Using keyword IN for a Panda DataFrame - must convert to list first!
df = pd.DataFrame([1,2,3],columns=['a'])
if 2 in df['a'].tolist():
    print('ITS IN THERE!')
    pass


'''DICTIONARIES'''
dict = {'a':1,'b':2,'c':10,'d':20}

#Value/Keys in dictionary
print(dict.values())
print(dict.keys())
print(1 in dict.keys())

#List into Dictionaries
lst =[('a',[1,2]),('b',[3,4])]
#dct = dict(lst)

'''TUPLE'''
#Single tuple
a = '1',
#print(type(a))

#String Tuple
a =  tuple('word')
#print(a)

#Zip method
a = tuple('abcdefghi')
b = [0,2,3]
#print(zip(a,b))

#Compare tuples
a=(1,2,3)
b = (4,5,6)
#print(a < b)


#Tuple unpacking/paralell assignment
coord = (19,20)
x,y = coord
print('TUPLE UNPACKING')
print(x)
print(y)


#Slicing: tuple{start:stop:step]
s = 'randomness'
print('tuple slicing')
print(s[::2])




'''RANDOM CLASSES'''

#Seed the random object so the same random numbers come up
np.random.seed(1)





'''CLASS'''
#Default methods
class MyClass(object):

    def __init__(self):
        pass

    def __str__(self):
        #String representation of the object, used by the print() method
        pass

    def __str__(self, param):
        #Overloaded version of the Str() method
        pass

    def __lt__(self, other):
        #Same as < boolean operator (Less Than)
        pass

    def __ge__(self, other):
        #Same as >= operator (Greater Than or Equals)
        pass

    def __add__(self, other):
        pass

    def __cmp__(self, other):
        pass

#Object is the Superclass for every class
print('Base Class')
print(MyClass.__class__.__base__)



'''NUMPY'''
#initialization
a  = np.array(([1,3,4],[5,22,4],[11,232,2]))
b = np.empty([10,2])
b = np.zeros([10,2])
#print(b)

#Shape
#print(a.shape[0])

#Convert into array, nothin if already array
c = np.asanyarray(a)
#print(type(c))

#Convert datatypes in arrays
d = b.astype(np.float64)
#print(d.dtype)

#Boolean Mask
t = np.array([1,3,4,4,1,1,1])
#print(a[a<3])

#Transpose
print(a.T)

#Unique
print(np.unique(a))

#Combining by rows or columns
a1 = np.array([1,2,3,])
a2 = np.array([4,5,6])
a3 = np.r_[a1,a2]
a4 = np.c_[a1,a2]
zipped = zip(a1,a2)

#Broadcasting
#Normalizing data
arr = np.array([[1,2,3],[4,5,6]])
mean =  arr.mean(axis=1)
mean = mean.reshape((2,1))
scaled = arr - mean
print(scaled)


#Apply Along Axis
def my_func(a):

    return a[0]

b = np.array([[1,2,3], [4,5,6], [7,8,9]])
print(np.apply_along_axis(my_func,1,b))




'''CONCURRENCY/THREADING'''
#Running processes on multiple threads
import time
import threading
def countdown(count):
   arr1 = np.zeros((10000,10000))
   return arr1 * arr1

start = time.time()
a = countdown(10)
b = countdown(10)
end = time.time()
print(end-start)


start = time.time()
t1 = threading.Thread(target=countdown,args=(10,))
t1.start()
t2 = threading.Thread(target=countdown,args=(10,))
t2.start()
end = time.time()
print(end - start)

#Have a thread wait for other threads to finish
t1.join()

#if a thread runs continuously in the background make it Daemonic. Otherwise the interpretor wait for the thread to finish
t1.setDaemon(True)

#Can extend the existing threading.thread class and overload the run() function to customize it
class CountdownThread(threading.Thread):
     def __init__(self,count):
        threading.Thread.__init__(self)
        self.count = count
     def run(self):
        while self.count > 0:
            print "Counting down", self.count
            self.count -= 1
            time.sleep(5)
        return



'''MISC FUNCTIONS'''
#Get the Pointer
x = [1,2,4]
print('memory address')
print(id(x))

#Test to see for a Type
x = pd.DataFrame()
assert isinstance(x,pd.DataFrame), "This has to be a DataFrame"
