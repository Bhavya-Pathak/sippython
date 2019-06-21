# -*- coding: utf-8 -*-


l1=[1,5,78]
type(l1)

tup=(1,5.0,'a')
type(tup)

d={1:'hi',2:'hello'}

s=set(['i','p'])
s1={'a','b','c','a'}
type(s1)
s1
str='zee'

for i in l1:
    print(i)
 #tuple-immutable   
for i in tup:
    print(i)
    
for i in range(1,10,2) :
    print(i,end=' ')
for i in range(1, 10, 2):
    print(i, end=' ')
    

tupfrozen1 = (1, 2, 3, 4, 5, 6) 
type(tupfrozen1)  

# converting tuple to frozenset #can make keys of dict. a a frozen set
fs1 = frozenset(tupfrozen1) 
fs1
type(tupfrozen1)
s1='a'
type(s1)
d
frozenset2 = frozenset(d)
type(frozenset2)
frozenset2
#keys of dictionary made as frozen set

#%%
#zip - map the similar index of multiple containers 
# initializing lists 
name = [ "A","B","C" ] 
rollno = [ 1,2,3 ] ,
marks = [ 90,0,2,52 ] 
mapped = zip(name, rollno, marks) 
mapped = set(mapped) 
mapped
namez, rollnoz, marksz = zip(*mapped)
namez


#%%
#numpy - array - same data type
import numpy as np
np1 = np.arange(1,10)
np1
type(np1)
np1
np2 = np.array([ 9, 150, 860, 70 ])
np2
np.sort(np2)

np3 = np.array([[1,4],[3,1]])
np3
np3.shape

#%%
#pandas - dataframe, excel like
import pandas as pd

df1 = pd.DataFrame({'rollno':[1,2,3,4], 'name': [ "Dhiraj", "Kounal", "Akhil", "Pooja" ], 'marks':[ 40, 50, 60, 70 ], 'gender':['M','M','M','F']})
df1
type(df1) 

df1.columns
df1.describe
df1.dtypes
df1.shape
df1.groupby('gender').size()
df1.groupby('gender')['marks'].mean()
df1.groupby('gender').aggregate({'marks': [np.mean, 'max']})

#%%
#Graphs https://python-graph-gallery.com/
import matplotlib.pyplot as plt
#https://matplotlib.org/
df1.groupby('gender').size().plot(kind='bar')

#https://seaborn.pydata.org/index.html
import seaborn as sns
# sns.set(style="ticks", color_codes=True)
iris = sns.load_dataset("iris")
sns.pairplot(iris)


#%%
#Load Inbuilt Datasets
import statsmodels.api as sm
#https://vincentarelbundock.github.io/Rdatasets/datasets.html
mtcars = sm.datasets.get_rdataset(dataname='mtcars', package='datasets')
mtcars.data.head()

#%%
#Load from Excel/ CSV and export to
data = mtcars.data
data.head()

data.to_csv('exportcsv1.csv')
data.to_excel('exportexcel1.xlsx','sheet1', header=False)

#load from CSV and Excel
data2a = pd.read_csv('exportcsv1.csv')
data2a.head()

data2b = pd.read_excel('exportexcel1.xlsx','sheet1')
data2b.head()


