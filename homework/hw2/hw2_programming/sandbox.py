import numpy as np
import os

os.chdir('/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/Homework/hw2/hw2_programming/')

data=np.genfromtxt("Digits089.csv",delimiter=",")
Xtrain=data[data[:,0]!=5,2:]
ytrain=data[data[:,0]!=5,1]
Xtest=data[data[:,0]==5,2:]
ytest=data[data[:,0]==5,1]


init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005] #intial indices
centers = Xtrain[init_idx]
cluster_assignment = np.zeros([len(Xtrain),]).astype('int')


for i in range(len(Xtrain)):
    
    distances = np.array(list(map(np.linalg.norm, Xtrain[i] - centers)))
    
    cluster_assignment[i] = np.argmin(distances) + 1
    

unique_clusters = np.unique(cluster_assignment)

centers[0] = np.mean(Xtrain[np.where(cluster_assignment == i)], axis = 0)

for i in range(len(centers)):
    centers[i] = np.mean(Xtrain[np.where(cluster_assignment == i+1)], axis = 0)

