import numpy as np
import os

os.chdir('/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/Homework/hw2/hw2_programming/')

data=np.genfromtxt("Digits089.csv",delimiter=",")
Xtrain=data[data[:,0]!=5,2:]
ytrain=data[data[:,0]!=5,1]
Xtest=data[data[:,0]==5,2:]
ytest=data[data[:,0]==5,1]

#----------for k means

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


#-------for PCA

num_dim = None
mean = np.zeros([1,784]) # means of training data
W = None # projection matrix

X_norm = Xtrain - np.mean(Xtrain)

X_cov = np.cov(X_norm, ddof = 1)

evalues, evectors = np.linalg.eigh(X_cov)

# Sort largest to smallest
# Per docs, both vals and vector come out asc, so reversing the order ensures they're still matched
evalues_sorted = np.sort(evalues)[::-1]

evectors_sorted = np.sort(evectors)[::-1]

# Find # of PCs needed to capture > 90% of variance
num_lambdas = 0
PoV = 0

while PoV < 0.9:
    PoV += evalues_sorted[num_lambdas]/np.sum(evalues_sorted)
    num_lambdas += 1

W = evectors_sorted[0:(num_lambdas - 1)]

# Get matching indices for matching evectors
#evector_indices = np.where(np.in1d(X_evalues, X_evalues_sorted[0:(num_lambdas-1)]))[0]


W = X_evectors[evector_indices]
    
    