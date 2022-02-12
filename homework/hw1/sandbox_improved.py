import numpy as np

import os
os.chdir('/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/Homework/hw1/hw1_programming/')

df = np.genfromtxt("training_data.txt",delimiter=",")
dftest = np.genfromtxt("test_data.txt",delimiter=",")
Xtrain = df[:,0:8] # 9th column (pos = 8) is label
ytrain = df[:, 8]
Xtest = dftest[:,0:8]
ytest = dftest[:,8]
k = 2
#p = [1.0/k for i in range(k)]
p = [0.3, 0.7]

#S = np.zeros((2, 8,8))
S = np.zeros((8,8))

#------------------------

# GOES IN FUNC 1, FIT

#------------------------

def class_mean(y_class):
    return np.mean(Xtrain[ytrain == y_class], axis = 0).tolist()

def compute_covariance3d(y_class, index):
    # 3d shape if class-dep
    S[index,:,:] = np.cov(Xtrain[ytrain == y_class].T, ddof = 0) # 0 is simple average
        
# compute the mean for each class
distinct_classes = np.unique(ytrain)
class_means = np.array(list(map(class_mean, distinct_classes)))

# class-dependent
list(map(compute_covariance3d, distinct_classes, range(k)))


#------------------------

# GOES IN FUNC 1, PREDICT

#------------------------

def S_inv(S):
    return np.linalg.inv(S)

def W(S):
    return -0.5 * S_inv(S)
    
def w(means, S = S):
    return S_inv(S) @ means

def w0(means, p, S = S):
    return -0.5 * (means.T @ S_inv(S) @ means) - 0.5 * np.log(np.linalg.det(S)) + np.log(p)
    

# class-dependent
# each contains a nested list for each class
Wi = list(map(W, S)) 
wi = list(map(w, class_means, S))
wi0 = list(map(w0, class_means, p, S))

g = np.zeros((Xtest.shape[0], k))

# calc g on test set using Wi, wi, wi0
for i in np.arange(Xtest.shape[0]):
    for c in np.arange(k):
        
        Xtest_obs = Xtest[i] #Xtest[i,:]   
        #g[i][c] = np.dot(Xtest_obs.T, np.dot(Wi[c], Xtest_obs)) + np.dot(wi[c].T, Xtest_obs) + wi0[c]
        g[i][c] = Xtest_obs.T @ Wi[c] @ Xtest_obs + wi[c].T @ Xtest_obs + wi0[c]

# make prediction based on largest g
def predict_class(i):
    return distinct_classes[g[i].argmax()]

preds = list(map(predict_class, range(Xtest.shape[0])))
    
predicted_class = np.array(preds).T

#=======checking results for class-indep, diagonal = predicted right
# 79% acc
conf = np.array([[sum((ytest==1) & (predicted_class==1)),sum((ytest==2) & (predicted_class==1))],
                 [sum((ytest==1) & (predicted_class==2)),sum((ytest==2) & (predicted_class==2))]])

# class-independent=======================================

S[:, :] = np.cov(Xtrain.T, ddof = 0)

def w0_shared(means, p):
    return -0.5 * (means.T @ S_inv(S) @ means) + np.log(p)

wi = list(map(w, class_means))
wi0 = list(map(w0_shared, class_means, p))

g = np.zeros((Xtest.shape[0], k))

# calc g on test set using Wi, wi, wi0
for i in np.arange(Xtest.shape[0]):
    for c in np.arange(k):
          
        g[i][c] = wi[c].T @ Xtest[i] + wi0[c]
        
preds = list(map(predict_class, range(Xtest.shape[0])))
predicted_class = np.array(preds).T

#89% acc
conf2 = np.array([[sum((ytest==1) & (predicted_class==1)),sum((ytest==2) & (predicted_class==1))],
                  [sum((ytest==1) & (predicted_class==2)),sum((ytest==2) & (predicted_class==2))]])

#------------------------

# GOES IN FUNC 2, FIT

#------------------------

# class-independent, diagonal ====================================
def class_mean(y_class):
    return np.mean(Xtrain[ytrain == y_class], axis = 0).tolist()

# compute the mean for each class
distinct_classes = np.unique(ytrain)
class_means = np.array(list(map(class_mean, distinct_classes)))

feature_sigmas = np.std(Xtrain, axis = 0)

#------------------------

# GOES IN FUNC 2, PREDICT

#------------------------
p = [0.3, 0.7]

g = np.zeros((Xtest.shape[0], k))

#-0.5 * np.sum(Xtest[1] - class_means[0] 

# calc g on test set using Wi, wi, wi0
for i in np.arange(Xtest.shape[0]):
    for c in np.arange(k):
          
        g[i][c] = -0.5 * np.sum( (Xtest[i] - class_means[c]) / feature_sigmas)**2 + np.log(p[c])


preds = list(map(predict_class, range(Xtest.shape[0])))
predicted_class = np.array(preds).T

#81% acc
conf3 = np.array([[sum((ytest==1) & (predicted_class==1)),sum((ytest==2) & (predicted_class==1))],
                  [sum((ytest==1) & (predicted_class==2)),sum((ytest==2) & (predicted_class==2))]])



