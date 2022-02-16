import numpy as np

# For quadratic (different Si, 0 assumptions)
# & linear discriminant (shared common sample covariance, 1 assumption)
class GaussianDiscriminant:
    
    def __init__(self,k=2,d=8,priors=None,shared_cov=False):
        
        self.mean = np.zeros((k,d)) # mean
        self.shared_cov = shared_cov # using class-independent covariance or not
        self.distinct_classes = np.zeros(k) # add a variable to store unique class labels in
        
        if self.shared_cov:
            self.S = np.zeros((d,d)) # class-independent covariance (S1=S2)
        else:
            self.S = np.zeros((k,d,d)) # class-dependent covariance (S1!=S2)
            
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
            
        self.k = k
        self.d = d

    def fit(self, Xtrain, ytrain):
        
        #==================================
        # compute the mean for each class
        #==================================
        
        def class_mean(y_class):
            return np.mean(Xtrain[ytrain == y_class], axis = 0).tolist()

        def compute_covariance3d(y_class, index):
            # 3d shape if class-dep
            self.S[index,:,:] = np.cov(Xtrain[ytrain == y_class].T, ddof = 0) # 0 is simple average
                
        
        self.distinct_classes = np.unique(ytrain)
        self.mean = np.array(list(map(class_mean, self.distinct_classes)))

        #============================================================
        # compute the class-independent covariance for each class
        #============================================================
        
        if self.shared_cov:
            self.S[:, :] = np.cov(Xtrain.T, ddof = 0)
           
        #============================================================
        # compute the class-dependent covariance for each class
        #============================================================
        else:
 
            list(map(compute_covariance3d, self.distinct_classes, range(self.k)))
            
            

    def predict(self, Xtest):
        
        #============================
        # Calculate weights up-front
        #============================
        
        def S_inv(S):
            return np.linalg.inv(S)

        def W(S):
            return -0.5 * S_inv(S)
            
        def w(means, S = self.S):
            return S_inv(S) @ means

        def w0(means, p, S = self.S):
            return -0.5 * (means.T @ S_inv(S) @ means) - 0.5 * np.log(np.linalg.det(S)) + np.log(p)
        
        # Class-independent
        def w0_shared(means, p):
            return -0.5 * (means.T @ S_inv(self.S) @ means) + np.log(p)
        
        if self.shared_cov:
            wi = list(map(w, self.mean))
            wi0 = list(map(w0_shared, self.mean, self.p))

            
        else:
            # each contains a nested list for each class
            Wi = list(map(W, self.S)) 
            wi = list(map(w, self.mean, self.S))
            wi0 = list(map(w0, self.mean, self.p, self.S))

        g = np.zeros((Xtest.shape[0], self.k))
        
        #==============================
        # Get predictions on test set
        #==============================

        for i in np.arange(Xtest.shape[0]): # for each test set example

            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                
                if self.shared_cov:
                    g[i][c] = wi[c].T @ Xtest[i] + wi0[c]
                    
                    
                else:
                    
                    Xtest_obs = Xtest[i] #Xtest[i,:]   
                    g[i][c] = Xtest_obs.T @ Wi[c] @ Xtest_obs + wi[c].T @ Xtest_obs + wi0[c]
        
        #=============================================================================================
        # Determine predicted class based on the values of the discriminant function
        #=============================================================================================
        
        def predict_class(i):
            return self.distinct_classes[g[i].argmax()]
        
        preds = list(map(predict_class, range(Xtest.shape[0])))
            
        predicted_class = np.array(preds).T

        return predicted_class

    def params(self):
        if self.shared_cov:
            return self.mean[0], self.mean[1], self.S
        else:
            return self.mean[0],self.mean[1],self.S[0,:,:],self.S[1,:,:]


class GaussianDiscriminant_Diagonal:
    
    def __init__(self,k=2,d=8,priors=None):
        self.mean = np.zeros((k,d)) # mean
        self.S = np.zeros((d,)) # variance
        self.distinct_classes = np.zeros(k) # add a variable to store unique class labels in
        
        if priors is not None:
            self.p = priors
        else:
            self.p = [1.0/k for i in range(k)] # assume equal priors if not given
            
        self.k = k
        self.d = d
        
    def fit(self, Xtrain, ytrain):
        # compute the mean for each class
        def class_mean(y_class):
            return np.mean(Xtrain[ytrain == y_class], axis = 0).tolist()
        
        self.distinct_classes = np.unique(ytrain)
        self.mean = np.array(list(map(class_mean, self.distinct_classes)))

        # compute the variance of different features
        self.S = np.var(Xtrain, axis = 0)

    def predict(self, Xtest):
        # predict function to get prediction for test set
        
        g = np.zeros((Xtest.shape[0], self.k))

        for i in np.arange(Xtest.shape[0]): # for each test set example
            # calculate the value of discriminant function for each class
            for c in np.arange(self.k):
                g[i][c] = -0.5 * np.sum( ((Xtest[i] - self.mean[c]) / np.sqrt(self.S))**2 ) + np.log(self.p[c])

            # determine the predicted class based on the values of discriminant function
        def predict_class(i):
            return self.distinct_classes[g[i].argmax()]
        preds = list(map(predict_class, range(Xtest.shape[0])))
        predicted_class = np.array(preds).T
        return predicted_class

    def params(self):
        return self.mean[0], self.mean[1], self.S
