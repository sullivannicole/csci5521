import numpy as np

class PCA():
    def __init__(self,num_dim=None):
        self.num_dim = num_dim
        self.mean = np.zeros([1,784]) # means of training data
        self.W = None # projection matrix

    def fit(self,X):
        # normalize the data to make it centered at zero (also store the means as class attribute)
        self.mean = np.mean(X)
        X_norm = X - self.mean


        # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)
        X_cov = np.cov(X_norm, ddof = 1)
        evalues, evectors = np.linalg.eigh(X_cov)
        
        evalues_sorted = np.sort(evalues)[::-1]
        evectors_sorted = np.sort(evectors)[::-1]
        
        if self.num_dim is None:
            # select the reduced dimension that keeps >90% of the variance
            num_lambdas = 0
            PoV = 0
            
            while PoV < 0.9:
                PoV += evalues_sorted[num_lambdas]/np.sum(evalues_sorted)
                num_lambdas += 1

            # store the projected dimension
            self.num_dim = num_lambdas

        # determine the projection matrix and store it as class attribute
        self.W = evectors_sorted[:self.num_dim]

        # project the high-dimensional data to low-dimensional one
        #X_pca = X # placeholder
        X_pca = self.mean + evalues_sorted[:self.num_dim] @ self.W

        return X_pca, self.num_dim

    def predict(self,X):
        # normalize the test data based on training statistics
        X_norm = X - self.mean
        
        # project the test data
        X_pca = X_norm @ self.W

        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim
