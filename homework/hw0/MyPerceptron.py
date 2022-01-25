"""
Below is the pseudo code that you may follow to create your python user defined function.

Your function is expected to return :-
    1. number of iterations / passes it takes until your weight vector stops changing
    2. final weight vector w
    3. error rate (fraction of training samples that are misclassified)

def keyword can be used to create user defined functions. Functions are a great way to
create code that can come in handy again and again. You are expected to submit a python file
with def MyPerceptron(X,y,w) function.

"""
# Hints
# one can use numpy package to take advantage of vectorization
# matrix multiplication can be done via nested for loops or
# matmul function in numpy package

# Header
import numpy as np
import random

def calculate_error_rate(X, y, w):
    
    x_1 = X[:, 0]
    x_2 = X[:, 1]
    
    w1 = w[0] # weight 1
    w2 = w[1]
    
    pred_numeric = np.multiply(w1, x_1) + np.multiply(w2, x_2)

    # Convert predictions to binary
    pred_binary = []
    
    for i in pred_numeric:
        if i >= 0:
            pred_binary.append(1)
        else:
            pred_binary.append(-1)

    # Error rate of algorithm
    inaccurate_predictions = 0

    for i, ele in enumerate(pred_binary):
        
        if y[i] != ele:
            inaccurate_predictions += 1

    error_rate = inaccurate_predictions/len(y)

    return error_rate

# Implement the Perceptron algorithm
def MyPerceptron(X, y, w0 = [1.0,-1.0]):
    
    #======================================
    
    # Hyperparameters and initializations
    
    k = 0 # initialize variable to store number of iterations it will take
          # for your perceptron to converge to a final weight vector
    w = w0
    error_rate = 1.00

    
    # Features
    x_1 = X[:, 0]
    x_2 = X[:, 1]
    
    # Weights
    w1 = w[0] # weight 1
    w2 = w[1] # weight 2
    
    
    eta = 0.005 # learning rate, p. 194
    epochs = 100
    
    k = 0 # ith epoch
    
    print_div = "*"*60
    
    #======================================
    
    
    for i in range(epochs):
        
        k += 1
        
        index = range(len(X))
        
        # Leave-one-out k-fold
        index_shuffled = random.sample(index, len(X)-1)
        
        for i in index_shuffled:
            
            r_i = w1 * x_1[i] + w2 * x_2[i] # numeric prediction for the ith datapoint
            
            # Convert to binary classification
            if r_i >= 0:
                pred_i = 1
            else:
                pred_i = -1
            
            # Update weights
            error = y[i] - pred_i # truth - prediction
            
            w1 = w1 + eta * error * x_1[i]
            w2 = w2 + eta * error * x_2[i]
        
        # Append weights at the end of every epoch
        w = np.vstack([w, [w1, w2]])
        error_rate = calculate_error_rate(X = X, y = y, w = w[k])
        
        
        print(print_div + "\n\nEnd of epoch {}\nWeights are: [{}, {}]\nError rate: {}\n".format(k, w1, w2, error_rate))
        
        if np.array_equal(w[k], w[k-1]):
            return w[k], k, error_rate
        
if __name__ == "__main__":
    
    alt_data = np.genfromtxt('/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/Homework/hw0_programming/AltData.csv', delimiter = ',')
    
    #====================
    # Inputs

    X = alt_data[:, :2]
    y = alt_data[:, 2]
    w = np.array([1, -1])
        
    #====================
    
    test_w, test_k, test_error = MyPerceptron(X = X, y = y)
    
    print("Final weights were: {}\nEpochs to converge: {}\nFinal error rate: {}".format(test_w, test_k, test_error))
    
    
