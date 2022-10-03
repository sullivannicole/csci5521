#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:50:21 2022

@author: nicolesullivan
"""

import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self,x):
        # compute the statistics of training samples (i.e., means and std)
        self.mean = np.mean(x, axis = 0) # Mean & std column-wise
        self.std = np.std(x, axis = 0)
        
    def normalize(self,x):
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        x = (x - self.mean)/(self.std + 1e-15)
        
        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    #one_hot = np.zeros([len(label),10])
    
    #n_classes = len(np.unique(label))
    one_hot = np.eye(10)[label] # Create ID matrix of classes, & then for each entry in train, slice to that row

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line

    f_x = (2.0 / (1.0 + np.exp(-2.0 * x))) - 1.0

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    f_x = np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True)
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid]) # 64 x 4 # w_h in p1
        self.bias_1 = np.random.random([1,num_hid]) # w0 1 x 4 # wh0 in p1
        
        self.weight_2 = np.random.random([num_hid,10]) # this is v, 4 x 10 #v_h in p1
        self.bias_2 = np.random.random([1,10]) # this is v0 1 x 10 # v0 in p1

    def fit(self,train_x,train_y, valid_x, valid_y):
        
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count <= 50:
            
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            # from x to y, pay attention to differences in activation functions
            
            z = tanh(self.bias_1 + (train_x @ self.weight_1))
            yt = softmax(self.bias_2 + (z @ self.weight_2)) # our prediction

            yt_rt = yt - train_y

            v_gradient = z.T @ yt_rt
            v0_gradient = np.sum(yt_rt, axis = 0)

            dz = 1-z**2 # dz/dbeta
            
            #w_inner_term = (yt_rt @ v_gradient.T) * dz #1000 x 4
            w_inner_term = (yt_rt @ self.weight_2.T) * dz
            #w_gradient = train_x.T @ w_inner_term # (w_inner_term.T @ train_x).T 
            w_gradient = (w_inner_term.T @ train_x).T
            w0_gradient = np.sum(w_inner_term, axis = 0)
            

            # update the parameters based on sum of gradients for all training samples
            self.weight_1 -= lr * w_gradient
            self.bias_1 -= lr * w0_gradient
            
            self.weight_2 -= lr * v_gradient
            self.bias_2 -= lr * v0_gradient

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        
        # 1. Get transformed z layer (hidden layer)
        z = self.get_hidden(x)
        
        # 2. Calculate y layer
        y_raw = self.bias_2 + (z @ self.weight_2)
        
        # 3. Transform y layer using activation function (generate the predicted probability of different classes)
        y_softmax = softmax(y_raw)
        
        # 4. Convert class probability to predicted labels by choosing max in each row
        y = np.argmax(y_softmax, axis = 1)

        return y

    def get_hidden(self,x):
        
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        # 1. Calculate z
        z_raw = self.bias_1 + (x @ self.weight_1)
        
        # 2. Transform z using activation function
        z = tanh(z_raw)

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
