#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 10:32:03 2022

@author: nicolesullivan
"""
import os

def process_label(label):
    # convert the labels into one-hot vector for training
    #one_hot = np.zeros([len(label),10])
    
    #n_classes = len(np.unique(label))
    one_hot = np.eye(10)[label] # Create ID matrix of classes, & then for each entry in train, slice to that row

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line

    f_x = (2 / (1 + np.exp(-2 * x))) - 1

    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    x_exp = np.exp(x)
    f_x = x_exp / np.sum(x_exp, axis = 1, keepdims = True)
    return f_x
    
    #return f_x

os.chdir('/Users/nicolesullivan/Documents/Academic/2021-2023/MS in DS/Coursework/2022/Spring/CSCI 5521/GitHub/csci5521/homework/hw3/')
#import libraries
import numpy as np

num_hid = 4


# initialize the weights
weight_1 = np.random.random([64,num_hid]) # 64 x 4 # w_h in p1
bias_1 = np.random.random([1,num_hid]) # w0 1 x 4 # wh0 in p1
        
weight_2 = np.random.random([num_hid,10]) # this is v, 4 x 10 #v_h in p1
bias_2 = np.random.random([1,10])

# read in data.
# training data
train_data = np.genfromtxt("optdigits_train.txt",delimiter=",")
train_x = train_data[:,:-1]
train_y = train_data[:,-1].astype('int')

# validation data
valid_data = np.genfromtxt("optdigits_valid.txt",delimiter=",")
valid_x = valid_data[:,:-1]
valid_y = valid_data[:,-1].astype('int')

# test data
test_data = np.genfromtxt("optdigits_test.txt",delimiter=",")
test_x = test_data[:,:-1]
test_y = test_data[:,-1].astype('int')

x_norm = (train_x - np.mean(train_x, axis = 0))/(np.std(train_x, axis = 0) + 1e-15)

z_raw = bias_1 + (x_norm @ weight_1)
z = tanh(z_raw)
y_raw = bias_2 + (z @ weight_2)
yt = softmax(y_raw)      
rt = process_label(train_y)

# np.sum(yt[0]) check
yt_rt = yt - rt

v_gradient = z.T @ yt_rt
v_gradient_other = yt_rt.T @ z
v0_gradient = np.sum(yt_rt, axis = 0)

dz = 1-z**2 # dz/dbeta
(1-tanh(z)**2).shape
w_gradient = (dz * (yt_rt @ v_gradient_other)).T @ x_norm
w_inner = (yt_rt @ v_gradient_other) * dz
w_gradient_other = w_inner.T @ x_norm
w0_gradient = np.sum((dz.T @ yt_rt) @ v_gradient.T, axis = 1)
