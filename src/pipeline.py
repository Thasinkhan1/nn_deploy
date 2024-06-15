import pandas as pd
import numpy as np
import math

from src.config import config


theta0 = [None]
theta = [None]

def initialize_layer_biases(num_units):
  return np.random.uniform(low=-1,high=1,size=(1,num_units))


def initialize_layer_weights(num_units_l_1, num_units_l): #(no. of neuron in  previous layer and in current layer)
    return np.random.uniform(low=-1,high=1,size=(num_units_l_1,num_units_l))



    

def initialize_paramerters():
   
   for l in range(1,config.NUM_LAYERS-1):
       
     theta0.append(initialize_layer_biases(config.P[l])/math.sqrt(config.P[l-1]))
     theta.append(initialize_layer_weights(config.P[l-1],config.P[l])/math.sqrt(config.P[l-1])) #p[l-1] is number of neouron in previous layer and pl = no. of neuron in current layer
   
   theta0.append(initialize_layer_biases(config.P[l])/math.sqrt(config.P[l-1]))
   theta.append(initialize_layer_weights(config.P[l],config.P[l])/math.sqrt(config.P[l-1]))
   
   
   

def mini_batch_training(X, Y, mini_batch_size=2):
    initialize_parameters()
    
    num_batches = math.ceil(len(X) / mini_batch_size)
    
    for epoch in range(config.NUM_INPUTS):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * mini_batch_size
            end_idx = min((batch_idx + 1) * mini_batch_size, len(X))
            
            X_batch = X[start_idx:end_idx]
            Y_batch = Y[start_idx:end_idx]
   