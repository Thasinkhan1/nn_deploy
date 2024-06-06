import pandas as pd
import numpy as np
from src.config import config
import src.preprocessing.preprocessor as pp
from src.preprocessing.preprocessor import preprosses_data #importing the class from preprocessor.py

from src.preprocessing.data_management import load_dataset,save_model,load_model
import pipeline as p



z = [None]*config.NUM_LAYERS #automatically have three time of None
h = [None]*config.NUM_LAYERS


del_fl_by_del_z = [None]*config.NUM_LAYERS
del_hl_by_del_theta0 = [None]*config.NUM_LAYERS
del_hl_by_del_theta = [None]*config.NUM_LAYERS
del_L_by_del_h = [None]*config.NUM_LAYERS
del_L_by_del_theta = [None] *config.NUM_LAYERS
del_L_by_del_theta0 =  [None] *config.NUM_LAYERS


def layer_neurons_weighted_sum(previous_layer_neurons_outputs, current_layer_neurons_biases, current_layer_neurons_weights):

  return current_layer_neurons_biases + np.matmul(previous_layer_neurons_outputs,current_layer_neurons_weights) # lth layer le liye weight sum ka answer dega



def layer_neurons_output(current_layer_neurons_weighted_sums, current_layer_neurons_activation_function):

  if current_layer_neurons_activation_function == "linear":
    return current_layer_neurons_weighted_sums

  elif current_layer_neurons_activation_function == "sigmoid":
    return 1/(1 + np.exp(-current_layer_neurons_weighted_sums)) # it is automaticLLY APLLY e^2

  elif current_layer_neurons_activation_function == "tanh":
    return (np.exp(current_layer_neurons_weighted_sums) - np.exp(-current_layer_neurons_weighted_sums))/ \
            (np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums))

  elif current_layer_neurons_activation_function == "relu":
    return np.max(np.zeros((current_layer_neurons_weighted_sums.shape[0],1)), current_layer_neurons_weighted_sums)



def del_layer_neurons_outputs_wrt_weighted_sums(current_layer_neurons_activation_function, current_layer_neurons_weighted_sums):

  if current_layer_neurons_activation_function == "linear":
    return np.ones_like(current_layer_neurons_weighted_sums) # it makes the mstrix of ones

  elif current_layer_neurons_activation_function == "sigmoid":
    current_layer_neurons_outputs = 1/(1 + np.exp(-current_layer_neurons_weighted_sums))
    return current_layer_neurons_outputs * (1 - current_layer_neurons_outputs)

  elif current_layer_neurons_activation_function == "tanh":
    return (2/(np.exp(current_layer_neurons_weighted_sums) + np.exp(-current_layer_neurons_weighted_sums)))**2

  elif fl == "relu":
    return current_layer_neurons_weighted_sums * (current_layer_neurons_weighted_sums > 0)



def del_layer_neurons_outputs_wrt_biases(current_layer_neurons_outputs_dels):

  return current_layer_neurons_outputs_dels


def del_layer_neurons_outputs_wrt_weights(previous_layer_neurons_outputs,current_layer_neurons_outputs_dels):

  return np.matmul(previous_layer_neurons_outputs.T,current_layer_neurons_outputs_dels)


def run_training(tol,epsilon):
    epoch_counter = 0
    mse = 1
    loss_per_epoch = list()
    loss_per_epoch.append(mse)
    
    training_data = load_dataset("train.csv")
    obj = pp.preprosses_data()
    obj.fit(training_data.iloc[:,0:2],training_data.iloc[:,2])
    
    X_train,Y_train = obj.transform(training_data.iloc[:,0:2],training_data.iloc[:,2])
    
    p.initialize_paramerters()
    
    while True:
      mse = 0
      for i in range(X_train.shape[0]):
 
          h[0] = X_train[i].reshape(1,X_train.shape[1]) # 0th index of h h[0] = (0,0)
    
          for l in range(1,config.NUM_LAYERS): #loop is run 2 times

    
            z[l] = layer_neurons_weighted_sum(h[l-1], p.theta0[l], p.theta[l])
            #print("z[{}].shape = {}".format(l,z[l].shape))
            h[l] = layer_neurons_output(z[l], config.f[l])
            #print("h[{}].shape = {}".format(l,h[l].shape))
    
            del_fl_by_del_z[l] = del_layer_neurons_outputs_wrt_weighted_sums(config.f[l],z[l])
            #print("del_fl_by_del_z[{}].shape = {}".format(l,del_fl_by_del_z[l].shape))
            del_hl_by_del_theta0[l] = del_layer_neurons_outputs_wrt_biases(del_fl_by_del_z[l])
            #print("del_hl_by_del_theta0[{}].shape = {}".format(l,del_hl_by_del_theta0[l].shape))
            del_hl_by_del_theta[l] = del_layer_neurons_outputs_wrt_weights(h[l-1],del_fl_by_del_z[l])
            #print("del_hl_by_del_theta[{}].shape = {}".format(l,del_hl_by_del_theta[l].shape))
    

      #print("\n")

      Y_train[i] = Y_train[i].reshape(Y_train[i].shape[0],1)
     #       print("y[{}].shape = {}".format(i,y[i].shape))
      L = (1/2)*(Y_train[i][0] - h[config.NUM_LAYERS-1][0,0])**2
            #The above expression is the expression of Loss Function in our case which is Squared Error.
      
      mse = mse + L
      
      del_L_by_del_h[config.NUM_LAYERS-1] = (h[config.NUM_LAYERS-1] - Y_train[0])
            #The above expression is the expression of derivative of the loss function with respect to the output of the Neural Network.
             # print("del_L_by_del_h[{}].shape = {}".format(NUM_LAYERS-1,del_L_by_del_h[NUM_LAYERS-1].shape))
      for l in range(config.NUM_LAYERS-2,0,-1):
      
          del_L_by_del_h[l] = np.matmul(del_L_by_del_h[l+1], (del_fl_by_del_z[l+1] * p.theta[l+1]).T)
              #print("del_L_by_del_h[{}].shape = {}".format(l,del_L_by_del_h[l].shape))
              #The expression shown above is computing the derivative of the Loss Function wrt to the output of neurons in a layer. For more information, please see
              #the first picture below.
      
      #print("\n\n\n")
      for l in range(1,config.NUM_LAYERS):
              del_L_by_del_theta0[l] = del_L_by_del_h[l] * del_hl_by_del_theta0[l]
              del_L_by_del_theta[l] = del_L_by_del_h[l] * del_hl_by_del_theta[l]
      
              p.theta0[l] = p.theta0[l] - (epsilon * del_L_by_del_theta0[l])
              p.theta[l] = p.theta[l] - (epsilon * del_L_by_del_theta[l])
              
              
      mse = mse/X_train.shape[0]
      epoch_counter = epoch_counter + 1
      loss_per_epoch.append(mse)
      print("Epoch # {}, Loss = {} ".format(epoch_counter,mse))
  
        #Now, we will be running Gradient Descent Algorithm (Backpropagation Algorithm) by computing the overall derivative of the Loss Function wrt the biases and
        #weights of each layer (as shown in the second picture below) and updating these parameters.
        
      if abs(loss_per_epoch[epoch_counter]-loss_per_epoch[epoch_counter-1]) < tol:
         break

if __name__ == "__main__":
    run_training(10**(-3),10**(-7))
    save_model(p.theta0,p.theta0)