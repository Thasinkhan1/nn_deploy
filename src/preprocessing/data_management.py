#we saving the parameters in this file 

#.hdf5, .h5 , .npz, .npy, .pkl, .keras # they all are the file format to store the information of the data

import os
import pandas as pd
import pickle

from src.config import config

def load_dataset(file_name):
    
    file_path = os.path.join(config.DATAPATH,file_name) #"/src/datasets/train.csv"
    data = pd.read_csv(file_path)
    return data

def save_model(theta0,theta):# it save the parameters or which type of activation function you used
    
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,"two_input_XOR_nn.pkl")
    
    
    with open(pkl_file_path,"wb") as file_handle:
        pickle.dump({"params":{"Biases:":theta0,"Weights:":theta},"activation":config.f},file_handle)
    
    print("Saved Model with file name {} at {}".format("two_input_XOR_nn.pkl",config.SAVED_MODEL_PATH))
    
def load_model(file_name): #load the saved model
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,file_name)
    
    with open(pkl_file_path,"rb") as file_handel:
        trained_params = pickle.load(file_handel)
        
        
    return load_model


