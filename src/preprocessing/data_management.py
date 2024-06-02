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

def save_model(theta0,theta):# it save the parameters or which type of activation you used
    
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,"two_input_XOR_nn.pkl")
    
    
    with open(pkl_file_path,"wb") as file_handle:
        file_handle.dump({"params":{"Biases:":theta0,"Weights:":theta},"activation":config.f})
    
    
def load_model(file_name): #load the saved model
    pkl_file_path = os.path.join(config.SAVED_MODEL_PATH,file_name)
    
    with open(pkl_file_path,"rb") as file_handel:
        trained_params = file_handel.load()
        
        
    return trained_params["Biases"],trained_params["Weights"]


