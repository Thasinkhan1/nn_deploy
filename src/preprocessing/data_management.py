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


