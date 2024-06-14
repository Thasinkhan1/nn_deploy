import os 
import pathlib #it provide the generator or function for feching the data 
import src


#IT IS A MODULE 
NUM_INPUTS = 2
NUM_LAYERS = 3
P = [NUM_INPUTS,2,1] # NO OF PERCEPTRON

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent # making src the root directory
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets") 
#output of this line ---> "/src/datasets"

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_model") 
# output of this line --> "/src/trained_model"

f = [None,"linear","sigmoid"] # list of activation function


LOSS_FUNCTION = ["mean squared error"]
MINI_BATCH_SIZE = 2
