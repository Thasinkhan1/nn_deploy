import os #operation related to our filesystem
import pathlib #it provide the generator or function for feching the data 
import src

#generator is a typr of function that are in lazy manner it never execute exactly after the running the function it create the dag of operation

#IT IS A MODULE 
NUM_INPUTS = 2
NUM_LAYERS = 3
P = [NUM_INPUTS,2,1] # NO OF PERCEPTRON

PACKAGE_ROOT = pathlib.Path(src.__file__).resolve().parent # making src the root directory
DATAPATH = os.path.join(PACKAGE_ROOT,"datasets") 
#"/src/datasets"

SAVED_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_model") 
#"/src/trained_model"

f = [None,"LINEAR","SIGMOID"] # list of activation

TASK_TYPE = "classification"

LOSS_FUNCTION = ["MEAN SQUARED ERROR"]
MINI_BATCH_SIZE = 1
# theta0 = [None]
# theta = [None]

# z = [None]*NUM_LAYERS #aoutomatically have three time of None
# h = [None]*NUM_LAYERS


# del_fl_by_del_z = [None]*NUM_LAYERS
# del_hl_by_del_theta0 = [None]*NUM_LAYERS
# del_hl_by_del_theta = [None]*NUM_LAYERS
# del_L_by_del_h = [None]*NUM_LAYERS
# del_L_by_del_theta = [None] *NUM_LAYERS
# del_L_by_del_theta0 =  [None] *NUM_LAYERS