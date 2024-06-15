import pandas as pd
import numpy as np
import src.preprocessing.preprocessor as pp
from src.config import config
from src.preprocessing.data_management import load_dataset,save_model,load_model
from sklearn.metrics import accuracy_score, f1_score



def predict(X):
    loaded_model = load_model("two_input_XOR_nn.pkl") # loading the trained data from pickling file
    
    preprocessor = pp.preprosses_data()
    preprocessor.fit(X)
    X_processed, _ = preprocessor.transform(X)

    
    num_samples = len(X_processed)
    predictions = []

    for i in range(0, num_samples, config.MINI_BATCH_SIZE):
        
        X_batch = X_processed[i:i + config.MINI_BATCH_SIZE]
        print("X_batch:", X_batch)
        
        weights = loaded_model["params"]["Weights:"][1]
        biases = loaded_model["params"]["Biases:"][1]
        print("Shapes before dot product - X_batch:", X_batch.shape, ", weights:", weights.shape)
        
        if weights.shape[0] != X_batch.shape[1]:
            weights = weights.T
            print("Reshaped weights to:", weights.shape)
            
        Z = np.dot(X_batch, weights) + biases 
        
        A = 1 / (1 + np.exp(-Z)) # Example activation function (sigmoid)
        
        predictions.extend(A.flatten()[:X_batch.shape[0]])
        
    print(f"Number of samples: {num_samples}, Number of predictions: {len(predictions)}")
    return predictions

def main():
    
    X_test = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y_test = np.array([0,1,1,0]) #corresponding XOR ouputs
    
    #make prediction
    prediction = predict(X_test)
    prediction = np.array(prediction).flatten()
    prediction  = (prediction >= 0.5).astype(int)
    
    print(f"Y_test: {Y_test}")
    print(f"Predictions: {prediction}")
    
    #calculating accuracy and f1-score
    accuracy = accuracy_score(Y_test,prediction)
    f1 = f1_score(Y_test,prediction)
    print(f"Accuracy is {accuracy} and the F1-Score is {f1}")   
    
if __name__ == "__main__":
      main()