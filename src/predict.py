import pandas as pd
import numpy as np
import src.preprocessing.preprocessor as pp


from src.config import config
from flask import Flask, request, jsonify
from src.preprocessing.data_management import load_dataset,save_model,load_model
from sklearn.metrics import accuracy_score, f1_score

app = Flask(__name__)

def predict(X):
    loaded_model = load_model("two_input_XOR_nn.pkl") # loading the trained data from pickling file
    
    preprocessor = pp.preprosses_data()
    preprocessor.fit(X)
    X_processed, _ = preprocessor.transform(X)

    
    num_samples = len(X_processed)
    predictions = []

    for i in range(0, num_samples, config.MINI_BATCH_SIZE):
        X_batch = X_processed[i:i + config.MINI_BATCH_SIZE]

        
        Z = np.dot(X_batch, loaded_model["params"]["Weights"][1]) + loaded_model["params"]["Biases"][1]
        A = 1 / (1 + np.exp(-Z)) # Example activation function (sigmoid)
        predictions.extend(A)

    return predictions

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json['inputs']
    data = np.array(data)
    predictions = predict(data)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)