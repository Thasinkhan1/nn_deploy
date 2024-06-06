import pandas as pd
import numpy as np

from src.config import config

import src.preprocessing.preprocessor as pp
from src.preprocessing.data_management import load_dataset,save_model,load_model
from src.train_pipeline import run_training

X_train,Y_train = load_model('train.csv')

Model = run_training()

Model.fit(X_train,Y_train)
y_pred = Model.predict(X_train)

# Calculate accuracy and F1-score
accuracy = accuracy_score(Y_train, y_pred)
f1 = f1_score(Y_train, y_pred)

if __name__ == "__main__":
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")