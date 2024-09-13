import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler as cumlStandardScaler
from cuml.svm import SVC as cumlSVC
from cuml.metrics import accuracy_score as cuml_accuracy_score
import numpy as np
import time
import json
import joblib

def load_prepared_data():
    print("Loading prepared data from 'prepared_data.pkl'...")
    X_train_scaled, X_test_scaled, y_train, y_test = joblib.load('prepared_data.pkl')
    return X_train_scaled, X_test_scaled, y_train, y_test

def evaluate_model(params):
    X_train_scaled, X_test_scaled, y_train, y_test = load_prepared_data()
    
    # Try-except to ensure clf is defined before deletion
    clf = None
    try:
        clf = cumlSVC(kernel='poly', **params, tol=1e-4, max_iter=500)
        print(f"Training the model with parameters: {params}")

        fit_start_time = time.time()
        clf.fit(X_train_scaled, y_train)
        fit_time = time.time() - fit_start_time

        print(f"Predicting with the model...")
        pred_start_time = time.time()
        y_pred = clf.predict(X_test_scaled)
        pred_time = time.time() - pred_start_time

        accuracy = cuml_accuracy_score(y_test, y_pred)
        
        # Return results to be printed and saved
        return accuracy, fit_time, pred_time
    finally:
        # Ensure clf is deleted if it was successfully created
        if clf is not None:
            del clf
        cp.get_default_memory_pool().free_all_blocks()

def main():
    # Read parameters from the JSON file
    param_file = 'params.json'
    with open(param_file, 'r') as f:
        params = json.load(f)

    accuracy, fit_time, pred_time = evaluate_model(params)

    # Print and save results
    print(f"--- Result Verification ---")
    print(f"Parameters: {params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model fitting time: {fit_time:.2f} seconds")
    print(f"Prediction time: {pred_time:.2f} seconds")
    print(f"----------------------------")

    with open('result.txt', 'a') as f:
        f.write(f'Parametros: {params}\n')
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Model fitting time: {fit_time:.2f} seconds\n')
        f.write(f'Prediction time: {pred_time:.2f} seconds\n\n')

    print(f"Results saved to 'result.txt'")

if __name__ == "__main__":
    main()
