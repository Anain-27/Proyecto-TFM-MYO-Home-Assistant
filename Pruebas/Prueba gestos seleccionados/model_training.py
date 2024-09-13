import cudf
import time
import json
import subprocess
from cuml.preprocessing import StandardScaler as cumlStandardScaler
from cuml.model_selection import train_test_split
from cuml.svm import SVC as cumlSVC  # Importación de cumlSVC
from cuml.metrics import accuracy_score as cuml_accuracy_score  # Importación de cuml_accuracy_score
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
import joblib

def load_and_prepare_data(input_file, sample_size=None):
    print("Loading and preparing data...")

    # Mide el tiempo para cargar y preparar los datos
    start_time = time.time()

    # Cargar el DataFrame desde el archivo Parquet
    df = cudf.read_parquet(input_file)

    # Aplicar sample size si se especifica
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Separar características y etiquetas
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == 'object':
        y = y.astype('category').cat.codes

    # Separar en conjuntos de entrenamiento y prueba
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar los datos
    print("Standardizing the data...")
    scaler = cumlStandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Mide el tiempo de preparación y estandarización
    prep_time = time.time() - start_time
    print(f'Time for preparation and standardization: {prep_time:.2f} seconds')

    # Guardar datos preparados para la evaluación
    joblib.dump((X_train_scaled, X_test_scaled, y_train, y_test), 'prepared_data.pkl')

def main():
    # Define la cuadrícula de hiperparámetros
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
        'coef0': [0.0, 0.5, 1.0]
    }

    # Define el script para la evaluación
    script_path = 'evaluate_model.py'

    # Mide el tiempo de búsqueda de hiperparámetros
    print("Starting hyperparameter search...")
    search_start_time = time.time()

    best_accuracy = 0
    best_params = None
    best_fit_time = None
    best_pred_time = None

    for params in ParameterGrid(param_grid):
        print(f'\nEvaluating parameters: {params}')

        # Guardar los parámetros en un archivo temporal
        param_file = 'params.json'
        with open(param_file, 'w') as f:
            json.dump(params, f)

        # Ejecutar el script de evaluación
        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        
        # Leer la salida del script
        output = result.stdout
        print(output)

        # Analizar la salida para obtener precisión y tiempos
        try:
            accuracy = float(output.split('Accuracy: ')[1].split()[0])
            fit_time = float(output.split('Model fitting time: ')[1].split()[0])
            pred_time = float(output.split('Prediction time: ')[1].split()[0])
        except (IndexError, ValueError) as e:
            print(f"Error parsing script output: {e}")
            continue

        print(f'Accuracy: {accuracy:.4f}')
        print(f'Model fitting time: {fit_time:.2f} seconds')
        print(f'Prediction time: {pred_time:.2f} seconds')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_fit_time = fit_time
            best_pred_time = pred_time

    search_time = time.time() - search_start_time
    print(f'\nTotal hyperparameter search time: {search_time:.2f} seconds')

    print(f'\nBest parameters found: {best_params}')
    print(f'Best Accuracy: {best_accuracy:.4f}')
    print(f'Time to fit the best model: {best_fit_time:.2f} seconds')
    print(f'Time to predict with the best model: {best_pred_time:.2f} seconds')

    # Entrenar el modelo con los mejores parámetros
    print("Training the model with the best parameters...")
    best_model = cumlSVC(kernel='poly', **best_params, tol=1e-4, max_iter=500)
    X_train_scaled, X_test_scaled, y_train, y_test = joblib.load('prepared_data.pkl')
    
    train_start_time = time.time()
    best_model.fit(X_train_scaled, y_train)
    train_time = time.time() - train_start_time
    print(f'Time to train the model with the best parameters: {train_time:.2f} seconds')

    # Guardar el modelo
    joblib.dump(best_model, 'clasificador_svm_poly_total.pkl')

    print("Model saved successfully.")

    # Evaluar el modelo
    print("Evaluating the model...")
    y_pred = best_model.predict(X_test_scaled)
    accuracy = cuml_accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy:.4f}")

    y_test_np = y_test.to_numpy()
    print("Classification Report:\n", classification_report(y_test_np, y_pred.to_numpy()))

if __name__ == "__main__":
    input_file = 'datos_gestos_seleccionados.parquet'
    # Puedes cambiar el sample_size aquí si deseas un subconjunto de datos
    load_and_prepare_data(input_file, sample_size=372118)  # Cambia el número según tus necesidades
    main()
