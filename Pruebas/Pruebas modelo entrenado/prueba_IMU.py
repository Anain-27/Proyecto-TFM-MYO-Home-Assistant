import cudf
import cupy as cp
from cuml.preprocessing import StandardScaler as cumlStandardScaler
from cuml.model_selection import train_test_split
from cuml.svm import SVC as cumlSVC
from cuml.metrics import accuracy_score as cuml_accuracy_score
from sklearn.metrics import classification_report
import joblib
import time

def load_and_prepare_data(input_file, sample_size=None):
    print("Loading and preparing data...")

    # Mide el tiempo para cargar y preparar los datos
    start_time = time.time()

    # Cargar el DataFrame desde el archivo Parquet
    df = cudf.read_parquet(input_file)

    # Aplicar sample size si se especifica
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    # Seleccionar las características que necesitemos
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

    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test):
    params = {'C': 0.1, 'coef0': 1.0, 'degree': 5, 'gamma': 0.1, 'kernel': 'poly'}
    
    # Entrenamiento del modelo
    print(f"Training the model with parameters: {params}")
    clf = cumlSVC(**params, tol=1e-4, max_iter=500)

    fit_start_time = time.time()
    clf.fit(X_train_scaled, y_train)
    fit_time = time.time() - fit_start_time

    # Predicción
    print(f"Predicting with the model...")
    pred_start_time = time.time()
    y_pred = clf.predict(X_test_scaled)
    pred_time = time.time() - pred_start_time

    # Calcular la precisión
    accuracy = cuml_accuracy_score(y_test, y_pred)

    # Mostrar resultados
    print(f"--- Result Verification ---")
    print(f"Parameters: {params}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model fitting time: {fit_time:.2f} seconds")
    print(f"Prediction time: {pred_time:.2f} seconds")
    print(f"----------------------------")

    # Guardar el modelo
    entrenador ='entrenador_C_01.pkl'
    joblib.dump(clf, entrenador)
    print(f"Model saved as {entrenador}.")

    # Mostrar el informe de clasificación
    y_test_np = y_test.to_numpy()
    print("Classification Report:\n", classification_report(y_test_np, y_pred.to_numpy()))

    # Liberar memoria
    del clf
    cp.get_default_memory_pool().free_all_blocks()

if __name__ == "__main__":
    input_file = 'datos_gestos_seleccionados.parquet'
    sample_size = None  # Si quieres usar todos los datos, deja `sample_size` como None.
    X_train_scaled, X_test_scaled, y_train, y_test = load_and_prepare_data(input_file, sample_size=sample_size)
    train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)
