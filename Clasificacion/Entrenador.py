import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os
import sys

# Comenzamos definiendo el path de los datos de entrada
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado\\datos\\Dinamicos\\'

# Definir max_per_label
max_per_label = 1000

# Nombre del archivo para guardar el registro
registro_filename = os.path.join(path, f'accuracy_{max_per_label}.txt')

# Abrir el archivo en modo append (para añadir al final)
with open(registro_filename, 'a') as f:
    # Redirigir la salida estándar a este archivo
    original_stdout = sys.stdout
    sys.stdout = f

    # Cargar datos desde el archivo Excel
    df = pd.read_eximport pandas as pd


# Definir las etiquetas de interés
etiquetas_interes = ["ARRIBA", "FINGERS_SPREAD", "FIST", "GIRO_IN", "GIRO_OUT", "REST", "THUMB_TO_PINKY", "WAVE_IN",
                     "WAVE_OUT"]

# Comenzamos definiendo el path de los datos de entrada
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado\\datos\\Con_IMU_Dinamicos\\'

# Definir max_per_label
max_per_label = 1000

# Nombre del archivo para guardar el registro
registro_filename = os.path.join(path, f'accuracy_{max_per_label}.txt')

# Abrir el archivo en modo append (para añadir al final)
with open(registro_filename, 'a') as f:
    # Redirigir la salida estándar a este archivo
    original_stdout = sys.stdout
    sys.stdout = f

    # Cargar datos desde el archivo Excel
    df = pd.read_excel(os.path.join(path, 'Datos_Limpios.xlsx'), sheet_name=None)

    # Obtener el nombre de la única hoja del archivo
    sheet_name = list(df.keys())[0]

    # Obtener el DataFrame de la única hoja del archivo
    df = df[sheet_name]

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Filtrar solo las etiquetas de interés
    df = df[df['pose'].isin(etiquetas_interes)]

    # Limitar a max datos por etiqueta
    counts = df['pose'].value_counts()
    keep_indices = []
    for label in counts.index:
        indices = df.index[df['pose'] == label].tolist()
        if len(indices) > max_per_label:
            indices = np.random.choice(indices, max_per_label, replace=False)
        keep_indices.extend(indices)
    df = df.loc[keep_indices]

    # Tamaño de la muestra inicial
    tamano_muestra_inicial = len(df)
    print(f"Tamaño de la muestra inicial: {tamano_muestra_inicial}")

    # Separar características (X) de etiquetas (y)
    X = df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última
    y = df.iloc[:, -1].values  # Todas las filas, solo la última columna

    # Separamos los datos en datos de entrenamiento y de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Guardar el scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler guardado como 'scaler.pkl'")

    # Crear clasificador SVM con núcleo polinomial
    model = SVC(C=100, coef0=1.0, degree=5, gamma='scale', kernel='poly')

    # Medir el tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el clasificador
    print('Comienza el entrenamiento del modelo SVM...')
    model.fit(X_train_scaled, y_train)

    # Calcular el tiempo tardado
    training_time = time.time() - start_time
    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')

    # Guardar el modelo entrenado en un archivo
    joblib.dump(model, 'entrenado_svm_poly_model_dinamico_IMU.pkl')
    print("Modelo guardado como 'entrenado_svm_poly_model_dinamico_IMU.pkl'")

    # Predecir etiquetas para los datos de prueba
    y_pred = model.predict(X_test_scaled)

    # Evaluar la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo SVM: {accuracy:.2f}")

    # Obtener el informe de clasificación solo para las etiquetas de interés
    report = classification_report(y_test, y_pred, target_names=etiquetas_interes)
    print("Informe de clasificación:\n", report)

    # Restaurar la salida estándar
    sys.stdout = original_stdout

# Mensaje final
print(f'Registro guardado en: {registro_filename}')
cel(os.path.join(path, 'Datos_Limpios.xlsx'), sheet_name=None)

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Limitar a 5000 datos por etiqueta
    counts = df.iloc[:, -1].value_counts()
    keep_indices = []
    for label in counts.index:
        indices = df.index[df.iloc[:, -1] == label].tolist()
        if len(indices) > max_per_label:
            indices = np.random.choice(indices, max_per_label, replace=False)
        keep_indices.extend(indices)
    df = df.loc[keep_indices]

    # Tamaño de la muestra inicial
    tamano_muestra_inicial = len(df)
    print(f"Tamaño de la muestra inicial: {tamano_muestra_inicial}")

    # Separar características (X) de etiquetas (y)
    X = df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última
    y = df.iloc[:, -1].values  # Todas las filas, solo la última columna

    # Separamos los datos en datos de entrenamiento y de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Estandarizar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Guardar el scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler guardado como 'scaler.pkl'")

    # Crear clasificador SVM con núcleo polinomial
    model = SVC(C=100, coef0=1.0, degree=5, gamma='scale', kernel='poly')

    # Medir el tiempo de entrenamiento
    start_time = time.time()

    # Entrenar el clasificador
    print('Comienza el entrenamiento del modelo SVM...')
    model.fit(X_train_scaled, y_train)

    # Calcular el tiempo tardado
    training_time = time.time() - start_time
    print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')

    # Guardar el modelo entrenado en un archivo
    joblib.dump(model, 'entrenado_svm_poly_model_dinamico.pkl')
    print("Modelo guardado como 'entrenado_svm_poly_model_dinamico.pkl'")

    # Predecir etiquetas para los datos de prueba
    y_pred = model.predict(X_test_scaled)

    # Evaluar la precisión del clasificador
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión del modelo SVM: {accuracy:.2f}")

    # Obtener el informe de clasificación
    report = classification_report(y_test, y_pred)
    print("Informe de clasificación:\n", report)

    # Restaurar la salida estándar
    sys.stdout = original_stdout

# Mensaje final
print(f'Registro guardado en: {registro_filename}')
