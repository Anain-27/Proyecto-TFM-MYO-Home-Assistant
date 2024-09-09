import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os
from collections import Counter
import time

# Definir el path de los datos de entrada y salida
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Prueba funcionamiento clasificador\\Datos_reducidos'
path_out = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Prueba funcionamiento clasificador\\Clasificadores de pruebas\\'

# Leer los datos preprocesados desde el archivo Excel
data_file = os.path.join(path, 'Datos_5000_17_gestos.xlsx')
df = pd.read_excel(data_file)
print(f'Datos cargados desde {data_file}.')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Separar características (X) de etiquetas (y)
X = df.iloc[:, :8].values
y = df.iloc[:, -1].values

# Imprimir la cantidad de datos por etiqueta
contador_etiquetas = Counter(y)
print("Cantidad de datos por etiqueta en el conjunto completo:")
for etiqueta, cantidad in contador_etiquetas.items():
    print(f"{etiqueta}: {cantidad}")

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler
scaler_path = os.path.join(path_out, 'Escaladores\\scaler_5000_param.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler guardado como '{scaler_path}'")

# Definir los posibles valores de hiperparámetros
param_grid = {
    'C': [0.1, 1],
    'degree': [2, 3, 4],
    'gamma': ['scale', 'auto'],
    'kernel': ['poly']
}

# Crear el clasificador
svc = SVC()

# Búsqueda de hiperparámetros con GridSearchCV
grid_search = GridSearchCV(svc, param_grid, cv=5, verbose=2)
start_time = time.time()
grid_search.fit(X_train_scaled, y_train)
training_time = time.time() - start_time
print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')

# Mejor modelo
best_model = grid_search.best_estimator_
print(f"Mejores hiperparámetros: {grid_search.best_params_}")

# Evaluar el modelo en el conjunto de prueba
y_pred = best_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Guardar el mejor modelo
model_path = os.path.join(path_out, 'Clasificadores\\clasificador_5000_17_gestos_param.pkl')
joblib.dump(best_model, model_path)
print(f"Modelo guardado como '{model_path}'")
