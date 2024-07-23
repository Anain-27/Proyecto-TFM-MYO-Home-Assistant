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
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Con_IMU_Dinamicos\\'

# Definir max_per_label, para tener el mismo número por label
max_per_label = 1000

# Cargar datos desde el archivo Excel
df = pd.read_excel(os.path.join(path, 'Datos_Limpios.xlsx'), sheet_name="Sheet1")

# Eliminar filas con valores nulos
df.dropna(inplace=True)

#Tomamos max_per_label por etiqueta
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

# Separamos los datos en datos de entrenamiento y de test, eligiendo un tamaño de la muestra de test y una semilla para reproducibilidad del aleatorio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler por si hemos usado uno modificado
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
joblib.dump(model, 'entrenado_prueba.pkl')
print("Modelo guardado como 'entrenado_prueba.pkl'")

# Predecir etiquetas para los datos de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM: {accuracy:.2f}")

# Obtener el informe de clasificación
report = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", report)