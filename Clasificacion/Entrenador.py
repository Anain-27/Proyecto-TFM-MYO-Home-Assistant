import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os
import glob
import matplotlib.pyplot as plt
from collections import Counter

# Definir el path de los datos de entrada
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados\\'

# Capturar todos los archivos que comienzan con 'Datos_Limpios_'
files_pattern = os.path.join(path, 'Datos_Limpios_*.xlsx')
files = glob.glob(files_pattern)
print(f"Archivos encontrados: {files}")

# Leer y concatenar todos los archivos en un único DataFrame
df_list = []
for file in files:
    df = pd.read_excel(file)
    df_list.append(df)
df = pd.concat(df_list, ignore_index=True)
print(f'Todos los archivos han sido leídos y combinados en un único DataFrame.')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Tamaño de la muestra inicial
tamano_muestra_inicial = len(df)
print(f"Tamaño de la muestra inicial: {tamano_muestra_inicial}")

# Separar características (X) de etiquetas (y)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Imprimir la cantidad de datos por etiqueta en el conjunto completo
contador_etiquetas = Counter(y)
print("Cantidad de datos por etiqueta en el conjunto completo:")
for etiqueta, cantidad in contador_etiquetas.items():
    print(f"{etiqueta}: {cantidad}")

# Separar los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimir la cantidad de datos por etiqueta en los conjuntos de entrenamiento y prueba
print("\nCantidad de datos por etiqueta en el conjunto de entrenamiento:")
print(Counter(y_train))

print("\nCantidad de datos por etiqueta en el conjunto de prueba:")
print(Counter(y_test))

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler
scaler_path = os.path.join(path, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"Scaler guardado como '{scaler_path}'")

# Crear clasificador SVM con núcleo polinomial
model = SVC(C=100, coef0=1.0, degree=5, gamma=0.1, kernel='poly')

# Medir el tiempo de entrenamiento
start_time = time.time()

# Entrenar el clasificador
print('Comienza el entrenamiento del modelo SVM...')
model.fit(X_train_scaled, y_train)

# Calcular el tiempo tardado
training_time = time.time() - start_time
print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')

# Guardar el modelo entrenado
model_path = os.path.join(path, 'prueba_mejores_parametros.pkl')
joblib.dump(model, model_path)
print(f"Modelo guardado como '{model_path}'")

# Evaluación del modelo para detectar sobreajuste
print("\nEvaluación del modelo para detectar sobreajuste:")
cv_results = cross_validate(model, X_train_scaled, y_train, cv=5, return_train_score=True)

train_score_mean = cv_results['train_score'].mean()
test_score_mean = cv_results['test_score'].mean()

print(f"Precisión en el conjunto de entrenamiento (media): {train_score_mean:.2f}")
print(f"Precisión en el conjunto de validación (media): {test_score_mean:.2f}")

if train_score_mean > test_score_mean:
    print("El modelo podría estar sobreajustado (overfitting).")

# Generar curvas de aprendizaje
train_sizes, train_scores, val_scores = learning_curve(model, X_train_scaled, y_train, cv=5)

train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Precisión en entrenamiento")
plt.plot(train_sizes, val_mean, label="Precisión en validación")
plt.xlabel("Tamaño del conjunto de entrenamiento")
plt.ylabel("Precisión")
plt.title("Curvas de aprendizaje")
plt.legend()
plt.show()

# Predecir etiquetas para los datos de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo SVM: {accuracy:.2f}")

# Imprimir la cantidad de predicciones por etiqueta
print("\nCantidad de predicciones por etiqueta en el conjunto de prueba:")
print(Counter(y_pred))

# Obtener el informe de clasificación
report = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", report)
