import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import joblib
import time
import os
import matplotlib.pyplot as plt
from collections import Counter

# Definir el path de los datos de entrada y salida
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Prueba funcionamiento clasificador\\Datos_reducidos'
path_out = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Prueba funcionamiento clasificador\\Clasificadores de pruebas\\'

# Definir el nombre del archivo Excel con los datos preprocesados
data_file = os.path.join(path, 'Datos_50000_17_gestos.xlsx')

# Leer los datos preprocesados desde el archivo Excel
df = pd.read_excel(data_file)
print(f'Datos cargados desde {data_file}.')

# Eliminar filas con valores nulos (por si acaso)
df.dropna(inplace=True)

# Separar características (X) de etiquetas (y)
X = df.iloc[:, :8].values
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
scaler_path = os.path.join(path_out, 'Escaladores\\scaler_prueba.pkl')
#joblib.dump(scaler, scaler_path)
print(f"Scaler guardado como '{scaler_path}'")

# Crear clasificador SVM con núcleo polinomial
model = SVC(C=0.1, coef0=1.0, degree=5, gamma= 0.1, kernel='poly')
# Medir el tiempo de entrenamiento
start_time = time.time()

# Entrenar el clasificador
print('Comienza el entrenamiento del modelo SVM...')
model.fit(X_train_scaled, y_train)

# Calcular el tiempo tardado
training_time = time.time() - start_time
print(f'Tiempo de entrenamiento: {training_time:.2f} segundos')

# Guardar el modelo entrenado
model_path = os.path.join(path_out, 'Clasificadores\\clasificador_50000_IMU_17_C_01.pkl')
#joblib.dump(model, model_path)
print(f"Modelo guardado como '{model_path}'")

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test_scaled)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"\nPrecisión del modelo: {accuracy:.2f}")

# Generar y mostrar el informe de clasificación
report = classification_report(y_test, y_pred, target_names=['BAJAR_DIAL', 'C', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FIST', 'GIRO_IN', 'GIRO_OUT', 'I','JUNTOS', 'L', 'REST', 'SUBIR_DIAL', 'TRES', 'UNO', 'WAVE_IN', 'WAVE_OUT'])

print("\nInforme de clasificación:")
print(report)

# Dibujar la matriz de confusión
ConfusionMatrixDisplay.from_estimator(model, X_test_scaled, y_test, display_labels= ['BAJAR_DIAL', 'C', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FIST', 'GIRO_IN', 'GIRO_OUT', 'I','JUNTOS', 'L', 'REST', 'SUBIR_DIAL', 'TRES', 'UNO', 'WAVE_IN', 'WAVE_OUT'])

plt.title("Matriz de Confusión")
plt.show()
