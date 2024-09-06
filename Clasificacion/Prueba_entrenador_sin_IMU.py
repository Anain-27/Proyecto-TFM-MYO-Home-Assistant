import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time
import os
import glob
from collections import Counter

# Definir el path de los datos de entrada
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados\\'

# Buscar todos los archivos que comiencen con 'Datos_Limpios_'
files = glob.glob(os.path.join(path, 'Datos_Limpios_*.xlsx'))

# Comprobar si se encontraron archivos
if not files:
    raise FileNotFoundError(f"No se encontraron archivos que coincidan con el patrón 'Datos_Limpios_*' en {path}")

# Leer y concatenar todos los archivos encontrados en un único DataFrame
df_list = []
for file in files:
    print(f"Leyendo archivo: {file}")
    df = pd.read_excel(file)
    df_list.append(df)

# Concatenar todos los DataFrames en uno solo
df = pd.concat(df_list, ignore_index=True)

# Eliminar filas con valores nulos
df.dropna(inplace=True)

# Definir el número máximo de muestras por etiqueta
max_samples_per_label = 12500  # Ajusta este valor según sea necesario

# Filtrar para asegurarse de que ninguna etiqueta tenga más de max_samples_per_label muestras
# Corregir el warning añadiendo include_group=False
balanced_df = df.groupby(df.columns[-1], group_keys=False, as_index=False).apply(
    lambda x: x.sample(min(len(x), max_samples_per_label)))


# Tamaño de la muestra después de balancear
tamano_muestra_balanceada = len(balanced_df)
print(f"Tamaño de la muestra balanceada: {tamano_muestra_balanceada}")

# **Seleccionar las primeras 8 columnas para X y la última columna para y**
X = balanced_df.iloc[:, :8].values  # Las primeras 8 columnas (índices 0 a 7)
y = balanced_df.iloc[:, -1].values  # La última columna

# Imprimir la cantidad de datos por etiqueta en el conjunto balanceado
contador_etiquetas = Counter(y)
print("Cantidad de datos por etiqueta en el conjunto balanceado:")
for etiqueta, cantidad in contador_etiquetas.items():
    print(f"{etiqueta}: {cantidad}")

# Separamos los datos en datos de entrenamiento y de test, eligiendo un tamaño de la muestra de test y una semilla para reproducibilidad del aleatorio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimir la cantidad de datos por etiqueta en el conjunto de entrenamiento y de prueba
print("\nCantidad de datos por etiqueta en el conjunto de entrenamiento:")
print(Counter(y_train))

print("\nCantidad de datos por etiqueta en el conjunto de prueba:")
print(Counter(y_test))

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar el scaler por si hemos usado uno modificado
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

# Guardar el modelo entrenado en un archivo
model_path = os.path.join(path, 'prueba_mejores_parametros.pkl')
joblib.dump(model, model_path)
print(f"Modelo guardado como '{model_path}'")

# Predecir etiquetas para los datos de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM: {accuracy:.2f}")

# Imprimir la cantidad de predicciones por etiqueta
print("\nCantidad de predicciones por etiqueta en el conjunto de prueba:")
print(Counter(y_pred))

# Obtener el informe de clasificación
report = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", report)
