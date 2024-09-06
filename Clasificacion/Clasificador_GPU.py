import h2o4gpu
import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Definir el path de los datos de entrada
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados\\'
# Obtener todos los archivos Datos_Limpios_*.xlsx
file_pattern = os.path.join(path, 'Datos_Limpios_*.xlsx')
files = glob.glob(file_pattern)

# Crear el dataframe donde añadiremos todos los datos
df = pd.DataFrame()

# Leer y combinar todos los archivos en un solo DataFrame
print("Comienzo a leer los archivos")
for file in files:
    df_temp = pd.read_excel(file)
    df = pd.concat([df, df_temp], ignore_index=True)
    print(f'Archivo leído: {file}')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

print('Datos captados')
# Mezclar los datos para conseguir una muestra más homogénea al evaluar solo algunos
df = shuffle(df)

# Separar características (X) de etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última
y = df.iloc[:, -1].values   # Todas las filas, solo la última columna

# Separar los datos en datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir a tipos de datos numpy
X_train_np = np.array(X_train_scaled)
y_train_np = np.array(y_train)
X_test_np = np.array(X_test_scaled)
y_test_np = np.array(y_test)

# Crear y entrenar el modelo SVM con H2O4GPU
model = h2o4gpu.SVC(kernel='poly')
model.fit(X_train_np, y_train_np)

# Predecir etiquetas para los datos de prueba
y_pred = model.predict(X_test_np)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test_np, y_pred)
print("Accuracy:", accuracy)

# Obtener el informe de clasificación
report = classification_report(y_test_np, y_pred)
print("Classification Report:\n", report)
