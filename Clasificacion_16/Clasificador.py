import numpy as np
import pandas as pd
import os
import glob
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, hamming_loss, jaccard_score
)
from sklearn.utils import shuffle
import joblib
import time

# Definir el path de los datos de entrada
path = '/home/scuser/Proyecto-TFM-MYO-Home-Assistant/Preprocesado_16/datos_procesados_16/Gestos_usados/'
#path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado_16\\datos_procesados_16\\Gestos_se'

# Obtener todos los archivos Datos_Limpios_*.xlsx
file_pattern = os.path.join(path, 'Datos_Limpios_*.xlsx')
files = glob.glob(file_pattern)

# Crear el dataframe donde añadiremos todos los datos
df = pd.DataFrame()

# Leer y combinar todos los archivos en un solo DataFrame
for file in files:
    df_temp = pd.read_excel(file)
    df = pd.concat([df, df_temp], ignore_index=True)
    print(f'Archivo leído: {file}')

# Eliminar filas con valores nulos
df.dropna(inplace=True)

print('Datos captados')
print(f'Tamaño del dataset: {df.shape}')

# Mezclar los datos para conseguir una muestra más homogénea al evaluar solo algunos
df = shuffle(df)

# Separar características (X) de etiquetas (y)
X_16 = df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última
y = df.iloc[:, -1].values   # Todas las filas, solo la última columna

# Separamos los datos en datos de entrenamiento y de prueba
X_train, X_test, y_train, y_test = train_test_split(X_16, y, test_size=0.3, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convertir X_train y y_train de nuevo a DataFrame para poder muestrear
X_train_df = pd.DataFrame(X_train_scaled)
y_train_df = pd.DataFrame(y_train)

# Usar una muestra aleatoria de 50,000 filas para pruebas iniciales
sample_size = 50000
X_train_sample = X_train_df.sample(n=sample_size, random_state=50)
y_train_sample = y_train_df.loc[X_train_sample.index]

# Convertir de nuevo a numpy arrays para el modelo
X_train_sample = X_train_sample.to_numpy()
y_train_sample = y_train_sample.to_numpy().ravel()

# Crear clasificador SVM
model = svm.SVC(kernel='poly', probability=True)  # Agregar probability=True para calcular ROC AUC

# Definir el grid de parámetros a buscar
param_grid = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'coef0': [0.0, 0.5, 1.0]
}

# Entrenar el clasificador
print('Comienza el entrenamiento')
start_time = time.time()  # Iniciar el cronómetro

# Crear y entrenar el modelo con GridSearchCV para encontrar los mejores hiperparámetros
grid = GridSearchCV(model, param_grid, cv=3, refit=True, verbose=2, n_jobs=7)
grid.fit(X_train_sample, y_train_sample)

# Guardar el modelo
joblib.dump(grid, 'clasificador_svm_poly_8_de_16.pkl')

# Predecir etiquetas para los datos de prueba
y_pred = grid.predict(X_test)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Obtener el informe de clasificación
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Métricas adicionales
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, grid.predict_proba(X_test), multi_class='ovr')
avg_precision = average_precision_score(y_test, grid.predict_proba(X_test), average='weighted')
logloss = log_loss(y_test, grid.predict_proba(X_test))
mcc = matthews_corrcoef(y_test, y_pred)
cohen_kappa = cohen_kappa_score(y_test, y_pred)
hamming = hamming_loss(y_test, y_pred)
jaccard = jaccard_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC Score: {roc_auc}")
print(f"Average Precision Score: {avg_precision}")
print(f"Log Loss: {logloss}")
print(f"Matthews Correlation Coefficient: {mcc}")
print(f"Cohen's Kappa Score: {cohen_kappa}")
print(f"Hamming Loss: {hamming}")
print(f"Jaccard Score: {jaccard}")

# Imprimir los mejores parámetros
print(f"Best Parameters: {grid.best_params_}")

# Imprimir el tiempo total transcurrido
end_time = time.time()  # Parar el cronómetro
total_time = end_time - start_time
print(f"Tiempo total transcurrido: {total_time} segundos")
