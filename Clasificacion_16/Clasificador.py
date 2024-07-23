import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import shuffle  # Importar shuffle
import joblib
import time  # Importar la librería time

# Comenzamos definiendo el path de los datos de entrada
path = '/home/scuser/Proyecto-TFM-MYO-Home-Assistant/Preprocesado_16/datos_procesados_16/Todos/'
#path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado_16\\datos_procesados_16\\'

print(path+'Datos_Limpios.xlsx')
# Cargar datos desde el archivo Excel
df = pd.read_excel(path+'Datos_Limpios.xlsx')
df.dropna(inplace=True)

print('Datos captados')
print(f'Tamaño del dataset: {df.shape}')

# Mezclamos los datos para conseguir una muestra más homogénea al evaluar solo algunos
df = shuffle(df)


# Separar características (X) de etiquetas (y)
#Tomando todas las filas como características
#X_16= df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última

# Especificamos las columnas que deseamos usar
columnas_8 = ['Channel_1', 'Channel_2','Channel_3','Channel_4','Channel_5','Channel_6','Channel_7','Channel_8','quat1','quat2','quat3','quat4','acc1','acc2','acc3','gyro1','gyro2','gyro3']

# Elegimos solo las primeras 8 columnas
X_8 = df[columnas_8].values  # Selecciona solo las columnas deseadas
y = df.iloc[:, -1].values   # Todas las filas, solo la última columna

# Separamos los datos en datos de entrenamiento y de test
#X_train, X_test, y_train, y_test = train_test_split(X_16, y, test_size=0.3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_8, y, test_size=0.3, random_state=42)


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
model = svm.SVC(kernel='poly')

# Definir el grid de parámetros a buscar
param_grid = {
    'C': [0.1, 1, 10, 100],
    'degree': [2, 3, 4, 5],
    'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    'coef0': [0.0, 0.5, 1.0]
}

# Entrenar el clasificador
print('Comienza el training')
start_time = time.time()  # Iniciar el cronómetro
# Crear y entrenar el modelo con GridSearchCV para encontrar los mejores hiperparámetros
grid = GridSearchCV(model, param_grid, cv=3, refit=True, verbose=2, n_jobs=7)
grid.fit(X_train_sample, y_train_sample)

# Guardamos el modelo
joblib.dump(grid, 'clasificador_svm_poly_8_de_16.pkl')

# Predecir etiquetas para los datos de prueba
y_pred = grid.predict(X_test)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Obtener el informe de clasificación
report = classification_report(y_test, y_pred)
print("Classification poly Report:\n", report)

# Predecir en el conjunto de prueba
y_pred = grid.predict(X_test)

# Evaluar la precisión
print(f"Best Parameters: {grid.best_params_}")

# Imprimir el tiempo total transcurrido
end_time = time.time()  # Parar el cronómetro
total_time = end_time - start_time
print(f"Tiempo total transcurrido: {total_time} segundos")
