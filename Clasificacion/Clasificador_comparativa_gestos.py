import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from itertools import combinations
import os
import sys

# Definir path de los datos de entrada y salida
path_entrada = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado\\datos_procesados\\Todos\\'
path_salida = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Clasificacion\\Comparativa\\'

# Definir max_per_label
max_per_label = 1000

# Definir etiquetas de interés
etiquetas_interes = ['ARRIBA', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FINGERS_SPREAD',
                     'FIST', 'GIRO_IN', 'GIRO_OUT', 'REST', 'TRES', 'UNO',
                     'WAVE_IN', 'WAVE_OUT']

# Nombre del archivo para guardar el registro
registro_filename = os.path.join(path_salida, f'accuracy_{max_per_label}.txt')

# Abrir el archivo en modo append (para añadir al final)
with open(registro_filename, 'a') as f:
    # Redirigir la salida estándar a este archivo
    original_stdout = sys.stdout
    sys.stdout = f

    print(path_entrada + 'Datos_Limpios.xlsx')

    # Cargar datos desde el archivo Excel
    df = pd.read_excel(path_entrada + 'Datos_Limpios.xlsx')

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    print('Datos captados')

    # Mezclar los datos para conseguir una muestra más homogénea
    df = df.sample(frac=1, random_state=42)

    # Dictionary para almacenar los datos seleccionados por etiqueta
    data_por_etiqueta = {}

    # Iterar sobre cada etiqueta de interés y seleccionar los primeros max_per_label datos
    for etiqueta in etiquetas_interes:
        df_etiqueta = df[df['pose'] == etiqueta].head(max_per_label)
        data_por_etiqueta[etiqueta] = df_etiqueta

    # Iterar sobre todas las combinaciones de etiquetas de interés
    for r in range(1, len(etiquetas_interes) + 1):
        for etiquetas_combinacion in combinations(etiquetas_interes, r):
            etiquetas_combinacion = list(etiquetas_combinacion)
            print(f'Combinación de etiquetas: {etiquetas_combinacion}')

            # Unir los datos seleccionados por cada etiqueta de la combinación actual
            dfs_combinados = [data_por_etiqueta[etiqueta] for etiqueta in etiquetas_combinacion]
            df_filtered = pd.concat(dfs_combinados)

            # Separar características (X) de etiquetas (y)
            X = df_filtered.iloc[:, :-1].values
            y = df_filtered.iloc[:, -1].values

            # Separar datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Diagnosticar número de clases
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                print(f"Warning: Only {len(unique_classes)} class(es) present in the training data for combination {etiquetas_combinacion}.")
                continue  # Saltar esta combinación si hay menos de 2 clases

            # Estandarizar los datos
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Crear clasificador SVM
            model = svm.SVC(kernel='poly')

            # Definir el grid de parámetros a buscar
            param_grid = {
                'C': [100],
                'degree': [5],
                'gamma': ['scale'],
                'coef0': [1.0]
            }

            # Entrenar el clasificador
            print('Comienza el training')
            # Crear y entrenar el modelo con GridSearchCV para encontrar los mejores hiperparámetros
            grid = GridSearchCV(model, param_grid, cv=3, refit=True, verbose=2, n_jobs=7)
            grid.fit(X_train_scaled, y_train)

            # Predecir etiquetas para los datos de prueba
            y_pred = grid.predict(X_test_scaled)

            # Evaluar la precisión del clasificador
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy para combinación {etiquetas_combinacion}: {accuracy}")

            # Obtener el informe de clasificación
            report = classification_report(y_test, y_pred)
            print(f"Classification poly Report para combinación {etiquetas_combinacion}:\n", report)

            # Imprimir combinación de gestos usados en el archivo de registro
            print(f"Combinación de gestos: {etiquetas_combinacion}", file=f)

    # Restaurar la salida estándar
    sys.stdout = original_stdout
