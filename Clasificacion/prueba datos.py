import pandas as pd
import os
from collections import Counter

# Definir el path de la carpeta que contiene los archivos
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados\\'

# Nombres de los archivos que queremos analizar
archivos = ['Datos_Limpios_1.xlsx', 'Datos_Limpios_2.xlsx']

# Inicializar un contador para las etiquetas
contador_etiquetas_total = Counter()

for archivo in archivos:
    # Construir la ruta completa del archivo
    file_path = os.path.join(path, archivo)
    print(f'Leemos el archivo: {file_path}')

    # Leer el archivo Excel
    df = pd.read_excel(file_path)
    print(f'Archivo leído: {file_path}')

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Obtener las etiquetas
    y = df.iloc[:, -1].values  # Todas las filas, solo la última columna

    # Contar las etiquetas en este archivo y acumular en el contador total
    contador_etiquetas_total.update(Counter(y))

# Imprimir la cantidad total de datos por etiqueta en ambos archivos
print("Cantidad total de datos por etiqueta en ambos archivos:")
for etiqueta, cantidad in contador_etiquetas_total.items():
    print(f"{etiqueta}: {cantidad}")
