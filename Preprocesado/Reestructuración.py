import pandas as pd
import glob
import os


# Función para retirar los tiempos que no nos interesan de los datos
def filtrado_tiempo(df, rangos):
    # Crear una máscara booleana para filtrar los rangos
    mask = pd.Series(False, index=df.index)
    for inicio, fin in rangos:
        mask |= (df['Tiempo'] >= inicio) & (df['Tiempo'] <= fin)

    # Filtrar el DataFrame basado en la máscara
    df_filtrado = df[mask].drop(columns=['Tiempo'])
    return df_filtrado


# Función para guardar DataFrame en múltiples archivos
def guardar_en_multiples_archivos(df, base_filename, max_filas=1000000):
    num_files = (len(df) // max_filas) + 1
    for i in range(num_files):
        start_row = i * max_filas
        end_row = (i + 1) * max_filas
        sub_df = df.iloc[start_row:end_row]
        filename = f"{base_filename}_{i + 1}.xlsx"
        sub_df.to_excel(filename, index=False)
        print(f'Guardado archivo modificado: {filename}')


# Definir los paths de entrada y salida
path_entrada = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_df_nuevo\\'
path_salida = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados'

# Definir los rangos de los que queremos captar los datos
rango = [(5, 10), (15, 20), (25, 30)]

# Crear el DataFrame donde añadiremos todos los datos
df = pd.DataFrame()

# Recorrer cada una de las carpetas en las que están los datos
for root, dirs, files in os.walk(path_entrada):
    # Buscar los archivos en los que hemos guardado los datos
    for dir_name in dirs:
        sub_path = os.path.join(root, dir_name)
        excel_files = glob.glob(os.path.join(sub_path, '*.xlsx'))

        # Para cada uno de los archivos, leerlos y transformarlos
        for file in excel_files:
            # Leer el archivo de Excel desde la hoja 'Datos'
            df_excel = pd.read_excel(file, sheet_name='Datos')

            # Añadir una columna para definir el gesto, que tenemos almacenado como nombre del directorio
            df_excel['pose'] = dir_name

            # Filtrar los datos para obtener solo los del gesto
            df_filtrado = filtrado_tiempo(df_excel, rango)

            df = pd.concat([df, df_filtrado], ignore_index=True)

            # Mostrar el nombre del archivo y el DataFrame resultante
            print(f'Archivo procesado: {file}')

# Crear el nombre base para los archivos modificados
base_file_name = os.path.join(path_salida, 'Datos_Completos')

# Guardar el DataFrame resultante en múltiples archivos de Excel
guardar_en_multiples_archivos(df, base_file_name)
