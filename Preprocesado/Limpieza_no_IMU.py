import pandas as pd
import numpy as np
import os
import glob


# Limpiar los datos para quitarles el ruido y adaptarlos para el clasificador
def integrar_df(df, window_size=10):
    integrated_signals = {}  # Crear un array en el que guardar las señales

    channel_columns = [f'Channel_{i}' for i in range(1, 9)]

    # Procesar solo las columnas de Channel_1 a Channel_8
    for col in channel_columns:
        if col in df.columns:
            # Rectificar la señal
            rectified_signal = np.abs(df[col].astype(float))
            # Integrar la señal
            #integrated_signals[col] = np.convolve(rectified_signal, np.ones(window_size) / window_size, mode='valid')

    # Crear el DataFrame para las señales integradas
    df_integrada = pd.DataFrame(rectified_signal)

    # Conservar otras columnas que no sean Channel_* (incluyendo 'pose')
    other_columns = df.drop(columns=channel_columns)

    # Concatenar df_integrada con otras columnas no modificadas
    df_final = pd.concat([df_integrada, other_columns.reset_index(drop=True)], axis=1)

    return df_final


# Comenzar definiendo el path de entrada y salida
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados\\'

# Obtener todos los archivos Datos_Completos_*.xlsx
file_pattern = os.path.join(path, 'Datos_Completos_*.xlsx')
files = glob.glob(file_pattern)

# Crear el dataframe donde añadiremos todos los datos
df = pd.DataFrame()

# Leer y combinar todos los archivos en un solo DataFrame
for file in files:
    df_temp = pd.read_excel(file)
    df = pd.concat([df, df_temp], ignore_index=True)
    print(f'Archivo leído: {file}')

# Hacer una integración a los datos para suavizarlos
df_limpio = integrar_df(df, 10)


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


# Crear el nombre base para los archivos modificados
base_file_name = os.path.join(path, 'Datos_Limpios_no_IMU')

# Guardar el DataFrame resultante en múltiples archivos de Excel
guardar_en_multiples_archivos(df_limpio, base_file_name)
