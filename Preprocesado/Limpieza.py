import pandas as pd
import numpy as np
import os


# Limpiaremos los datos para quitarles el ruido y adaptarlos para el clasificador
# Creamos una función para hacer la integración en cada uno de los canales.
def integrar_df(df, window_size=10):
    integrated_signals = {}  # Creamos un diccionario en el que guardar las señales

    # Rectificamos e integramos con promedio móvil cada uno de los canales
    for col in df.columns:
        if col != 'pose':
            rectified_signal = np.abs(df[col].astype(float))
            integrated_signals[col] = np.convolve(rectified_signal, np.ones(window_size) / window_size, mode='valid')

    df_integrada = pd.DataFrame(integrated_signals)
    df_integrada['pose'] = df['pose'].iloc[:len(df_integrada)]  # Asegúrate de que las longitudes coincidan
    return df_integrada


# Definimos el path de entrada y salida
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados\\'

file = os.path.join(path, 'Datos_Completos.csv')

# Creamos el dataframe leyendo el archivo CSV
df = pd.read_csv(file)

# Debemos hacer una integración a los datos para suavizarlos
df_limpio = integrar_df(df, 10)

# Crear el nombre para el archivo modificado
new_file_name = os.path.join(path, 'Datos_Limpios.csv')

# Guardar el DataFrame resultante en un nuevo archivo CSV
df_limpio.to_csv(new_file_name, index=False)
print(f'Guardado archivo modificado: {new_file_name}')
