import pandas as pd
import glob
import os
import numpy as np

# Limpiaremos los datos para quitarles el ruido y adaptarlos para el clasificador
#Creamos una función para hacer la integración en cada uno de los canales.
def integrar_df(df, window_size=10):
    integrated_signals = {} #creamos un array en el que guardar las señales

    #rectificamos e integramos con promedio movil cada uno de los canales
    for col in df.columns[0:]:
        if col != 'pose':
            rectified_signal = np.abs(df[col].astype(float))
            integrated_signals[col] = np.convolve(rectified_signal, np.ones(window_size)/window_size, mode='valid')
    df_integrada = pd.DataFrame(integrated_signals)
    df_integrada['pose'] = df['pose']
    return df_integrada

# Aplicar la integración a cada canal


#Comenzamos definiendo el path de entrada y salida
path= 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado\\datos_procesados \\Todos\\'

file  = path + 'Datos_Completos.xlsx'

#Creamos el dataframe donde añadiremos todos los datos
df = pd.read_excel(file)

#Debemos hacer una integración a los datos para suavizarlos
df_limpio = integrar_df(df,10)

# Crear el nombre para el archivo modificado
new_file_name = os.path.join(path, 'Datos_Limpios.xlsx')

# Guardar el DataFrame resultante en un nuevo archivo de Excel
df_limpio.to_excel(new_file_name, index=False)
print(f'Guardado archivo modificado: {new_file_name}')
