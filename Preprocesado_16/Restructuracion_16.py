import pandas as pd
import glob
import os

# Vamos a unir cada uno de los datos que hemos recolectado

'''
Comenzamos creando una función para retirar los tiempos que no nos interesan de los datos
'''
def filtrado_tiempo(df, rangos):

    # Crear una máscara booleana para filtrar los rangos
    mask = pd.Series(False, index=df.index)
    for inicio, fin in rangos:
        mask |= (df['Tiempo'] >= inicio) & (df['Tiempo'] <= fin)

    # Filtrar el DataFrame basado en la máscara
    df_filtrado = df[mask].drop(columns=['Tiempo'])

    return df_filtrado


#Comenzamos definiendo el path de entrada y salida
path_entrada = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado_16\\datos_df_nuevo_16'
path_salida = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado_16\\datos_procesados_16\\Gestos_seleccionados'

#Definimos los rangos de los que queremos captar los datos.
rango= [(5, 10), (15, 20), (25, 30)]

#Creamos el dataframe donde añadiremos todos los datos
df= pd.DataFrame()

# Recorreremos cada una de las carpetas en las que están los datos
for root, dirs, files in os.walk(path_entrada):
    # Buscamos los archivos en los que hemos guardados los datos.
    for dir_name in dirs:
        sub_path = os.path.join(root, dir_name)
        excel_files = glob.glob(os.path.join(sub_path, '*.xlsx'))

        # Para cada uno de los archivos vamos a leerlos transformarlos
        for file in excel_files:
            # Leer el archivo de Excel desde la hoja 'Datos'
            df_excel = pd.read_excel(file, sheet_name='Datos')

            # Añadimos una columna en la que definiremos el gesto, que tenemos almacenado como nombre del directorio
            df_excel['pose'] = dir_name

            # Filtramos los datos para obtener solo los del gesto
            df_filtrado = filtrado_tiempo(df_excel, rango)

            df = pd.concat([df, df_filtrado], ignore_index=True)

            # Mostrar el nombre del archivo y el DataFrame resultante
            print(f'Archivo procesado: {file}')

# Crear el nombre para el archivo modificado
new_file_name = os.path.join(path_salida, 'Datos_Completos.xlsx')

# Guardar el DataFrame resultante en un nuevo archivo de Excel
df.to_excel(new_file_name, index=False)
print(f'Guardado archivo modificado: {new_file_name}')
