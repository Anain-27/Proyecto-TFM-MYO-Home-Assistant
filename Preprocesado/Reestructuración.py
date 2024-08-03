import pandas as pd
import glob
import os

# Función para filtrar los datos según el tiempo
def filtrado_tiempo(df, rangos):
    mask = pd.Series(False, index=df.index)
    for inicio, fin in rangos:
        mask |= (df['Tiempo'] >= inicio) & (df['Tiempo'] <= fin)
    df_filtrado = df[mask].drop(columns=['Tiempo'])
    return df_filtrado

# Rutas
path_entrada = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_df_nuevo\\'
path_salida = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados'

# Rangos de tiempo
rango = [(5, 10), (15, 20), (25, 30)]

# Crear DataFrame vacío
df = pd.DataFrame()

# Leer archivos
for root, dirs, files in os.walk(path_entrada):
    for dir_name in dirs:
        sub_path = os.path.join(root, dir_name)
        excel_files = glob.glob(os.path.join(sub_path, '*.xlsx'))

        for file in excel_files:
            df_excel = pd.read_excel(file, sheet_name='Datos')
            df_excel['pose'] = dir_name
            df_filtrado = filtrado_tiempo(df_excel, rango)
            df = pd.concat([df, df_filtrado], ignore_index=True)
            print(f'Archivo procesado: {file}')

# Guardar el DataFrame resultante en un solo archivo CSV
csv_file_name = os.path.join(path_salida, 'Datos_Completos.csv')

try:
    df.to_csv(csv_file_name, index=False)
    print(f'Guardado archivo CSV modificado: {csv_file_name}')
except Exception as e:
    print(f'Error al guardar el archivo CSV: {e}')
