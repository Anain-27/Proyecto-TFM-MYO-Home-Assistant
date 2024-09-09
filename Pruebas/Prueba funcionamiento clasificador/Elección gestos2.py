import pandas as pd
import os
import glob
from collections import Counter

# Definir el path de los datos de entrada y salida
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado\\datos_procesados\\Gestos_seleccionados'
path_out = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Prueba funcionamiento clasificador\\Datos_reducidos'

# Capturar todos los archivos que comienzan con 'Datos_Limpios_'
files_pattern = os.path.join(path, 'Datos_Limpios_*.xlsx')
files = glob.glob(files_pattern)

if not files:
    print(f"No se encontraron archivos con el patrón: {files_pattern}")
else:
    print(f"Archivos encontrados: {files}")

    # Leer y concatenar todos los archivos en un único DataFrame
    df_list = []
    for file in files:
        df = pd.read_excel(file)
        df_list.append(df)
    df = pd.concat(df_list, ignore_index=True)
    print(f'Todos los archivos han sido leídos y combinados en un único DataFrame.')

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Definir el tamaño límite por etiqueta
    label_size = 5000  # Cambia este valor según lo que necesites

    # Especificar las etiquetas que deseas incluir
    etiquetas_deseadas = ['BAJAR_DIAL', 'TRES', 'FIST', 'GIRO_OUT', 'REST', 'UNO', 'WAVE_IN', 'WAVE_OUT']
    #etiquetas_deseadas = [ 'C', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FIST',  'I','JUNTOS', 'L', 'REST', 'TRES', 'UNO', 'WAVE_IN', 'WAVE_OUT']  # Cambia esto a las etiquetas que necesites


    # Filtrar el DataFrame para incluir solo las etiquetas deseadas
    df_filtrado = df[df[df.columns[-1]].isin(etiquetas_deseadas)]

    # Limitar el número de datos por cada etiqueta
    df_limited = df_filtrado.groupby(df_filtrado.columns[-1], group_keys=False).apply(lambda x: x.sample(min(len(x), label_size)))

    # Verificar que se han reducido correctamente las muestras por etiqueta
    contador_etiquetas_limited = Counter(df_limited.iloc[:, -1].values)
    print("Cantidad de datos por etiqueta después de limitar:")
    for etiqueta, cantidad in contador_etiquetas_limited.items():
        print(f"{etiqueta}: {cantidad}")

    # Guardar el nuevo DataFrame en un archivo Excel
    output_file = os.path.join(path_out, f'Datos_{label_size}_{len(etiquetas_deseadas)}_gestosSIN_CRUZAR_DEDOS.xlsx')
    df_limited.to_excel(output_file, index=False)
    print(f"Datos limitados guardados en '{output_file}'")
