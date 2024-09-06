import cudf
import pandas as pd
import os
import glob
import time

def preprocess_data(input_path, output_file):
    # Definir el patrón de archivo
    file_pattern = os.path.join(input_path, 'Datos_Limpios_*.xlsx')
    files = glob.glob(file_pattern)

    print("Comenzando la lectura y combinación de archivos...")

    # Medir tiempo de lectura y combinación de datos
    start_time = time.time()

    df = cudf.DataFrame()
    for file in files:
        df_temp = cudf.DataFrame.from_pandas(pd.read_excel(file))
        df = cudf.concat([df, df_temp], ignore_index=True)

    # Obtener las etiquetas únicas antes de cambiarlas
    if 'pose' in df.columns:
        etiquetas_unicas = df['pose'].unique().to_pandas()
        print(f"Etiquetas únicas encontradas: {etiquetas_unicas.tolist()}")


    df.dropna(inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Guardar el DataFrame en un archivo Parquet
    df.to_parquet(output_file)

    print(f"Datos leídos y combinados correctamente. Datos guardados en '{output_file}'.")

    return df

if __name__ == "__main__":
    # Definir el path de los datos de entrada
    input_path = '/mnt/c/Users/anita/Documents/GitHub/Proyecto-TFM-MYO-Home-Assistant/Preprocesado/datos_procesados/Gestos_seleccionados/'    
    output_file = 'datos_gestos_seleccionado.parquet'
    preprocess_data(input_path, output_file)
