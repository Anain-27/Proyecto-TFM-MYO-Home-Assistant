import os
import pandas as pd

def merge_emg_channels(emg_df):
    # Obtener el nombre de la columna de tiempo de EMG
    tiempo_col_emg = 'Tiempo'  # Asumiendo que 'Tiempo' es el nombre de la columna de tiempo
    if tiempo_col_emg not in emg_df.columns:
        raise ValueError("No se encontró la columna de tiempo en la hoja 'EMG'")

    # Combinar filas con el mismo tiempo
    merged_data = {}
    for _, row in emg_df.iterrows():
        time = row[tiempo_col_emg]
        if time not in merged_data:
            merged_data[time] = row.copy()
        else:
            # Añadir nuevos canales EMG al dict existente
            for i, col in enumerate(emg_df.columns[1:], start=9):  # Empezar desde el Channel_1 original
                new_channel = f'Channel_{i}'  # Nombre para el nuevo canal EMG (Channel_9 a Channel_16)
                merged_data[time][new_channel] = row[col]

    # Crear DataFrame a partir del diccionario
    merged_emg_df = pd.DataFrame(merged_data.values())

    return merged_emg_df

def convert_excel(input_filepath, output_directory):
    # Leer los datos del archivo de entrada
    with pd.ExcelFile(input_filepath) as xls:
        emg_df = pd.read_excel(xls, sheet_name='EMG')
        imu_df = pd.read_excel(xls, sheet_name='IMU')
        pose_df = pd.read_excel(xls, sheet_name='Pose')

    # Asegurarse de que los datos de IMU estén ordenados por tiempo
    imu_df.sort_values(by='Tiempo', inplace=True)

    # Fusionar canales de EMG si hay tiempos duplicados
    emg_df = merge_emg_channels(emg_df)

    # Crear una copia de EMG para los datos combinados
    combined_df = emg_df.copy()

    # Obtener el nombre de la columna de tiempo de EMG
    tiempo_col_emg = 'Tiempo'  # Asumiendo que 'Tiempo' es el nombre de la columna de tiempo

    # Añadir columnas vacías para IMU a los datos combinados
    for col in ["quat1", "quat2", "quat3", "quat4", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"]:
        combined_df[col] = None

    # Buscar los datos de IMU más cercanos anteriores para cada muestra de EMG
    imu_index = 0
    for i, emg_row in combined_df.iterrows():
        while imu_index < len(imu_df) and imu_df.iloc[imu_index]['Tiempo'] <= emg_row[tiempo_col_emg]:
            imu_index += 1
        if imu_index > 0:
            closest_imu_row = imu_df.iloc[imu_index - 1]
            for col in ["quat1", "quat2", "quat3", "quat4", "acc1", "acc2", "acc3", "gyro1", "gyro2", "gyro3"]:
                combined_df.at[i, col] = closest_imu_row[col]

    # Construir la ruta de salida y nombre de archivo
    relative_path = os.path.relpath(input_filepath, start=root_directory)
    output_filepath = os.path.join(output_directory, relative_path)

    # Crear directorios si no existen
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Guardar los datos combinados en un nuevo archivo Excel
    with pd.ExcelWriter(output_filepath, engine='openpyxl') as writer:
        emg_df.to_excel(writer, sheet_name='EMG', index=False)
        imu_df.to_excel(writer, sheet_name='IMU', index=False)
        pose_df.to_excel(writer, sheet_name='Pose', index=False)
        combined_df.to_excel(writer, sheet_name='Datos', index=False)

    print("Conversión completa. Excel guardado en:", output_filepath)

def process_directory(root_directory, output_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                try:
                    convert_excel(file_path, output_directory)
                except Exception as e:
                    print(f"Error procesando {file_path}: {e}")

# Uso del script
root_directory = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Extración_de_datos\\datos_usados'
output_directory = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Preprocesado_16\\datos_nuevo_df_16'
process_directory(root_directory, output_directory)
