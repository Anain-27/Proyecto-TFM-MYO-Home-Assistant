import os
import pandas as pd

def convert_excel(input_filepath):
    # Leer los datos del archivo de entrada
    with pd.ExcelFile(input_filepath) as xls:
        emg_df = pd.read_excel(xls, sheet_name='EMG')
        imu_df = pd.read_excel(xls, sheet_name='IMU')
        pose_df = pd.read_excel(xls, sheet_name='Pose')

    # Asegurarse de que los datos de IMU estén ordenados por tiempo
    imu_df.sort_values(by='Tiempo', inplace=True)

    # Crear una copia de EMG para los datos combinados
    combined_df = emg_df.copy()

    # Obtener el nombre de la columna de tiempo de EMG
    tiempo_col_emg = next((col for col in emg_df.columns if 'tiempo' in col.lower()), None)
    if tiempo_col_emg is None:
        raise ValueError("No se encontró la columna de tiempo en la hoja 'EMG'")

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

    # Guardar los datos combinados y originales en el mismo archivo Excel
    with pd.ExcelWriter(input_filepath, engine='openpyxl') as writer:
        emg_df.to_excel(writer, sheet_name='EMG', index=False)
        imu_df.to_excel(writer, sheet_name='IMU', index=False)
        pose_df.to_excel(writer, sheet_name='Pose', index=False)
        combined_df.to_excel(writer, sheet_name='Datos', index=False)

    print("Conversion complete. Excel saved at:", input_filepath)

def process_directory(root_directory):
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(root, file)
                try:
                    convert_excel(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Uso del script
root_directory = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Extración_de_datos\\datos_con_IMU'
process_directory(root_directory)
