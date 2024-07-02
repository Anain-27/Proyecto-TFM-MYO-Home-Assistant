import time
import multiprocessing
import numpy as np
from collections import Counter
from pyomyo import Myo, emg_mode
from joblib import load
from sklearn.preprocessing import StandardScaler
import subprocess

# Cargar el clasificador preentrenado
classifier = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Clasificacion\\entrenado_svm_poly_model.pkl')

# Cargar el scaler preentrenado
scaler = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Clasificacion\\scaler.pkl')


# Función para la limpieza de datos EMG
def limpieza(emg_data):
    # Rectificación (valor absoluto)
    emg_rectified = np.abs(emg_data)

    # Escalado
    emg_rectified = emg_rectified.reshape(1, -1)  # Reformar para que tenga la forma correcta
    emg_scaled = scaler.transform(emg_rectified)

    return emg_scaled


# Función para clasificar los datos EMG
def classify_emg(emg_data):
    emg_data = limpieza(emg_data)  # Aplicar la limpieza antes de clasificar
    label = classifier.predict(emg_data)
    return label[0]  # Asumimos que el clasificador devuelve una lista


# Función para encontrar la etiqueta más común y su porcentaje
def mas_comun(labels):
    if labels:
        total_count = len(labels)
        counter = Counter(labels)
        most_common_label, count = counter.most_common(1)[0]
        percentage = count / total_count * 100
        return most_common_label, percentage
    return None, 0


def data_worker(mode, seconds,t_delay):
    collect = True
    start_time = time.time()
    last_print_time = t_delay
    decisiones_clasificador = []  # Lista para almacenar las decisiones del clasificador

    # ------------ Myo Setup ---------------
    m = Myo(mode=mode)
    m.connect()

    def label_emg(emg, movement):
        nonlocal collect, last_print_time, decisiones_clasificador
        current_time = time.time()
        elapsed_time = current_time - start_time


        if current_time - last_print_time >= 1:
            # Calcular la etiqueta más común y su porcentaje hasta el momento
            most_common_label, percentage = mas_comun(decisiones_clasificador)
            if most_common_label is not None:
                print(f"Segundo {int(elapsed_time-t_delay)}: Etiqueta más común: {most_common_label}, Porcentaje: {percentage:.2f}%")

            last_print_time = current_time  # Actualizar el último tiempo de impresión

            # Limpiar la lista de decisiones para comenzar de nuevo
            decisiones_clasificador = []


        if elapsed_time >= t_delay:
            # Clasificar los datos EMG en tiempo real después de 1 segundo
            label = classify_emg(emg)
            decisiones_clasificador.append(label)  # Almacenar la decisión

        # Detener la recolección después de segundos segundos
        if elapsed_time >= seconds:
            collect = False

    m.add_emg_handler(label_emg)

    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    while collect:
        m.run()

    m.vibrate(2)
    m.disconnect()


if __name__ == '__main__':
    seconds = 6  # Total de 6 segundos de recolección (coimienzo segundo esperando + x segundos clasificando cada segundo + 1 segundo final)
    t_delay = 1  #Deley del principio de captación de datos
    mode = emg_mode.FILTERED
    p = multiprocessing.Process(target=data_worker, args=(mode, seconds, t_delay))

    p.start()