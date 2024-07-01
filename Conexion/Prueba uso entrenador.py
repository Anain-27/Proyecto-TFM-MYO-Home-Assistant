import time
import multiprocessing
import numpy as np
from collections import Counter
from pyomyo import Myo, emg_mode
from joblib import load
from sklearn.preprocessing import StandardScaler

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


# Función para encontrar las decisiones más comunes con sus porcentajes
def mas_comunes(labels):
    if labels:
        total_count = len(labels)
        counter = Counter(labels)
        sorted_labels = counter.most_common()
        return [(label, count / total_count * 100) for label, count in sorted_labels]
    return []


def data_worker(mode, seconds):
    collect = True
    decisions = []  # Lista para almacenar las decisiones del clasificador

    # ------------ Myo Setup ---------------
    m = Myo(mode=mode)
    m.connect()

    start_time = time.perf_counter()  # Definir start_time aquí

    def label_emg(emg, movement):
        nonlocal start_time  # Declarar start_time como nonlocal
        tiempo = time.perf_counter() - start_time
        if tiempo >= 1:
            # Clasificar los datos EMG en tiempo real después de 1 segundo
            label = classify_emg(emg)
            decisions.append(label)  # Almacenar la decisión
            print(f"Tiempo: {tiempo:.2f} segundos, EMG: {emg}, Decisión del clasificador: {label}")

    m.add_emg_handler(label_emg)

    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    print("Data Worker started to collect")
    start_time = time.perf_counter()

    while collect:
        if time.perf_counter() - start_time < seconds:
            m.run()
        else:
            collect = False
            collection_time = time.perf_counter() - start_time
            print("Finished collecting.")

            m.vibrate(2)
            m.disconnect()

            print(f"Collection time: {collection_time:.2f} segundos")

            # Encontrar y mostrar las decisiones más comunes con sus porcentajes
            common_labels = mas_comunes(decisions)
            for label, percentage in common_labels:
                print(f"Etiqueta: {label}, Porcentaje: {percentage:.2f}%")


if __name__ == '__main__':
    seconds = 2

    mode = emg_mode.FILTERED
    p = multiprocessing.Process(target=data_worker, args=(mode, seconds))

    p.start()
