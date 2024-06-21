import time
import multiprocessing
import numpy as np
import subprocess
from pyomyo import Myo, emg_mode
from joblib import load

# Cargar el clasificador preentrenado
classifier = load('path/to/your/classifier.pkl')

# Función para clasificar los datos EMG
def classify_emg(emg_data):
    emg_data = np.array(emg_data).reshape(1, -1)
    label = classifier.predict(emg_data)
    return label[0]  # Asumimos que el clasificador devuelve una lista

# Función para ejecutar un comando curl
def execute_curl(url):
    subprocess.run([curl_command, url], check=True)

#Definamos los diferentes curl
# Definir los argumentos del comando curl
url = "http://raspberrypi.local:8123/api/services/switch/turn_off"
auth_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJkZjNkNjY1OTA5Yzg0MTI4YTViYTRmMzliYmVkZGRjMSIsImlhdCI6MTcxNjE0MTgxMiwiZXhwIjoyMDMxNTAxODEyfQ.1gcdIXy9KmEtm3RTX5gvFqWzhksu410atZUg_dA1Ews"
json_file_path = "C:\\Users\\anita\\body.json"

# Crear la lista de argumentos
curl_command = [
    "curl",
    "-X", "POST",
    "-H", f"Authorization: Bearer {auth_token}",
    "-H", "Content-Type: application/json",
    "-d", f"@{json_file_path}",
    url
]

# Diccionario de acciones
def handle_label_1():
    execute_curl('http://example.com/endpoint1')

def handle_label_2():
    execute_curl('http://example.com/endpoint2')

def handle_label_3():
    execute_curl('http://example.com/endpoint3')

# Diccionario que simula un switch-case
label_actions = {
    'label_1': handle_label_1,
    'label_2': handle_label_2,
    'label_3': handle_label_3
}

def data_worker(mode, seconds):
    collect = True

    # ------------ Myo Setup ---------------
    m = Myo(mode=mode)
    m.connect()

    tiempo = 0

    def save_emg(emg, movement):


        # Clasificar los datos EMG en tiempo real
        label = classify_emg(emg)
        print(f"Tiempo: {tiempo}, EMG: {emg}, Label: {label}")

        # Ejecutar la acción correspondiente a la etiqueta
        action = label_actions.get(label)
        if action:
            action()
        else:
            print(f"Etiqueta no reconocida: {label}")

    m.add_emg_handler(save_emg)

    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    print("Data Worker started to collect")
    start_time = time.perf_counter()

    while collect:
        if time.perf_counter() - start_time < seconds:
            m.run()
            tiempo = time.perf_counter() - start_time
        else:
            collect = False
            vuelta = 2
            collection_time = time.perf_counter() - start_time
            print(f"Finished collecting.")

            m.vibrate(2)
            m.disconnect()

            print(f"Collection time: {collection_time}")

if __name__ == '__main__':

    seconds = 5

    mode = emg_mode.FILTERED
    p = multiprocessing.Process(target=data_worker, args=(mode, seconds))

    p.start()
