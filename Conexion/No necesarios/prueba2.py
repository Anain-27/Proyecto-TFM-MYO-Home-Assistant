import time
import requests
import numpy as np
from collections import Counter
from pyomyo import Myo, emg_mode
from joblib import load
from sklearn.preprocessing import StandardScaler

# Configuración para la solicitud HTTP
base_url = 'http://192.168.142.137:8123'
token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJlMWI5ODc3NThmYjI0ZDhjYTA4ZTY3ZGU4Y2IzZDZhNiIsImlhdCI6MTcyNjA5MTQxOSwiZXhwIjoyMDQxNDUxNDE5fQ.OgRT3vg7dRxqe3vuRqG211gSfsTDEQ6VaWpSfUuv7yk'
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json',
}

def get_state(entity_id):
    url = f'{base_url}/api/states/{entity_id}'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    state = response.json()['state']
    return state

def set_state(entity_id, state):
    url = f'{base_url}/api/states/{entity_id}'
    data = {'state': state}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Estado de {entity_id} cambiado a {state}")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

print("Cargando los clasificadores preentrenados...")
clasificador_TRES = load('clasificador_50000_no_IMU_8_SIN_CRUZAR_mejor.pkl')
clasificador_CRUZAR = load('/home/ana/Documents/Pruebas-raspberry/clasificador_50000_no_IMU_8_SIN_TRES_mejor.pkl')
print("Clasificadores cargados.")

print("Cargando el escalador preentrenado...")
scaler = load('/home/ana/Documents/Pruebas-raspberry/scaler_50000_no_IMU_8_SIN_TRES_mejor.pkl')
print("Escalador cargado.")

def limpieza(emg_data):
    emg_rectified = np.abs(emg_data)
    emg_rectified = emg_rectified.reshape(1, -1)
    emg_scaled = scaler.transform(emg_rectified)
    return emg_scaled

def classify_emg(emg_data, classifier):
    emg_data = limpieza(emg_data)
    label = classifier.predict(emg_data)
    return label[0]

def mas_comun(etiquetas):
    if etiquetas:
        total_etiquetas = len(etiquetas)
        contador = Counter(etiquetas)
        etiqueta_mas_comun, conteo = contador.most_common(1)[0]
        porcentaje = conteo / total_etiquetas * 100
        return etiqueta_mas_comun, porcentaje
    return None, 0

def capturar_durante_5_segundos(myo, clasificador):
    print('Entro en capturar_durante_5_segundos')
    tiempo_inicio = time.time()
    decisiones = []
    while time.time() - tiempo_inicio < 5:
        myo.run()
        if myo.emg_data is not None:
            etiqueta = classify_emg(myo.emg_data, clasificador)
            decisiones.append(etiqueta)
            time.sleep(0.1)
    etiqueta_mas_comun, porcentaje = mas_comun(decisiones)
    print(f"Resultado más común durante 5 segundos: {etiqueta_mas_comun}, {porcentaje:.2f}%")
    return etiqueta_mas_comun, porcentaje

def verificar_cruzar_dedos(decisiones):
    if len(decisiones) >= 20:
        etiquetas_ventana = decisiones[-200:]  # Aproximadamente 200 muestras
        etiqueta_mas_comun, _ = mas_comun(etiquetas_ventana)
        return etiqueta_mas_comun == "CRUZAR_DEDOS", etiqueta_mas_comun
    return False, None

def manejar_emg(emg, movimiento):
    global myo
    myo.emg_data = emg

def trabajador_datos(modo, t_captura_total):
    print("Inicio de trabajador_datos")
    tiempo_inicio = time.time()
    decisiones = []
    buscando_cruzar_dedos = True
    tiempo_ciclo = 10  # Intervalo para reconectar el dispositivo Myo
    tiempo_proximo_reconexion = tiempo_inicio + tiempo_ciclo

    myo = None

    while time.time() - tiempo_inicio < t_captura_total:
        if time.time() >= tiempo_proximo_reconexion:
            if myo:
                myo.disconnect()
                print("Desconectado del dispositivo Myo")

            # Reconectar
            myo = Myo(mode=modo)
            print("Conectando al dispositivo Myo...")
            myo.connect()
            myo.emg_data = None

            # Agregar el manejador de datos EMG
            myo.add_emg_handler(manejar_emg)

            tiempo_proximo_reconexion = time.time() + tiempo_ciclo  # Actualizar el tiempo para la próxima reconexión

        try:
            myo.run()
            if myo.emg_data is not None:
                etiqueta = classify_emg(myo.emg_data, clasificador_CRUZAR)
                decisiones.append(etiqueta)

                if buscando_cruzar_dedos:
                    verificar = verificar_cruzar_dedos(decisiones)
                    if verificar[0]:
                        print("CRUZAR_DEDOS detectado, esperando 1 segundo...")
                        time.sleep(1)

                        try:
                            gesto = capturar_durante_5_segundos(myo, clasificador_TRES)
                            if gesto[0] == 'TRES':
                                dispositivo = 'switch.virtual_switch'
                                estado = get_state(dispositivo)
                                if estado == 'on':
                                    print("Interruptor virtual está en ON.")
                                    set_state(dispositivo, 'off')
                                else:
                                    print("Interruptor virtual está en OFF.")
                                    set_state(dispositivo, 'on')
                        except Exception as e:
                            print(f"Error al manejar el gesto o el estado del interruptor: {e}")
                        decisiones = []
                        time.sleep(1)
        except Exception as e:
            print(f"Error en el bucle principal: {e}")

    if myo:
        myo.vibrate(2)
        print('Vibración finalizada')
        myo.disconnect()
        print('Desconectado del dispositivo Myo')

if __name__ == '__main__':
    t_captura_total = 30  # Total de tiempo de recolección
    modo = emg_mode.FILTERED
    trabajador_datos(modo, t_captura_total)
