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
    url = f'{base_url}/api/services/switch/{state}'
    data = {'entity_id': entity_id}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Acción {state} ejecutada para {entity_id}")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

def set_media_state(service, entity_id):
    url = f'{base_url}/api/services/media_player/{service}'
    data = {'entity_id': entity_id}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Acción {service} ejecutada para {entity_id}")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
    except Exception as err:
        print(f"Other error occurred: {err}")

print("Cargando los clasificadores preentrenados...")
clasificador_TRES = load('clasificador_50000_no_IMU_8_SIN_CRUZAR_mejor.pkl')
clasificador_CRUZAR = load('clasificador_50000_no_IMU_8_SIN_TRES_mejor.pkl')
print("Clasificadores cargados.")

print("Cargando el escalador preentrenado...")
scaler = load('scaler_50000_no_IMU_8_SIN_TRES_mejor.pkl')
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
    if len(decisiones) >= 200:
        etiquetas_ventana = decisiones[-200:]  # Aproximadamente 200 muestras
        etiqueta_mas_comun, porcentaje = mas_comun(etiquetas_ventana)
        return porcentaje > 30, etiqueta_mas_comun
    return False, None

def manejar_gesto_dispositivos(myo):
    try:
        # Captura el primer gesto para seleccionar el dispositivo
        gesto = capturar_durante_5_segundos(myo, clasificador_TRES)
        print(f"Gesto capturado para selección de dispositivo: {gesto[0]}")  # Imprime el primer gesto capturado

        # Asignar el dispositivo dependiendo del gesto
        if gesto[0] == 'BAJAR_DIAL':
            dispositivo = 'switch.virtual_switch'
        elif gesto[0] == 'FIST':
            dispositivo = 'switch.enchufe'
        elif gesto[0] in ['TRES', 'WAVE_OUT']:
            dispositivo = 'media_player.spotify_ana_cuevas'  # Cambiado aquí
        else:
            print(f"Gesto no reconocido: {gesto[0]}")
            dispositivo = None  # No hay dispositivo asociado si el gesto no es válido

        # Si se ha seleccionado un dispositivo válido
        if dispositivo:
            print(f"Dispositivo seleccionado: {dispositivo}. Ahora, captura un nuevo gesto para asignar su estado.")

            # Descanso de 2 segundos antes de capturar el segundo gesto
            print("Descansando 2 segundos...")
            time.sleep(2)

            # Captura otro gesto para decidir el estado
            segundo_gesto = capturar_durante_5_segundos(myo, clasificador_TRES)
            print(f"Gesto capturado para cambiar estado: {segundo_gesto[0]}")  # Imprime el segundo gesto capturado

            if dispositivo == 'media_player.spotify_ana_cuevas':  # Cambiado aquí
                # Definir las acciones para media_player según el gesto
                if segundo_gesto[0] == 'FIST':
                    # Verificar el estado del media_player (reproduciendo o en pausa)
                    estado = get_state(dispositivo)

                    if estado == 'playing':
                        print(
                            f"{dispositivo} está reproduciendo. Gesto capturado: {segundo_gesto[0]}, pausando reproducción.")
                        set_media_state('media_pause', dispositivo)
                    else:
                        print(
                            f"{dispositivo} está en pausa. Gesto capturado: {segundo_gesto[0]}, iniciando reproducción.")
                        set_media_state('media_play', dispositivo)
                elif segundo_gesto[0] == 'WAVE_IN':
                    print(f"Cambiando a la pista anterior en {dispositivo}.")
                    set_media_state('media_previous_track', dispositivo)
                elif segundo_gesto[0] == 'WAVE_OUT':
                    print(f"Cambiando a la pista siguiente en {dispositivo}.")
                    set_media_state('media_next_track', dispositivo)
                else:
                    print(f"Gesto no reconocido para cambiar estado del media_player: {segundo_gesto[0]}")

            else:
                # Si no es media_player, tratamos el dispositivo como interruptor
                if segundo_gesto[0] == 'FIST':
                    # Verificar el estado del dispositivo y cambiar entre on y off
                    estado = get_state(dispositivo)

                    if estado == 'on':
                        print(f"{dispositivo} está en ON. Gesto capturado: {segundo_gesto[0]}, cambiando a OFF.")
                        set_state(dispositivo, 'turn_off')  # Cambiado aquí para usar el servicio correcto
                    else:
                        print(f"{dispositivo} está en OFF. Gesto capturado: {segundo_gesto[0]}, cambiando a ON.")
                        set_state(dispositivo, 'turn_on')  # Cambiado aquí para usar el servicio correcto
                else:
                    print(f"Gesto no reconocido para cambiar estado: {segundo_gesto[0]}")
    except Exception as e:
        print(f"Error al manejar el gesto o el estado del dispositivo: {e}")


def trabajador_datos(modo, t_captura_total):
    print("Inicio de trabajador_datos")
    tiempo_inicio = time.time()
    decisiones = []
    buscando_cruzar_dedos = True

    myo = Myo(mode=modo)
    myo.connect()
    myo.emg_data = None

    def manejar_emg(emg, movimiento):
        myo.emg_data = emg

    myo.add_emg_handler(manejar_emg)
    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    while time.time() - tiempo_inicio < t_captura_total:
        try:
            myo.run()
            if myo.emg_data is not None:
                etiqueta = classify_emg(myo.emg_data, clasificador_CRUZAR)
                print(f"Etiqueta clasificada: {etiqueta}")
                decisiones.append(etiqueta)
                if buscando_cruzar_dedos:
                    verificar = verificar_cruzar_dedos(decisiones)
                    if verificar[0]:
                        print("CRUZAR_DEDOS detectado, esperando 2 segundos...")
                        time.sleep(2)
                        manejar_gesto_dispositivos(myo)
                        time.sleep(1)
                        decisiones = []
        except Exception as e:
            print(f"Error en el bucle principal: {e}")

    myo.vibrate(2)
    myo.disconnect()


if __name__ == '__main__':
    t_captura_total = 30  # Total de tiempo de recolección
    modo = emg_mode.FILTERED
    trabajador_datos(modo, t_captura_total)
