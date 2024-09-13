import time
import subprocess
import multiprocessing
import numpy as np
from collections import Counter
from pyomyo import Myo, emg_mode
from joblib import load
from sklearn.preprocessing import StandardScaler

# Cargar el clasificador preentrenado
clasificador = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Clasificacion\\entrenado_svm_poly_model.pkl')

# Cargar el escalador preentrenado
escalador = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Clasificacion\\scaler.pkl')

# Función para la limpieza de datos EMG
def limpieza(datos_emg):
    # Rectificación (valor absoluto)
    emg_rectificado = np.abs(datos_emg)
    # Escalado
    emg_rectificado = emg_rectificado.reshape(1, -1)  # Reformar para que tenga la forma correcta
    emg_escalado = escalador.transform(emg_rectificado)
    return emg_escalado

# Función para clasificar los datos EMG
def clasificar_emg(datos_emg):
    datos_emg = limpieza(datos_emg)  # Aplicar la limpieza antes de clasificar
    etiqueta = clasificador.predict(datos_emg)
    return etiqueta[0]  # Asumimos que el clasificador devuelve una lista

# Función para encontrar la etiqueta más común y su porcentaje
def mas_comun(etiquetas):
    if etiquetas:
        total_etiquetas = len(etiquetas)
        contador = Counter(etiquetas)
        etiqueta_mas_comun, conteo = contador.most_common(1)[0]
        porcentaje = conteo / total_etiquetas * 100
        return etiqueta_mas_comun, porcentaje
    return None, 0

# Función para imprimir la etiqueta más común y ejecutar el comando curl
def llamada(etiqueta_mas_comun, porcentaje):
    print(f"Resultado más común después de 2 segundos: Etiqueta: {etiqueta_mas_comun}, Porcentaje: {porcentaje:.2f}%")
    if etiqueta_mas_comun == "FINGERS_SPREAD":
        print('on')
        comando_curl = """curl -X POST -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJkZjNkNjY1OTA5Yzg0MTI4YTViYTRmMzliYmVkZGRjMSIsImlhdCI6MTcxNjE0MTgxMiwiZXhwIjoyMDMxNTAxODEyfQ.1gcdIXy9KmEtm3RTX5gvFqWzhksu410atZUg_dA1Ews" -H "Content-Type: application/json" -d @C:\\Users\\anita\\body.json "http://raspberrypi.local:8123/api/services/switch/turn_on\""""
    elif etiqueta_mas_comun != "REST":
        print('off')
        comando_curl = """curl -X POST -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJkZjNkNjY1OTA5Yzg0MTI4YTViYTRmMzliYmVkZGRjMSIsImlhdCI6MTcxNjE0MTgxMiwiZXhwIjoyMDMxNTAxODEyfQ.1gcdIXy9KmEtm3RTX5gvFqWzhksu410atZUg_dA1Ews" -H "Content-Type: application/json" -d @C:\\Users\\anita\\body.json "http://raspberrypi.local:8123/api/services/switch/turn_off\""""
    resultado = subprocess.run(comando_curl, shell=True, capture_output=True, text=True)
    print(resultado.stdout)
    print(resultado.stderr)

def verificar_fist(decisiones):
    etiquetas_ventana = decisiones[-int(200 * 1):]  # Aproximadamente 200 muestras por segundo
    etiqueta_mas_comun, _ = mas_comun(etiquetas_ventana)
    return etiqueta_mas_comun == "FIST"

def trabajador_datos(modo, t_captura_total, t_espera, t_captura_FIST):
    recolectar = True
    tiempo_inicio = time.time()
    ultimo_tiempo_impresion = tiempo_inicio
    decisiones = []  # Lista para almacenar las decisiones del clasificador
    buscando_fist = True
    tiempo_deteccion_fist = None
    en_delay = False
    en_periodo_captura = False

    # ------------ Configuración Myo ---------------
    myo = Myo(mode=modo)
    myo.connect()

    def manejar_emg(emg, movimiento):
        nonlocal recolectar, ultimo_tiempo_impresion, decisiones, buscando_fist, tiempo_deteccion_fist, en_delay, en_periodo_captura
        tiempo_actual = time.time()
        tiempo_transcurrido = tiempo_actual - tiempo_inicio

        # Clasificar los datos EMG en tiempo real
        etiqueta = clasificar_emg(emg)
        decisiones.append(etiqueta)  # Almacenar la decisión

        if buscando_fist and tiempo_transcurrido >= t_espera:
            if verificar_fist(decisiones):
                tiempo_deteccion_fist = tiempo_actual  # Registrar el tiempo de detección de FIST
                buscando_fist = False
                en_delay = True
                print("FIST detectado, esperando 1 segundo antes de capturar")

        if en_delay and tiempo_deteccion_fist:
            tiempo_pasado = tiempo_actual - tiempo_deteccion_fist
            if tiempo_pasado >= t_espera:
                en_delay = False
                en_periodo_captura = True
                tiempo_deteccion_fist = tiempo_actual  # Reiniciar el tiempo de detección para el periodo de captura

        if en_periodo_captura and tiempo_deteccion_fist:
            tiempo_pasado = tiempo_actual - tiempo_deteccion_fist
            if tiempo_pasado >= t_captura_FIST:
                if tiempo_actual - ultimo_tiempo_impresion >= 1:
                    etiqueta_mas_comun, porcentaje = mas_comun(decisiones)
                    llamada(etiqueta_mas_comun, porcentaje)  # Llamar a la función para imprimir y ejecutar curl
                    ultimo_tiempo_impresion = tiempo_actual
                    decisiones = []  # Reiniciar la lista de decisiones después de imprimir
                    tiempo_deteccion_fist = None
                    en_periodo_captura = False
                    buscando_fist = True

        # Detener la recolección después de t_captura_total segundos
        if tiempo_transcurrido >= t_captura_total:
            recolectar = False

    myo.add_emg_handler(manejar_emg)

    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    while recolectar:
        myo.run()

    myo.vibrate(2)
    myo.disconnect()

if __name__ == '__main__':
    t_captura_total = 20  # Total de tiempo de recolección
    t_espera = 1  # Delay del principio de captación de datos y delay antes de capturar después de detectar "FIST"
    t_captura_FIST = 2  # Capturar datos durante 2 segundos después del delay
    modo = emg_mode.FILTERED
    p = multiprocessing.Process(target=trabajador_datos, args=(modo, t_captura_total, t_espera, t_captura_FIST))
    p.start()
