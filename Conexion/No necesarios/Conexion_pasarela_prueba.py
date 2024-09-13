import time
import subprocess
import multiprocessing
import numpy as np
from collections import Counter
from pyomyo import Myo, emg_mode
from joblib import load
from sklearn.preprocessing import StandardScaler

# Cargar los clasificadores preentrenados
print("Cargando los clasificadores preentrenados...")
clasificador_TRES = load('clasificador_50000_no_IMU_8_SIN_CRUZAR_mejor.pkl')
clasificador_CRUZAR = load('clasificador_50000_no_IMU_8_SIN_TRES_mejor.pkl')
print("Clasificadores cargados.")

# Cargar el escalador preentrenado
print("Cargando el escalador preentrenado...")
scaler = load('scaler_50000_no_IMU_8_SIN_TRES_mejor.pkl')
print("Escalador cargado.")

# Función para la limpieza de datos EMG
def limpieza(emg_data):
    emg_rectified = np.abs(emg_data)
    emg_rectified = emg_rectified.reshape(1, -1)
    emg_scaled = scaler.transform(emg_rectified)
    return emg_scaled

# Función para clasificar los datos EMG
def classify_emg(emg_data, classifier):
    emg_data = limpieza(emg_data)
    label = classifier.predict(emg_data)
    return label[0]

# Función para encontrar la etiqueta más común
def mas_comun(etiquetas):
    if etiquetas:
        total_etiquetas = len(etiquetas)
        contador = Counter(etiquetas)
        etiqueta_mas_comun, conteo = contador.most_common(1)[0]
        porcentaje = conteo / total_etiquetas * 100
        return etiqueta_mas_comun, porcentaje
    return None, 0

# Función para capturar el gesto más común durante 5 segundos
def capturar_durante_5_segundos(myo, clasificador):
    tiempo_inicio = time.time()
    decisiones = []
    while time.time() - tiempo_inicio < 5:
        myo.run()
        if myo.emg_data is not None:
            etiqueta = classify_emg(myo.emg_data, clasificador)
            decisiones.append(etiqueta)
    etiqueta_mas_comun, porcentaje = mas_comun(decisiones)
    print(f"Resultado más común durante 5 segundos: {etiqueta_mas_comun}, {porcentaje:.2f}%")
    return etiqueta_mas_comun, porcentaje

# Función para verificar si se ha detectado "CRUZAR_DEDOS"
def verificar_cruzar_dedos(decisiones):
    etiquetas_ventana = decisiones[-int(200 * 1):]  # Aproximadamente 200 muestras por segundo
    etiqueta_mas_comun, _ = mas_comun(etiquetas_ventana)
    return etiqueta_mas_comun == "CRUZAR_DEDOS"

# Función principal del trabajador de datos
def trabajador_datos(modo, t_captura_total, t_espera):
    recolectar = True
    tiempo_inicio = time.time()
    decisiones = []
    buscando_cruzar_dedos = True
    tiempo_deteccion_cruzar_dedos = None
    en_periodo_captura = False

    # Configuración Myo
    myo = Myo(mode=modo)
    myo.connect()
    myo.emg_data = None

    def manejar_emg(emg, movimiento):
        myo.emg_data = emg

    myo.add_emg_handler(manejar_emg)
    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    while recolectar:
        myo.run()

        if myo.emg_data is not None:
            etiqueta = classify_emg(myo.emg_data, clasificador_CRUZAR)
            decisiones.append(etiqueta)

            if buscando_cruzar_dedos:
                if verificar_cruzar_dedos(decisiones):
                    print("CRUZAR_DEDOS detectado, esperando 1 segundo...")
                    time.sleep(1)  # Esperar 1 segundo
                    print("Iniciando captura durante 5 segundos con clasificador TRES.")
                    capturar_durante_5_segundos(myo, clasificador_TRES)
                    buscando_cruzar_dedos = False  # Resetear la búsqueda después de capturar
                    decisiones = []  # Limpiar decisiones después de capturar

        if (time.time() - tiempo_inicio) >= t_captura_total:
            recolectar = False

    myo.vibrate(2)
    myo.disconnect()

if __name__ == '__main__':
    t_captura_total = 20  # Total de tiempo de recolección
    t_espera = 1  # Delay antes de capturar después de detectar "CRUZAR_DEDOS"
    modo = emg_mode.FILTERED
    p = multiprocessing.Process(target=trabajador_datos, args=(modo, t_captura_total, t_espera))
    p.start()
    p.join()
