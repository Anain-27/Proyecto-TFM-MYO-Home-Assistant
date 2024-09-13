import time
import numpy as np
import joblib
from pyomyo import Myo, emg_mode
from collections import Counter

# Cargar el clasificador y el escalador preentrenados desde tu path
clasificador = joblib.load('Clasificadores de pruebas/Clasificadores/clasificador_5000_17_gestos_param.pkl')
escalador = joblib.load('Clasificadores de pruebas/Escaladores/scaler_5000_no_IMU_17.pkl')

def limpieza(emg_data):
    datos_rectificados = np.abs(emg_data)
    datos_rectificados = datos_rectificados.reshape(1, -1)
    datos_escalados = escalador.transform(datos_rectificados)
    return datos_escalados

def clasificar_emg_imu(emg_data):
    datos_combinados = limpieza(emg_data)
    etiqueta = clasificador.predict(datos_combinados)[0]
    return etiqueta

def main():
    # Conectar al Myo
    myo = Myo(mode=emg_mode.FILTERED)
    myo.connect()

    emg_data = None
    etiquetas_intervalo = []

    def manejar_emg(emg, movimiento):
        nonlocal emg_data
        emg_data = emg

    myo.add_emg_handler(manejar_emg)

    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    tiempo_total = 20  # Tiempo total de prueba en segundos
    tiempo_intervalo = 2  # Intervalo para imprimir el gesto más repetido en segundos

    try:
        start_time = time.time()
        intervalo_inicio = start_time

        while time.time() - start_time < tiempo_total:
            if time.time() - intervalo_inicio >= tiempo_intervalo:
                if etiquetas_intervalo:
                    conteo_etiquetas = Counter(etiquetas_intervalo)
                    etiqueta_mas_frecuente = conteo_etiquetas.most_common(1)[0][0]
                    print(f"Gesto más repetido en los últimos {tiempo_intervalo} segundos: {etiqueta_mas_frecuente}")
                else:
                    print("No se detectaron etiquetas en el intervalo.")

                etiquetas_intervalo = []  # Limpiar la lista para el siguiente intervalo
                intervalo_inicio = time.time()

            myo.run()
            if emg_data is not None:
                etiqueta = clasificar_emg_imu(emg_data)
                etiquetas_intervalo.append(etiqueta)

    except KeyboardInterrupt:
        print("Interrupción del usuario, finalizando...")
    finally:
        myo.disconnect()
        print("Myo desconectado.")

if __name__ == '__main__':
    main()
