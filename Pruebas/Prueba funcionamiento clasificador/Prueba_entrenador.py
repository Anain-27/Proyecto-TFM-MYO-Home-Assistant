import time
import numpy as np
import joblib
from pyomyo import Myo, emg_mode

# Lista de gestos
etiquetas_deseadas = ['BAJAR_DIAL', 'C', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FIST', 'GIRO_IN', 'GIRO_OUT', 'I', 'JUNTOS',
                      'L', 'REST', 'SUBIR_DIAL', 'TRES', 'UNO', 'WAVE_IN',
                      'WAVE_OUT']  # Cambia esto a las etiquetas que necesites

# Cargar el clasificador y el escalador preentrenados desde tu path
clasificador = joblib.load(
    'Clasificadores de pruebas\\Clasificadores\\clasificador_50000_3_gestos_grid_param.pkl')
escalador = joblib.load(
    'Clasificadores de pruebas\\Escaladores\\scaler_50000.pkl')


def limpieza(datos_combinados):
    datos_rectificados = np.abs(datos_combinados)
    datos_rectificados = datos_rectificados.reshape(1, -1)
    datos_escalados = escalador.transform(datos_rectificados)
    return datos_escalados


def clasificar_emg_imu(datos_combinados):
    datos_combinados = limpieza(datos_combinados)
    etiqueta = clasificador.predict(datos_combinados)[0]
    return etiqueta


def main():
    myo = Myo(mode=emg_mode.FILTERED)
    myo.connect()

    emg_data = None
    imu_data = None

    def manejar_emg(emg, movimiento):
        nonlocal emg_data
        emg_data = emg

    def manejar_imu(quat, acc, gyro):
        nonlocal imu_data
        imu_data = np.concatenate((quat, acc, gyro), axis=None)

    myo.add_emg_handler(manejar_emg)
    myo.add_imu_handler(manejar_imu)

    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    tiempo_total = 5  # Tiempo para cada gesto en segundos
    aciertos_totales = 0
    total_gestos = len(gestos)
    total_predicciones = 0

    try:
        for gesto in gestos:
            print(f"{gesto}")
            time.sleep(2)
            print(f"Realiza el gesto: {gesto}")
            tiempo_inicio = time.time()
            aciertos = 0
            total_predicciones=0
            while time.time() - tiempo_inicio < tiempo_total:
                myo.run()
                tiempo_actual = time.time() - tiempo_inicio
                if emg_data is not None and imu_data is not None:
                    combined_data = np.concatenate((emg_data, imu_data), axis=None)
                    etiqueta = clasificar_emg_imu(combined_data)
                    print(etiqueta)

                    if etiqueta == gesto:
                        aciertos += 1

                    total_predicciones += 1
                else:
                    print("Esperando datos de Myo...")

            if total_predicciones > 0:
                porcentaje_aciertos = (aciertos / total_predicciones) * 100
            else:
                porcentaje_aciertos = 0

            aciertos_totales += porcentaje_aciertos

            print(f"Gesto {gesto} completado. Porcentaje de aciertos: {porcentaje_aciertos:.2f}%")
            print(f"Descansa 2 segundos y realiza el siguiente gesto:")

    except KeyboardInterrupt:
        print("Interrupción del usuario, finalizando...")
    finally:
        myo.disconnect()
        print("Myo desconectado.")

        promedio_aciertos = aciertos_totales / total_gestos
        print(f"Porcentaje promedio de aciertos: {promedio_aciertos:.2f}%")


if __name__ == '__main__':
    main()
