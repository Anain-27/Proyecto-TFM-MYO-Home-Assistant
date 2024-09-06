import numpy as np
import joblib
import cupy as cp
from collections import Counter
import time
from pyomyo import Myo, emg_mode

# Cargar el clasificador entrenado
print("Cargando el clasificador preentrenado...")
clasificador = joblib.load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Pruebas modelo entrenado\\entrenador_C_01.pkl')
print("Clasificador cargado.")

# Suponiendo que guardaste el escalador también, cargar el escalador
print("Cargando el escalador preentrenado...")
escalador = joblib.load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Pruebas\\Pruebas modelo entrenado\\escalador_nuevo.pkl')
print("Escalador cargado.")

# Función para la limpieza de datos EMG e IMU
def limpieza(datos_combinados):
    print("Limpiando y rectificando todos los datos...")
    datos_rectificados = np.abs(datos_combinados)

    # Escalar los datos combinados
    datos_rectificados = datos_rectificados.reshape(1, -1)
    datos_escalados = escalador.transform(datos_rectificados)
    print("Datos limpiados, rectificados y escalados.")
    return datos_escalados

# Función para clasificar los datos EMG e IMU combinados
def clasificar_emg_imu(datos_combinados):
    print("Clasificando datos combinados...")
    datos_combinados = limpieza(datos_combinados)
    etiqueta = clasificador.predict(cp.asarray(datos_combinados))  # Convertir a formato cupy si es necesario
    return etiqueta[0]  # Asumimos que el clasificador devuelve una lista

def main():
    print("Iniciando conexión a Myo...")
    myo = Myo(mode=emg_mode.FILTERED)
    myo.connect()
    print("Myo conectado.")

    emg_data = None
    imu_data = None
    decisiones = []

    def manejar_emg(emg, movimiento):
        nonlocal emg_data
        emg_data = emg  # Guardar los datos EMG para combinarlos luego

    def manejar_imu(quat, acc, gyro):
        nonlocal imu_data
        imu_data = np.concatenate((quat, acc, gyro), axis=None)  # Guardar los datos IMU combinados

    myo.add_emg_handler(manejar_emg)
    myo.add_imu_handler(manejar_imu)

    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    tiempo_inicio = time.time()
    tiempo_total = 10  # Tiempo total de recolección en segundos

    try:
        print("Iniciando recolección de datos...")
        while time.time() - tiempo_inicio < tiempo_total:
            myo.run()
            if emg_data is not None and imu_data is not None:
                combined_data = np.concatenate((emg_data, imu_data), axis=None)
                etiqueta = clasificar_emg_imu(combined_data)
                decisiones.append(etiqueta)
                print(f"Etiqueta clasificada: {etiqueta}")

    except KeyboardInterrupt:
        print("Interrupción del usuario, finalizando...")
    finally:
        print("Desconectando Myo...")
        myo.vibrate(2)
        myo.disconnect()
        print("Myo desconectado.")

        # Calcular y mostrar porcentaje de cada etiqueta
        if decisiones:
            contador = Counter(decisiones)
            total_decisiones = len(decisiones)
            print("Porcentaje de etiquetas detectadas:")
            for etiqueta, conteo in contador.items():
                porcentaje = (conteo / total_decisiones) * 100
                print(f"Etiqueta: {etiqueta}, Porcentaje: {porcentaje:.2f}%")
        else:
            print("No se detectaron etiquetas.")

if __name__ == '__main__':
    main()
