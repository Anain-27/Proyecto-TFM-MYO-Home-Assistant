import numpy as np
from pyomyo import Myo, emg_mode
from joblib import load
from collections import Counter
import time

# Cargar el clasificador preentrenado
print("Cargando el clasificador preentrenado...")
clasificador = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Clasificacion\\Clasificador\\prueba_solo_channels.pkl')
print("Clasificador cargado.")

# Cargar el escalador preentrenado
print("Cargando el escalador preentrenado...")
escalador = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Clasificacion\\Clasificador\\scaler_channels.pkl')
print("Escalador cargado.")

# Función para la limpieza de datos EMG
def limpieza(datos_emg):
    print("Limpiando y rectificando datos EMG...")
    # Aplicar el valor absoluto a los datos EMG
    datos_rectificados = np.abs(datos_emg)

    # Escalado de los datos EMG
    datos_rectificados = datos_rectificados.reshape(1, -1)  # Reformar para que tenga la forma correcta
    datos_escalados = escalador.transform(datos_rectificados)
    print("Datos EMG limpiados, rectificados y escalados.")
    return datos_escalados

# Función para clasificar los datos EMG
def clasificar_emg(datos_emg):
    print("Clasificando datos EMG...")
    datos_emg = limpieza(datos_emg)  # Aplicar la limpieza antes de clasificar
    etiqueta = clasificador.predict(datos_emg)
    return etiqueta[0]  # Asumimos que el clasificador devuelve una lista

def main():
    print("Iniciando conexión a Myo...")
    myo = Myo(mode=emg_mode.FILTERED)
    myo.connect()
    print("Myo conectado.")

    emg_data = None  # Variable para almacenar temporalmente los datos EMG
    decisiones = []  # Lista para almacenar las decisiones del clasificador

    def manejar_emg(emg, movimiento):
        nonlocal emg_data
        emg_data = emg  # Guardar los datos EMG para clasificarlos

    myo.add_emg_handler(manejar_emg)

    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    tiempo_inicio = time.time()
    tiempo_total = 10  # Tiempo total de recolección en segundos

    try:
        print("Iniciando recolección de datos...")
        while time.time() - tiempo_inicio < tiempo_total:
            myo.run()
            if emg_data is not None:
                etiqueta = clasificar_emg(emg_data)
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
