import time
import subprocess
import multiprocessing
import numpy as np
from collections import Counter
from pyomyo import Myo, emg_mode
from joblib import load
from sklearn.preprocessing import StandardScaler

# Cargar el clasificador preentrenado
print("Cargando el clasificador preentrenado...")
clasificador = load(
    'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Clasificacion\\Clasificador\\prueba_mejores_parametros.pkl')
print("Clasificador cargado.")

# Cargar el escalador preentrenado
print("Cargando el escalador preentrenado...")
escalador = load('C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM-MYO-Home-Assistant\\Clasificacion\\Clasificador\\scaler.pkl')
print("Escalador cargado.")

# Función para la limpieza de datos EMG
def limpieza(datos_combinados):
    print("Limpiando datos combinados...")
    # Rectificación (valor absoluto) solo para la parte de EMG
    emg_rectificado = np.abs(datos_combinados[:8])  # Asumimos que los primeros 8 datos son EMG
    datos_combinados[:8] = emg_rectificado

    # Escalado de todos los datos combinados
    datos_combinados = datos_combinados.reshape(1, -1)  # Reformar para que tenga la forma correcta
    datos_escalados = escalador.transform(datos_combinados)
    print("Datos limpiados y escalados.")
    return datos_escalados

# Función para clasificar los datos EMG e IMU combinados
def clasificar_emg_imu(datos_combinados):
    print("Clasificando datos combinados...")
    datos_combinados = limpieza(datos_combinados)  # Aplicar la limpieza antes de clasificar
    etiqueta = clasificador.predict(datos_combinados)
    print(f"Etiqueta clasificada: {etiqueta[0]}")
    return etiqueta[0]  # Asumimos que el clasificador devuelve una lista

# Función para encontrar la etiqueta más común y su porcentaje
def mas_comun(etiquetas):
    print("Calculando la etiqueta más común...")
    if etiquetas:
        total_etiquetas = len(etiquetas)
        contador = Counter(etiquetas)
        etiqueta_mas_comun, conteo = contador.most_common(1)[0]
        porcentaje = conteo / total_etiquetas * 100
        print(f"Etiqueta más común: {etiqueta_mas_comun}, Porcentaje: {porcentaje:.2f}%")
        return etiqueta_mas_comun, porcentaje
    return None, 0

# Función para imprimir la etiqueta más común y ejecutar el comando curl
def llamada(etiqueta_mas_comun, porcentaje):
    print(f"Resultado más común después de 2 segundos: Etiqueta: {etiqueta_mas_comun}, Porcentaje: {porcentaje:.2f}%")

    if etiqueta_mas_comun == "FINGERS_SPREAD":
        print('Encendiendo...')
        comando_curl = """curl -X POST -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d @C:\\Users\\anita\\body.json "http://raspberrypi.local:8123/api/services/switch/turn_on\""""
    elif etiqueta_mas_comun != "REST":
        print('Apagando...')
        comando_curl = """curl -X POST -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d @C:\\Users\\anita\\body.json "http://raspberrypi.local:8123/api/services/switch/turn_off\""""

    resultado = subprocess.run(comando_curl, shell=True, capture_output=True, text=True)
    print(f"Resultado de curl: {resultado.stdout}")
    print(f"Errores de curl: {resultado.stderr}")

# Nueva función para imprimir solo la etiqueta más común
def llamada_2(etiqueta_mas_comun, porcentaje):
    print(f"Resultado: {etiqueta_mas_comun}, {porcentaje:.2f}%")
    # Aquí puedes agregar las acciones específicas para cada etiqueta

# Función para verificar la etiqueta CRUZAR_DEDOS
def verificar_cruzar_dedos(decisiones):
    print("Verificando 'CRUZAR_DEDOS'...")
    etiquetas_ventana = decisiones[-int(200 * 1):]  # Aproximadamente 200 muestras por segundo
    etiqueta_mas_comun, _ = mas_comun(etiquetas_ventana)
    es_cruzar_dedos = etiqueta_mas_comun == "CRUZAR_DEDOS"
    print(f"'CRUZAR_DEDOS' detectado: {es_cruzar_dedos}")
    return es_cruzar_dedos

def trabajador_datos(modo, t_captura_total, t_espera, t_captura_cruzar_dedos):
    print("Iniciando trabajador de datos...")
    recolectar = True
    tiempo_inicio = time.time()
    ultimo_tiempo_impresion = tiempo_inicio
    decisiones = []  # Lista para almacenar las decisiones del clasificador
    buscando_cruzar_dedos = True
    tiempo_deteccion_cruzar_dedos = None
    en_delay = False
    en_periodo_captura = False

    # ------------ Configuración Myo ---------------
    print("Conectando a Myo...")
    myo = Myo(mode=modo)
    myo.connect()
    print("Myo conectado.")

    emg_data = None  # Variable para almacenar temporalmente los datos EMG
    imu_data = None  # Variable para almacenar temporalmente los datos IMU

    def manejar_emg(emg, movimiento):
        nonlocal emg_data
        emg_data = emg  # Guardar los datos EMG para combinarlos luego
        print(f"Datos EMG recibidos: {emg}")

    def manejar_imu(quat, acc, gyro):
        nonlocal imu_data
        imu_data = np.concatenate((quat, acc, gyro), axis=None)  # Guardar los datos IMU combinados
        print(f"Datos IMU recibidos: {imu_data}")

    myo.add_emg_handler(manejar_emg)
    myo.add_imu_handler(manejar_imu)

    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    while recolectar:
        myo.run()

        # Solo clasificar si ambos datos EMG e IMU están presentes
        if emg_data is not None and imu_data is not None:
            print("Datos EMG e IMU disponibles para clasificar.")
            combined_data = np.concatenate((emg_data, imu_data), axis=None)
            etiqueta = clasificar_emg_imu(combined_data)
            decisiones.append(etiqueta)

            if buscando_cruzar_dedos and (time.time() - tiempo_inicio) >= t_espera:
                if verificar_cruzar_dedos(decisiones):
                    tiempo_deteccion_cruzar_dedos = time.time()
                    buscando_cruzar_dedos = False
                    en_delay = True
                    print("CRUZAR_DEDOS detectado, esperando 1 segundo antes de capturar")

            if en_delay and tiempo_deteccion_cruzar_dedos:
                if (time.time() - tiempo_deteccion_cruzar_dedos) >= t_espera:
                    en_delay = False
                    en_periodo_captura = True
                    tiempo_deteccion_cruzar_dedos = time.time()

            if en_periodo_captura and tiempo_deteccion_cruzar_dedos:
                if (time.time() - tiempo_deteccion_cruzar_dedos) >= t_captura_cruzar_dedos:
                    if time.time() - ultimo_tiempo_impresion >= 1:
                        etiqueta_mas_comun, porcentaje = mas_comun(decisiones)
                        llamada_2(etiqueta_mas_comun, porcentaje)
                        ultimo_tiempo_impresion = time.time()
                        decisiones = []
                        tiempo_deteccion_cruzar_dedos = None
                        en_periodo_captura = False
                        buscando_cruzar_dedos = True

        if (time.time() - tiempo_inicio) >= t_captura_total:
            recolectar = False

    print("Desconectando Myo...")
    myo.vibrate(2)
    myo.disconnect()
    print("Myo desconectado.")

if __name__ == '__main__':
    t_captura_total = 20  # Total de tiempo de recolección
    t_espera = 1  # Delay del principio de captación de datos y delay antes de capturar después de detectar "CRUZAR_DEDOS"
    t_captura_cruzar_dedos = 2  # Capturar datos durante 2 segundos después del Cruzar dedos
    modo = emg_mode.FILTERED
    print("Iniciando proceso de trabajador de datos...")
    p = multiprocessing.Process(target=trabajador_datos, args=(modo, t_captura_total, t_espera, t_captura_cruzar_dedos))
    p.start()
    p.join()  # Asegúrate de que el proceso hijo termine antes de finalizar el programa principal
