import time
import numpy as np
import joblib
import pandas as pd
from pyomyo import Myo, emg_mode
import os

# Lista de gestos a probar
#gestos = ['BAJAR_DIAL', 'C', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FIST', 'GIRO_IN', 'GIRO_OUT', 'I','JUNTOS', 'L', 'REST', 'SUBIR_DIAL', 'TRES', 'UNO', 'WAVE_IN', 'WAVE_OUT']
#gestos = [ 'C', 'CRUZAR_DEDOS', 'CUATRO', 'DOS', 'FIST', 'I','JUNTOS', 'L', 'REST', 'TRES', 'UNO', 'WAVE_IN', 'WAVE_OUT']
#gestos = ['C','I','L', 'JUNTOS','UNO','DOS','TRES','CUATRO']


# Cargar el clasificador y el escalador preentrenados desde tu path
clasificador = joblib.load('Clasificadores de pruebas/Clasificadores/clasificador_50000_no_IMU_17_param.pkl')
escalador = joblib.load('Clasificadores de pruebas/Escaladores/scaler_50000.pkl')

# Definir la ruta del archivo de resultados
excel_path = 'resultados_memoria.xlsx'


def limpieza(emg_data):
    datos_rectificados = np.abs(emg_data)
    datos_rectificados = datos_rectificados.reshape(1, -1)
    datos_escalados = escalador.transform(datos_rectificados)
    return datos_escalados


def clasificar_emg(emg_data):
    datos_combinados = limpieza(emg_data)
    etiqueta = clasificador.predict(datos_combinados)[0]
    return etiqueta


def main():
    # Solicitar la posición del brazo
    posicion_brazo = input("Introduce la posición del brazo (Estirado, 90 grados, Mano Abajo, Mano Arriba): ")

    # Conectar al Myo
    myo = Myo(mode=emg_mode.FILTERED)
    myo.connect()

    emg_data = None

    def manejar_emg(emg, movimiento):
        nonlocal emg_data
        emg_data = emg

    myo.add_emg_handler(manejar_emg)


    myo.set_leds([0, 128, 0], [0, 128, 0])
    myo.vibrate(1)

    tiempo_total = 5  # Tiempo para cada gesto en segundos
    resultados = {gesto: [] for gesto in gestos}
    detecciones_intervalo = []

    try:
        for gesto in gestos:
            print(f"Dos segundos de descanso, luego gesto {gesto}")
            time.sleep(2)
            print(f"Realiza el gesto: {gesto}")
            tiempo_inicio = time.time()
            detecciones = {g: 0 for g in gestos}
            total_predicciones = 0

            while time.time() - tiempo_inicio < tiempo_total:
                myo.run()
                if emg_data is not None :
                    etiqueta = clasificar_emg(emg_data)
                    if etiqueta in detecciones:
                        detecciones[etiqueta] += 1
                    total_predicciones += 1

            # Calcular porcentaje de aciertos para el gesto actual
            porcentaje_aciertos = (detecciones[gesto] / total_predicciones) * 100 if total_predicciones > 0 else 0
            resultados[gesto].append(porcentaje_aciertos)

            # Determinar el gesto más detectado
            gesto_mas_detectado = max(detecciones, key=detecciones.get)
            detecciones_intervalo.append(gesto_mas_detectado)

            print(f"Gesto {gesto} completado. Porcentaje de aciertos: {porcentaje_aciertos:.2f}%")
            print(f"Gesto más detectado: {gesto_mas_detectado}")

    except KeyboardInterrupt:
        print("Interrupción del usuario, finalizando...")
    finally:
        myo.disconnect()
        print("Myo desconectado.")

    # Guardar resultados en el archivo Excel
    nombre_hoja = '50000_no_IMU_17_C_100'

    if not os.path.exists(excel_path):
        # Si el archivo no existe, crearlo con los datos iniciales
        df_resultados = pd.DataFrame(resultados)
        df_resultados.index = [f'{posicion_brazo}']
        df_detecciones_intervalo = pd.DataFrame([detecciones_intervalo],columns=gestos)
        df_detecciones_intervalo.index = [f'gesto_mas_detectado']


        df_final = pd.concat([df_resultados, df_detecciones_intervalo], axis=0)
        df_final.to_excel(excel_path, sheet_name=nombre_hoja)
    else:
        # Si el archivo ya existe, cargarlo y actualizar los resultados
        with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='overlay') as writer:
            try:
                df_existente = pd.read_excel(excel_path, sheet_name=nombre_hoja, index_col=0)
            except ValueError:
                df_existente = pd.DataFrame()

            df_resultados = pd.DataFrame(resultados)
            df_resultados.index = [f'{posicion_brazo}']
            df_detecciones_intervalo = pd.DataFrame([detecciones_intervalo], columns=gestos)
            df_detecciones_intervalo.index = [f'gesto_mas_detectado']


            df_final = pd.concat([df_existente, df_resultados, df_detecciones_intervalo], axis=0)
            df_final.to_excel(writer, sheet_name=nombre_hoja)

        print(f"Resultados guardados en la hoja '{nombre_hoja}' para la posición de brazo: {posicion_brazo}")


if __name__ == '__main__':
    main()
