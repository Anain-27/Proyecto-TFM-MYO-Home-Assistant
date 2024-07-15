import re
import matplotlib.pyplot as plt
import pandas as pd

# Definir patrones para buscar las líneas relevantes en el archivo
patron_gestos = re.compile(r"CombinaciÃ³n de gestos: \[(.*?)\]")
patron_accuracy = re.compile(r"Accuracy para combinaciÃ³n \[(.*?)\]: (\d+\.\d+)")

# Inicializar listas para almacenar los datos
combinaciones = []
accuracies = []

# Nombre del archivo de texto a procesar
archivo = 'accuracy_1000.txt'

try:
    with open(archivo, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Buscar la combinación de gestos y la accuracy
            match_gestos = patron_gestos.search(line)
            if match_gestos:
                gestos = match_gestos.group(1).strip().strip("['']").split("', '")
                accuracy = float(match_gestos.group(2))
                print(f"Encontrada combinación de gestos: {gestos}")
                print(f"Accuracy encontrada: {accuracy}")
                combinaciones.append(gestos)
                accuracies.append(accuracy)

    # Crear un DataFrame con los datos obtenidos
    if combinaciones and accuracies:  # Verificar que haya datos válidos
        df = pd.DataFrame({'Combinación': combinaciones, 'Accuracy': accuracies})

        # Encontrar la combinación con el mejor accuracy para cada cantidad de gestos
        if not df.empty:
            best_combinations = df.loc[df.groupby(df['Combinación'].apply(len))['Accuracy'].idxmax()]

            # Mostrar la mejor combinación para cada cantidad de gestos
            print("Mejor combinación para cada cantidad de gestos:")
            print(best_combinations)

            # Graficar los resultados
            plt.figure(figsize=(10, 6))
            for index, row in best_combinations.iterrows():
                plt.scatter(len(row['Combinación']), row['Accuracy'], marker='o', s=100, label=f'{row["Combinación"]}')
            plt.xlabel('Número de gestos')
            plt.ylabel('Accuracy')
            plt.title('Mejor accuracy por número de gestos')
            plt.legend()
            plt.grid(True)
            plt.show()

except FileNotFoundError:
    print(f"No se encontró el archivo: {archivo}")
except Exception as e:
    print(f"Ocurrió un error: {e}")
