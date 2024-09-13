import os
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
# Nombre de la subcarpeta para guardar las gráficas
subcarpeta = 'grafica'

# Crear la subcarpeta si no existe
if not os.path.exists(subcarpeta):
    os.makedirs(subcarpeta)

try:
    with open(archivo, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Buscar la combinación de gestos y la accuracy
            match_gestos = patron_gestos.search(line)
            match_accuracy = patron_accuracy.search(line)
            if match_gestos:
                gestos = match_gestos.group(1).strip().strip("['']").split("', '")
                combinaciones.append(gestos)
            if match_accuracy:
                accuracy = float(match_accuracy.group(2))
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

            # Graficar los resultados generales
            plt.figure(figsize=(10, 6))
            for index, row in best_combinations.iterrows():
                plt.scatter(len(row['Combinación']), row['Accuracy'], marker='o', s=100, label=f'{row["Combinación"]}')
            plt.xlabel('Número de gestos')
            plt.ylabel('Accuracy')
            plt.title('Mejor accuracy por número de gestos')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small')
            plt.grid(True)
            plt.subplots_adjust(bottom=0.3)
            plt.savefig(os.path.join(subcarpeta, 'mejor_accuracy_por_numero_de_gestos.png'), bbox_inches='tight')
            plt.show()

            # Graficar los diez mejores accuracy para cada cantidad de gestos
            for num_gestos in sorted(df['Combinación'].apply(len).unique()):
                df_filtered = df[df['Combinación'].apply(len) == num_gestos]
                top_ten = df_filtered.nlargest(10, 'Accuracy')

                plt.figure(figsize=(10, 6))
                colors = plt.cm.tab10(range(len(top_ten)))  # Colores para los puntos
                for i, (index, row) in enumerate(top_ten.iterrows()):
                    plt.scatter([i + 1], [row['Accuracy']], color=colors[i], marker='o', s=100, label=f'{row["Combinación"]}')
                plt.xlabel('Ranking')
                plt.ylabel('Accuracy')
                plt.title(f'Top 10 Accuracy para {num_gestos} gestos')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize='small')
                plt.grid(True)
                plt.subplots_adjust(bottom=0.4)  # Ajustar el margen inferior

                # Guardar la gráfica en la subcarpeta
                plt.savefig(os.path.join(subcarpeta, f'top_10_accuracy_para_{num_gestos}_gestos.png'), bbox_inches='tight')
                plt.show()

except FileNotFoundError:
    print(f"No se encontró el archivo: {archivo}")
except Exception as e:
    print(f"Ocurrió un error: {e}")
