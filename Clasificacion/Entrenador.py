import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Comenzamos definiendo el path de los datos de entrada
path = 'C:\\Users\\anita\\Documents\\GitHub\\Proyecto-TFM\\Preprocesado\\datos\\'

# Cargar datos desde el archivo Excel
df = pd.read_excel(path + 'Datos_Limpios.xlsx')
df.dropna(inplace=True)

# Tamaño de la muestra inicial
tamano_muestra_inicial = len(df)
print(f"Tamaño de la muestra inicial: {tamano_muestra_inicial}")

# Separar características (X) de etiquetas (y)
X = df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última
y = df.iloc[:, -1].values   # Todas las filas, solo la última columna

# Separamos los datos en datos de entrenamiento y de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Guardar el scaler
joblib.dump(scaler, 'scaler.pkl')
print("Scaler guardado como 'scaler.pkl'")

# Crear clasificador SVM con núcleo polinomial
model = svm.SVC(C=100, coef0=1.0, degree=5, gamma='scale', kernel='poly')

# Entrenar el clasificador
print('Comienza el entrenamiento del modelo SVM...')
model.fit(X_train_scaled, y_train)

# Guardar el modelo entrenado en un archivo
joblib.dump(model, 'entrenado_svm_poly_model.pkl')
print("Modelo guardado como 'entrenado_svm_poly_model.pkl'")

# Predecir etiquetas para los datos de prueba
y_pred = model.predict(X_test_scaled)

# Evaluar la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo SVM: {accuracy:.2f}")

# Obtener el informe de clasificación
report = classification_report(y_test, y_pred)
print("Informe de clasificación:\n", report)
