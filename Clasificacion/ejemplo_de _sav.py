import pandas as pd
import pyreadstat

# Cargar el archivo .sav
df, meta = pyreadstat.read_sav("C:\\Users\\anita\\Desktop\\Nueva carpeta\\EMG-model.sav")

# Ver las primeras filas del dataset
print(df.head())
