import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = os.getcwd()
print(path)

data = pd.read_csv("SeoulBikeData_utf8.csv")
print(data.head())

# Pasar la variable de interés al final del DataFrame
v_interes = data.pop("Rented Bike Count")
data["Rented Bike Count"] = v_interes
print(data)

# Convertir la columna fecha en formato fecha
data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
print(data)

# Valores faltantes
print(data.isnull().sum())

# Estadísticas descriptivas
print(data.describe())

# Tipos de datos
types = data.dtypes.value_counts()
print(f"Tipos de variables en los datos: \n{types}")

print(data.columns)

# Variables tipo objeto
print(data["Seasons"].unique())
print(data["Seasons"].value_counts())
print(data["Holiday"].unique())
print(data["Holiday"].value_counts())
print(data["Functioning Day"].unique())
print(data["Functioning Day"].value_counts())
# Convertir en enteros las variables tipo objeto
data["Holiday"] = data["Holiday"].map({"Holiday": 1, "No Holiday":0})
data["Functioning Day"] = data["Functioning Day"].map({"Yes": 1, "No":0})
print(data)
# Variable Seasons
data["Winter"] = data["Seasons"].map({"Winter":1, "Spring":0, "Summer":0, "Autumn":0})
data["Spring"] = data["Seasons"].map({"Winter":0, "Spring":1, "Summer":0, "Autumn":0})
data["Summer"] = data["Seasons"].map({"Winter":0, "Spring":0, "Summer":1, "Autumn":0})
data["Autumn"] = data["Seasons"].map({"Winter":0, "Spring":0, "Summer":0, "Autumn":1})
data.drop("Seasons", axis=1, inplace=True)

v_interes = data.pop("Rented Bike Count")
data["Rented Bike Count"] = v_interes
print(data)

# Verificar que se hizo la transformación
print(data["Winter"].value_counts())
print(data["Spring"].value_counts())
print(data["Summer"].value_counts())
print(data["Autumn"].value_counts())
print(data["Holiday"].unique())
print(data["Holiday"].value_counts())
print(data["Functioning Day"].unique())
print(data["Functioning Day"].value_counts())

# Creación de variables explicativas y variable de interés 
col = "Rented Bike Count"
X = data.drop(col, axis=1)
Y = data[col]