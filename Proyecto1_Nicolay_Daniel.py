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