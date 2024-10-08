import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import scipy.stats as stats

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

# Pasar la variable de interés al final del DataFrame
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

# Fechas de cada estación
invierno = [data[data["Winter"]==1]["Date"].min(), data[data["Winter"]==1]["Date"].max()]
primavera = [data[data["Spring"]==1]["Date"].min(), data[data["Spring"]==1]["Date"].max()]
verano = [data[data["Summer"]==1]["Date"].min(), data[data["Summer"]==1]["Date"].max()]
otoño = [data[data["Autumn"]==1]["Date"].min(), data[data["Autumn"]==1]["Date"].max()]
print(f"Invierno: {invierno}")
print(f"Primavera: {primavera}")
print(f"Verano: {verano}")
print(f"Otoño: {otoño}")

# Creación de variables explicativas y variable de interés 
col = "Rented Bike Count"
X = data.drop(col, axis=1)
Y = data[col]

# Graficar la renta de bicicletas en cada fecha por hora
data.plot(x="Date",y="Rented Bike Count")
plt.title("Renta de bicicletas por cada hora de cada fecha")
plt.xlabel("Fecha")
plt.ylabel("Bicicletas rentadas")
plt.grid()
plt.show()

# Graficar el comportamiento de la renta de bicicletas promedio por fecha
rented_by_date = data.groupby("Date")["Rented Bike Count"].describe()
plt.figure(figsize=(15,8))
plt.plot(rented_by_date["mean"])
plt.title("Renta de bicicletas promedio por fecha")
plt.xlabel("Fecha")
plt.ylabel("Bicicletas rentadas")
plt.grid()
plt.show()

# Distribución de la renta de bicicletas
plt.hist(data["Rented Bike Count"], bins=30)
plt.title("Distribución de la renta de bicicletas")
plt.grid()
plt.show()
"""
"""# Comportamiento de las variables explicativas y la variable de interés
sns.pairplot(data=data)
plt.show()

# Mapa de calor de la correlación entre variables
correlacion = data.corr()
plt.figure(figsize=(15,8))
sns.heatmap(correlacion, cmap="Blues", annot=True)
plt.title("Correlación entre variables")
plt.show()

# Mapa de calor de correlación de las variables con la variable de interés
correlacion2 = pd.DataFrame(X.corrwith(Y))
sns.heatmap(correlacion2, cmap="Blues", annot=True)
plt.title("Correlación de las variables con la variable de interés")
plt.show()

sns.pairplot(data, x_vars=data[data.columns[[1,2,3]]], y_vars="Rented Bike Count", height=7, kind="reg", plot_kws={"line_kws":{"color":"red"}})
sns.pairplot(data, x_vars=data[data.columns[[4,5,6,7]]], y_vars="Rented Bike Count", height=7, kind="reg", plot_kws={"line_kws":{"color":"red"}})
sns.pairplot(data, x_vars=data[data.columns[[8,9,10,11]]], y_vars="Rented Bike Count", height=7, kind="reg", plot_kws={"line_kws":{"color":"red"}})
sns.pairplot(data, x_vars=data[data.columns[[12,13,14,15]]], y_vars="Rented Bike Count", height=7, kind="reg", plot_kws={"line_kws":{"color":"red"}})
plt.show()

# Modelo de regresión

#Seleccion de variables para el modelo ideal 1 
features = ['Temperature(C)', 'Humidity(%)', 'Wind speed (m/s)', 'Visibility (10m)', 'Rainfall(mm)', 'Snowfall (cm)', 'Holiday' , 'Functioning Day' ]
X = data[features]
Y = data['Rented Bike Count']
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1)

# Agregar constante explíticamente
X_train = sm.add_constant(X_train)
# regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS) 
model = sm.OLS(y_train, X_train).fit()
# resumen de resultados
print(model.summary())

#Eliminación de variables con una significancia relevante
features = ['Temperature(C)', 'Wind speed (m/s)', 'Rainfall(mm)', 'Holiday' , 'Functioning Day' ]
X = data[features]
y = data['Rented Bike Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# agregar constante explíticamente
X_train = sm.add_constant(X_train)
# regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS) 
model = sm.OLS(y_train, X_train).fit()
# resumen de resultados
print(model.summary())

# disntacia de Cook
model_cooksd = model.get_influence().cooks_distance[0]
# get length of df to obtain n
n = X_train.shape[0]
# umbral
critical_d = 4/n
print('Umbral con distancia de Cook:', critical_d)

# puntos que podrían ser ourliers con alta influencia
outliers = model_cooksd > critical_d
print(X_train.index[outliers], "\n", model_cooksd[outliers])

# Se eliminan los outliers de los datos de entrenamiento
x_train_nuevo = X_train.drop(X_train.index[outliers], axis=0)
y_train_nuevo = y_train.drop(y_train.index[outliers], axis=0)

model_nuevo = sm.OLS(y_train_nuevo,x_train_nuevo).fit()
print(model_nuevo.summary())

#Aplicar transformación de raíz cuadrada
y_transformed = np.sqrt(y_train_nuevo)

# Ajusta el modelo con los datos transformados
model_ideal = sm.OLS(y_transformed, x_train_nuevo).fit()
print(model_ideal.summary())

# Prueba Shapiro-Wilk
shapiro, p_value = stats.shapiro(model_ideal.resid)
print(f"p_value: {p_value}")

# Prueba de multicolinealidad
vif = [variance_inflation_factor(x_train_nuevo.values, i) for i in range(x_train_nuevo.shape[1])]
for i in range(len(vif)):
    print(f"VIF de {x_train_nuevo.columns[i]}: {vif[i]}")
