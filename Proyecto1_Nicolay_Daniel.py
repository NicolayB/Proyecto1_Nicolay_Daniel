import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
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
plt.show()

# Distribución de la renta de bicicletas
plt.hist(data["Rented Bike Count"])
plt.title("Distribución de la renta de bicicletas")
plt.show()

# Comportamiento de las variables explicativas y la variable de interés
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
# Convertir las fechas en float
X["Date"] = (X["Date"]-pd.Timestamp("2017-11-30"))/pd.Timedelta(days=1)
x_train, x_test, y_train, y_test = train_test_split(X, Y , random_state=0)
x_train = sm.add_constant(x_train)
model = sm.OLS(y_train, x_train).fit()
print(model.summary())

# Prueba de multicolinealidad
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
for i in range(0,X.shape[1]):
    print(f"VIF de {X.columns[i]}:",vif[i])

# Puntos de influencia
# Distancia de cook
model_cook = model.get_influence().cooks_distance[0]

n = x_train.shape[0]

# Umbral
critical_d = 4/n
print("Umbral con distancia de Cook:", critical_d)

# Posibles outliers con influencia
outliers = model_cook > critical_d
print(x_train.index[outliers])

# Se eliminan los outliers de los datos de entrenamiento
x_train_nuevo = x_train.drop(x_train.index[outliers], axis=0)
y_train_nuevo = y_train.drop(y_train.index[outliers], axis=0)

# Inicialización modelo óptimo
X2 = X.drop(x_train.index[outliers], axis=0)
Y2 = Y.drop(y_train.index[outliers], axis=0)

model_nuevo = sm.OLS(y_train_nuevo,x_train_nuevo).fit()
print(model_nuevo.summary())

# Prueba de multicolinealidad
vif = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
for i in range(0,X.shape[1]):
    print(f"VIF de {X.columns[i]}:",vif[i])

# Promedio Temperature y Dew point temperature
x_train = x_train_nuevo
y_train = y_train_nuevo
#x_train["Temperature_avr"] = (x_train["Temperature(C)"]+x_train["Dew point temperature(C)"])/2
# Eliminación de la variables
x_train = x_train.drop(["Date","Temperature(C)", "Visibility (10m)"], axis=1)
model_prueba1 = sm.OLS(y_train, x_train).fit()
print(model_prueba1.summary())

x_train = x_train.drop(["Autumn"], axis=1)
model_prueba1 = sm.OLS(y_train, x_train).fit()
print(model_prueba1.summary())

# Modelo ideal 1
X_ideal1 = X2.drop(["Date","Temperature(C)", "Visibility (10m)","Autumn"], axis=1)
Y_ideal1 = Y2

x_train, x_test, y_train, y_test = train_test_split(X_ideal1, Y_ideal1, random_state=0)
x_train = sm.add_constant(x_train)
model_ideal1 = sm.OLS(y_train, x_train).fit()
print(model_ideal1.summary())