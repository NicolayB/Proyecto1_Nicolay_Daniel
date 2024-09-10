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

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px

path = os.getcwd()
print(path)

data = pd.read_csv("SeoulBikeData_utf8.csv")

# Pasar la variable de interés al final del DataFrame
v_interes = data.pop("Rented Bike Count")
data["Rented Bike Count"] = v_interes
print(data)

# Convertir la columna fecha en formato fecha
data["Date"] = pd.to_datetime(data["Date"], format="%d/%m/%Y")
print(data)

# Convertir en enteros las variables tipo objeto
data["Holiday"] = data["Holiday"].map({"Holiday": 1, "No Holiday":0})
data["Functioning Day"] = data["Functioning Day"].map({"Yes": 1, "No":0})
print(data)

v_interes = data.pop("Rented Bike Count")
data["Rented Bike Count"] = v_interes
print(data)

# Creación de variables explicativas y variable de interés 
col = "Rented Bike Count"
X = data.drop(col, axis=1)
Y = data[col]

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

#Eliminación de variables con una significancia relevante
features = ['Temperature(C)', 'Wind speed (m/s)', 'Rainfall(mm)', 'Holiday' , 'Functioning Day' ]
X = data[features]
y = data['Rented Bike Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# agregar constante explíticamente
X_train = sm.add_constant(X_train)
# regresión usando mínimos cuadrados ordinarios (ordinary least squares - OLS) 
model = sm.OLS(y_train, X_train).fit()

# disntacia de Cook
model_cooksd = model.get_influence().cooks_distance[0]
# get length of df to obtain n
n = X_train.shape[0]
# umbral
critical_d = 4/n

# puntos que podrían ser ourliers con alta influencia
outliers = model_cooksd > critical_d
#print(X_train.index[outliers], "\n", model_cooksd[outliers])

# Se eliminan los outliers de los datos de entrenamiento
x_train_nuevo = X_train.drop(X_train.index[outliers], axis=0)
y_train_nuevo = y_train.drop(y_train.index[outliers], axis=0)

model_nuevo = sm.OLS(y_train_nuevo,x_train_nuevo).fit()
#print(model_nuevo.summary())

#Aplicar transformación de raíz cuadrada
y_transformed = np.sqrt(y_train_nuevo)

# Carga del modelo entrenado
model_ideal = sm.OLS(y_transformed, x_train_nuevo).fit()


# Crear la aplicación Dash
app = dash.Dash(__name__)


# Layout de la aplicación
app.layout = html.Div([
    # Sección de visualizaciones
    html.Div([
        dcc.Graph(id='histogram'),
        dcc.Graph(id='bikes-by-season'),
        dcc.Graph(id='bikes-over-time')
    ])
])

# Callback para actualizar el histograma
@app.callback(
    Output('histogram', 'figure'),
    [Input('submit-button', 'n_clicks')]
)
def update_histogram(n_clicks):
    fig = px.histogram(data, x="Rented Bike Count", nbins=20, title="Histograma de Bicicletas Rentadas")
    return fig

# Callback para actualizar el gráfico de bicicletas por estación
@app.callback(
    Output('bikes-by-season', 'figure'),
    [Input('submit-button', 'n_clicks')]
)

def update_boxplot(n_clicks):
    fig = px.box(data, x="Seasons", y="Rented Bike Count",
                title="Relación entre la estación del año y el número de bicicletas rentadas")
    return fig

# Callback para actualizar el gráfico de línea de bicicletas a lo largo del tiempo
@app.callback(
    Output('bikes-over-time', 'figure'),
    [Input('submit-button', 'n_clicks')]
)
def update_time_series(n_clicks):
    fig = px.line(data, x="Date", y="Rented Bike Count", title="Evolución de las Bicicletas Rentadas")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)