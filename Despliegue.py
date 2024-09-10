import os
import pandas as pd
import numpy as np
import requests
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


# Cargar archivo desde el repositorio de Git
data = pd.read_csv('https://raw.githubusercontent.com/NicolayB/Proyecto1_Nicolay_Daniel/main/SeoulBikeData_utf8.csv') 

#data = pd.read_csv("SeoulBikeData_utf8.csv")

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
    # Sección de filtros
    dcc.Dropdown(
        id='holiday-filter',
        options=[
            {'label': 'Todos los días', 'value': 'all'},
            {'label': 'Días festivos', 'value': 'holiday'},
            {'label': 'Días no festivos', 'value': 'non-holiday'}
        ],
        value='all'
    ),
    # Sección de visualizaciones
    html.Div(id='graphs-container', children=[
        dcc.Graph(id='histogram'),
        dcc.Graph(id='bikes-by-season'),
        dcc.Graph(id='bikes-over-time'),
        dcc.Graph(id='regression-graph')
    ])
])

# Callback para actualizar el histograma
@app.callback(
    Output('histogram', 'figure'),
    [Input('holiday-filter', 'value')]
)

def update_histogram(selected_holiday):
    filtered_data = data
    if selected_holiday == 'holiday':
        filtered_data = data[data['Holiday'] == True]  # Ajusta la columna 'Holiday' según tu DataFrame
    elif selected_holiday == 'non-holiday':
        filtered_data = data[data['Holiday'] == False]

    fig = px.histogram(filtered_data, x="Rented Bike Count", nbins=20, title="Histograma de Bicicletas Rentadas")
    return fig

# Callback para actualizar el gráfico de bicicletas por estación
@app.callback(
    Output('bikes-by-season', 'figure'),
    [Input('holiday-filter', 'value')]
)
def update_boxplot(selected_holiday):
    filtered_data = data
    if selected_holiday == 'holiday':
        filtered_data = data[data['Holiday'] == True]
    elif selected_holiday == 'non-holiday':
        filtered_data = data[data['Holiday'] == False]

    fig = px.box(filtered_data, x="Seasons", y="Rented Bike Count",
                 title="Relación entre la estación del año y el número de bicicletas rentadas")
    return fig

# Callback para actualizar el gráfico de línea de bicicletas a lo largo del tiempo
@app.callback(
    Output('bikes-over-time', 'figure'),
    [Input('holiday-filter', 'value')]
)
def update_time_series(selected_holiday):
    filtered_data = data
    if selected_holiday == 'holiday':
        filtered_data = data[data['Holiday'] == True]
    elif selected_holiday == 'non-holiday':
        filtered_data = data[data['Holiday'] == False]

    fig = px.line(filtered_data, x="Date", y="Rented Bike Count", title="Evolución de las Bicicletas Rentadas")
    return fig

# Callback para actualizar el gráfico de regresión
@app.callback(
    Output('regression-graph', 'figure'),
    [Input('holiday-filter', 'value')]
)

def update_regression_graph(selected_holiday):
    # Realizar predicciones
    predictions = model.predict(x_train_nuevo)
    df_predictions = pd.DataFrame({'Valor objetivo': y_transformed, 'predictions': predictions})

    # Crear el gráfico
    fig = px.scatter(df_predictions, y='Valor objetivo',
                    trendline='ols', trendline_color_override='red',
                    title='Regresión Lineal')
    return fig

if __name__ == '__main__':
    app.run_server( host = " 0.0.0.0 ", debug = True )