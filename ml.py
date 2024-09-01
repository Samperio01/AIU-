from flask import Flask, render_template, request
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Entrena el modelo inicialmente para evitar cargarlo en cada solicitud
def train_model():
    # URL de la API para obtener datos climáticos de Londres
    api_url = "http://api.worldweatheronline.com/premium/v1/weather.ashx"
    params = {
        'key': '679f5acc04e94339b0314215243108',  # Tu API key
        'q': 'London',
        'format': 'json',
        'num_of_days': '5'
    }

    # Realizar la solicitud a la API
    response = requests.get(api_url, params=params)
    weather_data = response.json()

    # Extraer datos relevantes para el DataFrame
    current_condition = weather_data['data']['current_condition'][0]
    forecast = weather_data['data']['weather'][0]['hourly']

    # Crear un DataFrame a partir de las condiciones actuales y el pronóstico
    df = pd.DataFrame(forecast)

    # Seleccionar columnas relevantes para el modelo
    columns_of_interest = ['tempC', 'windspeedKmph', 'humidity', 'visibility']
    cleaned_data = df[columns_of_interest].dropna()

    # Definir variables independientes (X) y la variable dependiente (y)
    X = cleaned_data.drop('tempC', axis=1)
    y = cleaned_data['tempC']

    # Entrenar un modelo de Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# Entrenar el modelo al inicio
model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Supongamos que queremos predecir la temperatura usando datos ingresados por el usuario
    windspeed = float(request.form['windspeed'])
    humidity = float(request.form['humidity'])
    visibility = float(request.form['visibility'])

    # Crear un DataFrame con los datos ingresados
    input_data = pd.DataFrame([[windspeed, humidity, visibility]],
                              columns=['windspeedKmph', 'humidity', 'visibility'])

    # Realizar la predicción
    prediction = model.predict(input_data)[0]

    # Mostrar el resultado en la página
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
