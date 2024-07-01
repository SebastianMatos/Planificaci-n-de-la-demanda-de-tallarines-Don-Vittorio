import os
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
import matplotlib.pyplot as plt
import warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas.tseries.offsets import DateOffset
import itertools
warnings.filterwarnings("ignore")

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear el directorio de subida si no existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Variable global para almacenar los datos cargados desde el archivo Excel
loaded_data = None

@app.route('/', methods=['GET', 'POST'])
def index():
    global loaded_data
    global data_table

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Leer el archivo Excel
            df = pd.read_excel(file_path)

            # Crear la lista de salida
            data = ["Periodo,Demanda de fideos,Kilos exportados,Precio en promocion,Pedidos semolas de trigo,Pedidos Harina de trigo,Venta unitaria de pastas al interior del pais,Pedidos de mercaderia,Indice de distribucion numerica,Participacion en la publicidad"]
            for _, row in df.iterrows():
                formatted_row = f"{row['Periodo']},{row['Demanda de fideos']},{row['Kilos exportados']},{row['Precio en promocion']},{row['Pedidos semolas de trigo']},{row['Pedidos Harina de trigo']},{row['Venta unitaria de pastas al interior del pais']},{row['Pedidos de mercaderia']},{row['Indice de distribucion numerica']},{row['Participacion en la publicidad']}"
                data.append(formatted_row)

            # Convertir a DataFrame y ajustar tipos de datos
            data_list = [line.split(",") for line in data[1:]]
            df = pd.DataFrame(data_list, columns=data[0].split(","))
            df["Demanda de fideos"] = pd.to_numeric(df["Demanda de fideos"])

            # Convertir la columna "Periodo" a formato de fecha y hora
            df["Periodo"] = pd.to_datetime(df["Periodo"], errors='coerce', format='%Y-%m')

            # Eliminar filas con valores de fecha y hora no válidos
            df = df.dropna(subset=['Periodo'])

            # Establecer la columna "Periodo" como índice
            df.set_index("Periodo", inplace=True)

            # Guardar los datos cargados globalmente
            loaded_data = df
            data_table=data[1:]

            return render_template('index.html', data=data[1:], show_data_button=True, show_predict_button=True)

    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global loaded_data

    if loaded_data is None:
        return 'No data loaded'
    # División de datos en entrenamiento y prueba
    train_size = int(len(loaded_data) * 0.8)
    train_data, test_data = loaded_data[:train_size], loaded_data[train_size:]

    # Búsqueda de hiperparámetros óptimos
    p = d = q = range(0, 3)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None

    warnings.filterwarnings("ignore") # Ignorar warnings de ajuste del modelo
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = SARIMAX(train_data["Demanda de fideos"], order=param, seasonal_order=param_seasonal)
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
            except:
                continue

    print(f'Mejor AIC: {best_aic}')
    print(f'Mejor pdq: {best_pdq}')
    print(f'Mejor seasonal pdq: {best_seasonal_pdq}')

    # Definición del modelo SARIMA con los mejores hiperparámetros
    sarima_model = SARIMAX(train_data["Demanda de fideos"], order=best_pdq, seasonal_order=best_seasonal_pdq)

    # Ajuste del modelo SARIMA
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sarima_model_fit = sarima_model.fit(disp=False)

    # Predicciones para fechas futuras (marzo 2024 - diciembre 2025)
    future_dates = [loaded_data.index[-1] + DateOffset(months=x+1) for x in range(21)]
    future_df = pd.DataFrame(index=future_dates, columns=loaded_data.columns)
    extended_df = pd.concat([loaded_data, future_df])

    future_predictions = sarima_model_fit.get_forecast(steps=len(extended_df) - len(loaded_data))
    future_mean = future_predictions.predicted_mean
    future_ci = future_predictions.conf_int()

    # Evaluación del modelo en datos de prueba
    predictions = sarima_model_fit.get_forecast(steps=len(test_data))
    predicted_mean = predictions.predicted_mean
    mse = mean_squared_error(test_data["Demanda de fideos"], predicted_mean)
    print(f"Error cuadrático medio (MSE) en datos de prueba: {mse}")

    # Crear tabla de comparación
    comparison_df = pd.DataFrame({
        "Periodo": test_data.index,
        "Observado": test_data["Demanda de fideos"],
        "Predicho": predicted_mean
    })

    print("\nComparación entre lo Observado y lo Predicho:")
    print(comparison_df)

    predicciones_path = os.path.join(app.config['UPLOAD_FOLDER'], 'predicciones.png')
    observado_path = os.path.join(app.config['UPLOAD_FOLDER'], 'observado.png')
    
    # Visualización de resultados
    plt.figure(figsize=(12, 8))
    plt.plot(loaded_data.index, loaded_data["Demanda de fideos"], label="Histórico")
    plt.plot(extended_df.index[len(loaded_data):], future_mean, label="Predicción SARIMA", color='red')
    plt.fill_between(extended_df.index[len(loaded_data):], future_ci.iloc[:, 0], future_ci.iloc[:, 1], color='k', alpha=0.1)
    plt.xlabel("Periodo")
    plt.ylabel("Demanda de fideos")
    plt.title("Predicción de la demanda de fideos con SARIMA")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig(predicciones_path)
    plt.close()

    # Resultados de las pruebas en datos numéricos
    print("\nResultados de las pruebas:")
    print(f"Error cuadrático medio (MSE) en datos de prueba: {mse}\n")

    # Visualización de resultados de las pruebas
    plt.figure(figsize=(10, 6))
    plt.plot(test_data.index, test_data["Demanda de fideos"], label="Datos Observados")
    plt.plot(test_data.index, predicted_mean, label="Predicción SARIMA", color='red')
    plt.fill_between(test_data.index, future_ci.iloc[:, 0][:len(test_data)], future_ci.iloc[:, 1][:len(test_data)], color='k', alpha=0.1)
    plt.xlabel("Periodo")
    plt.ylabel("Demanda de fideos")
    plt.title("Comparación de Predicción SARIMA con Datos Observados")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.savefig(observado_path)
    plt.close()

    return render_template('index.html', data=data_table ,graph1='uploads/observado.png', graph2='uploads/predicciones.png')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv("PORT", default=5000))
