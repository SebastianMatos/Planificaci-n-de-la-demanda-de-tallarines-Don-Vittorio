import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
import warnings
import itertools

# Configuración de semilla aleatoria para reproducibilidad
np.random.seed(42)

# Carga de datos desde xlsx
file_path = 'data.xlsx'
df = pd.read_excel(file_path)

# Preprocesamiento de datos
df["Periodo"] = pd.to_datetime(df["Periodo"], format='%Y-%m')
df.set_index("Periodo", inplace=True)

# Exploración de datos
print(df.describe())

# División de datos en entrenamiento y prueba
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

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
future_dates = [df.index[-1] + DateOffset(months=x+1) for x in range(21)]
future_df = pd.DataFrame(index=future_dates, columns=df.columns)
extended_df = pd.concat([df, future_df])

future_predictions = sarima_model_fit.get_forecast(steps=len(extended_df) - len(df))
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

# Visualización de resultados
plt.figure(figsize=(12, 8))
plt.plot(df.index, df["Demanda de fideos"], label="Histórico")
plt.plot(extended_df.index[len(df):], future_mean, label="Predicción SARIMA", color='red')
plt.fill_between(extended_df.index[len(df):], future_ci.iloc[:, 0], future_ci.iloc[:, 1], color='k', alpha=0.1)
plt.xlabel("Periodo")
plt.ylabel("Demanda de fideos")
plt.title("Predicción de la demanda de fideos con SARIMA")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

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
plt.show()

# Guardar el modelo entrenado (opcional)
# sarima_model_fit.save("sarima_model.pkl")

# Guardar la tabla de comparación en un archivo CSV
comparison_df.to_csv("observado_vs_predicho.csv", index=False)