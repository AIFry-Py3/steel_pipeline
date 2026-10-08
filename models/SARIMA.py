import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_csv("data/text_extracted_data/lag_rolling_cleaned.csv")

data = df[[
    'period_year', 'period_month', 'consumption_value',
    'lag_1', 'lag_2', 'lag_3',
    'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_9', 'rolling_mean_12'
]].copy()

# ------------------------------------------------------------------------------
# Split by year (data must already be sorted chronologically)
# train: year < 2025 | test: year == 2025 | validation: year > 2025
# ------------------------------------------------------------------------------
train = data[data['period_year'] <= 2025]
test = data[data['period_year'] > 2025]
# val = data[data['period_year'] > 2025]

print(f"Train rows: {len(train)}, Test rows: {len(test)} ")
# , Val rows: {len(val)}")

train_series = train['consumption_value']
test_series = test['consumption_value']
# val_series = val['consumption_value']

# ------------------------------------------------------------------------------
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    p_value = result[1]
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {p_value}')
    print('Stationary' if p_value < 0.05 else 'Non-Stationary')


check_stationarity(train_series)

plot_acf(train_series)
plot_pacf(train_series)
plt.show()

# ------------------------------------------------------------------------------
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 12

model = SARIMAX(train_series, order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

# ------------------------------------------------------------------------------
def evaluate(name, n_periods, observed):
    forecast = results.forecast(steps=n_periods)
    mae = mean_absolute_error(observed, forecast)
    mse = mean_squared_error(observed, forecast)
    rmse = root_mean_squared_error(observed, forecast)
    mape = 100*mean_absolute_percentage_error(observed, forecast)
    r2 = r2_score(observed, forecast)
    print(f"[{name}] \nMAE: {mae:.4f}  \nMSE: {mse:.4f}  \nMAPE: {mape:.4f}  \nRMSE: {rmse:.4f}  \nR2: {r2:.4f}")
    return forecast


test_forecast = evaluate("TEST (2026)", len(test_series), test_series)
# val_forecast = evaluate("VALIDATION (>2025)", len(val_series), val_series)