import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score

df = pd.read_csv("current_final_data\lagged_after_interpolation_diff_log.csv")

data = df[[
    'period_year', 'period_month', 'consumption_value', 'consumption_value_source'
]].copy()

# ------------------------------------------------------------------------------
# Split by year (data must already be sorted chronologically)
# train: year < 2025 | test: year == 2025 | validation: year > 2025
# ------------------------------------------------------------------------------
# train = data[data['period_year'] <= 2024]
# test = data[data['period_year'] > 2024]
# val = data[data['period_year'] > 2025]

train = data[
    (data['period_year'] < 2024) | 
    ((data['period_year'] == 2024) & (data['period_month'] == 'January'))
]

test = data[
    (data['period_year'] == 2024) & 
    (data['period_month'].isin(['February','March','April','May','June','July']))
]

# sanity check — confirm every test row is real
print(test[['period_month','period_year','consumption_value_source']])

print(f"Train rows: {len(train)}, Test rows: {len(test)}")
# , Val rows: {len(val)}


train_series = train['consumption_value']
test_series = test['consumption_value']
# val_series = val['consumption_value']

# ------------------------------------------------------------------------------
# ETS: trend + seasonal components (additive; seasonal period = 12 months)
model = ExponentialSmoothing(
    train_series,
    trend='mul',
    seasonal='add',
    seasonal_periods=12
)
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

print(test_forecast)


plt.plot(train_series.values, label='Train')
plt.plot(range(len(train_series), len(train_series) + len(test_series)), test_series.values, label='Test actual')
plt.plot(range(len(train_series), len(train_series) + len(test_series)), test_forecast.values, label='Test forecast')
plt.legend()
plt.show()