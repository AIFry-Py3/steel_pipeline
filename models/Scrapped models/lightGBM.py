import lightgbm as lgb
import pandas as pd

lrc = r'final_data/final_data_non_stationary.csv' # lag_rolling_cleaned
df = pd.read_csv(lrc)

# adding month_number to the csv, to use instead of period_month
month_dict = {  "January": 1, "February": 2, "March":3, "April": 4, "May": 5, "June":6, "July":7,
                "August": 8, "September": 9, "October": 10, "November": 11, "December": 12  }
df['month_number'] = [month_dict[df.iloc[i]["period_month"]] for i in range(len(df))]
df.to_csv(lrc, index=False)

# Re-reading updated csv.
df = pd.read_csv(lrc)

train = df[df['period_year'] <= 2025]
test = df[df['period_year'] > 2025]

features = ['period_year', 'month_number']
target = 'consumption_value'


X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

model = lgb.LGBMRegressor(max_depth=10)
model.fit(X_train, y_train)

predictions = model.predict(X_test)


# Evaluating the model - using MAE, RMSE
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

# mae = mean_absolute_error(y_test, predictions)
# rmse = np.sqrt(mean_squared_error(y_test, predictions))

# print(f"MAE: {mae}")
# print(f"RMSE: {rmse}")


observed = y_test
forecast = predictions
mae = mean_absolute_error(observed, forecast)
mse = mean_squared_error(observed, forecast)
rmse = root_mean_squared_error(observed, forecast)
mape = 100*mean_absolute_percentage_error(observed, forecast)
r2 = r2_score(observed, forecast)
print(f"[Test 2026] \nMAE: {mae:.4f}  \nMSE: {mse:.4f}  \nMAPE: {mape:.4f}  \nRMSE: {rmse:.4f}  \nR2: {r2:.4f}")


# Plot the forecast
import matplotlib.pyplot as plt
