import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error, r2_score


prophet_data = r'data\prophet_data\interpolation_lagroll_prophet_data.csv'
df = pd.read_csv(prophet_data)

df_train = df[pd.to_datetime(df['ds']) <= pd.to_datetime('2026-01-01')]
df_test = df[pd.to_datetime(df['ds']) > pd.to_datetime('2026-01-01')]

m = Prophet()
print('Fitting training data:')
m.fit(df_train)

# This is not the prediction: A dataframe extension into the future - a specified number of months/days. 
# here, months. 
# Note that future doesn't contain a 'y' column.
# future = m.make_future_dataframe(periods=12, freq='ME') 
# print(future.tail(16))


forecast = m.predict(df_test)
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# try:
#     test_dict = dict(zip(pd.to_datetime(df_test['ds']), df_test['y']))
#     forecast_dict = dict(zip(forecast['ds'], forecast['yhat']))

#     # Mean Squared Error Logic!
#     # diff_list = []
#     # for ds in test_dict.keys():
#     #     diff_list.append(abs(test_dict[ds] - forecast_dict[ds]))
#     # loss = sum(diff_list)/len(test_dict.keys()) 
#     from sklearn.metrics import mean_absolute_error as mae
#     loss = mae(list(test_dict.values()), list(forecast_dict.values()))
#     print(loss)
#
# except Exception as inst:
#     print(type(inst))
#     print(inst.args)
#     print(inst)


mae = mean_absolute_error(list(df_test['y']), list(forecast['yhat']))
mse = mean_squared_error(list(df_test['y']), list(forecast['yhat']))
rmse = root_mean_squared_error(list(df_test['y']), list(forecast['yhat']))
mape = 100*mean_absolute_percentage_error(list(df_test['y']), list(forecast['yhat']))
r2 = r2_score(list(df_test['y']), list(forecast['yhat']))
print(f"[Test 2026] \nMAE: {mae:.4f}  \nMSE: {mse:.4f}  \nMAPE: {mape:.4f}  \nRMSE: {rmse:.4f}  \nR2: {r2:.4f}")

# print(f"MAE: {loss}")


# fig1 = m.plot(forecast)

# fig2 = m.plot_components(forecast)


# from prophet.plot import plot_plotly, plot_components_plotly
# plot_plotly(m, forecast)
# plot_components_plotly(m, forecast)