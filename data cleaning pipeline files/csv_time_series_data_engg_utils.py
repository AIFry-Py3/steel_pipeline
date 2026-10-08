import pandas as pd

filled_dsm = 'data/filled_dsm.csv'
lagged_fdsm ='data/lagged_fdsm.csv'

data = pd.read_csv(filled_dsm)

data['lag_1'] = data['consumption_value'].shift(1)
data['lag_2'] = data['consumption_value'].shift(2)
data['lag_3'] = data['consumption_value'].shift(3)
# ------------------------------------------------------------------------------
# THIS IS THE REASON FOR DATA LEAKAGE FROM THE FUTURE: Moving average considers 
# data from the future too.
# data['rolling_mean_3'] = data['consumption_value'].shift(1).rolling(3).mean()
# data['rolling_std_3'] = data['consumption_value'].shift(1).rolling(3).std()
# data['rolling_mean_6'] = data['consumption_value'].shift(1).rolling(6).mean()
# data['rolling_std_6'] = data['consumption_value'].shift(1).rolling(6).std()
# data['rolling_mean_9'] = data['consumption_value'].shift(1).rolling(9).mean()
# data['rolling_std_9'] = data['consumption_value'].shift(1).rolling(9).std()
# data['rolling_mean_12'] = data['consumption_value'].shift(1).rolling(12).mean()
# data['rolling_std_12'] = data['consumption_value'].shift(1).rolling(12).std()
# ------------------------------------------------------------------------------



# NOT DOING DROPNA
# data = data.dropna(subset=['lag_1', 'lag_2', 'lag_3'])
# 'rolling_mean_3', 'rolling_std_3', 'rolling_mean_6', 'rolling_std_6', 'rolling_mean_9', 'rolling_std_9', 'rolling_mean_12', 'rolling_std_12'])

data.to_csv(lagged_fdsm, index=False)
# data.to_parquet(lagged_fdsm, index=False)