# OBJECTIVE is to transform the data mentioned below into stationary data.
# Expected features: 
# 1. Add a column month_year to the "current_final_data\lagged_after_interpolation.csv".
# 2. Visualize the data: consumption_value by month_year using matplotlib.
# 3. difference once - create the column
# 4. 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

lagged_after_interpolation = 'current_final_data\lagged_after_interpolation_diff_log.csv'
df = pd.read_csv(lagged_after_interpolation)

# non-stationary data
plt.plot(df['consumption_value'], label='consumption_value')
plt.show()

# diff
df['diff'] = df['consumption_value'].diff()
plt.plot(df['diff'])
plt.show()



# log
df["log_value"] = np.log(df["consumption_value"])
plt.plot(df["log_value"])
plt.show()


# diff has already been added to the csv.
# log has already been added to the csv.
df.to_csv(lagged_after_interpolation, index=False)




# FOR Parquet data:
lagged_after_interpolation = 'current_final_data\lagged_after_interpolation_diff_log.parquet'
df = pd.read_parquet(lagged_after_interpolation)

df['diff'] = df['consumption_value'].diff()
df["log_value"] = np.log(df["consumption_value"])


df.to_parquet(lagged_after_interpolation, index=False)
