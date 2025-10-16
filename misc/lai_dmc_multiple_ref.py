# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['day'])

# %% 2.0 Set up reference and logged time series

stage_data = stage_data[stage_data['flag'] == 0].copy()
logged_id = '14_500'
reference_ids = ['14_612', '14_538', '15_409']
logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
logged_date = pd.to_datetime('2/1/2024')
print(logged_date)

# %% 2.1 Set up the reference time series

reference_ts = stage_data[stage_data['well_id'].isin(reference_ids)].copy()
reference_ts.sort_values(['well_id', 'day'], inplace=True)


# Line plot colored by well_id
plt.figure(figsize=(8, 6))
for wid, df in reference_ts.groupby('well_id'):
    plt.plot(df['day'], df['well_depth'], label=wid)
plt.xlabel('Date')
plt.ylabel('Well Depth')
plt.title('Reference Well Depth Time Series')
plt.legend(title='well_id', ncol=2)
plt.grid(True)
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.tight_layout()
plt.show()

# %% 2.2 Average the reference time series across wells

# Drop any days that don't have data for all reference wells
reference_ts = reference_ts.groupby('day').filter(lambda x: len(x) == len(reference_ids))

# Adjust well depth to be relative to the minimum well depth for each well
reference_ts['well_stage_zeroed'] = reference_ts.groupby('well_id')['well_depth'].transform(lambda x: x - x.min())

# Get the cumulative well stage across all reference wells
reference_ts_avg = reference_ts.groupby('day').agg({'well_stage_zeroed': 'mean'}).reset_index()

plt.figure(figsize=(8, 6))
plt.plot(reference_ts_avg['day'], reference_ts_avg['well_stage_zeroed'], label='Reference (Average)')
plt.xlabel('Date')
plt.ylabel('Well Stage (Average)')
plt.title('Reference Well Stage Time Series (Average)')
plt.legend()
plt.grid(True)

# %% 3.0 Plot the reference and logged time series

logged_ts['well_stage_zeroed'] = logged_ts['well_depth'] - logged_ts['well_depth'].min()

plt.figure(figsize=(8, 6))
plt.plot(reference_ts_avg['day'], reference_ts_avg['well_stage_zeroed'], label=f'Reference')
plt.plot(logged_ts['day'], logged_ts['well_stage_zeroed'], label=f'Logged ({logged_id})')
plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
plt.xlabel('Date')
plt.ylabel('Well Stage (Average)')
plt.title(f'Well Stage Time Series - Reference: {reference_ids}, Logged: {logged_id}')
plt.legend()
plt.grid(True)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.show()

# %% 

comparison = pd.merge(
    reference_ts_avg[['day', 'well_stage_zeroed']], 
    logged_ts[['day', 'well_stage_zeroed']], 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
)
print(comparison.head(10))
comparison['cum_stage_ref'] = comparison['well_stage_zeroed_ref'].cumsum()
comparison['cum_stage_log'] = comparison['well_stage_zeroed_log'].cumsum()

# NOTE: Fit only on pre-logged date data
partial = comparison[comparison['day'] < pd.to_datetime(logged_date)].copy()
x_full = comparison['cum_stage_ref'].to_numpy()
y_full = comparison['cum_stage_log'].to_numpy()
x_fit = partial['cum_stage_ref'].to_numpy()
y_fit = partial['cum_stage_log'].to_numpy()
result_partial = np.linalg.lstsq(x_fit[:, None], y_fit, rcond=None)
result_full = np.linalg.lstsq(x_full[:, None], y_full, rcond=None)
m_partial = result_partial[0][0]
m_full = result_full[0][0]

# %%

plt.figure(figsize=(8, 8))
plt.scatter(comparison['cum_stage_ref'], comparison['cum_stage_log'], alpha=0.7)
plt.plot(x_full, m_full * x_full, color='red', linewidth=2, label=f'Fit Full Data (slope = {m_full:.2f})')
plt.plot(x_full, m_partial * x_full, color='green', linewidth=2, label=f'Fit Pre-Logged Data (slope = {m_partial:.2f})')
plt.xlabel('Reference Cumulative Stage')
plt.ylabel('Logged Cumulative Stage')
plt.legend()
plt.grid(True)
plt.show()

# %%

comparison['predicted_log_full'] = comparison['cum_stage_ref'] * m_full
comparison['predicted_log_partial'] = comparison['cum_stage_ref'] * m_partial
comparison['residual_full'] = comparison['cum_stage_log'] - comparison['predicted_log_full']
comparison['residual_partial'] = comparison['cum_stage_log'] - comparison['predicted_log_partial']

plt.figure(figsize=(10, 6))
plt.plot(comparison['day'], comparison['residual_full'], linewidth=2, color='red', label='Residual Full Fit')
plt.plot(comparison['day'], comparison['residual_partial'], linewidth=2, color='green', label='Residual Pre-Logged Fit')
plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
plt.axhline(0, color='black', linestyle='-', label='Zero Line')
plt.xlabel('Date')
plt.ylabel('Residual (Logging - Predicted)')
plt.title(f'Residual Time Series - Reference: {reference_ids}, Logged: {logged_id}')
plt.legend()
plt.grid(True)
plt.show()

# %%
