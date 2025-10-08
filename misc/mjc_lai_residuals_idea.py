# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['day'])

# %%
stage_data = stage_data[stage_data['flag'] == 0].copy()
reference_id = '5_546'
reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()
logged_id = '9_77'
logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
logged_date = pd.to_datetime('1/1/2024')

# %% Plot the reference and logged time series

plt.figure(figsize=(12, 6))
plt.plot(reference_ts['day'], reference_ts['well_depth'], label=f'Reference ({reference_id})')
plt.plot(logged_ts['day'], logged_ts['well_depth'], label=f'Logged ({logged_id})')
plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
plt.xlabel('Date')
plt.ylabel('Well Depth')
plt.title(f'Well Depth Time Series - Reference: {reference_id}, Logged: {logged_id}')
plt.legend()
plt.grid(True)
plt.show()

# %% 

reference_ts['well_stage_zeroed'] = reference_ts['well_depth'] - reference_ts['well_depth'].min()
logged_ts['well_stage_zeroed'] = logged_ts['well_depth'] - logged_ts['well_depth'].min()

comparison = pd.merge(
    reference_ts[['day', 'well_stage_zeroed']], 
    logged_ts[['day', 'well_stage_zeroed']], 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
)

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

plt.figure(figsize=(12, 6))
plt.plot(comparison['day'], comparison['residual_full'], linewidth=2, color='red', label='Residual Full Fit')
plt.plot(comparison['day'], comparison['residual_partial'], linewidth=2, color='green', label='Residual Pre-Logged Fit')
plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
plt.axhline(0, color='black', linestyle='-', label='Zero Line')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.title(f'Residual Time Series - Reference: {reference_id}, Logged: {logged_id}')
plt.legend()
plt.grid(True)
plt.show()

# %%
