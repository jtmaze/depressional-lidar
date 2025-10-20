# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lai_wy_scripts.dmc_vis_functions import plot_stage_ts, remove_flagged_buffer, plot_ts

# Load stage and climate data
stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
climate_path = "D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/ERA-5_daily_mean.csv"
stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['day'])
climate_ts = pd.read_csv(climate_path)[['date_local', 'pet_m', 'precip_m']]
climate_ts.rename(columns={'date_local': 'day'}, inplace=True)
# PET is negative in the data, changed to positive. 
climate_ts['pet_m'] = climate_ts['pet_m'] * -1

# Table for reference and logged wells
logging_info = pd.DataFrame([
    {'referend_id': '13_410', 'logged_id': '13_267', 'logged_date': '2/15/2023'}, #0
    {'referend_id': '15_409', 'logged_id': '15_268', 'logged_date': '2/15/2024'}, #1
    {'referend_id': '3_34', 'logged_id': '3_173', 'logged_date': '7/15/2023'}, #2
    {'referend_id': '5_546', 'logged_id': '9_77', 'logged_date': '7/15/2023'}, #3
    {'referend_id': '14_612', 'logged_id': '14_500', 'logged_date': '2/15/2024'}, #4 NOTE: Fairly gradual LAI decline
    {'referend_id': '9_332', 'logged_id': '9_439', 'logged_date': '12/15/2023'}, #5 
    {'referend_id': '14_15', 'logged_id': '7_626', 'logged_date': '10/15/2022'}, #6
    {'referend_id': '3_21', 'logged_id': '7_341', 'logged_date': '8/15/2022'}, #7
    {'referend_id': '6_300', 'logged_id': '6_20', 'logged_date': '10/15/2022'}, #8
    {'referend_id': '5_546', 'logged_id': '5_510', 'logged_date': '12/15/2023'}, #9
    {'referend_id': '3_23', 'logged_id': '3_311', 'logged_date': '5/15/2023'}, #10
    {'referend_id': '14_15', 'logged_id': '14_610', 'logged_date': '12/15/2023'}, #11
])

# %%
selected_idx = 3
logged_id = logging_info.iloc[selected_idx]['logged_id']
reference_id = logging_info.iloc[selected_idx]['referend_id']
logged_date = pd.to_datetime(logging_info.iloc[selected_idx]['logged_date'])

logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()

plot_stage_ts(logged_ts, reference_ts, logged_date)

# %% Clean the time series and calculate cumulative stage

logged_ts = remove_flagged_buffer(logged_ts, buffer_days=1)
reference_ts = remove_flagged_buffer(reference_ts, buffer_days=1)
reference_ts['well_stage_zeroed'] = reference_ts['well_depth'] - reference_ts['well_depth'].min()
logged_ts['well_stage_zeroed'] = logged_ts['well_depth'] - logged_ts['well_depth'].min()

# NOTE: Using an inner join to ensure both time series have data on the same days
comparison = pd.merge(
    reference_ts, 
    logged_ts, 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
).drop(columns=['flag_ref', 'flag_log', 'well_depth_ref', 'well_depth_log'])

plot_stage_ts(
    comparison[['day', 'well_stage_zeroed_log']], 
    comparison[['day', 'well_stage_zeroed_ref']], 
    logged_date
)

comparison['cum_stage_ref'] = comparison['well_stage_zeroed_ref'].cumsum()
comparison['cum_stage_log'] = comparison['well_stage_zeroed_log'].cumsum()
comparison['is_post_logging'] = comparison['day'] >= logged_date

# %% Merge the

start_date = comparison['day'].min()
climate_ts['day'] = pd.to_datetime(climate_ts['day'])
climate_ts = climate_ts[climate_ts['day'] >= start_date].copy()
climate_ts['p_pet'] = climate_ts['precip_m'] - climate_ts['pet_m']
climate_ts['p_pet_adj'] = climate_ts['precip_m'] - (climate_ts['pet_m'] * 0.04) 
plot_ts(climate_ts, 'p_pet_adj')
plot_stage_ts
comparison = pd.merge(comparison, climate_ts, how='left', on='day')
comparison['cum_precip'] = comparison['precip_m'].cumsum()
comparison['cum_p_pet_adj'] = comparison['p_pet_adj'].cumsum()
plot_ts(comparison, 'cum_p_pet_adj')

# %%
tgt_met_var = 'cum_precip' #NOTE can change this

x_full = comparison[tgt_met_var].to_numpy()

pre_logged = comparison[comparison['day'] < logged_date].copy()
x_pre = pre_logged[tgt_met_var].to_numpy()
y_pre_log = pre_logged['cum_stage_log'].to_numpy()
y_pre_ref = pre_logged['cum_stage_ref'].to_numpy()
result_pre_log = np.linalg.lstsq(x_pre[:, None], y_pre_log, rcond=None)
result_pre_ref = np.linalg.lstsq(x_pre[:, None], y_pre_ref, rcond=None)
m_pre_log = result_pre_log[0][0]
m_pre_ref = result_pre_ref[0][0]

plt.figure(figsize=(8, 8))
plt.scatter(comparison[tgt_met_var], comparison['cum_stage_ref'], alpha=0.7, label=f"Reference {reference_id} Cumulative Stage")
plt.scatter(comparison[tgt_met_var], comparison['cum_stage_log'], alpha=0.7, label=f"Logged {logged_id} Cumulative Stage")

# Add vertical line at logging date
logging_cum_precip = comparison[comparison['day'] == logged_date][tgt_met_var].values
if len(logging_cum_precip) > 0:
    logging_cum_precip = logging_cum_precip[0]
else:
    logging_cum_precip = comparison[comparison['day'] <= logged_date][tgt_met_var].iloc[-1]
plt.axvline(logging_cum_precip, color='red', linestyle='--', linewidth=2, 
            label=f'Logging Date ({logged_date.strftime("%Y-%m-%d")})')

# Plot pre-logging fit lines
plt.plot(x_full, m_pre_log * x_full, color='orange', linestyle='--', label=f"Pre-Logging Fit (Logged {logged_id})")
plt.plot(x_full, m_pre_ref * x_full, color='blue', linestyle='--', label=f"Pre-Logging Fit (Reference {reference_id})")

plt.xlabel('Cumulative (Precip) [m]')
plt.ylabel('Cumulative Stage [m]')
plt.legend()
plt.grid(True)
plt.show()

# %%

comparison['predicted_log'] = comparison[tgt_met_var] * m_pre_log
comparison['predicted_ref'] = comparison[tgt_met_var] * m_pre_ref
comparison['residual_log'] = comparison['cum_stage_log'] - comparison['predicted_log']
comparison['residual_ref'] = comparison['cum_stage_ref'] - comparison['predicted_ref']


fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(comparison['day'], comparison['residual_ref'], label=f"Reference {reference_id} Residuals")
ax.plot(comparison['day'], comparison['residual_log'], label=f"Logged {logged_id} Residuals")
ax.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
ax.axhline(0, color='black', linestyle='--', linewidth=1)

# Clean date formatting
locator = mdates.AutoDateLocator(minticks=5, maxticks=8)
formatter = mdates.ConciseDateFormatter(locator)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
# Optional: minor ticks for months
ax.xaxis.set_minor_locator(mdates.MonthLocator())

ax.set_xlabel('Date')
ax.set_ylabel('Residual [m]')
ax.legend()
ax.grid(True, which='both', linestyle=':', alpha=0.5)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %%
