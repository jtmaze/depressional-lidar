# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dmc_vis_functions import plot_stage_ts, remove_flagged_buffer

stage_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/daily_waterlevel_Fall2025.csv"
climate_path = "D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/ERA-5_daily_mean.csv"
stage_data = pd.read_csv(stage_path)
stage_data['day'] = pd.to_datetime(stage_data['day'])

# %%

reference_id = '15_409'
logged_id = '15_268'
logged_date = '2/1/2024'
logged_date = pd.to_datetime(logged_date)

logged_ts = stage_data[stage_data['well_id'] == logged_id].copy()
reference_ts = stage_data[stage_data['well_id'] == reference_id].copy()

plot_stage_ts(logged_ts, reference_ts, logged_date)

# %%

logged_ts = remove_flagged_buffer(logged_ts)
reference_ts = remove_flagged_buffer(reference_ts)

# NOTE: Using an inner join to ensure both time series have data on the same days
comparison = pd.merge(
    reference_ts[['day', 'well_depth']], 
    logged_ts[['day', 'well_depth']], 
    how='inner', 
    on='day', 
    suffixes=('_ref', '_log')
)
plot_stage_ts(comparison[['day', 'well_depth_log']], comparison[['day', 'well_depth_ref']], logged_date)


# %%
