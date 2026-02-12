# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

lai_buffer_dist = 150 # Options are 1) 150m 2) 250m
data_set = '' # Options are 1) no_dry_days 2) wtd_above0

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'
models_path = data_dir + f'/model_info/all_wells_model_estimates_LAI{lai_buffer_dist}m_domain_{data_set}.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
model_data = pd.read_csv(models_path)
model_data = model_data[model_data['model_type'] == 'OLS']

# %% 2.0 Filter to strong model fits

print(len(model_data)) 

strong_pairs = model_data[
    (model_data['data_set'] == data_set) & 
    (model_data['r2_joint'] >= 0.5)
][['log_id', 'log_date', 'ref_id']]

print(len(strong_pairs))

# %% 3.0 Write the output

strong_pairs.to_csv(f'{data_dir}/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv', index=False)

# %% 4.0 Diagnostic plots of model fits

plot_data = model_data.copy()
datasets = plot_data['data_set'].unique()

# %% 4.1 Histograms of joint r-squared values for each dataset

