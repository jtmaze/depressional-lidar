# %% 1.0 Libraries and file paths

import pandas as pd

lai_buffer_dist = 150 # Options are 1) 150m 2) 250m
data_set = 'no_dry_days' # Options are 1) no_dry_days 2) wtd_above0_25 3)full_obs

data_dir = "D:/depressional_lidar/data/bradford/out_data/"
wetland_pairs_path = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_all_wells.csv'
models_path = data_dir + f'/model_info/model_estimates_LAI{lai_buffer_dist}m_domain_{data_set}.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
model_data = pd.read_csv(models_path)
print(model_data['model_type'].unique())
model_data = model_data[model_data['model_type'] == 'OLS'] #NOTE: HuberRLM doesn't optimize for r-squared like OLS, so r-squared will always be lower. 

# %% 2.0 Filter to strong model fits

print(len(model_data)) 

strong_pairs = model_data[
    (model_data['data_set'] == data_set) & 
    (model_data['r2_joint'] >= 0.3)
][['log_id', 'log_date', 'ref_id']].copy()

print(len(strong_pairs))

# Well data was problematic
#strong_pairs = strong_pairs[strong_pairs['log_id'] != '15_516']

# Outlier of model output
print(len(strong_pairs))
strong_pairs = strong_pairs[
    ~((strong_pairs['log_id'] == '7_341') & (strong_pairs['ref_id'] == '15_4'))
]
print(len(strong_pairs))

# %% 3.0 Write the output

strong_pairs.to_csv(f'{data_dir}/strong_ols_models_{lai_buffer_dist}m_domain_{data_set}.csv', index=False)


# %%
