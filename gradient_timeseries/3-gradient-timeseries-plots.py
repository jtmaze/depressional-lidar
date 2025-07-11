# %% 1.0 Libaries and directories
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient_timeseries import gradient_viz

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')
catchment = 'jl'
gradients_path = f'./delmarva/out_data/{catchment}_gradient_timeseries.csv'
gradient_ts = pd.read_csv(gradients_path)
gradient_ts['Date'] = pd.to_datetime(gradient_ts['Date'])

if catchment == 'jl':
    # NOTE: A little extra data cleaning
    gradient_ts = gradient_ts[gradient_ts['head_gradient_cm_m'] < 1.0]
    gradient_ts = gradient_ts[gradient_ts['head_gradient_cm_m'] > -4]

if catchment == 'bc':
    gradient_ts = gradient_ts[gradient_ts['head_gradient_cm_m'] > -1]

# %% 2.0 Get a summary table for counts of each pair type

pair_type_summary = gradient_ts.groupby('pair_type').agg(
    pair_type_count=('well_pair', lambda x: len(x.unique()))
).reset_index().sort_values(
    by='pair_type_count',
    ascending=False
)
print(pair_type_summary)

# Define a canonical function
def orderless_pair(pair):
    a, b = pair.split('__to__')
    return '__to__'.join(sorted([a, b]))

# Create a new column using the canonical mapping
pair_type_summary['orderless_pair'] = pair_type_summary['pair_type'].apply(orderless_pair)

# Group by the canonical pair and sum the counts
orderless_grouped = pair_type_summary.groupby('orderless_pair', as_index=False)['pair_type_count'].sum()
print(orderless_grouped)

del orderless_grouped, pair_type_summary
# %% 3.0 

well_pairs_to_plot = gradient_ts['well_pair'].unique()

gradient_viz.gradient_ts_plot(
    gradient_ts, 
    well_pairs_to_plot, 
    y_var='head_gradient_cm_m', 
    color_arg='same_color', 
    y_lim=None)

# %% 

gradient_viz.gradient_ts_plot(
    gradient_ts,
    well_pairs_to_plot,
    y_var='head_gradient_cm_m',
    color_arg='by_elevation_gradient',
    y_lim=None,
)

# %%

temp = gradient_ts[gradient_ts['elevation_gradient_cm_m'] <= 2.5]
well_pairs_to_plot = temp['well_pair'].unique()

gradient_viz.gradient_ts_plot(
    temp,
    well_pairs_to_plot,
    y_var='head_gradient_cm_m',
    color_arg='by_elevation_gradient',
    y_lim=None,
)
# %%

temp = gradient_ts[gradient_ts['pair_type'] == 'SW__to__SW']
well_pairs_to_plot = temp['well_pair'].unique()

gradient_viz.gradient_ts_plot(
    temp,
    well_pairs_to_plot,
    y_var='head_gradient_cm_m',
    color_arg='multi',
    y_lim=None,
)

gradient_viz.gradient_ts_plot(
    temp, 
    well_pairs_to_plot, 
    y_var='adj_gradient',
    color_arg='multi',
    y_lim=None
)

# %%

temp = gradient_ts[gradient_ts['pair_type'] == 'UW__to__UW']
well_pairs_to_plot = temp['well_pair'].unique()

gradient_viz.gradient_ts_plot(
    temp,
    well_pairs_to_plot,
    y_var='head_gradient_cm_m',
    color_arg='multi',
    y_lim=None,
)

gradient_viz.gradient_ts_plot(
    temp, 
    well_pairs_to_plot, 
    y_var='adj_gradient',
    color_arg='multi',
    y_lim=None
)


# %%

temp = gradient_ts[gradient_ts['pair_type'] == 'CH__to__CH']
well_pairs_to_plot = temp['well_pair'].unique()

gradient_viz.gradient_ts_plot(
    temp,
    well_pairs_to_plot,
    y_var='head_gradient_cm_m',
    color_arg='multi',
    y_lim=None,
)

gradient_viz.gradient_ts_plot(
    temp, 
    well_pairs_to_plot, 
    y_var='adj_gradient',
    color_arg='multi',
    y_lim=None
)
# %%
