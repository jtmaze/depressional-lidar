# %% 1.0 Libraries and file paths

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient_timeseries import gradient_viz

os.chdir('D:/depressional_lidar/')

bc_path = f'./delmarva/out_data/bc_gradient_timeseries.csv'
bc = pd.read_csv(bc_path)
jl_path = f'./delmarva/out_data/jl_gradient_timeseries.csv'
jl = pd.read_csv(jl_path)

gradient_ts = pd.concat([bc, jl], ignore_index=True)

gradient_ts['Date'] = pd.to_datetime(gradient_ts['Date'])
gradient_ts = gradient_ts[(gradient_ts['Date'] >= '2021-03-11') &
                          (gradient_ts['Date'] <= '2023-10-01')]

# %% 2.0 Summarize gradient stats by well_pair

def summarize_head_gradient(df):
    def cv(x):
        # Avoid division by zero
        return x.std() / x.mean() if x.mean() != 0 else None

    summary = df.groupby(['well_pair']).agg(
        pair_type=('pair_type', 'first'),
        mean_head_gradient=('head_gradient_cm_m', 'mean'),
        std_head_gradient=('head_gradient_cm_m', 'std'),
        cv_head_gradient=('head_gradient_cm_m', lambda x: cv(x)),
        mean_adj_gradient=('adj_gradient', 'mean'),
        elevation_gradient=('elevation_gradient_cm_m', 'first'),
    ).reset_index()
    return summary

head_gradient_site_summary = summarize_head_gradient(gradient_ts)


# %% 3.0 Correlations between elevation and head in CH, SW, UW pairs

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['CH__to__CH'],
    line_color='red'
)

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['SW__to__SW'],
    line_color='red'
)

gradient_viz.summary_correlations_plot(
    summary_df=head_gradient_site_summary,
    pair_type_filter=['UW__to__UW'],
    line_color='red'
)

# %% 4.0 Slope and correlations based on water level data. 

temp = gradient_ts[gradient_ts['pair_type'] == 'CH__to__CH'].copy()
temp['abs_wl0'] = temp['rel_wl0'] - temp['z0'] / 100
temp['abs_wl1'] = temp['rel_wl1'] - temp['z1'] / 100

wells = list(
    set(temp['well0'].unique()).union(set(temp['well1'].unique()))
)

# Create dataframe with dates and wells
dates = pd.date_range(start=temp['Date'].min(), end=temp['Date'].max(), freq='D')
custom_wells_wl_df = pd.DataFrame([(date, well) for date in dates for well in wells],
                                 columns=['Date', 'well'])

# Map absolute water levels to the custom DataFrame
def find_abs_wl(row):
    matching_row = temp[(temp['Date'] == row['Date']) & (temp['well0'] == row['well'])]
    if not matching_row.empty:
        return matching_row['abs_wl0'].values[0]
    else:
        other_matching_row = temp[(temp['Date'] == row['Date']) & (temp['well1'] == row['well'])]
        return other_matching_row['abs_wl1'].values[0] if not other_matching_row.empty else None
    
custom_wells_wl_df['abs_wl'] = custom_wells_wl_df.apply(find_abs_wl, axis=1)

# %%

agg_wl_custom = custom_wells_wl_df.groupby(['Date']).agg(
    mean_abs_wl=('abs_wl', 'mean'),
).reset_index()

# %% Plot histogram of mean absolute water levels

low_wl_cut = -0.60
high_wl_cut = -0.05

# Plot the histogram with shaded regions
plt.figure(figsize=(8, 6))
hist, bins, _ = plt.hist(agg_wl_custom['mean_abs_wl'], bins=42, edgecolor='k', alpha=0.7, color='gray')

# Add shading for low water level region (below low_wl_cut)
plt.axvspan(min(bins), low_wl_cut, color='red', alpha=0.2, label='Low WL')

# Add shading for moderate water level region (between low_wl_cut and high_wl_cut)
plt.axvspan(low_wl_cut, high_wl_cut, color='green', alpha=0.2, label='Moderate WL')

# Add shading for high water level region (above high_wl_cut)
plt.axvspan(high_wl_cut, max(bins), color='blue', alpha=0.2, label='High WL')

plt.xlabel('Mean Absolute Water Level')
plt.ylabel('Days')
plt.title('Histogram of Mean Absolute Water Level in Channel Wells (Mar 2021 - Oct 2022)')
plt.legend()
plt.show()

# %% Select dates based on water level thresholds

low_wl_dates = agg_wl_custom[agg_wl_custom['mean_abs_wl'] < low_wl_cut]['Date']
high_wl_dates = agg_wl_custom[agg_wl_custom['mean_abs_wl'] > high_wl_cut]['Date']
moderate_wl_dates = agg_wl_custom[
    (agg_wl_custom['mean_abs_wl'] >= low_wl_cut) & 
    (agg_wl_custom['mean_abs_wl'] <= high_wl_cut)
]['Date']

low_wl_gradient_ts = gradient_ts[gradient_ts['Date'].isin(low_wl_dates)]
high_wl_gradient_ts = gradient_ts[gradient_ts['Date'].isin(high_wl_dates)]
moderate_wl_gradient_ts = gradient_ts[gradient_ts['Date'].isin(moderate_wl_dates)]

# %% 5.0 Plot correlations for low and high water level gradient timeseries
high_grandient_site_summary = summarize_head_gradient(high_wl_gradient_ts)
low_gradient_site_summary = summarize_head_gradient(low_wl_gradient_ts)
moderate_gradient_site_summary = summarize_head_gradient(moderate_wl_gradient_ts)

gradient_viz.summary_correlations_plot(
    low_gradient_site_summary,
    pair_type_filter=['CH__to__CH'],
    line_color='red',
)

gradient_viz.summary_correlations_plot(
    high_grandient_site_summary,
    pair_type_filter=['CH__to__CH'],
    line_color='blue',  
)

gradient_viz.summary_correlations_plot(
    moderate_gradient_site_summary,
    pair_type_filter=['CH__to__CH'],
    line_color='green',
)

# %%
