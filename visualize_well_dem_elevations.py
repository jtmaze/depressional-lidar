# %% 1.0 Libraries and directories

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

well_metrics = pd.read_csv('./out_data/well_dem_metrics_wells_v1.csv')
well_metrics_from_rtk_pts = pd.read_csv('./out_data/well_dem_metrics_wells_RTK.csv')
rtk_digitized = pd.read_csv('./out_data/rtk_data_digitized.csv')
rtk_digitized_clean = rtk_digitized[rtk_digitized['vert_accuracy_mm'] < 500]

# %% Convert data to meters if you're going to use it. 
well_metrics['mean_elevation_m'] = well_metrics['mean_elevation'] * 0.3048
well_metrics['filtered_mean_elevation_m'] = well_metrics['filtered_mean_elevation'] * 0.3048
well_metrics['filtered_dif_m'] = well_metrics['mean_elevation_m'] - well_metrics['filtered_mean_elevation_m']
well_metrics['cv_elevation_m'] = well_metrics['cv_elevation'] * 0.3048

well_metrics_from_rtk_pts['mean_elevation_m'] = well_metrics_from_rtk_pts['mean_elevation'] * 0.3048
well_metrics_from_rtk_pts['filtered_mean_elevation_m'] = well_metrics_from_rtk_pts['filtered_mean_elevation'] * 0.3048
well_metrics_from_rtk_pts['filtered_dif_m'] = well_metrics_from_rtk_pts['mean_elevation_m'] - well_metrics_from_rtk_pts['filtered_mean_elevation_m']
well_metrics_from_rtk_pts['cv_elevation_m'] = well_metrics_from_rtk_pts['cv_elevation'] * 0.3048


rtk_digitized = rtk_digitized[
    ['site_id', 'latitude', 'longitude', 'orthometric_height_m', 'horizontal_accuracy_mm', 'vert_accuracy_mm']
]

# %% Plot the coefficient of variation of elevation for different buffer sizes

plt.figure(figsize=(12, 7))

# Create a scatter plot colored by buffer_size
sns.scatterplot(x='site_id', y='mean_elevation_m', hue='buffer_size', 
                palette='viridis', data=well_metrics, s=100, alpha=0.7)

plt.xticks(rotation=90)
plt.ylabel('Elevation (m)')
plt.title('Coefficient of Variation of Elevation (m) by Buffer Size')
plt.legend(title='Buffer Size (m)')
plt.tight_layout()
plt.show()
# %%

plt.figure(figsize=(12, 5))
sns.barplot(
    data=rtk_digitized,
    x='site_id',
    y='vert_accuracy_mm'
)
plt.xticks(rotation=90)
plt.title("RTK GPS Vertical Accuracy")
plt.ylabel("millimeters")
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(
    data=rtk_digitized_clean,
    x='site_id',
    y='vert_accuracy_mm'
)
plt.xticks(rotation=90)
plt.title("Cleaned RTK GPS Vertical Accuracy")
plt.ylabel("millimeters")
plt.show()

# %% Get DEM data with multiple buffer sizes
buffer_sizes = [1.5, 2, 5, 10, 25]
dfs = []

for buffer in buffer_sizes:
    well_metrics_at_buff = well_metrics_from_rtk_pts[well_metrics_from_rtk_pts['buffer_size'] == buffer]
    well_metrics_at_buff = well_metrics_at_buff[
        ['site_id', 'buffer_size', 'mean_elevation_m', 'filtered_mean_elevation_m']
    ]
    
    merged = pd.merge(well_metrics_at_buff, rtk_digitized, on='site_id')
    merged = merged[
        ['site_id', 'buffer_size', 'mean_elevation_m', 'filtered_mean_elevation_m', 
         'orthometric_height_m', 'vert_accuracy_mm']
    ]
    merged['DEM_dif_RTK'] = merged['mean_elevation_m'] - merged['orthometric_height_m']
    dfs.append(merged)

# Combine all data
merged_concat = pd.concat(dfs)

# %%
#merged_concat = merged_concat[merged_concat['vert_accuracy_mm'] < 500]

# Melt the data to prepare for plotting
df_melted = merged_concat.melt(
    id_vars=['site_id', 'buffer_size'],
    value_vars=['DEM_dif_RTK'],
    var_name='Difference_Type', 
    value_name='Difference_Value'
)

plt.figure(figsize=(12,6))
sns.barplot(x='site_id', y='Difference_Value', hue='buffer_size', data=df_melted, palette='coolwarm')
plt.xlabel('Site ID')
plt.ylabel('Difference (RTK - DEM) Elevation (m)')
plt.title('DEM Differences by Site and Buffer')
plt.xticks(rotation=90)
plt.legend(title='Buffer Size (m)')
plt.tight_layout()
plt.show()
# %%

buff_5 = merged_concat[merged_concat['buffer_size'] == 5]
buff_5 = buff_5[buff_5['vert_accuracy_mm'] < 500]

buff_5['abs_DEM_dif_RTK'] = abs(buff_5['DEM_dif_RTK'])

# Create the scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='vert_accuracy_mm', y='abs_DEM_dif_RTK', data=buff_5)
plt.xlabel('Vertical Accuracy (mm)')
plt.ylabel('Absolute DEM Difference (m)')
plt.title('Relationship between Vertical Accuracy and Absolute DEM Difference (Buffer Size 5)')
plt.grid(True)
plt.tight_layout()
plt.show()
# %%
site_ids = ['6_93', '15_409', '6_629', '14_500', '5a_582', '13_267']
best_well_estimates = well_metrics[
    (well_metrics['buffer_size'] == 5) & 
    (well_metrics['site_id'].isin(site_ids))

].copy()
best_well_estimates = best_well_estimates[['site_id', 'mean_elevation_m', 'filtered_mean_elevation_m']]
print(best_well_estimates)