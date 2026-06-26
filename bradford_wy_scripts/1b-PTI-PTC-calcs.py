# %% 1.0 Libraries and file paths

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway

data_dir = "D:/depressional_lidar/data/bradford/"

est_spills_path = f"{data_dir}/out_data/bradford_estimated_basin_spills.csv"
well_data_path = f"{data_dir}/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
connectivity_path = f"{data_dir}/bradford_wetland_connect_logging_key.xlsx"

est_spills = pd.read_csv(est_spills_path)
well_data = pd.read_csv(well_data_path)
well_data = well_data[~well_data['wetland_id'].isin(['Donor_wetland', 'Receiver_wetland'])]
connect = pd.read_excel(connectivity_path)

# %% 2.0 Select columns of interest 

est_spills = est_spills[['wetland_id', 'min_elev', 'well_elev', 'max_fill_elev']]
connect = connect[['wetland_id', 'connectivity']]
# %% 3.0 Bottomed out proportion for each well

flag2_summary = (
    well_data.groupby('wetland_id')
    .apply(lambda x: (x['flag'] == 2).sum() / len(x))
    .reset_index()
    .rename(columns={0: 'proportion_flag2'})
    .sort_values('proportion_flag2', ascending=False)
)

min_depth_table = (
    well_data.groupby('wetland_id')
    .apply(lambda x: x['well_depth_m'].min())
    .reset_index()
    .rename(columns={0: 'min_depth_m'})
    .sort_values('min_depth_m', ascending=False)
)




# %% 4.0 Join connect and est_spills to the well_data

well_data = well_data.merge(connect, on='wetland_id', how='left')
well_data = well_data.merge(est_spills, on='wetland_id', how='left')

print(well_data.head())
print(well_data.columns)

# %% 5.0 Calculate PTI for each wetland_id

well_data['wse'] = well_data['well_depth_m'] + well_data['well_elev']
well_data['inundated'] = well_data['wse'] > well_data['min_elev']

inundated_summary = (
    well_data.groupby('wetland_id')
    .apply(lambda x: x['inundated'].sum() / len(x))
    .reset_index()
    .rename(columns={0: 'proportion_inundated'})
    .sort_values('proportion_inundated', ascending=False)
)

print(inundated_summary)

# %% 6.0 Calculate PTC for each wetland_id

well_data['connected'] = well_data['wse'] > (well_data['max_fill_elev'])

ptc_summary = (
    well_data.groupby(['wetland_id', 'connectivity'])['connected']
    .mean()
    .reset_index(name='ptc')
)

print(ptc_summary)

# %% 7.0 Boxplot of Percent time connected (PTC)

connectivity_config = {
    "first order": {"color": "#6C5B7B", "label": "Ditch connected"},
    "giw": {"color": "#1B7F79", "label": "Unditched"},
    "flow-through": {"color": "#C46A1A", "label": "Flow-through connected"},
}

connect_order = ['giw', 'first order', 'flow-through']
connect_labels = [connectivity_config[c]['label'] for c in connect_order]

ptc_data = [
    ptc_summary.loc[ptc_summary['connectivity'] == c, 'ptc'].dropna().values * 100
    for c in connect_order
]

fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(
    ptc_data,
    labels=connect_labels,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
)

for patch, c in zip(box['boxes'], connect_order):
    patch.set_facecolor(connectivity_config[c]['color'])
    patch.set_alpha(0.65)

for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)

for i, c in enumerate(connect_order, start=1):
    class_data = ptc_summary.loc[ptc_summary['connectivity'] == c]
    vals = class_data['ptc'].dropna().values * 100
    x = np.random.normal(i, 0.05, size=len(vals))
    ax.scatter(x, vals, color=connectivity_config[c]['color'], edgecolor='white', linewidth=0.6, alpha=0.8, s=45, zorder=3)
    for x_val, y_val, wid in zip(x, vals, class_data['wetland_id'].dropna().values):
        ax.annotate(str(wid), (x_val, y_val), textcoords='offset points', xytext=(4, 3), fontsize=7, alpha=0.9, zorder=4)

ax.set_ylabel('PTC (%)', fontsize=12)
ax.set_xlabel('Connectivity', fontsize=12)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.show()

# %% 8.0 Summary Connectivity statistics for each class

stats_rows = []
for c in connect_order:
    vals = ptc_summary.loc[ptc_summary['connectivity'] == c, 'ptc'].dropna().to_numpy() * 100
    q1, q3 = np.percentile(vals, [25, 75])
    stats_rows.append({
        'connectivity': connectivity_config[c]['label'],
        'mean': np.mean(vals),
        'sd': np.std(vals, ddof=1),
        'iqr': q3 - q1,
        'n': len(vals),
    })

summary_stats = pd.DataFrame(stats_rows)
print(summary_stats)

# %% 9.0 1-way ANOVA on Connectivity for each class

anova_groups = [
    ptc_summary.loc[ptc_summary['connectivity'] == c, 'ptc'].dropna().to_numpy() * 100
    for c in connect_order
]
f_stat, p_val = f_oneway(*anova_groups)
print(f"PTC ANOVA: F={f_stat:.3f}, p={p_val:.4g}")

# %%
