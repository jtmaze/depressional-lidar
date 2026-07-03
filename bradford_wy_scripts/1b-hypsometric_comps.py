# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data_dir = 'D:/depressional_lidar/data/bradford/'

connectivity_path = f'{data_dir}/bradford_wetland_connect_logging_key.xlsx'
spills_path = f'{data_dir}/out_data/bradford_estimated_basin_spills.csv'
hypsometry_path = f'{data_dir}/out_data/bradford_hypsometry_curves.csv'

# %% 2.0 Read the data

spills = pd.read_csv(spills_path)
connect = pd.read_excel(connectivity_path)
hypsometry_data = pd.read_csv(hypsometry_path)

# %% 3.0 Find the low elevations for each wetland

summary_elevations = hypsometry_data.groupby(['wetland_id']).agg(
    pct10_elev=('elev_bin_center', lambda x: (np.percentile(x, 10))),
    pct90_elev=('elev_bin_center', lambda x: (np.percentile(x, 90)))
)

spills = spills.merge(summary_elevations, on='wetland_id')
print(spills.columns)

# %% 3.1 Add a boxplot of interdecile range for hypsometry

# Compute interdecile range from hypsometry and merge connectivity.
summary_elevations = summary_elevations.reset_index()
summary_elevations['interdecile_range'] = (
    (summary_elevations['pct90_elev'] - summary_elevations['pct10_elev']) * 100
)
print(summary_elevations['interdecile_range'].mean())
print(summary_elevations['interdecile_range'].std())

# %% 3.2 Boxplot of elevation interdecile range.

summary_elevations = summary_elevations.merge(
    connect[['wetland_id', 'connectivity']],
    on='wetland_id',
    how='left'
)

hypsometry_idr = summary_elevations.dropna(subset=['interdecile_range', 'connectivity']).copy()
connect_order = ['giw', 'first order', 'flow-through']
connect_colors = {
    'giw': '#1B7F79',
    'first order': '#6C5B7B',
    'flow-through': '#C46A1A'
}

box_data = [
    hypsometry_idr.loc[hypsometry_idr['connectivity'] == conn, 'interdecile_range'].values
    for conn in connect_order
]

fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(
    box_data,
    tick_labels=['Unditched', 'Ditch connected', 'Flow-through connected'],
    patch_artist=True,
    widths=0.55,
    showfliers=False
)

for patch, conn in zip(box['boxes'], connect_order):
    patch.set_facecolor(connect_colors[conn])
    patch.set_alpha(0.65)

for median in box['medians']:
    median.set_color('black')
    median.set_linewidth(2)

for idx, conn in enumerate(connect_order, start=1):
    class_idr = hypsometry_idr.loc[
        hypsometry_idr['connectivity'] == conn,
        'interdecile_range'
    ].dropna().values

    x_jitter = np.random.normal(loc=idx, scale=0.05, size=len(class_idr))
    ax.scatter(
        x_jitter,
        class_idr,
        color=connect_colors[conn],
        edgecolor='white',
        linewidth=0.6,
        alpha=0.8,
        s=40
    )

ax.set_ylabel('P90-P10 (cm)')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.show()

# %% 4.0 ANOVA with means for each connectivity class

# Summarize means by connectivity class for quick interpretation.
anova_df = hypsometry_idr[hypsometry_idr['connectivity'].isin(connect_order)].copy()
class_summary = (
    anova_df.groupby('connectivity')['interdecile_range']
    .agg(['count', 'mean', 'std', 'median'])
    .reindex(connect_order)
)

print('\nInterdecile range summary by connectivity class (cm):')
print(class_summary)

anova_groups = [
    anova_df.loc[anova_df['connectivity'] == conn, 'interdecile_range'].dropna().values
    for conn in connect_order
]

f_stat, p_val = stats.f_oneway(*anova_groups)
print('\nOne-way ANOVA (interdecile range ~ connectivity)')
print(f'F = {f_stat:.3f}, p = {p_val:.4g}')

# %%
