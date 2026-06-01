# %% 1.0 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sites = ['bradford', 'osbs', 'delmarva']

# %% Read the data

hypsometry = []
area = []
tai = []

for i in sites:
    data_dir = f'D:/depressional_lidar/data/{i}/out_data/basin_tai_stats/'
    hyps_path = f'{data_dir}/hypsometry_cdfs_long.csv'
    area_path = f'{data_dir}/area_timeseries_long.csv'
    tai_path = f'{data_dir}/tai_timeseries_long.csv'

    hyps_data = pd.read_csv(hyps_path)
    hyps_data['site'] = i

    area_data = pd.read_csv(area_path)
    area_data['site'] = i

    tai_data = pd.read_csv(tai_path)
    tai_data['site'] = i

    hypsometry.append(hyps_data)
    area.append(area_data)
    tai.append(tai_data)

hypsometry_df = pd.concat(hypsometry)
area_df = pd.concat(area)
tai_df = pd.concat(tai)

# %% Zero elevations and normalize area

hypsometry_df['rel_elev'] = hypsometry_df.groupby('wetland_id')['elevation_m'].transform(
    lambda x: x - x.min()
)
hypsometry_df['norm_area'] = hypsometry_df.groupby('wetland_id')['cum_area_m2'].transform(
    lambda x: x / x.max()
)

# %% Three-panel hypsometry plot

site_colors = {'bradford': 'orange', 'delmarva': 'green', 'osbs': 'blue'}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)

for ax, site in zip(axes, sites):
    site_df = hypsometry_df[hypsometry_df['site'] == site]
    color = site_colors[site]

    # Individual wetland curves
    for wid, grp in site_df.groupby('wetland_id'):
        ax.plot(grp['rel_elev'], grp['norm_area'], color=color, alpha=0.4, linewidth=0.5)

    # Average curve: interpolate onto common relative-elevation grid, then average
    common_elev = np.linspace(0, site_df['rel_elev'].max(), 200)
    interp_areas = []
    for wid, grp in site_df.groupby('wetland_id'):
        grp_sorted = grp.sort_values('rel_elev')
        interp_area = np.interp(common_elev, grp_sorted['rel_elev'], grp_sorted['norm_area'])
        interp_areas.append(interp_area)

    mean_area = np.mean(interp_areas, axis=0)
    ax.plot(common_elev, mean_area, color='black', linewidth=2.5, label='Mean')

    ax.set_title(site)
    ax.set_xlabel('Relative Elevation (m)')
    if ax == axes[0]:
        ax.set_ylabel('Normalized Cumulative Area')
    ax.legend()

plt.tight_layout()
plt.show()

# %% Inundated area CDF plot
from matplotlib.ticker import MaxNLocator
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)

for ax, site in zip(axes, sites):
    site_df = area_df[area_df['site'] == site]
    color = site_colors[site]

    # Individual wetland CDFs
    all_sorted = []
    for wid, grp in site_df.groupby('wetland_id'):
        vals = np.sort(grp['area_m2'].dropna().values)
        if len(vals) > 1:
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=color, alpha=0.4, linewidth=0.5)
            all_sorted.append(vals)

    # Aggregate CDF across all wetlands at this site
    all_vals = np.sort(site_df['area_m2'].dropna().values)
    all_cdf = np.arange(1, len(all_vals) + 1) / len(all_vals)
    ax.plot(all_vals, all_cdf, color='black', linewidth=2.5, label='All wetlands')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_title(site)
    ax.set_xlabel('Inundated Area (m²)')
    if ax == axes[0]:
        ax.set_ylabel('CDF')
    ax.legend()

plt.tight_layout()
plt.show()

# %% TAI area CDF Plot

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)

for ax, site in zip(axes, sites):
    site_df = tai_df[tai_df['site'] == site]
    color = site_colors[site]

    # Individual wetland CDFs
    for wid, grp in site_df.groupby('wetland_id'):
        vals = np.sort(grp['tai_m2'].dropna().values)
        if len(vals) > 1:
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=color, alpha=0.4, linewidth=0.5)

    # Aggregate CDF across all wetlands at this site
    all_vals = np.sort(site_df['tai_m2'].dropna().values)
    all_cdf = np.arange(1, len(all_vals) + 1) / len(all_vals)
    ax.plot(all_vals, all_cdf, color='black', linewidth=2.5, label='All wetlands')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    ax.set_title(site)
    ax.set_xlabel('TAI Area (m²)')
    if ax == axes[0]:
        ax.set_ylabel('CDF')
    ax.legend()

plt.tight_layout()
plt.show()
# %%
