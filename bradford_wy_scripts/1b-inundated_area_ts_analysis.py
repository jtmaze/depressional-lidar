# %% 1.0 Libraries and file paths

import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import f_oneway

data_dir = "D:/depressional_lidar/data/bradford/"

est_spills_path = f"{data_dir}/out_data/bradford_estimated_basin_spills.csv"
well_data_path = f"{data_dir}/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
connectivity_path = f"{data_dir}/bradford_wetland_connect_logging_key.xlsx"
hypsometry_path = f"{data_dir}/out_data/bradford_hypsometry_curves.csv"

era5_path = f'{data_dir}/in_data/hydro_forcings_and_LAI/ERA5LAND_daily_mean.csv'

est_spills = pd.read_csv(est_spills_path)
est_spills = est_spills[['wetland_id', 'well_elev']]

well_data = pd.read_csv(well_data_path)
well_data['date'] = pd.to_datetime(well_data['date'])
well_data = well_data[~well_data['wetland_id'].isin(['Donor_wetland', 'Receiver_wetland'])]

connect = pd.read_excel(connectivity_path)
hyps = pd.read_csv(hypsometry_path)

# %% 2.0 Merge data frames of inundated area timeseries

# %% 2.1 Pre data by determining well's min depth and formatting hypsometry data
min_depth_table = (
    well_data.groupby('wetland_id', as_index=False)['well_depth_m']
    .min()
    .rename(columns={'well_depth_m': 'min_well_depth_m'})
    .sort_values('min_well_depth_m', ascending=False)
)

flag2_summary = (
    well_data.groupby('wetland_id')['flag']
    .apply(lambda s: (s == 2).mean())
    .reset_index(name='proportion_flag2')
    .sort_values('proportion_flag2', ascending=False)
)

est_spills = est_spills.merge(min_depth_table, on='wetland_id')
well_data = well_data.merge(est_spills, on='wetland_id')
well_data['wse'] = well_data['well_elev'] + well_data['well_depth_m']

# Add min-max scaled inundated area for each wetland
hyps['inundated_area_scaled'] = hyps.groupby('wetland_id')['inundated_area'].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
hyps.rename(columns={'elev_bin_center': 'wse'}, inplace=True)

hyps_bounds = (
    hyps.groupby('wetland_id', as_index=False)['wse']
    .agg(hyps_wse_min='min', hyps_wse_max='max')
)

# %% 2.2 Determine if well "no observation" days will be an issue for inundation calcs

hyps_p25 = (
    hyps.groupby('wetland_id')
    .apply(lambda group: group.iloc[(group['inundated_area_scaled'] - 0.25).abs().argmin()][['wse', 'inundated_area_scaled']], include_groups=False)
    .reset_index()
)
hyps_p25.columns = ['wetland_id', 'wse_at_25p_inundated', '__']

hyps_p25.drop(columns=['__'], inplace=True)

est_spills = est_spills.merge(hyps_p25, on='wetland_id')
est_spills['min_observable_elev'] = est_spills['well_elev'] + est_spills['min_well_depth_m']

est_spills['p25_bad'] = est_spills['min_observable_elev'] > est_spills['wse_at_25p_inundated']

print(len(est_spills[est_spills['p25_bad'] == True])) # NOTE no wells bottom out before 25% of basin flooded

# %% 2.2 Well data with hypsometry for inundated area timeseries

well_data['wse'] = well_data['wse'].round(2)
hyps['wse'] = hyps['wse'].round(2)

well_data = well_data.merge(hyps_bounds, on='wetland_id', how='left')
well_data = well_data.merge(hyps, on=['wetland_id', 'wse'], how='left')

# Fill out-of-range WSE values after merge using wetland-specific hypsometric bounds.
above_max = well_data['inundated_area_scaled'].isna() & (well_data['wse'] > well_data['hyps_wse_max'])
below_min = well_data['inundated_area_scaled'].isna() & (well_data['wse'] < well_data['hyps_wse_min'])

well_data.loc[above_max, 'inundated_area_scaled'] = 1.0
well_data.loc[below_min, 'inundated_area_scaled'] = 0.0

well_data['inundated_area_scaled'] = np.where(
    well_data['flag'] == 2, 
    0,
    well_data['inundated_area_scaled']
)

well_data = well_data.drop(columns=['hyps_wse_min', 'hyps_wse_max'])

del above_max, below_min

# %% 3.0 Show the coefficient of variation for each inundated area timeseries as a boxplot

connect = connect[['wetland_id', 'connectivity']]

cv_summary = (
    well_data.groupby('wetland_id', as_index=False)['inundated_area_scaled']
    .agg(
        mean_scaled='mean',
        sd_scaled=lambda s: s.std(ddof=1)
    )
)

# CV is undefined when mean is zero.
cv_summary['cv_scaled'] = np.where(
    cv_summary['mean_scaled'] > 0,
    cv_summary['sd_scaled'] / cv_summary['mean_scaled'],
    np.nan
)

cv_summary = cv_summary.merge(connect, on='wetland_id', how='left')

connectivity_config = {
    "first order": {"color": "#6C5B7B", "label": "Ditch connected"},
    "giw": {"color": "#1B7F79", "label": "Unditched"},
    "flow-through": {"color": "#C46A1A", "label": "Flow-through connected"},
}

connect_order = ['giw', 'first order', 'flow-through']
connect_labels = [connectivity_config[c]['label'] for c in connect_order]

cv_data = [
    cv_summary.loc[cv_summary['connectivity'] == c, 'cv_scaled'].dropna().values * 100
    for c in connect_order
]

fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(
    cv_data,
    tick_labels=connect_labels,
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
    class_data = cv_summary.loc[cv_summary['connectivity'] == c]
    vals = class_data['cv_scaled'].dropna().values * 100
    x = np.random.normal(i, 0.05, size=len(vals))
    ax.scatter(x, vals, color=connectivity_config[c]['color'], edgecolor='white', linewidth=0.6, alpha=0.8, s=45, zorder=3)
    for x_val, y_val, wid in zip(x, vals, class_data['wetland_id'].dropna().values):
        ax.annotate(str(wid), (x_val, y_val), textcoords='offset points', xytext=(4, 3), fontsize=7, alpha=0.9, zorder=4)

ax.set_ylabel('CV of inundated (%)', fontsize=12)
ax.set_xlabel('Connectivity', fontsize=12)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.show()

print(cv_summary['cv_scaled'].mean() * 100)
print(cv_summary['cv_scaled'].std() * 100)

# %% 4.0 Investigate inundated area and water level correlations

# %% 4.1 Inundated area correlations
ids = well_data['wetland_id'].unique()

pairs = list(itertools.combinations(ids, 2))
print(pairs)
from scipy.stats import pearsonr

correlations = []
for w1, w2 in pairs:
    ts1 = (
        well_data.loc[well_data['wetland_id'] == w1, ['date', 'inundated_area_scaled']]
        .set_index('date')
        .rename(columns={'inundated_area_scaled': 'w1'})
    )
    ts2 = (
        well_data.loc[well_data['wetland_id'] == w2, ['date', 'inundated_area_scaled']]
        .set_index('date')
        .rename(columns={'inundated_area_scaled': 'w2'})
    )
    merged = ts1.join(ts2, how='inner').dropna()

    r, p = pearsonr(merged['w1'], merged['w2'])
    correlations.append({'w1': w1, 'w2': w2, 'r': r, 'p': p})

corr_df = pd.DataFrame(correlations)
print(f"Mean Pearson's r:   {corr_df['r'].mean():.3f}")
print(f"Median Pearson's r: {corr_df['r'].median():.3f}")
print(corr_df.sort_values('r', ascending=False).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
mean_r = corr_df['r'].mean()
ax.hist(corr_df['r'], bins=30, color='gray', edgecolor='black', alpha=0.7)
ax.axvline(mean_r, color='red', linestyle='--', linewidth=2, label=f"Mean r = {mean_r:.3f}")
ax.set_xlabel("Pearson's r", fontsize=11)
ax.set_ylabel('Pair count', fontsize=11)
ax.grid(alpha=0.25, axis='y')
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% 4.2 Water level correlations

correlations_wse = []
for w1, w2 in pairs:
    well_data = well_data[~well_data['flag'].isin([1, 2])]
    ts1 = (
        well_data.loc[well_data['wetland_id'] == w1, ['date', 'well_depth_m']]
        .set_index('date')
        .rename(columns={'well_depth_m': 'w1'})
    )
    ts2 = (
        well_data.loc[well_data['wetland_id'] == w2, ['date', 'well_depth_m']]
        .set_index('date')
        .rename(columns={'well_depth_m': 'w2'})
    )
    merged = ts1.join(ts2, how='inner').dropna()
    r, p = pearsonr(merged['w1'], merged['w2'])
    correlations_wse.append({'w1': w1, 'w2': w2, 'r': r, 'p': p, 'n': len(merged)})

corr_wse_df = pd.DataFrame(correlations_wse)
print(f"Mean Pearson's r (WSE):   {corr_wse_df['r'].mean():.3f}")
print(f"Median Pearson's r (WSE): {corr_wse_df['r'].median():.3f}")
print(corr_wse_df.sort_values('r', ascending=False).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
mean_r_wse = corr_wse_df['r'].mean()
ax.hist(corr_wse_df['r'], bins=30, color='gray', edgecolor='black', alpha=0.7)
ax.axvline(mean_r_wse, color='red', linestyle='--', linewidth=2, label=f"Mean r = {mean_r_wse:.3f}")
ax.set_xlabel("Pearson's r", fontsize=11)
ax.set_ylabel('Pair count', fontsize=11)
ax.set_title("Water depth correlations between wetlands", fontsize=12, fontweight='bold')
ax.grid(alpha=0.25, axis='y')
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# %% 5.0 Compare the mean inundated fraction across water years. 

water_years = {
    'WY2022': ('2021-10-01', '2022-09-30'),
    'WY2023': ('2022-10-01', '2023-09-30'),
    'WY2024': ('2023-10-01', '2024-09-30'),
    'WY2025': ('2024-10-01', '2025-09-30'),
}
era5_data = pd.read_csv(era5_path)
era5_data = era5_data[['date_local', 'precip_m', 'pet_m']]
era5_data['date_local'] = pd.to_datetime(era5_data['date_local'])

wy_results = []
for wy_name, (start_date, end_date) in water_years.items():

    mask = (well_data['date'] >= start_date) & (well_data['date'] <= end_date)
    wy_data = well_data.loc[mask]
    mean_inund = wy_data['inundated_area_scaled'].mean()

    era5_mask = (era5_data['date_local'] >= start_date) & (era5_data['date_local'] <= end_date)
    era5_wy = era5_data.loc[era5_mask]
    
    total_precip = era5_wy['precip_m'].sum()
    total_pet = era5_wy['pet_m'].sum()
    
    wy_results.append({
        'Water Year': wy_name,
        'Start Date': start_date,
        'End Date': end_date,
        'Mean Inundated Fraction': mean_inund,
        'Total Precip (m)': total_precip,
        'Total PET (m)': total_pet,
    })

wy_summary = pd.DataFrame(wy_results)
print(wy_summary.to_string(index=False))

# %% 5.0 Plot inundated area scaled timeseries for target wetland IDs

# Define target wetland IDs to visualize inundation timeseries
tgt_ids = ['6_629', '9_609']

# Plot each target wetland's timeseries
for wid in tgt_ids:
    wid_data = well_data[well_data['wetland_id'] == wid].sort_values('date')
    
    fig, ax = plt.subplots(figsize=(12, 3))
    line1 = ax.plot(
        wid_data['date'],
        wid_data['inundated_area_scaled'],
        linewidth=1.5,
        color='steelblue',
        label='Inundated fraction'
    )
    ax2 = ax.twinx()
    line2 = ax2.plot(
        wid_data['date'],
        wid_data['wse'],
        linewidth=1.5,
        color='red',
        alpha=0.85,
        label='WSE'
    )
    ax.set_title(f'{wid}', fontsize=11, fontweight='bold')
    ax.set_ylabel('Inundated fraction (0-1)', fontsize=10)
    ax2.set_ylabel('WSE (m)', fontsize=10, color='red')
    ax2.tick_params(axis='y', colors='red')
    ax.set_xlabel('Date', fontsize=10)
    ax.grid(alpha=0.3)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()

# %% 6.0 Every inundated fraction timeseries colored by connectivity class. 

plot_df = (
    well_data[['date', 'wetland_id', 'inundated_area_scaled']]
    .merge(connect[['wetland_id', 'connectivity']].drop_duplicates(), on='wetland_id', how='left')
)

fig, ax = plt.subplots(figsize=(12, 6))

for c in connect_order:
    class_df = plot_df.loc[plot_df['connectivity'] == c].copy()
    if class_df.empty:
        continue

    # Plot each wetland timeseries with low alpha so class-level pattern is visible.
    for _, wetland_ts in class_df.groupby('wetland_id'):
        wetland_ts = wetland_ts.sort_values('date')
        ax.plot(
            wetland_ts['date'],
            wetland_ts['inundated_area_scaled'],
            color=connectivity_config[c]['color'],
            alpha=0.5,
            linewidth=0.3,
            zorder=1,
        )

    class_mean = (
        class_df.groupby('date', as_index=False)['inundated_area_scaled']
        .mean()
        .sort_values('date')
    )
    ax.plot(
        class_mean['date'],
        class_mean['inundated_area_scaled'],
        color=connectivity_config[c]['color'],
        linewidth=3,
        alpha=1.0,
        label=f"{connectivity_config[c]['label']} mean",
        zorder=3,
    )

ax.set_ylabel('Inundated fraction (0-1)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylim(0, 1)
ax.set_xlim(pd.Timestamp("2021-10-01"), pd.Timestamp("2026-01-01"))
ax.grid(alpha=0.25)
fig.autofmt_xdate()
plt.tight_layout()
plt.show()

# %% 7.0 Boxplot for proportion of days with over 25% flooded area

flooded_25_summary = (
    well_data.groupby('wetland_id', as_index=False)['inundated_area_scaled']
    .agg(prop_days_over_25=lambda s: (s > 0.25).mean())
)

flooded_25_summary = flooded_25_summary.merge(connect, on='wetland_id', how='left')

flooded_25_data = [
    flooded_25_summary.loc[
        flooded_25_summary['connectivity'] == c,
        'prop_days_over_25'
    ].dropna().values * 100
    for c in connect_order
]

fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(
    flooded_25_data,
    tick_labels=connect_labels,
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
    class_data = flooded_25_summary.loc[flooded_25_summary['connectivity'] == c]
    vals = class_data['prop_days_over_25'].dropna().values * 100
    x = np.random.normal(i, 0.05, size=len(vals))
    ax.scatter(
        x,
        vals,
        color=connectivity_config[c]['color'],
        edgecolor='white',
        linewidth=0.6,
        alpha=0.8,
        s=45,
        zorder=3,
    )
    for x_val, y_val, wid in zip(x, vals, class_data['wetland_id'].dropna().values):
        ax.annotate(
            str(wid),
            (x_val, y_val),
            textcoords='offset points',
            xytext=(4, 3),
            fontsize=7,
            alpha=0.9,
            zorder=4,
        )

ax.set_ylabel('Proportion of days with >25% flooded area (%)', fontsize=12)
ax.set_xlabel('Connectivity', fontsize=12)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.ylim(0, 100)
plt.show()

# %% 8.0 Boxplot for mean inundated fraction

mean_inund_summary = (
    well_data.groupby('wetland_id', as_index=False)['inundated_area_scaled']
    .agg(mean_inundated_fraction='mean')
)

mean_inund_summary = mean_inund_summary.merge(connect, on='wetland_id', how='left')

mean_inund_data = [
    mean_inund_summary.loc[
        mean_inund_summary['connectivity'] == c,
        'mean_inundated_fraction'
    ].dropna().values * 100
    for c in connect_order
]

fig, ax = plt.subplots(figsize=(8, 6))
box = ax.boxplot(
    mean_inund_data,
    tick_labels=connect_labels,
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
    class_data = mean_inund_summary.loc[mean_inund_summary['connectivity'] == c]
    vals = class_data['mean_inundated_fraction'].dropna().values * 100
    x = np.random.normal(i, 0.05, size=len(vals))
    ax.scatter(
        x,
        vals,
        color=connectivity_config[c]['color'],
        edgecolor='white',
        linewidth=0.6,
        alpha=0.8,
        s=45,
        zorder=3,
    )
    for x_val, y_val, wid in zip(x, vals, class_data['wetland_id'].dropna().values):
        ax.annotate(
            str(wid),
            (x_val, y_val),
            textcoords='offset points',
            xytext=(4, 3),
            fontsize=7,
            alpha=0.9,
            zorder=4,
        )

ax.set_ylabel('Mean inundated fraction (%)', fontsize=12)
ax.set_xlabel('Connectivity', fontsize=12)
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.ylim(0, 100)
plt.show()

# %% 9.0 Summary stats on mean inundated fractions

stats_rows = []
for c in connect_order:
    vals = mean_inund_summary.loc[
        mean_inund_summary['connectivity'] == c,
        'mean_inundated_fraction'
    ].dropna().to_numpy() * 100
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

print(mean_inund_summary['mean_inundated_fraction'].mean())
print(mean_inund_summary['mean_inundated_fraction'].std())

# %% 9.1 1-way ANOVA on Connectivity for mean inundated fraction

anova_groups = [
    mean_inund_summary.loc[
        mean_inund_summary['connectivity'] == c,
        'mean_inundated_fraction'
    ].dropna().to_numpy() * 100
    for c in connect_order
]
f_stat, p_val = f_oneway(*anova_groups)
print(f"Mean inundated fraction ANOVA: F={f_stat:.3f}, p={p_val:.4g}")

# %%
