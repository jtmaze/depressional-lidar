# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data_dir = 'D:/depressional_lidar/data/bradford/'

distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_LAI150m_domain_no_dry_days.csv'
connectivity_path = f'{data_dir}/bradford_wetland_connect_logging_key.xlsx'
spills_path = f'{data_dir}/out_data/bradford_estimated_basin_spills_no_smooth.csv'
strong_models_path = f'{data_dir}/out_data/strong_ols_models_150m_domain_no_dry_days.csv'
hypsometry_path = f'{data_dir}/out_data/bradford_hypsometry_curves.csv'

# %% 2.0 Read the data

spills = pd.read_csv(spills_path)
distributions = pd.read_csv(distributions_path)
connect = pd.read_excel(connectivity_path)
hypsometry_data = pd.read_csv(hypsometry_path)

strong_pairs = pd.read_csv(strong_models_path)
distributions = distributions.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id'],
    right_on=['log_id', 'ref_id'],
    how='inner'
)

distributions = distributions.merge(
    connect[['wetland_id', 'connectivity']],
    left_on='log_id',
    right_on='wetland_id',
    how='left'
)
# %% 3.0 Find the low elevations for each wetland

summary_elevations = hypsometry_data.groupby(['wetland_id']).agg(
    pct10_elev=('elev_bin_center', lambda x: np.percentile(x, 10)),
    pct90_elev=('elev_bin_center', lambda x: np.percentile(x, 90))
)

spills = spills.merge(summary_elevations, on='wetland_id')
print(spills)

# %% 3.1 Add a boxplot of interdecile range for hypsometry

# Compute interdecile range from hypsometry and merge connectivity.
summary_elevations = summary_elevations.reset_index()
summary_elevations['interdecile_range'] = summary_elevations['pct90_elev'] - summary_elevations['pct10_elev']
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
    labels=['Unconnected', 'Ditch connected', 'Flow-through connected'],
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
    if len(class_idr) == 0:
        continue
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

ax.set_ylabel('Hypsometry interdecile range (m)')
ax.set_xlabel('Connectivity')
ax.grid(axis='y', alpha=0.25)
plt.tight_layout()
plt.show()

# %% 3.2 Print statistics and ANOVAs for interdecile range across categories.

from scipy import stats

for conn in connect_order:
    vals = hypsometry_idr.loc[hypsometry_idr['connectivity'] == conn, 'interdecile_range'].dropna().values
    print(
        f"{conn}: n={len(vals)}, mean={vals.mean():.3f}, sd={vals.std(ddof=1):.3f}, "
        f"p25={np.percentile(vals, 25):.3f}, p75={np.percentile(vals, 75):.3f}, "
        f"iqr={(np.percentile(vals, 75) - np.percentile(vals, 25)):.3f}"
    )

all_vals = hypsometry_idr['interdecile_range'].dropna().values
print(f"all wetlands: n={len(all_vals)}, mean={all_vals.mean():.3f}, sd={all_vals.std(ddof=1):.3f}")

f_stat, p_val = stats.f_oneway(
    *(hypsometry_idr.loc[hypsometry_idr['connectivity'] == conn, 'interdecile_range'].dropna().values
      for conn in connect_order)
)
print(f"ANOVA: F={f_stat:.3f}, p={p_val:.4g}")

# %% 4.0 Adjust the distribution depths to the basin depths

spills['offset'] = spills['well_elev'] - spills['pct10_elev'] # NOTE also test min_elev

distributions = distributions.merge(
    spills[['wetland_id', 'offset']],
    left_on='log_id',
    right_on='wetland_id',
    how='left'
)
distributions['pre_adj'] = distributions['pre'] + distributions['offset']
distributions['post_adj'] = distributions['post'] + distributions['offset']

distributions['pre_adj_above'] = np.where(distributions['pre_adj'] < 0, np.nan, distributions['pre_adj'])
distributions['post_adj_above'] = np.where(distributions['post_adj'] < 0, np.nan, distributions['post_adj'])

# %% 5.0 Bar plot pre and post coefficient of variation by log_id


# summarize to one row per wetland and compute coefficient of variation
summary = distributions.groupby(['log_id', 'connectivity'], as_index=False).agg(
    pre_mean=('pre_adj_above', 'mean'),
    pre_std=('pre_adj_above', 'std'),
    post_mean=('post_adj_above', 'mean'),
    post_std=('post_adj_above', 'std')
)

summary['pre_cv'] = np.where(summary['pre_mean'] != 0, summary['pre_std'] / summary['pre_mean'], np.nan)
summary['post_cv'] = np.where(summary['post_mean'] != 0, summary['post_std'] / summary['post_mean'], np.nan)

plot_df = summary[['log_id', 'connectivity', 'pre_cv', 'post_cv']].copy()

# connectivity order and colors
connect_order = ['giw', 'first order', 'flow-through']
connect_colors = {
    'giw': '#1B7F79',
    'first order': '#6C5B7B',
    'flow-through': '#C46A1A'
}

plot_df['connectivity'] = pd.Categorical(
    plot_df['connectivity'],
    categories=connect_order,
    ordered=True
)

# order wetlands by connectivity class, then by log_id
plot_df = plot_df.sort_values(['connectivity', 'log_id']).reset_index(drop=True)

# long format for bar plotting
long_df = plot_df.melt(
    id_vars=['log_id', 'connectivity'],
    value_vars=['pre_cv', 'post_cv'],
    var_name='stage',
    value_name='value'
)

stage_order = ['pre_cv', 'post_cv']
x_positions = np.arange(len(plot_df))
bar_width = 0.35

fig, ax = plt.subplots(figsize=(14, 6))

for idx, row in plot_df.iterrows():
    color = connect_colors[row['connectivity']]
    pre_val = row['pre_cv']
    post_val = row['post_cv']

    ax.bar(
        x_positions[idx] - bar_width / 2,
        pre_val,
        width=bar_width,
        color=color,
        alpha=0.85
    )
    ax.bar(
        x_positions[idx] + bar_width / 2,
        post_val,
        width=bar_width,
        color=color,
        alpha=0.45
    )

    # ax.plot(
    #     [x_positions[idx] - bar_width / 2, x_positions[idx] + bar_width / 2],
    #     [pre_val, post_val],
    #     color='k',
    #     linewidth=0.7,
    #     alpha=0.35
    # )

ax.axhline(0, color='black', linewidth=1)
ax.set_xticks(x_positions)
ax.set_xticklabels(plot_df['log_id'], rotation=90, fontsize=8)
ax.set_ylabel('Coefficient of variation')
ax.set_xlabel('log_id')

# legend for connectivity classes
legend_handles = [
    plt.Line2D([0], [0], color=c, lw=8, label=lab)
    for lab, c in connect_colors.items()
]
ax.legend(handles=legend_handles, title='Connectivity', frameon=False)

plt.tight_layout()
plt.show()




