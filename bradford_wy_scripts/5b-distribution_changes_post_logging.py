# %% 1.0 Libraries and file paths

"""
NOTE: I needed to contemplate a minor issue.
The number of unique wells dramatically decreases at low and high spill depths. 
For example at -1.5m below, the spill depth there might only be a few logged wetlands with data.
How can I ensure the upper and lower bounds of the Q-Q plot aren't warped by data from a single well?
Maybe truncate the curves whenever less than three are represented?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lai_buffer_dist = 150
data_set = 'no_dry_days'

data_dir = "D:/depressional_lidar/data/bradford/"
distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_wetlandLAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_wetland{lai_buffer_dist}m_domain_{data_set}.csv'
connectivity_key_path = data_dir + '/bradford_wetland_connect_logging_key.xlsx'
est_spills_path = data_dir + '/out_data/bradford_estimated_basin_spills.csv'

# %% 2.0 Read and munge the data

spills = pd.read_csv(est_spills_path)
distributions = pd.read_csv(distributions_path)
connect = pd.read_excel(connectivity_key_path)

# Only keep strong models
strong_pairs = pd.read_csv(strong_wetland_pairs_path)
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

# %% 2.1 Adjust the distributions so that well data is relative to spill depth

spills = spills[['wetland_id', 'well_elev', 'max_fill_delineated', 'max_fill_elev']]
spills['well_to_spill'] = spills['well_elev'] - (spills['max_fill_elev'] + 0.15) # NOTE: Added adjustment for the Spill here
spills = spills[['wetland_id', 'well_to_spill']]


distributions = distributions.merge(
    spills,
    left_on='log_id',
    right_on='wetland_id',
    how='left'
)

distributions['pre_adj'] = distributions['pre'] + distributions['well_to_spill']
distributions['post_adj'] = distributions['post'] + distributions['well_to_spill']

distributions_clean = distributions.copy()

# %% 3.0 Q-Q plot for all wells independent of connectivity class. 

dist_all = distributions_clean[
    (distributions_clean['pre_adj'] >= -2.1) & (distributions_clean['pre_adj'] <= 0.75) &
    (distributions_clean['post_adj'] >= -2.1) & (distributions_clean['post_adj'] <= 0.75)
].copy()

pre_all = dist_all['pre_adj'].dropna()
post_all = dist_all['post_adj'].dropna()

n_quantiles = 1000
quantiles_all = np.linspace(0, 1, n_quantiles)
pre_q_all = np.quantile(pre_all, quantiles_all)
post_q_all = np.quantile(post_all, quantiles_all)

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

ax.scatter(pre_q_all, post_q_all, s=30, alpha=0.8, color='black', label='All wells (aggregate)')
ax.plot([pre_q_all.min(), pre_q_all.max()],
        [pre_q_all.min(), pre_q_all.max()],
        linestyle='--', linewidth=2, color='black', label='1:1 line')
ax.axvline(0, color='grey', linestyle='-', linewidth=10, alpha=0.5, label='Spill Depth')

ax.set_xlabel('Pre-Logging [m]', fontsize=15)
ax.set_ylabel('Post-Logging [m]', fontsize=15)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-2, 0.75)
ax.set_ylim(-2, 0.75)
ax.legend(fontsize=10, loc='upper left', markerscale=1.1)

plt.tight_layout()
plt.show()

# %% 4.0 Q-Q plot aggregated by connectivity class

connectivity_config = {
    'first order': {'color': '#6C5B7B', 'label': 'Ditch Connected', 'marker': 's'},
    'giw': {'color': '#1B7F79', 'label': 'Unditched', 'marker': '^'}, 
    'flow-through': {'color': '#C46A1A', 'label': 'Flow-through connected', 'marker': 'X'}
}

# %% 5.0 Q-Q plot for each individual wetland

wetland_filtered = distributions_clean[
    (distributions_clean['pre_adj'] >= -2.1) & (distributions_clean['pre_adj'] <= 0.75) &
    (distributions_clean['post_adj'] >= -2.1) & (distributions_clean['post_adj'] <= 0.75)
].copy()

wetland_ids = sorted(wetland_filtered['log_id'].unique())

xmin, xmax = -2, 0.75

for wetland_id in wetland_ids:
    wetland_df = wetland_filtered[wetland_filtered['log_id'] == wetland_id]
    wetland_df = wetland_df.dropna(subset=['pre_adj', 'post_adj'])

    conn_class = wetland_df['connectivity'].iloc[0]
    conn_style = connectivity_config.get(
        conn_class,
        {'color': '#666666', 'label': 'Unknown'}
    )

    q_local = np.linspace(0, 1, 250)
    pre_q = np.quantile(wetland_df['pre_adj'], q_local)
    post_q = np.quantile(wetland_df['post_adj'], q_local)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    ax.scatter(pre_q, post_q, s=15, alpha=0.7, color=conn_style['color'])
    ax.plot([xmin, xmax], [xmin, xmax], linestyle='--', linewidth=1.5, color='black', label='1:1 line')
    ax.axvline(0, color='grey', linestyle='-', linewidth=8, alpha=0.35, label='Spill Depth')

    ax.set_title(f'Wetland {wetland_id} | {conn_style["label"]}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pre-Logging [m]', fontsize=12)
    ax.set_ylabel('Post-Logging [m]', fontsize=12)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.show()


# %% 4.0 Q-Q plot aggregated by logged wetland, then by connectivity class

# Draw order controls overlap; last drawn is on top.
draw_order = ['flow-through', 'first order', 'giw']

# Legend order shown to reader.
legend_conn_order = ['giw', 'first order', 'flow-through']

# Filter to plotting domain
dist_filtered = distributions_clean[
    (distributions_clean['pre_adj'] >= -2.1) & (distributions_clean['pre_adj'] <= 0.75) &
    (distributions_clean['post_adj'] >= -2.1) & (distributions_clean['post_adj'] <= 0.75)
].copy()

n_quantiles = 200
quantiles = np.linspace(0, 1, n_quantiles)

# %% 4.1 Calculate one Q-Q curve per logged wetland

wetland_quantile_dfs = []

for log_id, wetland_df in dist_filtered.groupby('log_id'):

    conn_values = wetland_df['connectivity'].unique()
    conn_class = conn_values[0]

    pre_q_cm = np.quantile(wetland_df['pre_adj'], quantiles) * 100
    post_q_cm = np.quantile(wetland_df['post_adj'], quantiles) * 100

    n_model_pairs = wetland_df[['log_id', 'ref_id']].drop_duplicates().shape[0]

    wetland_quantile_dfs.append(
        pd.DataFrame({
            'log_id': log_id,
            'connectivity': conn_class,
            'quantile': quantiles,
            'pre_q_cm': pre_q_cm,
            'post_q_cm': post_q_cm,
            'n_model_pairs': n_model_pairs
        })
    )

wetland_quantiles = pd.concat(wetland_quantile_dfs, ignore_index=True)

# %% 4.2 Average the wetland Q-Q curves across all wetlands

pooled_wetland_quantiles = (
    wetland_quantiles
    .groupby('quantile', as_index=False)
    .agg(
        pre_q_cm=('pre_q_cm', 'mean'),
        post_q_cm=('post_q_cm', 'mean'),
        n_wetlands=('log_id', 'nunique')
    )
)

pre_q_all_wetland = pooled_wetland_quantiles['pre_q_cm'].to_numpy()
post_q_all_wetland = pooled_wetland_quantiles['post_q_cm'].to_numpy()
diff_all_wetland = post_q_all_wetland - pre_q_all_wetland

idx_neg05 = np.argmin(np.abs(pre_q_all_wetland - (-50)))
idx_spill = np.argmin(np.abs(pre_q_all_wetland - 0.0))

n_wetlands_all = int(pooled_wetland_quantiles['n_wetlands'].max())

print(f"\nAll wetlands pooled at wetland level | n logged wetlands = {n_wetlands_all}")

print(f"Quantile difference at pre = -0.5 m: {diff_all_wetland[idx_neg05]:.3f} cm  "
      f"(pre={pre_q_all_wetland[idx_neg05]:.3f} cm, "
      f"post={post_q_all_wetland[idx_neg05]:.3f} cm)")

print(f"Quantile difference at pre =  0.0 m: {diff_all_wetland[idx_spill]:.3f} cm  "
      f"(pre={pre_q_all_wetland[idx_spill]:.3f} cm, "
      f"post={post_q_all_wetland[idx_spill]:.3f} cm)")

# %% 4.3 Aggregate wetland Q-Q curves by connectivity

class_quantiles = (
    wetland_quantiles
    .groupby(['connectivity', 'quantile'], as_index=False)
    .agg(
        pre_q_cm=('pre_q_cm', 'mean'),
        post_q_cm=('post_q_cm', 'mean'),
        n_wetlands=('log_id', 'nunique')
    )
)

# %% 4.4 Render the Q-Q plot 

fig, ax = plt.subplots(1, 1, figsize=(9, 8))

axis_label_fs = 17
tick_label_fs = 14

for conn_class in draw_order:
    config = connectivity_config[conn_class]
    class_df = class_quantiles[class_quantiles['connectivity'] == conn_class].copy()

    pre_q = class_df['pre_q_cm'].to_numpy()
    post_q = class_df['post_q_cm'].to_numpy()

    diff = post_q - pre_q

    # idx_neg05 = np.argmin(np.abs(pre_q - (-50)))
    # idx_spill = np.argmin(np.abs(pre_q - 0.0))


    # print(f"Quantile difference at pre = -0.5 m: {diff[idx_neg05]:.3f} cm  "
    #       f"(pre={pre_q[idx_neg05]:.3f} cm, post={post_q[idx_neg05]:.3f} cm)")
    # print(f"Quantile difference at pre =  0.0 m: {diff[idx_spill]:.3f} cm  "
    #       f"(pre={pre_q[idx_spill]:.3f} cm, post={post_q[idx_spill]:.3f} cm)")

    ax.scatter(
        pre_q,
        post_q,
        alpha=1,
        s=30,
        color=config['color'],
        label=f"{config['label']}",
        zorder=4
    )

# 1:1 line
ax.plot(
    [-120, 70],
    [-120, 70],
    linestyle='--',
    linewidth=2,
    color='black',
    label='1:1 line',
    zorder=2
)

# Spill depth threshold
ax.axvline(
    0,
    color='red',
    linestyle='-',
    linewidth=4,
    label='Spill Depth',
    alpha=0.7,
    zorder=1
)

ax.set_xlabel('h$_{L,\\ pre}$ (cm)', fontsize=axis_label_fs)
ax.set_ylabel('h$_{L,\\ post}$ (cm)', fontsize=axis_label_fs)

ax.tick_params(axis='both', labelsize=tick_label_fs)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-120, 70)
ax.set_ylim(-120, 70)

# Ordered legend
handles, labels = ax.get_legend_handles_labels()
label_to_handle = {label: handle for handle, label in zip(handles, labels)}

ordered_labels = [connectivity_config[c]['label'] for c in legend_conn_order]
ordered_labels += [label for label in labels if label not in ordered_labels]

ordered_handles = [label_to_handle[label] for label in ordered_labels if label in label_to_handle]

ax.legend(
    ordered_handles,
    ordered_labels,
    loc='upper right',
    fontsize=12,
    frameon=True,
    framealpha=1,
    markerscale=2
)

plt.tight_layout()
plt.show()

# %% 5.0 Quantify the shift in wetland pre and post logging distributions for depth increments (not Quantiles)

depth_grid_cm = np.arange(-100, 31, 5)
q_depth = np.linspace(0, 1, 500)

depth_shift_dfs = []

for log_id, wetland_df in dist_filtered.groupby('log_id'):

    wetland_df = wetland_df.dropna(subset=['pre_adj', 'post_adj'])

    conn_class = wetland_df['connectivity'].iloc[0]

    pre_q_cm = np.quantile(wetland_df['pre_adj'], q_depth) * 100
    post_q_cm = np.quantile(wetland_df['post_adj'], q_depth) * 100

    post_on_depth_cm = np.interp(
        depth_grid_cm,
        pre_q_cm,
        post_q_cm,
        left=np.nan,
        right=np.nan
    )

    depth_shift_dfs.append(
        pd.DataFrame({
            'log_id': log_id,
            'connectivity': conn_class,
            'depth_cm': depth_grid_cm,
            'post_on_depth_cm': post_on_depth_cm,
            'delta_cm': post_on_depth_cm - depth_grid_cm
        })
    )

depth_shifts = pd.concat(depth_shift_dfs, ignore_index=True)

# %% 

target_depth_shifts = depth_shifts[
    depth_shifts['depth_cm'].isin([-100, -75, -50, 0, 10])
].copy()

target_summary = (
    target_depth_shifts
    .groupby('depth_cm', as_index=False)
    .agg(
        mean_delta_cm=('delta_cm', 'mean'),
        sd_delta_cm=('delta_cm', 'std'),
        median_delta_cm=('delta_cm', 'median'),
        n_wetlands=('delta_cm', 'count')
    )
)

print("\nWetland-level fixed-depth shifts:")
print(target_summary)
# %%

target_summary_by_conn = (
    target_depth_shifts
    .groupby(['connectivity', 'depth_cm'], as_index=False)
    .agg(
        mean_delta_cm=('delta_cm', 'mean'),
        sd_delta_cm=('delta_cm', 'std'),
        median_delta_cm=('delta_cm', 'median'),
        n_wetlands=('delta_cm', 'count')
    )
)

print("\nWetland-level fixed-depth shifts by connectivity:")
print(target_summary_by_conn)

# %% 5.X Pair-level fixed-depth shifts

pair_depth_shift_dfs = []

for (log_id, ref_id), pair_df in dist_filtered.groupby(['log_id', 'ref_id']):

    conn_class = pair_df['connectivity'].iloc[0]

    pre_q_cm = np.quantile(pair_df['pre_adj'], q_depth) * 100
    post_q_cm = np.quantile(pair_df['post_adj'], q_depth) * 100

    post_on_depth_cm = np.interp(
        depth_grid_cm,
        pre_q_cm,
        post_q_cm,
        left=np.nan,
        right=np.nan
    )

    pair_depth_shift_dfs.append(
        pd.DataFrame({
            'log_id': log_id,
            'ref_id': ref_id,
            'connectivity': conn_class,
            'depth_cm': depth_grid_cm,
            'post_on_depth_cm': post_on_depth_cm,
            'delta_cm': post_on_depth_cm - depth_grid_cm
        })
    )

pair_depth_shifts = pd.concat(pair_depth_shift_dfs, ignore_index=True)

# %%

target_depth_cm = -75
rng = np.random.default_rng(42)

# Wetland-level points
wetland_plot = target_depth_shifts[
    target_depth_shifts['depth_cm'] == target_depth_cm
].copy()

wetland_plot['x'] = rng.normal(0, 0.03, size=len(wetland_plot))

# Pair-level points
pair_plot = pair_depth_shifts[
    pair_depth_shifts['depth_cm'] == target_depth_cm
].copy()

pair_plot['x'] = rng.normal(0, 0.03, size=len(pair_plot))

fig, ax = plt.subplots(1, 1, figsize=(2, 3))

# Pair-level dots in background
for conn_class, config in connectivity_config.items():
    pair_df = pair_plot[pair_plot['connectivity'] == conn_class].copy()

    ax.scatter(
        pair_df['x'],
        pair_df['delta_cm'],
        s=5,
        alpha=0.5,
        color=config['color'],
        zorder=1
    )

# Wetland-level dots on top
for conn_class, config in connectivity_config.items():
    plot_df = wetland_plot[wetland_plot['connectivity'] == conn_class].copy()

    ax.scatter(
        plot_df['x'],
        plot_df['delta_cm'],
        s=75,
        alpha=0.9,
        color=config['color'],
        label=config['label'],
        edgecolor='black',
        zorder=3
    )

ax.axhline(0, color='black', linestyle='--', linewidth=1.2, zorder=0)

ax.set_xlim(-0.18, 0.18)
ax.set_xticks([0])

ax.set_xticklabels([f'{target_depth_cm} cm'])
#ax.set_ylabel('Diff (cm)', fontsize=11)

ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.ylim(-50, 100)
y_min, y_max = ax.get_ylim()
y_min_rounded = np.floor(y_min / 5) * 5
y_max_rounded = np.ceil(y_max / 5) * 5
ax.set_yticks([int(y_min_rounded), int((y_min_rounded + y_max_rounded) / 2), int(y_max_rounded)])
plt.show()
# %%



# %%
