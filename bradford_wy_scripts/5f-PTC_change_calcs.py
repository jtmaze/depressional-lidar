# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

lai_buffer_dist = 150
data_set = 'no_dry_days'

data_dir = 'D:/depressional_lidar/data/bradford/'

distributions_path = f'{data_dir}/out_data/modeled_logging_stages/hypothetical_distributions_wetlandLAI{lai_buffer_dist}m_domain_{data_set}.csv'
strong_wetland_pairs_path = f'{data_dir}/out_data/strong_ols_models_wetland{lai_buffer_dist}m_domain_{data_set}.csv'
connectivity_key_path = f'{data_dir}/bradford_wetland_connect_logging_key.xlsx'
est_spills_path = f'{data_dir}/out_data/bradford_estimated_basin_spills_no_smooth.csv'
agg_shift_data_path = f'{data_dir}/out_data/modeled_logging_stages/shift_results_wetlandLAI{lai_buffer_dist}m_domain_{data_set}.csv'


# %% 2.0 Read data and filter for strong models

spills = pd.read_csv(est_spills_path)
distributions = pd.read_csv(distributions_path)
distributions = distributions[distributions['log_id'] != '9_332']
connect = pd.read_excel(connectivity_key_path)
n_bottomed = pd.read_csv(agg_shift_data_path)
n_bottomed = n_bottomed[['log_id', 'ref_id', 'total_obs', 'n_bottomed_out']]

# Only keep strong models
strong_pairs = pd.read_csv(strong_wetland_pairs_path)
distributions = distributions.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id'],
    right_on=['log_id', 'ref_id'],
    how='inner'
)


def swap_dry_days(depths, not_modeled_pct):
    """Replace random values with -1.5 based on proportion of dry days"""
    swap_depths = depths.copy().to_numpy()
    proportion = not_modeled_pct / 100

    n_to_swap = int(len(depths) * proportion)

    if n_to_swap > 0:
        swap_idx = np.random.choice(len(depths), size=n_to_swap, replace=False)
        swap_depths[swap_idx] = -1.5 # NOTE this values is arbitrary, but far below DEM elevations

    return swap_depths

# %% 3.0 Determine the PTC for in pre and post logging for each wetland

ptc_results = []

for i in distributions['log_id'].unique():

    wetland_data = distributions[distributions['log_id'] == i].copy()
    wetland_spill = spills[spills['wetland_id'] == i].copy()
    wetland_info = connect[connect['wetland_id'] == i].copy()
    wetland_connect = wetland_info['connectivity'].iloc[0]

    well_z = wetland_spill['well_elev'].iloc[0]
    spill_z = wetland_spill['max_fill_elev'].iloc[0]
    basin_min_z = wetland_spill['min_elev'].iloc[0]                                        
    spill_depth = wetland_spill['max_fill_delineated'].iloc[0]

    well_to_bottom = well_z - basin_min_z

    # NOTE:
    # Due to ditching, the elevation with the deepest spill depth (i.e., lowest depression point) might not be
    # the delineated area's absolute lowest point. Therefore, we need to find the difference between the basin's
    # absolute lowest point, and the lowest point in a filled depression. Most of the time, the lowest basin point has
    # the deepest spill depths, but other cases ditching interferes. 
    spill_lowest_z_abs_lowest = (spill_z - spill_depth) - basin_min_z
    spill_depth_adj = spill_depth - spill_lowest_z_abs_lowest

    wetland_data['pre_adj'] = wetland_data['pre'] + well_to_bottom
    wetland_data['post_adj'] = wetland_data['post'] + well_to_bottom

    for r in wetland_data['ref_id'].unique():

        n_bottomed_out = n_bottomed[
            (n_bottomed['log_id'] == i) & (n_bottomed['ref_id'] == r)
        ].copy()
        total_obs = n_bottomed_out['total_obs'].iloc[0]
        inval_obs = n_bottomed_out['n_bottomed_out'].iloc[0]

        not_modeled_pct = inval_obs / total_obs * 100

        pre_adj = wetland_data[wetland_data['ref_id'] == r]['pre_adj']
        post_adj = wetland_data[wetland_data['ref_id'] == r]['post_adj']

        pre_depth_with_dry = swap_dry_days(pre_adj, not_modeled_pct)
        post_depth_with_dry = swap_dry_days(post_adj, not_modeled_pct)

        pre_ptc = sum(pre_depth_with_dry > spill_depth_adj) / len(pre_adj) * 100
        print(pre_ptc)

        post_ptc = sum(post_depth_with_dry > spill_depth_adj) / len(post_adj) * 100
        print(post_ptc)

        d_ptc = post_ptc - pre_ptc

        result = {
            'log_id': i,
            'ref_id': r,
            'pre_ptc': pre_ptc,
            'post_ptc': post_ptc,
            'd_ptc': d_ptc,
            'connect': wetland_connect
        }

        ptc_results.append(result)

# %% 3.0 Visualize the change in percent time connected

ptc = pd.DataFrame(ptc_results)

ptc_summary = ptc.groupby('log_id').agg(
    pre_ptc=('pre_ptc', 'mean'),
    pre_sd=('pre_ptc', 'std'),
    post_ptc=('post_ptc', 'mean'),
    post_sd=('post_ptc', 'std'),
    d_ptc=('d_ptc', 'mean'),
    connect=('connect', 'first')
)
print(ptc_summary)

# %% 4.0 Visualize the pre versus post percent time connected

connectivity_config = {
    'first order': {'color': '#6C5B7B', 'label': 'Ditch connected', 'marker': 's'},
    'giw': {'color': '#1B7F79', 'label': 'Unconnected', 'marker': '^'},
    'flow-through': {'color': '#C46A1A', 'label': 'Flow-through connected', 'marker': 'X'}
}

conn_order = ['first order', 'giw', 'flow-through']

plot_df = ptc_summary.reset_index().copy()
plot_df['connect'] = pd.Categorical(plot_df['connect'], categories=conn_order, ordered=True)
plot_df = plot_df.sort_values(['connect', 'd_ptc']).reset_index(drop=True)

x = np.arange(len(plot_df))
bar_width = 0.4

bar_colors = [connectivity_config.get(c, {'color': '#888888'})['color'] for c in plot_df['connect']]

fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.55 * len(plot_df)), 6.5))

# Pre is solid fill; post is white fill with colored diagonal hatch.
ax.bar(
    x - bar_width / 2,
    plot_df['pre_ptc'],
    width=bar_width,
    yerr=plot_df['pre_sd'],
    capsize=3,
    color=bar_colors,
    edgecolor='black',
    linewidth=0.8,
    label='Pre PTC'
)

ax.bar(
    x + bar_width / 2,
    plot_df['post_ptc'],
    width=bar_width,
    yerr=plot_df['post_sd'],
    capsize=3,
    color='white',
    edgecolor=bar_colors,
    linewidth=1.3,
    hatch='///',
    label='Post PTC'
)

ax.set_xticks(x)
ax.set_xticklabels(plot_df['log_id'], rotation=65, ha='right')
ax.set_xlabel('log_id', fontsize=14)
ax.set_ylabel('Percent Time Connected (%)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=13)
ax.set_title('Pre vs Post Percent Time Connected by Wetland', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)

# Optional visual separators between connectivity blocks after sorting.
group_sizes = plot_df.groupby('connect', observed=True).size()
boundary = 0
for size in group_sizes[:-1]:
    boundary += size
    ax.axvline(boundary - 0.5, color='grey', linewidth=1.0, alpha=0.5)

connect_handles = [
    Patch(facecolor=cfg['color'], edgecolor='black', label=cfg['label'])
    for cfg in connectivity_config.values()
]
hatch_handles = [
    Patch(facecolor='lightgray', edgecolor='black', label='Pre PTC'),
    Patch(facecolor='white', edgecolor='black', hatch='///', label='Post PTC')
]

legend_handles = connect_handles + hatch_handles
ax.legend(handles=legend_handles, fontsize=10, ncol=3, loc='upper left', frameon=True)

plt.tight_layout()
plt.show()

# %% 5.0 Change in percent time connected grouped by connectivity

conn_summary = ptc_summary.groupby('connect', observed=True).agg(
    pre_ptc=('pre_ptc', 'mean'),
    pre_sd=('pre_ptc', 'std'),
    post_ptc=('post_ptc', 'mean'),
    post_sd=('post_ptc', 'std'),
    n_wetlands=('pre_ptc', 'size')
).reset_index()

chunk5_conn_order = ['giw', 'first order', 'flow-through']
conn_summary['connect'] = pd.Categorical(conn_summary['connect'], categories=chunk5_conn_order, ordered=True)
conn_summary = conn_summary.sort_values('connect').reset_index(drop=True)
conn_summary[['pre_sd', 'post_sd']] = conn_summary[['pre_sd', 'post_sd']].fillna(0)

x = np.arange(len(conn_summary))
bar_width = 0.34

conn_colors = [connectivity_config.get(c, {'color': '#888888'})['color'] for c in conn_summary['connect']]
conn_labels = [connectivity_config.get(c, {'label': str(c)})['label'] for c in conn_summary['connect']]

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for row in conn_summary.itertuples(index=False):
    print(
        f"{row.connect}: n={row.n_wetlands}, "
        f"pre={row.pre_ptc:.2f} +/- {row.pre_sd:.2f}, "
        f"post={row.post_ptc:.2f} +/- {row.post_sd:.2f}, "
        f"diff={(row.post_ptc - row.pre_ptc):.2f}, "
        f"rel_change={((row.post_ptc - row.pre_ptc)/row.pre_ptc):.2f}, "
    )

# Match chunk 4 aesthetics: solid pre bars and white hatched post bars.
ax.bar(
    x - bar_width / 2,
    conn_summary['pre_ptc'],
    width=bar_width,
    capsize=4,
    color=conn_colors,
    edgecolor='black',
    linewidth=0.9,
    label='Pre PTC'
)

ax.bar(
    x + bar_width / 2,
    conn_summary['post_ptc'],
    width=bar_width,
    capsize=4,
    color='white',
    edgecolor=conn_colors,
    linewidth=1.4,
    hatch='///',
    label='Post PTC'
)

ax.set_xticks(x)
ax.set_xticklabels(conn_labels)
ax.set_ylabel('Percent Time Connected (%)', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=15)
ax.grid(True, axis='y', alpha=0.3)

# connect_handles = [
#     Patch(facecolor=cfg['color'], edgecolor='black', label=cfg['label'])
#     for cfg in connectivity_config.values()
# ]
hatch_handles = [
    Patch(facecolor='lightgray', edgecolor='black', label='Pre PTC'),
    Patch(facecolor='white', edgecolor='black', hatch='///', label='Post PTC')
]

ax.legend(handles=hatch_handles, fontsize=18, ncol=1, loc='upper right', frameon=True)

plt.tight_layout()
plt.show()

# %% 6.0 Run an ANOVA

from scipy import stats


def _normalize_connectivity(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace('_', ' ', regex=False)
    )


def run_two_group_anova(
    df: pd.DataFrame,
    group_col: str = 'connect',
    baseline_col: str = 'pre_ptc',
    diff_col: str = 'd_ptc',
    group_a: str = 'giw',
    group_b: str = 'first order'
):
    work = df.copy()
    work['_group_norm'] = _normalize_connectivity(work[group_col])

    group_a = group_a.replace('_', ' ').lower()
    group_b = group_b.replace('_', ' ').lower()

    subset = work[work['_group_norm'].isin([group_a, group_b])].copy()

    a_base = subset.loc[subset['_group_norm'] == group_a, baseline_col].dropna().to_numpy()
    b_base = subset.loc[subset['_group_norm'] == group_b, baseline_col].dropna().to_numpy()

    a_diff = subset.loc[subset['_group_norm'] == group_a, diff_col].dropna().to_numpy()
    b_diff = subset.loc[subset['_group_norm'] == group_b, diff_col].dropna().to_numpy()

    f_base, p_base = stats.f_oneway(a_base, b_base)
    f_diff, p_diff = stats.f_oneway(a_diff, b_diff)

    print('ANOVA: giw vs first order (flow-through excluded)')
    print(f"baseline ({baseline_col}) n_giw={len(a_base)}, n_first_order={len(b_base)}, F={f_base:.4f}, p={p_base:.4g}")
    print(f"diff ({diff_col}) n_giw={len(a_diff)}, n_first_order={len(b_diff)}, F={f_diff:.4f}, p={p_diff:.4g}")


run_two_group_anova(ptc_summary, group_col='connect')



# %%
