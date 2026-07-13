# %% 1.0 Libraries, function imports and filepaths

import sys

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from scipy.stats import t
import statsmodels.formula.api as smf

from bradford_wy_scripts.functions.lai_vis_functions import read_concatonate_lai

lai_buffer_dist = '150'
model_type = 'OLS'  
data_set = 'no_dry_days'  

data_dir = "D:/depressional_lidar/data/bradford/"
#f'/in_data/hydro_forcings_and_LAI/basin_buffer_{lai_buffer_dist}m_maskedwetland/' OR
lai_dir = data_dir + f'/in_data/hydro_forcings_and_LAI/basin_buffer_{lai_buffer_dist}m_maskedwetland/'

# Path to results
shift_path = data_dir + f'out_data/modeled_logging_stages/shift_results_wetlandLAI{lai_buffer_dist}m_domain_{data_set}.csv'
models_path = data_dir + f'out_data/model_info/model_estimates_wetlandLAI_{lai_buffer_dist}m.csv'

# Path to wetland pairs, connnectivity key and strong model fits
wetland_pairs_path = data_dir + f'/in_data/hydro_forcings_and_LAI/log_ref_pairs_{lai_buffer_dist}m_wetland_basins.csv'
connectivity_key_path = data_dir + 'bradford_wetland_connect_logging_key.xlsx'
strong_pairs = data_dir + f'out_data/strong_ols_models_wetland{lai_buffer_dist}m_domain_{data_set}.csv'

wetland_pairs = pd.read_csv(wetland_pairs_path)
strong_pairs = pd.read_csv(strong_pairs)
shift_data = pd.read_csv(shift_path)

# NOTE: Filters shift data where model fits were worse than threshold
shift_data = shift_data.merge(
    strong_pairs[['log_id', 'ref_id', 'log_date']],
    left_on=['log_id', 'ref_id', 'logging_date'],
    right_on=['log_id', 'ref_id', 'log_date'],
    how='inner'
)


print(len(shift_data))
shift_data = shift_data[shift_data['mean_depth_change'] < 1.0]
#shift_data = shift_data[shift_data['log_id'] != '9_332']
print(len(shift_data))
print(len(strong_pairs))

# %% 2.0 Histogram of inundation shift data for a specific model and dataset

plot_df = shift_data[
    (shift_data['model_type'] == model_type) & (shift_data['data_set'] == data_set)
]

mean_change = plot_df['mean_depth_change'].mean()
median_change = plot_df['mean_depth_change'].median()
std_change = plot_df['mean_depth_change'].std()
print(std_change)

# count how many pairs have non‑negative or negative mean depth change
increase_num = (plot_df['mean_depth_change'] >= 0).sum()
decrease_num = (plot_df['mean_depth_change'] < 0).sum()
print(increase_num, decrease_num)

fig, ax = plt.subplots(figsize=(10, 5))

ax.hist(
    plot_df['mean_depth_change'], 
    bins=20, 
    edgecolor='black', 
    alpha=0.7, 
    color='steelblue',
    linewidth=1.2
)

ax.axvline(0, color='grey', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(mean_change, color='navy', linestyle='--', linewidth=4, alpha=1)

ax.set_title('Modeled Stage Increase (Unlogged - Logged)', fontsize=18, fontweight='bold')
ax.set_xlabel('Mean Depth Difference [m]', fontsize=16)
ax.set_ylabel('Number of Pairs', fontsize=18)
ax.tick_params(axis='both', labelsize=14) 

# Add stats text
stats_text = (
    f'Mean = {mean_change:.3f}m\n'
    f'Median = {median_change:.3f}m\n'
    f'Std = {std_change:.3f}m'
)
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=18,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# %% 3.0 LAI change biplot (unclassified by connectivity)

# %% 3.1 Calculate the LAI differences for every pair

lai_roll_diffs = []

for idx, p in wetland_pairs.iterrows():

    date = p['planet_logging_date']
    log_id = p['logged_id']
    logged_lai = read_concatonate_lai(
        lai_dir,
        log_id,
        "na",
        upper_bound=5.5,
        lower_bound=0.5
    )
    logged_lai = logged_lai[logged_lai['date'] >= '2019-01-01']

    ref_id = p['reference_id']
    reference_lai = read_concatonate_lai(
        lai_dir, 
        ref_id,
        "na",
        upper_bound=5.5,
        lower_bound=0.5
    )
    reference_lai = reference_lai[reference_lai['date'] >= '2019-01-01']

    merged_lai = logged_lai[['date', 'roll_yr']].merge(
        reference_lai[['date', 'roll_yr']],
        on='date', 
        suffixes=('_log', '_ref'),
        how='outer'
    )
    merged_lai['roll_yr_diff'] = merged_lai['roll_yr_log'] - merged_lai['roll_yr_ref']

    pre_mean = merged_lai[merged_lai['date'] <= date]['roll_yr_diff'].mean()
    post_mean = merged_lai[merged_lai['date'] > date]['roll_yr_diff'].mean()

    roll_diff_change = pre_mean - post_mean

    lai_roll_diffs.append(roll_diff_change)

wetland_pairs['roll_diff_change'] = lai_roll_diffs

# %% 3.2 Merge the water level shift data with the LAI data

shift_data = shift_data.merge(
    wetland_pairs[['logged_id', 'reference_id', 'roll_diff_change']],
    how='left',
    left_on=['log_id','ref_id'],
    right_on=['logged_id', 'reference_id']
)


# %% 3.3 Visualize LAI ~ Depth change NOT factoring connectivity

plot_df = shift_data[
    (shift_data['model_type'] == model_type) & 
    (shift_data['data_set'] == data_set)
].copy()

res = stats.linregress(
    plot_df['roll_diff_change'], 
    plot_df['mean_depth_change']
)

slope = res.slope
intercept = res.intercept
r_value = res.rvalue
p_value = res.pvalue
n = len(plot_df)

intercept_t = intercept / res.intercept_stderr
intercept_p = 2 * stats.t.sf(np.abs(intercept_t), n - 2)

print(intercept_p)

fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter(
    plot_df['roll_diff_change'], 
    plot_df['mean_depth_change'],
    alpha=0.6, 
    s=80, 
    edgecolors='black', 
    linewidth=0.5,
    color='steelblue'
)

x_range = np.linspace(plot_df['roll_diff_change'].min(), plot_df['roll_diff_change'].max(), 100)
y_pred = slope * x_range + intercept
ax.plot(x_range, y_pred, 'r--', linewidth=2, 
        label=f'y = {slope:.3f}x + {intercept:.3f}\nR² = {r_value**2:.3f}, p = {p_value:.3f}')

ax.axhline(0, color='gray', linestyle=':', alpha=1)
ax.axhline(plot_df['mean_depth_change'].mean(), color='blue', linestyle=':', linewidth=2, alpha=0.8, label=f'Mean Stage Increase {plot_df['mean_depth_change'].mean():.2f} [m]')
ax.axvline(plot_df['roll_diff_change'].mean(), color='orange', linestyle=':', linewidth=2, label=f"Mean LAI change {plot_df['roll_diff_change'].mean():.2f}")
ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=12)
ax.set_ylabel('Modeled Stage Increase (Post - Pre) [m]', fontsize=12)
ax.set_title(f'LAI Loss vs Wetland Depth Change (n={len(strong_pairs)} pairs)', 
             fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=11)

plt.tight_layout()
plt.show()

# %% 4.0 Make boxplot normalized by LAI

# %% 4.1 Data preparation

connectivity_key = pd.read_excel(connectivity_key_path)

connectivity_config = {
    'first order': {'color': '#6C5B7B', 'label': 'Ditch connected', 'marker': 's'},
    'giw': {'color': '#1B7F79', 'label': 'Unditched', 'marker': '^'}, 
    'flow-through': {'color': '#C46A1A', 'label': 'Flow-through connected', 'marker': 'X'}
}

shift_data['LAI_normalized_depth'] = (shift_data['mean_depth_change'] / shift_data['roll_diff_change']) * 100 # NOTE for cm

shift_data['log_connected'] = shift_data.apply(
    lambda row: connectivity_key.loc[
        connectivity_key['wetland_id'] == row['log_id'], 'connectivity'
    ].values[0],
    axis=1
)
shift_data['ref_connected'] = shift_data.apply(
    lambda row: connectivity_key.loc[
        connectivity_key['wetland_id'] == row['ref_id'], 'connectivity'
    ].values[0],
    axis=1
)

interaction_stats = shift_data.groupby(['log_connected', 'ref_connected'])['LAI_normalized_depth'].agg(
    ['mean', 'sem', 'count']
).reset_index()

# %% 5.0 Boxplot LAI normalized depth change aggregated by logged connectivity (panel a) and colored by reference connectivity (panel b)

plot_connectivity_order = ['giw', 'first order', 'flow-through']
ref_connectivity_order = ['giw', 'first order', 'flow-through']

box_df = shift_data.dropna(subset=['log_connected', 'ref_connected', 'LAI_normalized_depth']).copy()

box_df = box_df[
    box_df['log_connected'].isin(plot_connectivity_order)
    & box_df['ref_connected'].isin(ref_connectivity_order)
]

rng = np.random.default_rng(42)

# Panel A: all references pooled by logged connectivity, black box/whiskers + black jitter points
panel_a_data = [
    box_df.loc[box_df['log_connected'] == conn, 'LAI_normalized_depth'].values
    for conn in plot_connectivity_order
]

fig, ax_a = plt.subplots(1, 1, figsize=(8, 10))

ax_a.boxplot(
    panel_a_data,
    positions=np.arange(len(plot_connectivity_order)),
    widths=0.6,
    patch_artist=False,
    showfliers=False,
    boxprops=dict(color='black', linewidth=2),
    whiskerprops=dict(color='black', linewidth=1.8),
    capprops=dict(color='black', linewidth=1.8),
    medianprops=dict(color='black', linewidth=2.2)
)

for i, y_vals in enumerate(panel_a_data):
    x_jitter = rng.normal(loc=i, scale=0.06, size=len(y_vals))
    ax_a.scatter(x_jitter, y_vals, color='black', s=20, alpha=0.45, zorder=3)


ax_a.set_ylabel('Depth Increase (cm) / LAI Decrease', fontsize=16)
ax_a.set_xticks(np.arange(len(plot_connectivity_order)))
ax_a.set_xticklabels(
    [connectivity_config[c]['label'] for c in plot_connectivity_order],
    fontsize=16,
    rotation=20,
    ha='right'
)
ax_a.axhline(0, color='black', linestyle='--', linewidth=2)
ax_a.tick_params(axis='y', labelsize=14)
#ax_a.set_ylim(-100, 100)

plt.tight_layout()
plt.show()

# Panel B: grouped by reference connectivity for each logged-connectivity class
center_spacing = 1.35
centers = np.arange(len(plot_connectivity_order)) * center_spacing
group_width = 0.78
n_ref = len(ref_connectivity_order)
box_width = group_width / n_ref * 0.85
offsets = np.linspace(
    -group_width / 2 + box_width / 2,
    group_width / 2 - box_width / 2,
    n_ref
)

fig, ax_b = plt.subplots(1, 1, figsize=(8, 8))

for j, ref_conn in enumerate(ref_connectivity_order):
    grouped_data = [
        box_df.loc[
            (box_df['log_connected'] == log_conn) & (box_df['ref_connected'] == ref_conn),
            'LAI_normalized_depth'
        ].values
        for log_conn in plot_connectivity_order
    ]

    positions = centers + offsets[j]
    ref_color = connectivity_config[ref_conn]['color']

    ax_b.boxplot(
        grouped_data,
        positions=positions,
        widths=box_width,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=ref_color, edgecolor=ref_color, alpha=0.9, linewidth=1.8),
        whiskerprops=dict(color=ref_color, linewidth=1.4),
        capprops=dict(color=ref_color, linewidth=1.4),
        medianprops=dict(color='black', linewidth=1.8)
    )

ax_b.set_xticks(centers)
ax_b.set_xticklabels(
    [connectivity_config[c]['label'] for c in plot_connectivity_order],
    fontsize=16,
    rotation=20,
)
ax_b.set_ylabel('Depth Increase (cm) / LAI Decrease', fontsize=16)

legend_handles = [
    plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor=connectivity_config[ref_conn]['color'],
        edgecolor=connectivity_config[ref_conn]['color'],
        alpha=0.9,
        label=f"Reference: {connectivity_config[ref_conn]['label']}"
    )
    for ref_conn in ref_connectivity_order
]
ax_b.axhline(0, color='grey', linestyle='--', linewidth=2, label='No Depth Change')
ax_b.legend(handles=legend_handles, loc='lower left', fontsize=16)
ax_b.tick_params(axis='y', labelsize=14)
plt.ylim(top=45)
plt.tight_layout()
plt.show()

# %% 5.1 Print stats assocaited with boxplots
print("\nPanel A: LAI-normalized depth by logged connectivity")
for conn in plot_connectivity_order:
    vals = box_df.loc[box_df["log_connected"] == conn, "LAI_normalized_depth"].dropna()
    print(
        f"{connectivity_config[conn]['label']:<14} "
        f"n={len(vals):>3} "
        f"mean={vals.mean():>8.3f} "
        f"median={vals.median():>8.3f} "
        f"std={vals.std():>8.3f}"
    )

print("\nPanel B: LAI-normalized depth by logged x reference connectivity")
for log_conn in plot_connectivity_order:
    for ref_conn in ref_connectivity_order:
        vals = box_df.loc[
            (box_df["log_connected"] == log_conn) &
            (box_df["ref_connected"] == ref_conn),
            "LAI_normalized_depth"
        ].dropna()

        print(
            f"log={connectivity_config[log_conn]['label']:<14} "
            f"ref={connectivity_config[ref_conn]['label']:<14} "
            f"n={len(vals):>3} "
            f"mean={vals.mean():>8.3f} "
            f"median={vals.median():>8.3f} "
            f"std={vals.std():>8.3f}"
        )

# %% 6.0 LAI biplot factored by connectivity

plot_df_conn = shift_data[
    (shift_data['model_type'] == model_type) &
    (shift_data['data_set'] == data_set)
].copy()

fig, ax = plt.subplots(figsize=(10, 8))

for connectivity_level, config in connectivity_config.items():
    subset = plot_df_conn[plot_df_conn['log_connected'] == connectivity_level].copy()

    subset_clean = subset.dropna(subset=['roll_diff_change', 'mean_depth_change', 'log_id'])
    
    ax.scatter(
        subset_clean['roll_diff_change'],
        subset_clean['mean_depth_change'],
        alpha=0.5,
        s=20,
        edgecolors='black',
        linewidth=0.5,
        color=config['color'],
        marker=config['marker']
    )

    if connectivity_level == 'first order':
        for _, row in subset_clean.iterrows():
            ax.annotate(
                str(row['log_id']),
                xy=(row['roll_diff_change'], row['mean_depth_change']),
                xytext=(6, 4),
                textcoords='offset points',
                fontsize=9,
                alpha=0.9,
                bbox=dict(boxstyle='round,pad=0.1', fc='white', alpha=0.6)
            )

    x = subset_clean['roll_diff_change']
    y = subset_clean['mean_depth_change']

    slope, intercept, r_value, p_value, std_err = stats.linregress(
        x,
        y
    )

    x_min = subset_clean['roll_diff_change'].min()
    x_max = subset_clean['roll_diff_change'].max()

    x_range = np.linspace(x_min, x_max, 100)

    y_pred = slope * x_range + intercept

    ax.plot(x_range, y_pred, '-', linewidth=2, color=config['color'],
            label=f"{config['label']}")
    
    n = len(x)
    dof = n - 2 
    t_val = t.ppf(0.95, dof) 

    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean)**2)
    residuals = y - (slope * x + intercept)
    s_res = np.sqrt(np.sum(residuals**2) / dof)
    
    se = s_res * np.sqrt(1/n + (x_range - x_mean)**2 / sxx)
    ci = t_val * se

    ax.fill_between(x_range, y_pred - ci, y_pred + ci, 
                    alpha=0.2, color='grey')

# # Fourth trendline: flow-through excluding outlier log_ids
# ft_subset = plot_df_conn[
#     (plot_df_conn['log_connected'] == 'flow-through') &
#     (~plot_df_conn['log_id'].astype(str).isin(['7_341', '9_77']))
# ].copy()
# ft_clean = ft_subset.dropna(subset=['roll_diff_change', 'mean_depth_change', 'log_id'])

# x_ft = ft_clean['roll_diff_change']
# y_ft = ft_clean['mean_depth_change']

# slope_ft, intercept_ft, r_ft, p_ft, std_err_ft = stats.linregress(x_ft, y_ft)

# x_range_ft = np.linspace(x_ft.min(), x_ft.max(), 100)
# y_pred_ft = slope_ft * x_range_ft + intercept_ft

# ax.plot(x_range_ft, y_pred_ft, '--', linewidth=2, color='darkred',
#         label=f"flow-through (excl. outliers)")

# n_ft = len(x_ft)
# dof_ft = n_ft - 2
# t_val_ft = t.ppf(0.95, dof_ft)

# x_mean_ft = np.mean(x_ft)
# sxx_ft = np.sum((x_ft - x_mean_ft)**2)
# residuals_ft = y_ft - (slope_ft * x_ft + intercept_ft)
# s_res_ft = np.sqrt(np.sum(residuals_ft**2) / dof_ft)

# se_ft = s_res_ft * np.sqrt(1/n_ft + (x_range_ft - x_mean_ft)**2 / sxx_ft)
# ci_ft = t_val_ft * se_ft

# ax.fill_between(x_range_ft, y_pred_ft - ci_ft, y_pred_ft + ci_ft,
#                 alpha=0.2, color='grey')

# Formatting
ax.set_xlabel('Relative LAI Decrease (Pre - Post)', fontsize=18)
ax.set_ylabel('Modeled Depth Change (Post - Pre) [m]', fontsize=18)
# ax.set_title(f'LAI Loss vs Wetland Depth Change by Connectivity', 
#             fontsize=20, fontweight='bold')
ax.tick_params(axis='both', labelsize=14)
ax.legend(loc='best', fontsize=14, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 6.1 Print the stats for each connectivity class on the regression plot
plot_df_conn = shift_data[
    (shift_data['model_type'] == model_type) &
    (shift_data['data_set'] == data_set)
].copy()

print(f"\n{'Connectivity':<25} {'n pairs':>5} {'slope':>8} {'intercept':>10} {'R²':>8} {'p-value':>10}")
print("-" * 70)

for connectivity_level, config in connectivity_config.items():
    subset = plot_df_conn[plot_df_conn['log_connected'] == connectivity_level].copy()
    subset_clean = subset.dropna(subset=['roll_diff_change', 'mean_depth_change'])

    #print('connectivity_level')
    x = subset_clean['roll_diff_change']
    y = subset_clean['mean_depth_change']
    #print(connectivity_level, y.mean(), y.std())


    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print(f"{config['label']:<25} {len(x):>5} {slope:>8.4f} {intercept:>10.4f} {r_value**2:>8.4f} {p_value:>10.4f}")



# %% 7.0 Mixed effects model to test slope differences between GIWs and 1st order ditched. 

# %% 7.1 Data prep
lme_data = plot_df_conn[
    ['log_id', 'ref_id', 'mean_depth_change', 'roll_diff_change', 'log_connected', 'ref_connected']
].copy()

lme_data['mean_depth_change'] = lme_data['mean_depth_change'] * 100

lme_data = lme_data[lme_data['log_connected'].isin(['giw', 'first order'])].copy()
print(len(lme_data))
lme_data['log_id'] = lme_data['log_id'].astype('category')
lme_data['ref_id'] = lme_data['ref_id'].astype('category')
lme_data['log_connected'] = pd.Categorical(lme_data['log_connected'], categories=['giw', 'first order',])


df = lme_data.copy()
# Between-logged effect (mean across references for each logged wetland)
df["roll_log_mean"] = df.groupby("log_id", observed=True)["roll_diff_change"].transform("mean")
# Within-logged deviation (pair-level deviation from that logged wetland mean)
df["roll_within"] = df["roll_diff_change"] - df["roll_log_mean"]

# Optional centering for stability / interpretation
df["roll_log_mean_c"] = df["roll_log_mean"] - df["roll_log_mean"].mean()
df["roll_within_c"]   = df["roll_within"] - df["roll_within"].mean()


# %% 7.2 Simple logged-only mixed effect model

# md = smf.mixedlm(
#     "mean_depth_change ~ roll_log_mean_c * C(log_connected)",
#     data=df,
#     groups="log_id",
#     re_formula="1"
# )
# m = md.fit(reml=False, method="lbfgs")
# print(m.summary())

# %% 7.3 GLM with cluster errors 

import statsmodels.api as sm
glm_mod = smf.glm(
    "mean_depth_change ~ roll_log_mean_c * C(log_connected) + roll_within_c",
    data=df,
    family=sm.families.Gaussian()
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["log_id"]}
)

print(glm_mod.summary())

#print(df['roll_diff_change'].mean())


# %% 7.4 Simplest OLS model by log-id

log_level = (
    df.groupby(["log_id", "log_connected"], observed=True)
      .agg(
          mean_depth_change=("mean_depth_change", "mean"),
          roll_log_mean=("roll_log_mean", "first")
      )
      .reset_index()
)

log_level["roll_log_mean_c"] = (
    log_level["roll_log_mean"] - log_level["roll_log_mean"].mean()
)

mod = smf.ols(
    "mean_depth_change ~ roll_log_mean_c * C(log_connected)",
    data=log_level
).fit()

print(mod.summary())

# %% 7.5 Testing a boot-strapping approach to ascertian the interaction. 

formula = "mean_depth_change ~ roll_log_mean_c * C(log_connected)"
coef_name = "roll_log_mean_c:C(log_connected)[T.first order]"

# Original fit on one-row-per-log_id data
orig_mod = smf.ols(formula, data=log_level).fit()
orig_est = orig_mod.params[coef_name]

rng = np.random.default_rng(123)
boot_est = []

n_boot = 5000

for b in range(n_boot):
    boot_df = log_level.sample(
        n=len(log_level),
        replace=True,
        random_state=rng.integers(0, 1_000_000_000)
    )

    boot_mod = smf.ols(formula, data=boot_df).fit()
    boot_est.append(boot_mod.params[coef_name])


boot_est = np.array(boot_est)

ci_low, ci_high = np.percentile(boot_est, [2.5, 97.5])

print(f"Original interaction estimate: {orig_est:.3f}")
print(f"Bootstrap 95% CI: {ci_low:.3f} to {ci_high:.3f}")
print(f"Successful bootstrap fits: {len(boot_est)}")

# %% 7.6 Sensitivity to one-at-a-time connectivity reclassification ???


# %%
