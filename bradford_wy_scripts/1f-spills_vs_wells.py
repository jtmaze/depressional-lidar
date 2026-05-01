# %% 1.0 Libraries and file paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/bradford/'

est_spills_path = f'{data_dir}/out_data/bradford_estimated_basin_spills.csv'
hyps_path = f'{data_dir}/out_data/bradford_hypsometry_curves.csv'
well_data_path = f'{data_dir}/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv'
connectivity_path = f'{data_dir}/bradford_wetland_connect_logging_key.xlsx'

est_spills = pd.read_csv(est_spills_path)
hyps = pd.read_csv(hyps_path)
well_data = pd.read_csv(well_data_path)


# %% 2.0 Adjust different spill thresholds to depth relative to the the well's elevation

"""
NOTE:
# Due to ditching, the elevation with the deepest spill depth (i.e., lowest depression point) might not be
# the delineated area's absolute lowest point. Therefore, we need to find the difference between the basin's
# absolute lowest point, and the lowest point in a filled depression.
"""

est_spills['well_to_min_delineated'] = est_spills['well_elev'] - est_spills['min_elev'] # Differene between well and basin low point
# NOTE max_fill_elev is the elevation of the highest spill (dem + fill), and max_fill_delineated is the depth value assocated with that spill
est_spills['spill_min_to_basin_min'] = (est_spills['max_fill_elev'] - est_spills['max_fill_delineated']) - est_spills['min_elev'] 
est_spills['delineated_spill_h_min'] = est_spills['max_fill_delineated'] + est_spills['spill_min_to_basin_min']

est_spills['well_to_min150'] = est_spills['well_elev'] - est_spills['basin150_min'] # Difference between well and radius low point
# NOTE for 150m, 200m, and 250m radii this is already the elevation unfilled, where spill depth is greatest
est_spills['spill_min_to_basin_min150'] = est_spills['basin150_dem_z'] - est_spills['basin150_min']
est_spills['150spill_h_min'] = est_spills['max_fill150'] + est_spills['spill_min_to_basin_min150']

est_spills['well_to_min200'] = est_spills['well_elev'] - est_spills['basin200_min']
est_spills['spill_min_to_basin_min200'] = est_spills['basin200_dem_z'] - est_spills['basin200_min']
est_spills['200spill_h_min'] = est_spills['max_fill200'] + est_spills['spill_min_to_basin_min200']

est_spills['well_to_min250'] = est_spills['well_elev'] - est_spills['basin250_min']
est_spills['spill_min_to_basin_min250'] = est_spills['basin250_dem_z'] - est_spills['basin250_min']
est_spills['250spill_h_min'] = est_spills['max_fill250'] + est_spills['spill_min_to_basin_min250']

print(est_spills.head())


# %% 2.0 Modal water depth vs spill thresholds across all wetlands

def compute_modal_depth(depths):
    vals = depths.dropna().values

    edges = np.arange(vals.min(), vals.max() + 0.05, 0.05)
    counts, _ = np.histogram(vals, bins=edges)
    idx = np.argmax(counts)
    modal = (edges[idx] + edges[idx + 1]) / 2
    median_depth = np.median(vals)
    # NOTE a quick catch here, becuase one well had mode occur at low water due to oscilations at low water table.
    if modal < median_depth:
        above_median_vals = vals[vals >= median_depth]
        edges = np.arange(above_median_vals.min(), above_median_vals.max() + 0.05, 0.05)
        counts, _ = np.histogram(above_median_vals, bins=edges)
        idx = np.argmax(counts)
        modal = (edges[idx] + edges[idx + 1]) / 2

    return modal

records = []
for wid in est_spills['wetland_id'].unique():
    sp = est_spills[est_spills['wetland_id'] == wid]

    w = well_data[well_data['wetland_id'] == wid].copy()
    w = w[w['flag'] == 0]

    record = {'wetland_id': wid}
    for offset_col, modal_col in [
        ('well_to_min_delineated', 'modal_depth_delineated'),
        ('well_to_min150',         'modal_depth_150'),
        ('well_to_min200',         'modal_depth_200'),
        ('well_to_min250',         'modal_depth_250'),
    ]:
        offset = sp[offset_col].iloc[0]
        record[modal_col] = compute_modal_depth(w['well_depth_m'] + offset)

    record['delineated_spill_h_min'] = sp['delineated_spill_h_min'].iloc[0]
    record['150spill_h_min']         = sp['150spill_h_min'].iloc[0]
    record['200spill_h_min']         = sp['200spill_h_min'].iloc[0]
    record['250spill_h_min']         = sp['250spill_h_min'].iloc[0]

    records.append(record)

modal_df = pd.DataFrame(records)

thresholds = [
    ('delineated_spill_h_min', 'modal_depth_delineated', 'firebrick',    'Delineated spill'),
    ('150spill_h_min',         'modal_depth_150',        'royalblue',    '150 m spill'),
    ('200spill_h_min',         'modal_depth_200',        'darkgreen',    '200 m spill'),
    ('250spill_h_min',         'modal_depth_250',        'saddlebrown',  '250 m spill'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (col, modal_col, color, label) in zip(axes, thresholds):
    x = modal_df[col].values
    y = modal_df[modal_col].values

    ax.scatter(x, y, color=color, alpha=0.7, edgecolors='k', linewidths=0.5, zorder=3)

    # Linear trendline
    mask = np.isfinite(x) & np.isfinite(y)

    m, b = np.polyfit(x[mask], y[mask], 1)
    y_pred = m * x[mask] + b
    ss_res = np.sum((y[mask] - y_pred) ** 2)
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
    ax.plot(x_line, m * x_line + b, color=color, linewidth=2,
            label=f'Trend (slope={m:.2f}, r²={r2:.3f})')

    # 1:1 line
    all_vals = np.concatenate([x[mask], y[mask]])
    lim = (all_vals.min() - 0.05, all_vals.max() + 0.05)
    ax.plot(lim, lim, 'k--', linewidth=1.5, label='1:1')

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f'{label} depth (m)', fontsize=12)
    ax.set_title(label, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

axes[0].set_ylabel('Modal water depth (m)', fontsize=12)
axes[2].set_ylabel('Modal water depth (m)', fontsize=12)
fig.suptitle('Modal well water depth vs spill thresholds', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()


# %% 5.0 Make the same line plot by connectivity class for each spill estimate

connect = pd.read_excel(connectivity_path)

# Join to modal_df based on wetland_id to get 'connectivity' column. 
modal_with_connect = modal_df.merge(
    connect[['wetland_id', 'connectivity']],
    on='wetland_id',
    how='left'
)

connectivity_classes = sorted(modal_with_connect['connectivity'].dropna().unique())

for spill_col, modal_col, color, spill_label in thresholds:
    plot_df = modal_with_connect.dropna(subset=['connectivity', spill_col, modal_col]).copy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    global_vals = np.concatenate([
        plot_df[spill_col].values,
        plot_df[modal_col].values
    ])
    lim = (global_vals.min() - 0.05, global_vals.max() + 0.05)

    for i, ax in enumerate(axes):

        connectivity_class = connectivity_classes[i]
        class_df = plot_df[plot_df['connectivity'] == connectivity_class]
        x = class_df[spill_col].values
        y = class_df[modal_col].values

        ax.scatter(x, y, color=color, alpha=0.7, edgecolors='k', linewidths=0.5, zorder=3)

        for xi, yi, wid in zip(x, y, class_df['wetland_id'].values):
            ax.annotate(str(wid), (xi, yi), textcoords='offset points', xytext=(4, 4), fontsize=6)

        mask = np.isfinite(x) & np.isfinite(y)

        m, b = np.polyfit(x[mask], y[mask], 1)
        y_pred = m * x[mask] + b
        ss_res = np.sum((y[mask] - y_pred) ** 2)
        ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
        r2 = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
        x_line = np.linspace(x[mask].min(), x[mask].max(), 100)
        trend_label = f'Trend (slope={m:.2f}, r²={r2:.3f})' if np.isfinite(r2) else f'Trend (slope={m:.2f})'
        ax.plot(x_line, m * x_line + b, color=color, linewidth=2, label=trend_label)

        ax.plot(lim, lim, 'k--', linewidth=1.5, label='1:1')
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f'{spill_label} depth (m)', fontsize=12)
        ax.set_title(f'Connectivity {connectivity_class} (n={len(class_df)})', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect('equal')

    axes[0].set_ylabel('Modal water depth (m)', fontsize=12)
    fig.suptitle(f'Modal well water depth vs {spill_label.lower()} by connectivity class', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


# %%
