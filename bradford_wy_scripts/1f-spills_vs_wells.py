# %% 1.0 Libraries and file paths

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/bradford/'

est_spills_path = f'{data_dir}/out_data/bradford_estimated_basin_spills.csv'
hyps_path = f'{data_dir}/out_data/bradford_hypsometry_curves.csv'
well_data_path = f'{data_dir}/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv'

est_spills = pd.read_csv(est_spills_path)
hyps = pd.read_csv(hyps_path)
well_data = pd.read_csv(well_data_path)

# Adjust the spill elevations to a common vertical datumn (the lowest point in the delineated basin)
est_spills['perimeter_smoothed_spill_depth'] = est_spills['perimeter_smoothed_spill_elev'] - est_spills['min_elev']
est_spills['contiguous_spill_depth'] = est_spills['contiguous_spill_elev'] - est_spills['min_elev']
est_spills['well_to_low'] = est_spills['well_elev'] - est_spills['min_elev']

# %% 2.0 Plot spill thresholds, and hypsometry relative to water depth PDF

id = '6_300'

spills = est_spills[est_spills['wetland_id'] == id].copy()
well_to_low_offset = spills['well_to_low'].iloc[0]

# Adjust the cdf to the basin low value
min_elev = spills['min_elev'].iloc[0]

# Adjust well data to the minimum elevation
well_to_low_offset = spills['well_to_low'].iloc[0]
well = well_data[well_data['wetland_id'] == id].copy()
well = well[well['flag'] ==0]
well['water_depth_m'] = well['well_depth_m'] + well_to_low_offset

# Extract thresholds for vertical lines
smoothed_perimeter = spills['perimeter_smoothed_spill_depth'].iloc[0]
contiguous_spill = spills['contiguous_spill_depth'].iloc[0]
conventional_spill = spills['max_fill_delineated'].iloc[0]

# %% 3.0 Simple timeseries plot of water depth m

well_ts = well.sort_values('date').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(well_ts.index, well_ts['water_depth_m'], color='steelblue', linewidth=1, label='Water depth')

p90_depth = well_ts['water_depth_m'].quantile(0.9)
ax.axhline(p90_depth, color='red', linestyle='--', linewidth=2, label=f'90th percentile depth ({p90_depth:.2f} m)')

# Mode as the center of the most common 5cm bin
well_vals_ts = well_ts['water_depth_m'].dropna().values
mode_bin_edges = np.arange(well_vals_ts.min(), well_vals_ts.max() + 0.05, 0.05)
mode_counts, _ = np.histogram(well_vals_ts, bins=mode_bin_edges)
mode_bin_idx = np.argmax(mode_counts)
mode_depth = (mode_bin_edges[mode_bin_idx] + mode_bin_edges[mode_bin_idx + 1]) / 2
ax.axhline(mode_depth, color='green', linestyle=':', linewidth=2, label=f'Mode (most common 5cm bin, {mode_depth:.2f} m)')

ax.set_xlabel('Days', fontsize=13)
ax.set_ylabel('Water depth above basin low (m)', fontsize=13)
ax.set_title(f'Wetland {id} — Water depth time series', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% 4.0 PDF plot: well water depth vs hypsometry depth

# --- Load flat hypsometry and filter to wetland ---
cdf = hyps[hyps['wetland_id'] == id].copy()
depth_vals = cdf['elev_bin_center'].values - min_elev  # convert to depth above basin low

well_vals = well['water_depth_m'].dropna().values

# --- Shared bin edges at 0.02 m intervals ---
bin_width = 0.02
bin_edges = np.arange(
    min(depth_vals.min(), well_vals.min()),
    max(depth_vals.max(), well_vals.max()) + bin_width,
    bin_width
)

# --- Hypsometry: counts per bin -> mean inundated area (m²) per bin ---
# Use the raw area values from the flat hypsometry table, binned by depth
hyps_counts, _ = np.histogram(depth_vals, bins=bin_edges,
                               weights=cdf['inundated_area'].values)
bin_counts_hyps, _ = np.histogram(depth_vals, bins=bin_edges)
hyps_area_per_bin = np.where(bin_counts_hyps > 0,
                              hyps_counts / bin_counts_hyps, 0)

# --- Well: counts per bin -> % of days ---
n_days = len(well_vals)
well_counts, _ = np.histogram(well_vals, bins=bin_edges)
well_pct = well_counts / n_days * 100  # % of days in each bin

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# --- Dual-axis plot ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

ax1.step(bin_centers, hyps_area_per_bin, where='mid',
         color='steelblue', linewidth=2, label='Inundated area (m²)')
ax2.step(bin_centers, well_pct, where='mid',
         color='darkorange', linewidth=2, label='% of days')

# Vertical threshold lines (drawn on ax1, visible on both)
ax1.axvline(smoothed_perimeter, color='purple', linewidth=7,
            label=f'Smoothed perimeter spill ({smoothed_perimeter:.2f} m)', alpha=0.5)
ax1.axvline(contiguous_spill, color='green', linewidth=7,
            label=f'Contiguous spill ({contiguous_spill:.2f} m)', alpha=0.5)
ax1.axvline(conventional_spill, color='red', linewidth=7,
            label=f'Conventional spill ({conventional_spill:.2f} m)', alpha=0.5)

ax1.set_xlabel('Depth above basin low (m)', fontsize=13)
ax1.set_ylabel('Inundated area (m²)', fontsize=13, color='steelblue', weight='bold')
ax1.tick_params(axis='y', labelcolor='steelblue')
ax2.set_ylabel('% of days', fontsize=13, color='darkorange', weight='bold')
ax2.tick_params(axis='y', labelcolor='darkorange')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=11)

ax1.set_title(f'Wetland {id} — Well depth vs hypsometry', fontsize=14)
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% 5.0 Modal water depth vs spill thresholds across all wetlands

records = []
for wid in est_spills['wetland_id'].unique():
    sp = est_spills[est_spills['wetland_id'] == wid]
    if sp.empty:
        continue

    w = well_data[well_data['wetland_id'] == wid].copy()
    w = w[w['flag'] == 0]
    if w.empty:
        continue

    well_to_low = sp['well_to_low'].iloc[0]
    w['water_depth_m'] = w['well_depth_m'] + well_to_low

    vals = w['water_depth_m'].dropna().values
    if len(vals) < 5:
        continue

    edges = np.arange(vals.min(), vals.max() + 0.05, 0.05)
    counts, _ = np.histogram(vals, bins=edges)
    idx = np.argmax(counts)
    modal = (edges[idx] + edges[idx + 1]) / 2

    records.append({
        'wetland_id': wid,
        'modal_depth': modal,
        'smoothed_perimeter': sp['perimeter_smoothed_spill_depth'].iloc[0],
        'contiguous_spill':   sp['contiguous_spill_depth'].iloc[0],
        'conventional_spill': sp['max_fill_delineated'].iloc[0],
    })

modal_df = pd.DataFrame(records)
modal_df = modal_df[modal_df['wetland_id'] != '14_418']

thresholds = [
    ('smoothed_perimeter', 'purple', 'Smoothed perimeter spill'),
    ('contiguous_spill',   'green',  'Contiguous spill'),
    ('conventional_spill', 'red',    'Conventional spill (OG method)'),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, (col, color, label) in zip(axes, thresholds):
    x = modal_df[col].values
    y = modal_df['modal_depth'].values

    ax.scatter(x, y, color=color, alpha=0.7, edgecolors='k', linewidths=0.5, zorder=3)

    # Linear trendline
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 1:
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
fig.suptitle('Modal well water depth vs spill thresholds', fontsize=14, y=1.01)
plt.tight_layout()
plt.show()

# %%
