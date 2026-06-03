# %% 1.0 Libraries and filepaths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

well_ts_path = "D:/depressional_lidar/data/bradford/in_data/stage_data/bradford_daily_well_depth_Winter2025.csv"
gauges_ts_path = 'D:/depressional_lidar/data/bradford/in_data/ancillary_data/sam_howley_streams.csv'

# %% 2.0 Read the data

well_ts = pd.read_csv(well_ts_path)

# reformat the gauges
gauge_ts = pd.read_csv(gauges_ts_path)
gauge_ts = gauge_ts.rename(
    columns={
        'Date': 'date',
        'Site_ID': 'gauge_id',
        'gauge_depth_m': 'depth'
    }
)
gauge_ts['gauge_id'] = gauge_ts['gauge_id'].astype(str)
gauge_ts = gauge_ts[['date', 'gauge_id', 'depth']]


# %% 3.0 Plot correlations for a few pairs of wetlands and gauges
test_pairs = [ #wetland_id, gauge_id
   ('5a_598', '5a'), ('6_93', '6a'), ('9_77', '9'), ('15_268', '15')
]

well_ts['date'] = pd.to_datetime(well_ts['date'], utc=True, format='mixed').dt.normalize()
gauge_ts['date'] = pd.to_datetime(gauge_ts['date'], utc=True, format='mixed').dt.normalize()

for wetland_id, gauge_id in test_pairs:
    well = well_ts[well_ts['wetland_id'] == wetland_id][['date', 'well_depth_m']]
    gauge = gauge_ts[gauge_ts['gauge_id'] == gauge_id][['date', 'depth']]

    merged = well.merge(gauge, on='date').dropna()

    fig, ax = plt.subplots()
    ax.scatter(merged['depth'], merged['well_depth_m'], alpha=0.6, s=20)

    m, b = np.polyfit(merged['depth'], merged['well_depth_m'], 1)
    r = np.corrcoef(merged['depth'], merged['well_depth_m'])[0, 1]
    x = np.linspace(merged['depth'].min(), merged['depth'].max(), 100)
    ax.plot(x, m * x + b, color='red', linewidth=1)
    ax.text(
        0.05,
        0.95,
        f'slope = {m:.3f}',
        transform=ax.transAxes,
        va='top',
        ha='left',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
    )

    ax.set_xlabel(f'Gauge {gauge_id} depth (m)')
    ax.set_ylabel(f'Wetland {wetland_id} well depth (m)')
    ax.set_title(f'Wetland {wetland_id} vs Gauge {gauge_id}  |  r = {r:.2f}')
    plt.tight_layout()
    plt.show()


# %% 4.0 Boxplot for each stream gauges variability

gauge_order = gauge_ts.groupby('gauge_id')['depth'].median().sort_values().index
gauge_ts_plot = gauge_ts.copy()
gauge_ts_plot['gauge_id'] = pd.Categorical(gauge_ts_plot['gauge_id'], categories=gauge_order, ordered=True)

fig, ax = plt.subplots(figsize=(10, 5))
gauge_ts_plot.boxplot(column='depth', by='gauge_id', ax=ax)
ax.set_xlabel('Gauge ID')
ax.set_ylabel('Depth (m)')
ax.set_title('Gauge Depth Variability by Site')
plt.suptitle('')
plt.tight_layout()
plt.show()




# %%
