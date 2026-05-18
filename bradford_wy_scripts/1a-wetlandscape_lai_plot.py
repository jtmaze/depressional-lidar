# %% 1.0 Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

landscape_lai_path = 'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/landscape_lai_timeseries.csv'

landscape_lai = pd.read_csv(landscape_lai_path)[['date', 'lai_bradford', 'lai_upland', 'lai_wetland']]

# %% 2.0 Quick timeseries of landscape lai

# Convert date to datetime
landscape_lai['date'] = pd.to_datetime(landscape_lai['date'])
landscape_lai = landscape_lai.sort_values('date').reset_index(drop=True)

colors = ['black', "#CCB486", "#649600"]  

fig, ax = plt.subplots(figsize=(8, 5))

# # Plot raw data with transparency
# plt.plot(landscape_lai['date'], landscape_lai['lai_bradford'], label='landscape (raw)', 
#          color=colors[0], alpha=0.2, linestyle=':')
# plt.plot(landscape_lai['date'], landscape_lai['lai_upland'], label='upland (raw)', 
#          color=colors[1], alpha=0.2, linestyle=':')
# plt.plot(landscape_lai['date'], landscape_lai['lai_wetland'], label='wetland (raw)', 
#          color=colors[2], alpha=0.2, linestyle=':')

# Plot 3-month rolling averages
# ax.plot(landscape_lai['date'], landscape_lai['lai_bradford'].rolling(3, center=True, min_periods=1).mean(), 
#     label='Full Landscape', color=colors[0], linewidth=7)
plt.plot(landscape_lai['date'], landscape_lai['lai_upland'].rolling(3, center=True, min_periods=1).mean(), 
         label='Upland Areas', color=colors[1], linewidth=7)
plt.plot(landscape_lai['date'], landscape_lai['lai_wetland'].rolling(3, center=True, min_periods=1).mean(), 
         label='Wetland Areas', color=colors[2], linewidth=7)

# ax.set_xlabel('Date', fontsize=26)
ax.set_ylabel('LAI', fontsize=48)
#ax.legend(fontsize=18)
ax.tick_params(axis='both', labelsize=48)

ax.set_xticks([])
ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

ax.yaxis.set_major_locator(MaxNLocator(1))

ax.grid(False)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_linewidth(2.5)
ax.spines['bottom'].set_linewidth(2.5)

plt.tight_layout()
plt.show()
# %%
