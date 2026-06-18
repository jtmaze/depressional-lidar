# %% 1.0 File paths and data

import pandas as pd
import matplotlib.pyplot as plt

data_dir = 'D:/depressional_lidar/data/bradford/'
stream_gauges_path = f'{data_dir}/in_data/ancillary_data/sam_howley_streams.csv'
out_path = f'{data_dir}/in_data/ancillary_data/synthetic_stream_timeseries.csv'


# %% 2.0 Read the data

gauge_data = pd.read_csv(stream_gauges_path)
gauge_data = gauge_data[['Date', 'Site_ID', 'sensor_depth', 'depth']]
gauge_data['Site_ID'] = gauge_data['Site_ID'].astype('str')

# Parse dates, strip timezone info, and normalize to date-only (no time component)
gauge_data['Date'] = pd.to_datetime(gauge_data['Date'], utc=True).dt.tz_convert(None).dt.normalize()

print(gauge_data.head(10))
print(f"\nDate dtype: {gauge_data['Date'].dtype}")
print(f"Date range: {gauge_data['Date'].min()} to {gauge_data['Date'].max()}")

print(gauge_data.head(10))

# %% 3.0 Quick time series plot of sensor_depth OR depth

variable = 'depth'  # Change to 'depth' to plot the other variable

# Create plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot each site separately
for site in gauge_data['Site_ID'].unique():
    site_data = gauge_data[gauge_data['Site_ID'] == site].sort_values('Date')
    ax.plot(site_data['Date'], site_data[variable], linewidth=1, label=site, alpha=0.8)

ax.set_xlabel('Date')
ax.set_ylabel(variable)
ax.set_title(f'Time Series of {variable}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 4.0 Interpolate any data gaps in the average timeseries

daily_avg = gauge_data.groupby('Date')[variable].mean()

full_date_range = pd.date_range(daily_avg.index.min(), daily_avg.index.max(), freq='D')
daily_avg_interp = daily_avg.reindex(full_date_range)

print(f"Total days in range: {len(full_date_range)}")
print(f"Missing days filled: {daily_avg_interp.isna().sum()}")

daily_avg_interp = daily_avg_interp.interpolate(method='linear')

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(daily_avg.index, daily_avg.values, linewidth=2.5, color='orange', alpha=0.8, label='Daily Mean')
#ax.plot(daily_avg_interp.index, daily_avg_interp.values, linewidth=1, color='red', alpha=0.6, label='Interpolated', linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel(f'Daily Average {variable}')
ax.set_title(f'Daily Average Stream {variable} — Interpolated')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# %% 5.0 Write the interpolated stream gauge timeseries to csv

output_df = pd.DataFrame({
    'date': daily_avg_interp.index,
    'depth': daily_avg_interp.values
})

output_df.to_csv(out_path, index=False)


# %%
