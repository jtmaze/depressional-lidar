# %% 1.0 Import packages and establish list of wetland ids
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bradford_wy_scripts.functions.lai_vis_functions import read_concatonate_lai, visualize_lai

buffer_dist = 150

lai_dir = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/well_buffer_{buffer_dist}m_nomasking/'
lai_method = f'well_buffer_{buffer_dist}m_nomasking'
upper_bound = 5.6
lower_bound = 0.5

# NOTE: Paramters to adjust
ref_slope_threshold = -0.05  # LAI / yr
start_date = '2019-01-01'

wetland_connectivity_key = pd.read_excel(
    'D:/depressional_lidar/data/bradford/bradford_wetland_connect_logging_key.xlsx'
)

candidate_ids = wetland_connectivity_key['wetland_id'].unique()

# %% 2.0 Read the LAI timeseriess and determine which ids are suitable references. 

ref_dfs = []
ref_ids = []
log_dfs = []
log_ids = []

for i in candidate_ids:

    lai = read_concatonate_lai(lai_dir, i, lai_method, upper_bound, lower_bound)
    connect = wetland_connectivity_key[wetland_connectivity_key['wetland_id'] == i].iloc[0]['connectivity']
    print(connect)
                                                                                         
    # Check for trend with theil-sen slope regression
    ref_check = lai.loc[lai['date'] >= pd.Timestamp(start_date), ['date', 'roll_yr']].dropna()

    def _theil_sen_per_year(df):
        
        first_date = df['date'].iloc[0]
        x_months = (
            (df['date'].dt.year - first_date.year) * 12
            + (df['date'].dt.month - first_date.month)
        ).to_numpy(dtype=float)  
        y_vals = df['roll_yr'].to_numpy(dtype=float)

        month_slope = theilslopes(y=y_vals, x=x_months).slope
        yr_slope = month_slope * 12
        return yr_slope
    
    slope = _theil_sen_per_year(ref_check)
    if slope > ref_slope_threshold:
        lai['wetland_id'] = i
        ref_ids.append(i)
        ref_dfs.append(lai)
        print(f"Assigning {i} as a reference ID ||| Theil-Sen slope = {slope:.2f} LAI/yr")
        visualize_lai(lai[lai['date'] >= start_date], wetland_id=i, show=True)
    else:
        print(f"Assigning {i} as a logging ID ||| Theil-Sen slope = {slope:.2f} LAI/yr")
        lai['wetland_id'] = i
        log_ids.append(i)
        log_dfs.append(lai)

ref = pd.concat(ref_dfs)
log = pd.concat(log_dfs)

print(len(log_ids))
print(len(ref_ids))


# %% 3.0 Visually estimate the logging dates

"""
NOTE: This used this to estimate logging dates based on LAI timeseries. 
After getting an estimate, we verified the logging date using Planet Imagery.



aggregate_ref = ref.groupby('date')['roll_yr'].mean().reset_index()
aggregate_ref.rename(columns={'roll_yr': 'roll_yr_ref'}, inplace=True)

log_ids_dict = {
    '15_268': '2023-12-01',
    '15_516': '2023-01-01',
    '14_418': '2024-01-01',
    '14_115': '2022-05-01',
    '14_500': '2023-10-01',
    '14_610': '2023-09-01',
    '5_597': '2023-03-01', 
    '5a_598': '2021-06-01',
    '5_161': '2020-04-01',
    '5_510': '2023-09-01',
    '9_439': '2023-10-01',
    '9_508': '2023-10-01',
    '13_263': '2021-08-01',
    '13_271': '2022-10-01',
    '6_20': '2021-12-01',
    '7_626': '2022-06-01',
    '7_243': '2022-07-01',
    '3_311': '2023-01-01',
    '3_173': '2023-03-01',
    '3_244': '2023-02-01', 
    '6_93': '2024-09-01',

    # Flow-through well_ids
    '14_612': '2020-08-01',
    '14.9_168': '2019-12-01',
    '14.9_601': '2021-10-01',
    '9_77': '2023-04-01',
    '7_622': '2022-03-01',
    '3_23': '2023-10-01',
    '9_609': '2020-01-01',
    '13_267': '2022-09-01', 
    '7_341': '2022-01-01',

    '14.9_527': '2020-02-01',
    '6_300': '2024-02-01',

    # Only at 250m
    '9_609': '2023-10-01'


}

for i in log_ids:

    temp = log[log['well_id'] == i]
    temp = temp.merge(aggregate_ref,
                      on='date',
                      how='outer')

    temp['log_diff_1yr'] = temp['roll_yr'] - temp['roll_yr_ref']
    temp = temp[temp['date'] >= start_date]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top left: Full time series of roll_yr
    axes[0, 0].plot(temp['date'], temp['roll_yr'])
    axes[0, 0].set_title(f'Rolling Year LAI for {i}')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('roll_yr')
    
    logging_date = pd.Timestamp(log_ids_dict[i])
    axes[0, 0].axvline(logging_date, color='red', linestyle='--', label='Logging Date')
    axes[0, 0].legend()

    # Top right: Zoomed roll_yr
    start_zoom = logging_date - pd.DateOffset(months=8)
    end_zoom = logging_date + pd.DateOffset(months=8)
    temp_zoom = temp[(temp['date'] >= start_zoom) & (temp['date'] <= end_zoom)]
    
    axes[0, 1].plot(temp_zoom['date'], temp_zoom['roll_yr'])
    axes[0, 1].axvline(logging_date, color='red', linestyle='--', label='Logging Date')
    axes[0, 1].set_title(f'Rolling Year LAI for {i} (±8 months)')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Logged 1-yr LAI')
    axes[0, 1].legend()
    
    axes[1, 0].plot(temp['date'], temp['log_diff_1yr'])
    axes[1, 0].set_title(f'Reference Rolling Year LAI')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Reference 1-yr rolling LAI')
    axes[1, 0].axvline(logging_date, color='red', linestyle='--', label='Logging Date')
    axes[1, 0].legend()
    
    axes[1, 1].plot(temp_zoom['date'], temp_zoom['log_diff_1yr'])
    axes[1, 1].axvline(logging_date, color='red', linestyle='--', label='Logging Date')
    axes[1, 1].set_title(f'Reference Rolling Year LAI (±8 months)')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Rolling 1yr difference (log - ref)')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

print('done')

"""

# %% 4.0 Determine if logged wetlands were sufficiently instrumented prior to harvest

logged_summary = wetland_connectivity_key[['wetland_id', 'planet_log_date']].copy()
logged_summary['hydro_sufficient'] = pd.to_datetime(logged_summary['planet_log_date']) >= pd.to_datetime('2022-05-01')

# %% 4.1 Timeseries plot for roll_yr LAI. Use all well_ids color by log and ref

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14), height_ratios=[4, 1])

# Plot individual lines as dashed
for wid in log_ids:
    df_w = log[log['wetland_id'] == wid].sort_values('date')
    df_w = df_w[df_w['date'] >= '2019-01-01']
    ax1.plot(df_w['date'], df_w['roll_yr'], color='red', alpha=0.4, linewidth=1.5, linestyle='--')

for wid in ref_ids:
    df_w = ref[ref['wetland_id'] == wid].sort_values('date')
    df_w = df_w[df_w['date'] >= '2019-01-01']
    ax1.plot(df_w['date'], df_w['roll_yr'], color='blue', alpha=0.4, linewidth=1.5, linestyle='--')

# Calculate and plot averages as thick solid lines
log_avg = log[log['date'] >= '2019-01-01'].groupby('date')['roll_yr'].mean().reset_index()
ref_avg = ref[ref['date'] >= '2019-01-01'].groupby('date')['roll_yr'].mean().reset_index()

ax1.plot(log_avg['date'], log_avg['roll_yr'], color='red', linewidth=3, label=f'Logged (n={len(log_ids)})')
ax1.plot(ref_avg['date'], ref_avg['roll_yr'], color='blue', linewidth=3, label=f'Reference (n={len(ref_ids)})')

ax1.set_ylabel('Landsat 8 LAI', fontsize=22)
ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=24)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.grid(True, alpha=0.3)

# Set x-axis limits for alignment
start_date = pd.to_datetime('2019-01-01')
end_date = pd.to_datetime('2026-01-01')
ax1.set_xlim(start_date, end_date)
ax1.tick_params(axis='x', labelbottom=False)

# Second subplot: Plot logging dates as stacked dots by quarter
# Collect all logging dates first
logging_dates = []
for wid in log_ids:
    log_date = logged_summary[logged_summary['wetland_id'] == wid]['planet_log_date']
    if not log_date.empty:
        log_date = pd.to_datetime(log_date.iloc[0])
        if log_date >= start_date:
            logging_dates.append(log_date)

# Create quarterly bins and count events per quarter
if logging_dates:
    logging_df = pd.DataFrame({'date': logging_dates})
    logging_df['quarter'] = logging_df['date'].dt.to_period('Q')
    quarter_counts = logging_df['quarter'].value_counts().sort_index()
    
    # Plot stacked dots for each quarter
    max_count = quarter_counts.max() if len(quarter_counts) > 0 else 1
    
    for quarter, count in quarter_counts.items():
        # Use middle of quarter for x-position
        quarter_start = quarter.start_time
        quarter_end = quarter.end_time
        quarter_mid = quarter_start + (quarter_end - quarter_start) / 2
        
        # Stack dots vertically
        for i in range(count):
            y_pos = (i - (count-1)/2) * 0.15  # Center the stack around y=0
            ax2.scatter(quarter_mid, y_pos, marker='o', color='red', s=80, alpha=0.8, edgecolor='black', linewidth=1)

ax2.set_ylabel('Logging\nEvents\n  ', fontsize=22)
ax2.set_xlabel('Year', fontsize=18)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_ylim(-1, 1) 
ax2.set_yticks([])
ax2.set_xlim(start_date, end_date)  # Match x-axis limits

# Add green shading for hydrologic monitoring period
monitoring_start = pd.to_datetime('2021-11-20')
ax2.axvspan(monitoring_start, end_date, color='green', alpha=0.2, zorder=0)
ax2.text(monitoring_start + (end_date - monitoring_start) / 2, 0.7, 
         'Hydrologic monitoring period', 
         ha='center', va='center', fontsize=16, 
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=1))

ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% 5.0 Generate a combination of every logging and reference wetland. 

combinations_list = []
for ref_id in ref_ids:
    for log_id in log_ids:

        # Get hydro data sufficiency and connectivity classification values
        hydro_sufficient = logged_summary[
            logged_summary['wetland_id'] == log_id
        ].iloc[0]['hydro_sufficient']

        planet_log = logged_summary[
            logged_summary['wetland_id'] == log_id
        ].iloc[0]['planet_log_date']

        ref_connect = wetland_connectivity_key[
            wetland_connectivity_key['wetland_id'] == ref_id
        ].iloc[0]['connectivity']

        log_connect = wetland_connectivity_key[
            wetland_connectivity_key['wetland_id'] == log_id
        ].iloc[0]['connectivity']

        combinations_list.append({
            'reference_id': ref_id,
            'logged_id': log_id,
            'planet_logging_date': planet_log, 
            'ref_connect': ref_connect,
            'log_connect': log_connect,
            'logged_hydro_sufficient': hydro_sufficient,

        })

combinations_df = pd.DataFrame(combinations_list)
print(len(combinations_df))

bad_log_ids = [
    '3_173', # The wetland itself was logged. 
    '3_244', # The wetland was logged and the well was snapped in the process
    '13_263', # well position is not stable. 
]

combinations_df = combinations_df[~combinations_df['logged_id'].isin(bad_log_ids)]
print(len(combinations_df))
print(len(combinations_df['logged_id'].unique()))
print(len(combinations_df['reference_id'].unique()))

combinations_df = combinations_df[combinations_df['logged_hydro_sufficient'] == True]
print(len(combinations_df))
print(len(combinations_df['logged_id'].unique()))
print(len(combinations_df['reference_id'].unique()))

# %% 6.0 Write the output

combinations_df.to_csv(
    f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{buffer_dist}m_all_wells.csv',
    index=False
)


# %%
