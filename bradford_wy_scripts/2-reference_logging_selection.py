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

candidate_ids = wetland_connectivity_key['well_id'].unique()

# %% 2.0 Read the LAI timeseriess and determine which ids are suitable references. 

ref_dfs = []
ref_ids = []
log_dfs = []
log_ids = []

for i in candidate_ids:

    lai = read_concatonate_lai(lai_dir, i, lai_method, upper_bound, lower_bound)
    connect = wetland_connectivity_key[wetland_connectivity_key['well_id'] == i].iloc[0]['connectivity']
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
        lai['well_id'] = i
        ref_ids.append(i)
        ref_dfs.append(lai)
        print(f"Assigning {i} as a reference ID ||| Theil-Sen slope = {slope:.2f} LAI/yr")
        visualize_lai(lai[lai['date'] >= start_date], well_id=i, show=True)
    else:
        print(f"Assigning {i} as a logging ID ||| Theil-Sen slope = {slope:.2f} LAI/yr")
        lai['well_id'] = i
        log_ids.append(i)
        log_dfs.append(lai)

ref = pd.concat(ref_dfs)
log = pd.concat(log_dfs)

print(len(log_ids))
print(len(ref_ids))

# %% 2.1 Timeseries plot for roll_yr LAI. Use all well_ids color by log and ref
fig, ax = plt.subplots(figsize=(10, 8))

# Plot individual lines as dashed
for wid in log_ids:
    df_w = log[log['well_id'] == wid].sort_values('date')
    df_w = df_w[df_w['date'] >= '2019-01-01']
    ax.plot(df_w['date'], df_w['roll_yr'], color='red', alpha=0.4, linewidth=1.5, linestyle='--')

for wid in ref_ids:
    df_w = ref[ref['well_id'] == wid].sort_values('date')
    df_w = df_w[df_w['date'] >= '2019-01-01']
    ax.plot(df_w['date'], df_w['roll_yr'], color='blue', alpha=0.4, linewidth=1.5, linestyle='--')

# Calculate and plot averages as thick solid lines
log_avg = log[log['date'] >= '2019-01-01'].groupby('date')['roll_yr'].mean().reset_index()
ref_avg = ref[ref['date'] >= '2019-01-01'].groupby('date')['roll_yr'].mean().reset_index()

ax.plot(log_avg['date'], log_avg['roll_yr'], color='red', linewidth=3, label=f'Logged Average (n={len(log_ids)})')
ax.plot(ref_avg['date'], ref_avg['roll_yr'], color='blue', linewidth=3, label=f'Reference Average (n={len(ref_ids)})')

ax.set_title('LAI by Well ID', fontsize=18, fontweight='bold')
ax.set_xlabel('Date', fontsize=16)
ax.set_ylabel('Landsat 8 LAI', fontsize=16)
ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

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

logged_summary = wetland_connectivity_key[['well_id', 'planet_log_date']].copy()
logged_summary['hydro_sufficient'] = pd.to_datetime(logged_summary['planet_log_date']) >= pd.to_datetime('2022-05-01')


# %% 5.0 Generate a combination of every logging and reference wetland. 

combinations_list = []
for ref_id in ref_ids:
    for log_id in log_ids:

        # Get hydro data sufficiency and connectivity classification values
        hydro_sufficient = logged_summary[
            logged_summary['well_id'] == log_id
        ].iloc[0]['hydro_sufficient']

        planet_log = logged_summary[
            logged_summary['well_id'] == log_id
        ].iloc[0]['planet_log_date']

        ref_connect = wetland_connectivity_key[
            wetland_connectivity_key['well_id'] == ref_id
        ].iloc[0]['connectivity']

        log_connect = wetland_connectivity_key[
            wetland_connectivity_key['well_id'] == log_id
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
