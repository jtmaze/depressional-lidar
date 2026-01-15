# %% 1.0 Import packages and establish list of wetland ids
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import theilslopes

PROJECT_ROOT = r"C:\Users\jtmaz\Documents\projects\depressional-lidar"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bradford_wy_scripts.functions.lai_vis_functions import read_concatonate_lai

buffer_dist = 150

lai_dir = f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/well_buffer_{buffer_dist}m_nomasking/'
lai_method = f'well_buffer_{buffer_dist}m_nomasking'
upper_bound = 5.6
lower_bound = 0.5

# NOTE: Paramters to adjust
ref_slope_threshold = -0.05  # LAI / yr
start_date = '2019-01-01'

# Keep wetlands
keep_ids = [
    "15_268", "15_409", "15_516", "15_4",
    "14_418", "14_115", "14_500", "14_538", "14_610", "14_616",
    "5_597", "5a_550", "5a_598", "5_161", "5_510", "5_573", "5_546",
    "9_439", "9_332", "9_508",
    "13_263", "13_271", "13_410", "13_274",
    "6a_17", "6_300", "6_20",
    "7_626", "7_243",
    "3_311", "3_173", "3_34", "3_638", "3_244",
    "6_93"
]

# Omit wetlands
omit_wetlands = [
    "14_612", "14_15", "14.9_527", "14.9_168", "14.9_601",
    "5_560", "5_321",
    "9_77", "7_622",
    "3_21", "3_23"
]

# Unsure wetlands
unsure_wetlands = [
    "5a_582", "9_609", "13_267", "6a_530", "6_629", "7_341"
]

candidate_ids = keep_ids

print(len(candidate_ids))

#print(len(combinations))
# %% 3.0 Read the LAI timeseriess and determine which ids are suitable references. 

ref_dfs = []
ref_ids = []
log_dfs = []
log_ids = []

for i in candidate_ids:

    lai = read_concatonate_lai(lai_dir, i, lai_method, upper_bound, lower_bound)

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
        #visualize_lai(lai[lai['date'] >= start_date], well_id=i, show=True)
    else:
        print(f"Assigning {i} as a logging ID ||| Theil-Sen slope = {slope:.2f} LAI/yr")
        lai['well_id'] = i
        log_ids.append(i)
        log_dfs.append(lai)

ref = pd.concat(ref_dfs)
log = pd.concat(log_dfs)

print(len(log_ids))
print(len(ref_ids))
# %% 3.0 Visually estimate the logging dates

aggregate_ref = ref.groupby('date')['roll_yr'].mean().reset_index()
aggregate_ref.rename(columns={'roll_yr': 'roll_yr_ref'}, inplace=True)

log_ids_dict = {
    '15_268': '2023-12-01',
    '15_516': '2023-03-01',
    #'15_4': '2020-09-01',
    '14_418': '2024-02-01',
    '14_115': '2022-05-01',
    '14_500': '2023-10-01',
    '14_610': '2023-09-01',
    #'14_616': '2023-09-01',
    '5_597': '2023-05-01', # NOTE: changed from 2023-01-01
    '5a_598': '2021-06-01',
    '5_161': '2020-04-01',
    '5_510': '2023-08-01',
    '9_439': '2023-10-01',
    #'9_332': '2020-02-01',
    '9_508': '2023-10-01',
    '13_263': '2022-01-01',
    '13_271': '2022-11-01',
    #'6a_17': '2023-06-01',
    '6_300': '2024-03-01', 
    '6_20': '2021-12-01',
    '7_626': '2022-07-01',
    #'7_243': '2022-07-01',
    '3_311': '2023-02-01',
    '3_173': '2023-03-01',
    '3_244': '2023-02-01', # NOTE: hydro data is suspect
    '6_93': '2024-11-01'
    # NOTE: Additional logged wetlands, originally not included due to connectivity
    #'14_612': '2020-08-01',
    #'14.9_527': '2020-06-01',
    #'14.9_168': '2020-01-01',
    #'14.9_601': '2023-08-01',
    #'9_77': '2023-03-01',
    #'7_622': '2022-02-01',
    #'3_23': '2023-03-01',
    #'9_609': '2020-01-01',
    #'13_267': '2022-09-01', 
    #'7_341': '2022-01-01',
}

for i in log_ids:

    temp = log[log['well_id'] == i]
    temp = temp.merge(aggregate_ref,
                      on='date',
                      how='outer')

    temp['log_diff_1yr'] = temp['roll_yr'] - temp['roll_yr_ref']
    temp = temp[temp['date'] >= start_date]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(temp['date'], temp['log_diff_1yr'])
    axes[0].set_title(f'Log Difference for {i}')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('log_diff_1yr')
    
    logging_date = pd.Timestamp(log_ids_dict[i])
    axes[0].axvline(logging_date, color='red', linestyle='--', label='Logging Date')
    axes[0].legend()

    start_zoom = logging_date - pd.DateOffset(months=8)
    end_zoom = logging_date + pd.DateOffset(months=8)
    temp_zoom = temp[(temp['date'] >= start_zoom) & (temp['date'] <= end_zoom)]
    
    axes[1].plot(temp_zoom['date'], temp_zoom['log_diff_1yr'])
    axes[1].axvline(logging_date, color='red', linestyle='--', label='Logging Date')
    axes[1].set_title(f'Log Difference for {i} (±8 months)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('log_diff_1yr')
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()

print('done')

# %% 4.0 Determine if logged wetlands were sufficiently instrumented prior to harvest

logged_summary = pd.DataFrame.from_dict(
    log_ids_dict, orient='index', columns=['logging_date']
).reset_index()
logged_summary.rename(columns={'index': 'well_id'}, inplace=True)
logged_summary['hydro_sufficient'] = pd.to_datetime(logged_summary['logging_date']) >= pd.to_datetime('2022-06-01')

valid_log_ids = logged_summary[logged_summary['hydro_sufficient']]['well_id'].to_list()

# %% 5.0 Generate a combination of every logging and reference wetland. 

combinations_list = []
for ref_id in ref_ids:
    for log_id in valid_log_ids:
        combinations_list.append({
            'reference_id': ref_id,
            'logged_id': log_id,
            'logging_date': log_ids_dict[log_id]
        })


combinations_df = pd.DataFrame(combinations_list)

# %% 6.0 Write the output

combinations_df.to_csv(
    f'D:/depressional_lidar/data/bradford/in_data/hydro_forcings_and_LAI/log_ref_pairs_{buffer_dist}m.csv',
    index=False
)


# %%
