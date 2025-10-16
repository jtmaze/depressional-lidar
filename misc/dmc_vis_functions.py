

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_stage_ts(logged_df: pd.DataFrame, reference_df: pd.DataFrame, logged_date: pd.Timestamp):

    if 'well_id' not in logged_df.columns or 'well_id' not in reference_df.columns:
        reference_id = 'reference'
        logged_id = 'logged'
        well_depth_col_ref = 'well_depth_ref'
        well_depth_col_log = 'well_depth_log'   
    else:
        reference_id = reference_df['well_id'].iloc[0]
        logged_id = logged_df['well_id'].iloc[0]
        well_depth_col_ref = 'well_depth'
        well_depth_col_log = 'well_depth'

    plt.figure(figsize=(8, 6))
    plt.plot(reference_df['day'], reference_df[well_depth_col_ref], label=f'Reference ({reference_id})')
    plt.plot(logged_df['day'], logged_df[well_depth_col_log], label=f'Logged ({logged_id})')
    plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
    plt.xlabel('Date')
    plt.ylabel('Well Depth')
    plt.title(f'Well Depth Time Series - Reference: {reference_id}, Logged: {logged_id}')
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

def remove_flagged_buffer(ts_df, buffer_days=2):
    """
    Remove ±buffer_days from records where flag != 0
    """
    
    flagged_dates = ts_df[ts_df['flag'] != 0]['day']
    
    # Create set of dates to remove (±buffer_days around each flagged date)
    dates_to_remove = set()
    for date in flagged_dates:
        for offset in range(-buffer_days, buffer_days + 1):  # -2, -1, 0, 1, 2 days
            dates_to_remove.add(date + pd.Timedelta(days=offset))
    
    filtered_df = ts_df[~ts_df['day'].isin(dates_to_remove)]
    
    print(f"Removed {len(ts_df) - len(filtered_df)} records around flagged periods")
    return filtered_df