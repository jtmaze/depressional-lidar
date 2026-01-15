import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def read_concatonate_lai(
        dir_path: str, 
        well_id: str, 
        lai_method: str, 
        upper_bound: float,
        lower_bound: float
    ):
    """
    Reads and concatonates LAI csv files. The lai_method argument adds a column to indicate the 
    calculation method
    """
    # Read files
    file_list = glob.glob(f"{dir_path}/*.csv")
    pattern = rf"_{well_id}_"
    well_files = [f for f in file_list if re.search(pattern, f)]
    dfs = [pd.read_csv(f) for f in well_files]
    out_df = pd.concat(dfs, ignore_index=True)

    # Quick column cleaning
    out_df = out_df.drop(columns=['system:index', '.geo'])
    out_df = out_df.rename(columns={'mean_LAI': 'LAI'})

    # TODO: Ensure that missing monthly observatoins are filled with np.nan
    out_df['date'] = pd.to_datetime(
        out_df['year'].astype(str) + '-' + out_df['month'].astype(str) + '-01'
    )

    full_date_range = pd.date_range(
        start=out_df['date'].min(),
        end=out_df['date'].max(),
        freq='MS'
    )
    
    complete_dates = pd.DataFrame({'date': full_date_range})
    out_df = complete_dates.merge(out_df, on='date', how='left')

    # Use bounds to trim the LAI data
    if upper_bound:
        out_df.loc[out_df['LAI'] > upper_bound, 'LAI'] = np.nan
    if lower_bound:
        out_df.loc[out_df['LAI'] < lower_bound, 'LAI'] = np.nan

    # Specify the LAI method
    out_df['method'] = lai_method

    out_df = apply_moving_averages(out_df)
    
    return out_df

def apply_moving_averages(lai_df: pd.DataFrame):

    out_df = lai_df.copy()

    out_df['roll5'] = out_df['LAI'].rolling(
        window=5,
        center=True,
        min_periods=3
    ).mean()

    out_df['roll9'] = out_df['LAI'].rolling(
        window=9, 
        center=True, 
        min_periods=5
    ).mean()

    out_df['roll_yr'] = out_df['LAI'].rolling(
        window=12, 
        center=True, 
        min_periods=7
    ).mean()

    return out_df


def visualize_lai(lai_df: pd.DataFrame, 
                  well_id: str, 
                  show: bool=True, 
                  ax=None, 
                  title_suffix: str = ""
    ):
    """
    Plot LAI timeseries - can be used standalone or as part of subplots
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(lai_df['date'], lai_df['LAI'], 
               color='red', alpha=0.7, s=30, label='LAI')
    
    ax.plot(lai_df['date'], lai_df['roll5'], 
            color='blue', linewidth=2, label='5-month')
    
    ax.plot(lai_df['date'], lai_df['roll9'], 
            color='green', linewidth=2, label='9-month')
    
    ax.plot(lai_df['date'], lai_df['roll_yr'], 
            color='orange', linewidth=2, label='1-year')
    
    # Formatting
    ax.set_title(f"Wetland ID: {well_id} - LAI Timeseries {title_suffix}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('LAI', fontsize=12)
    ax.legend(frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    if ax is None and show:
        plt.tight_layout()
        plt.show()

def lai_comparison_vis(
        logged_lai_df: pd.DataFrame,
        reference_lai_df: pd.DataFrame,
        logged_id: str, 
        reference_id: str
    ):

    """
    Create three-panel LAI comparison visualization:
    1. Logged wetland LAI
    2. Reference wetland LAI  
    3. Difference in rolling averages
    """
    merged_df = logged_lai_df[['date', 'LAI', 'roll_yr']].merge(
        reference_lai_df[['date', 'LAI', 'roll_yr']], 
        on='date', 
        suffixes=('_logged', '_ref'),
        how='outer'
    )
    
    # Calculate differences
    merged_df['1month_diff'] = merged_df['LAI_logged'] - merged_df['LAI_ref']
    merged_df['roll_yr_diff'] = merged_df['roll_yr_logged'] - merged_df['roll_yr_ref']

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    # Panel 1: Logged wetland
    visualize_lai(logged_lai_df, logged_id, show=True, ax=ax1, title_suffix="(Logged)")
    
    # Panel 2: Reference wetland  
    visualize_lai(reference_lai_df, reference_id, show=True, ax=ax2, title_suffix="(Reference)")

    ax3.bar(merged_df['date'], merged_df['1month_diff'], width=15,
             color='cyan', edgecolor='black', alpha=0.4, label='monthly difference')
    ax3.plot(merged_df['date'], merged_df['roll_yr_diff'], 
             color='black', linewidth=4, label='1-year difference')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    ax3.set_title(f"LAI Difference: Logged - Reference", fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('LAI Difference', fontsize=12)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def prep_for_logging_eval(log_df: pd.DataFrame, ref_df: pd.DataFrame):

    merged = log_df[['date', 'roll_yr']].merge(
        ref_df[['date', 'roll_yr']],
        on='date',
        suffixes=('_log', '_ref'),
        how='outer'
    )

    merged['roll_yr_diff'] = merged['roll_yr_log'] - merged['roll_yr_ref']

    return merged

