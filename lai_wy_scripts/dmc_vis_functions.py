import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.dates as mdates

def plot_ts(df: pd.DataFrame, y_col: str):

    plt.figure(figsize=(10, 6))
    plt.plot(df['day'], df[y_col], marker='o', markersize=2, linewidth=1)
    plt.xlabel('Date')
    plt.ylabel(y_col)
    plt.title(f'Time Series of {y_col}')
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.show()

def plot_stage_ts(
        logged_df: pd.DataFrame, 
        reference_df: pd.DataFrame, 
        logged_date: pd.Timestamp,
        y_label: str
    ):

    if 'well_id' not in logged_df.columns or 'well_id' not in reference_df.columns:
        reference_id = 'reference'
        logged_id = 'logged'
        well_depth_col_ref = 'wetland_depth_ref'
        well_depth_col_log = 'wetland_depth_log'
    else:
        reference_id = reference_df['well_id'].iloc[0]
        logged_id = logged_df['well_id'].iloc[0]
        well_depth_col_ref = 'well_depth'
        well_depth_col_log = 'well_depth'

    # Sort dataframes by date to ensure proper gap detection
    reference_df = reference_df.sort_values('day')
    logged_df = logged_df.sort_values('day')

    plt.figure(figsize=(8, 6))
    
    # Plot with markers and detect gaps
    plt.plot(reference_df['day'], reference_df[well_depth_col_ref], 
             label=f'Reference ({reference_id})', marker='o', markersize=2, linewidth=1)
    plt.plot(logged_df['day'], logged_df[well_depth_col_log], 
             label=f'Logged ({logged_id})', marker='o', markersize=2, linewidth=1)
    
    plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date')
    plt.xlabel('Date')
    plt.ylabel(f'{y_label}')
    plt.title(f'{y_label} Time Series - Reference: {reference_id}, Logged: {logged_id}')
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
    
    flagged_dates = ts_df[ts_df['flag'] == 2]['day']
    
    # Create set of dates to remove (±buffer_days around each flagged date)
    dates_to_remove = set()
    for date in flagged_dates:
        for offset in range(-buffer_days, buffer_days + 1):  # -2, -1, 0, 1, 2 days
            dates_to_remove.add(date + pd.Timedelta(days=offset))
    
    filtered_df = ts_df[~ts_df['day'].isin(dates_to_remove)]
    
    print(f"Removed {len(ts_df) - len(filtered_df)} records around bottomed periods")
    return filtered_df

def plot_correlations(
        comparison_df: pd.DataFrame, 
        x_series_name: str, 
        y_series_name: str, 
        log_date: pd.Timestamp,
        filter_obs: tuple = None
    ):
    
    if filter_obs:
        min_val, max_val = filter_obs
        comparison_df = comparison_df[
            (comparison_df[x_series_name] >= min_val) &
            (comparison_df[x_series_name] <= max_val) &
            (comparison_df[y_series_name] >= min_val) &
            (comparison_df[y_series_name] <= max_val)
        ]

    comparison_df['pre_logging'] = comparison_df['day'] <= log_date
    
    # Split data into pre and post logging
    pre_df = comparison_df[comparison_df['pre_logging']]
    post_df = comparison_df[~comparison_df['pre_logging']]
    
    # Calculate regressions
    pre_slope, pre_intercept, pre_r, pre_p, pre_stderr = stats.linregress(
        pre_df[x_series_name], pre_df[y_series_name]
    )
    post_slope, post_intercept, post_r, post_p, post_stderr = stats.linregress(
        post_df[x_series_name], post_df[y_series_name]
    )
    
    # Create plot
    plt.figure(figsize=(8, 8))
    plt.scatter(
        pre_df[x_series_name],
        pre_df[y_series_name], 
        color='black',
        label='Pre-logging',
        alpha=0.6
    )
    plt.scatter(
        post_df[x_series_name],
        post_df[y_series_name], 
        color='red',
        label='Post-logging',
        alpha=0.6
    )
    
    # Plot trendlines
    x_range = comparison_df[x_series_name]
    plt.plot(x_range, pre_slope * x_range + pre_intercept, 
                'black', label=f'Pre: m={pre_slope:.3f}, r_sq={pre_r:.3f}')
    plt.plot(x_range, post_slope * x_range + post_intercept, 
                'red', linestyle='--', label=f'Post: m={post_slope:.3f}, r_sq={post_r:.3f}')
    
    plt.xlabel(x_series_name, fontsize=14)
    plt.ylabel(y_series_name, fontsize=14)
    plt.title(f'Pre vs Post Logging Correlation')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return {
        'pre': {'slope': pre_slope, 'intercept': pre_intercept, 'p_value': pre_p, 'r': pre_r},
        'post': {'slope': post_slope, 'intercept': post_intercept, 'p_value': post_p, 'r': post_r}
    }


def plot_dmc(
        comparison_df: pd.DataFrame, 
        x_series_name: str, 
        y_series_name: str, 
        log_date: pd.Timestamp, 
    ):

    pre_logged = comparison_df[comparison_df['day'] <= log_date]
    if log_date in comparison_df['day'].values:
        log_x_value = comparison_df.loc[comparison_df['day'] == log_date, x_series_name].values[0]
    else:
        # Find the nearest date and get its x value
        nearest_idx = (comparison_df['day'] - log_date).abs().argsort()[0]
        log_x_value = comparison_df.iloc[nearest_idx][x_series_name]
        nearest_date = comparison_df.iloc[nearest_idx]['day']
        print(f"Exact log date not found. Using nearest date: {nearest_date}")
 
    x_pre = pre_logged[x_series_name].to_numpy()
    y_pre = pre_logged[y_series_name].to_numpy()
    x_full = comparison_df[x_series_name].to_numpy()


    result = np.linalg.lstsq(x_pre[:, None], y_pre, rcond=None)
    m = result[0][0]

    plt.figure(figsize=(8, 8))
    plt.scatter(comparison_df[x_series_name], comparison_df[y_series_name], label=f"DMC")
    plt.plot(x_full, m * x_full, color='black', linestyle='--', label=f'Pre-logging fit')
    plt.axvline(log_x_value, color='red', linestyle='-', label='logging date')
    plt.xlabel('Cummulative Reference')
    plt.ylabel('Cummulative Logged')
    ax = plt.gca()
    ax.text(0.02, 0.98, f"m = {m:.3f}", transform=ax.transAxes, ha='left', va='top')
    plt.show()

    return m


def plot_dmc_residual(
        comparison_df: pd.DataFrame,
        x_series_name: str,
        y_series_name: str,
        log_date: pd.Timestamp
    ):

    pass

