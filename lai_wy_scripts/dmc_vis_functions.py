import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
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
        ].copy()

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
                'red', label=f'Post: m={post_slope:.3f}, r_sq={post_r:.3f}')
    
    plt.xlabel(x_series_name, fontsize=14)
    plt.ylabel(y_series_name, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_correlations_from_model(
        comparison_df: pd.DataFrame, 
        x_series_name: str, 
        y_series_name: str, 
        log_date: pd.Timestamp,
        model_results: dict,
    ):
    """
    Plot correlations using pre-fitted interaction model parameters instead of
    recalculating regressions. Useful for consistent visualization after ANCOVA analysis.
    
    Args:
        comparison_df: DataFrame with time series data
        x_series_name: Name of x-variable column
        y_series_name: Name of y-variable column  
        log_date: Date when logging occurred
        model_results: Dictionary from fit_interaction_model() containing parameters
    """

    comparison_df['pre_logging'] = comparison_df['day'] <= log_date
    
    # Split data into pre and post logging
    pre_df = comparison_df[comparison_df['pre_logging']]
    post_df = comparison_df[~comparison_df['pre_logging']]
    
    # Extract model parameters
    pre_intercept = model_results['pre']['intercept']
    pre_slope = model_results['pre']['slope']
    post_intercept = model_results['post']['intercept']
    post_slope = model_results['post']['slope']
    
    # Calculate R² for each period (for display)
    pre_r_sq = np.corrcoef(pre_df[x_series_name], pre_df[y_series_name])[0,1]**2
    post_r_sq = np.corrcoef(post_df[x_series_name], post_df[y_series_name])[0,1]**2
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plots
    ax.scatter(
        pre_df[x_series_name],
        pre_df[y_series_name], 
        color='black',
        label='Pre-logging',
        alpha=0.6,
        s=40
    )
    ax.scatter(
        post_df[x_series_name],
        post_df[y_series_name], 
        color='red',
        label='Post-logging',
        alpha=0.6,
        s=40
    )
    
    # Create smooth line range for plotting regression lines
    x_min, x_max = comparison_df[x_series_name].min(), comparison_df[x_series_name].max()
    x_smooth = np.linspace(x_min, x_max, 100)
    
    # Plot regression lines using model parameters
    ax.plot(x_smooth, pre_slope * x_smooth + pre_intercept, 
            'black', linewidth=2, linestyle='--',
            label=f'Pre: m={pre_slope:.3f}, R²={pre_r_sq:.3f}')
    ax.plot(x_smooth, post_slope * x_smooth + post_intercept, 
            'red', linewidth=2, linestyle='--',
            label=f'Post: m={post_slope:.3f}, R²={post_r_sq:.3f}')
    
    # Add significance indicators
    p_slope = model_results['tests']['p_slope_diff']
    p_intercept = model_results['tests']['p_intercept_diff']
    slope_change = post_slope - pre_slope
    intercept_change = post_intercept - pre_intercept
    
    # Add text box with test results
    textstr = f'Slope change: {slope_change:+.3f} (p={p_slope:.3f})\n'
    textstr += f'Intercept change: {intercept_change:+.3f} (p={p_intercept:.3f})\n'
    #textstr += f'Joint test p-value: {model_results["tests"]["joint_p"]:.3f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Formatting
    ax.set_xlabel(x_series_name, fontsize=14)
    ax.set_ylabel(y_series_name, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Model summary (n={model_results['model_fit']['n']}):")
    print(f"  Pre-logging:  y = {pre_slope:.3f}x + {pre_intercept:.3f}")
    print(f"  Post-logging: y = {post_slope:.3f}x + {post_intercept:.3f}")
    print(f"  R² = {model_results['model_fit']['r2']:.3f}")
    
    if p_slope < 0.05:
        print(f"  *** Significant slope change: {slope_change:+.3f} (p={p_slope:.3f})")
    if p_intercept < 0.05:
        print(f"  *** Significant intercept change: {intercept_change:+.3f} (p={p_intercept:.3f})")

def fit_interaction_model(
    comparison_df: pd.DataFrame,
    x_series_name: str,
    y_series_name: str,
    log_date: pd.Timestamp,
    cov_type: str = "HC3"
    ):
    """
    Asses the significance and magnitude of logging change on the hydro variable
    """

    df = comparison_df.copy()
    df["pre_logging"] = df["day"] <= log_date
    df["group"] = (~df["pre_logging"]).astype(int)  # 0 = pre, 1 = post

    y = df[y_series_name].astype(float).to_numpy()
    x = df[x_series_name].astype(float).to_numpy()
    G = df["group"].to_numpy()

    X = np.column_stack([
        np.ones_like(x),  # For the dummy and interaction variable    
        x,                   
        G,                   
        x * G  # interaction
    ])
    model = sm.OLS(y, X).fit(cov_type=cov_type) 

    b0, b1, bg, bint = model.params
    V = model.cov_params()
    post_intercept = b0 + bg
    post_slope = b1 + bint
    se_post_intercept = np.sqrt(V[0,0] + V[2,2] + 2*V[0,2])
    se_post_slope = np.sqrt(V[1,1] + V[3,3] + 2*V[1,3])

    p_intercept_diff = model.pvalues[2]  # H0: bg = 0  (no intercept shift)
    p_slope_diff = model.pvalues[3]

    R = np.array([
    [0, 0, 1, 0],  # bg = 0
    [0, 0, 0, 1]   # bint = 0
    ])
    joint = model.f_test(R)
    joint_df = (int(joint.df_num), int(joint.df_denom))

    results = {
        "pre": {
            "intercept": float(b0),
            "slope": float(b1),
            "se_intercept": float(np.sqrt(V[0,0])),
            "se_slope": float(np.sqrt(V[1,1]))
        },
        "post": {
            "intercept": float(post_intercept),
            "slope": float(post_slope),
            "se_intercept": float(se_post_intercept),
            "se_slope": float(se_post_slope)
        },
        "tests": {
            "p_intercept_diff": float(p_intercept_diff),
            "p_slope_diff": float(p_slope_diff),
            "joint_F": float(joint.fvalue),
            "joint_df": joint_df,
            "joint_p": float(joint.pvalue)
        },
        "model_fit": {
            "r2": float(model.rsquared),
            "n": int(len(df)),
            "cov_type": cov_type
        }
    }

    return results, model

def sample_reference_ts(df: pd.DataFrame, only_pre_log: bool, column_name: str="wetland_depth_ref", n: int=1000):
    """
    Draw n samples from the empirical distribution F of the reference series.
    Expects columns: 'wetland_depth_ref' and 'pre_logging' (bool).
    """
    
    if only_pre_log:
        sample_df = df[df['pre_logging']].copy()
    else:
        sample_df = df.copy()
    
    sample = np.random.choice(sample_df[column_name].dropna().values, size=n, replace=True)
    
    return sample

def generate_model_distributions(f_dist: np.ndarray, models: dict):

    """
    Use the interaction-model summaries to create predicted stage draws
    for pre and post regimes over the same covariate (reference stage)
    """

    pre_intercept = models['pre']['intercept']
    pre_slope = models['pre']['slope']
    post_intercept = models['post']['intercept']
    post_slope = models['post']['slope']
    

    y_pre = pre_intercept + pre_slope * f_dist
    y_post = post_intercept + post_slope * f_dist

    return {'pre': y_pre, 'post': y_post}

def plot_hypothetical_distributions(model_distributions: dict, f_dist: np.ndarray, bins:int = 50):

    pre_model_distributions = model_distributions['pre']
    post_model_distributions = model_distributions['post']

    # Determine global x-axis limits across all distributions
    all_data = np.concatenate([pre_model_distributions, post_model_distributions])
    if f_dist is not None:
        all_data = np.concatenate([all_data, f_dist])
    x_min, x_max = all_data.min(), all_data.max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6))
    
    # Top panel: Pre and Post distributions
    ax1.hist(pre_model_distributions, bins=bins, alpha=0.6, label='Pre Logging Regime', color='black', range=(x_min, x_max))
    ax1.hist(post_model_distributions, bins=bins, alpha=0.6, label='Post Logging Regime', color='red', range=(x_min, x_max))
    ax1.axvline(np.mean(pre_model_distributions), color='black', linestyle='--', linewidth=2, label='Pre Mean')
    ax1.axvline(np.mean(post_model_distributions), color='red', linestyle='--', linewidth=2, label='Post Mean')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylabel('Density')
    ax1.tick_params(labelbottom=False)
    
    # Bottom panel: Reference distribution
    if f_dist is not None:
        ax2.hist(f_dist, bins=bins, alpha=0.6, color='blue', label=' Actual Reference (F)', range=(x_min, x_max))
        ax2.axvline(np.mean(f_dist), color='blue', linestyle='--', linewidth=2, label='Reference Mean')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylabel('Density')
    
    ax2.set_xlabel('Stage Regimes')
    
    # Collect all legend handles and labels
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if f_dist is not None else ([], [])
    
    # Create single legend below subplots
    fig.legend(handles1 + handles2, labels1 + labels2, loc='lower center', 
               ncol=3, bbox_to_anchor=(0.5, -0.15))
    
    plt.tight_layout()
    plt.show()

def summarize_depth_shift(model_distributions: dict):

    pre_model_distributions = model_distributions['pre']
    post_model_distributions = model_distributions['post']
    pre_mean = pre_model_distributions.mean()
    post_mean = post_model_distributions.mean()
    delta_mean = post_mean - pre_mean

    # TODO: Should I do some bootstrapping for confidence intervals on these estimates??

    return {
        "mean_pre": pre_mean,
        "mean_post": post_mean,
        "delta_mean": delta_mean
    }

def summarize_inundation_shift(model_distributions: dict, z_thresh: float=0.0):

    pre_model_distributions = model_distributions['pre']
    post_model_distributions = model_distributions['post']
 
    pre_above = (pre_model_distributions > z_thresh).mean()
    post_above = (post_model_distributions > z_thresh).mean()
    delta_inundation = post_above - pre_above

    # TODO: Should I do some bootstrapping for confidence intervals on these estimates??

    return {
        'pre_inundation': pre_above,
        'post_inundation': post_above,
        'delta_inundation': delta_inundation,
        'threshold': z_thresh
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

def plot_dmc_residuals(
        comparison_df: pd.DataFrame,
        x_series_name: str,
        y_series_name: str,
        dmc_slope: float,
        log_date: pd.Timestamp,
        stage: bool
    ):
    
    if stage:
        units = 'm'
    else:
        units = 'm3'

    comparison_df = comparison_df.copy()
    comparison_df['predicted'] = comparison_df[x_series_name] * dmc_slope
    comparison_df['residual'] = comparison_df[y_series_name] - comparison_df['predicted']

    comparison_df = comparison_df.sort_values('day')

    date_range = pd.date_range(start=comparison_df['day'].min(),
                               end=comparison_df['day'].max(),
                               freq='D')
    
    filled_df = comparison_df.set_index('day').reindex(date_range)
    filled_df['rolling_residual_change'] = filled_df['residual'].diff(periods=3) / 3

    def _scale_depth(df: pd.DataFrame):

        df = df.copy()
        df['mean_wetland_depth'] = (df['wetland_depth_ref'] + df['wetland_depth_log']) / 2
        min_d = df['mean_wetland_depth'].min()
        max_d = df['mean_wetland_depth'].max()
        range_d = max_d - min_d

        df['scaled_depth'] = df['mean_wetland_depth'] / range_d

        return df

    filled_df = _scale_depth(filled_df)
    plot_df = filled_df.reset_index().rename(columns={'index': 'day'})

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    # Top panel is cummulative residuals
    ax1.plot(plot_df['day'], plot_df['residual'], linewidth=2.5, marker='o')
    ax1.axvline(log_date, color='red', linestyle='-', label='Logged Date')
    ax1.axhline(0, color='black', linestyle='--', linewidth=1)
    ax1.set_ylabel(f'Cummulative Residual ({units})')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelbottom=False)

    ax2.plot(plot_df['day'], plot_df['rolling_residual_change'], 'g-', linewidth=2.5, marker='o')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.axvline(log_date, color='red', linestyle='-', linewidth=2, label='Logging Date')
    ax2.set_ylabel(f'3-Day Change in Residuals ({units} / d)')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelbottom=False)
    
    # Bottom panel rolling change in residuals
    plot_df_valid = plot_df.dropna(subset=['rolling_residual_change', 'scaled_depth'])
    
    # Create scatter plot with color mapping
    scatter = ax3.scatter(
        plot_df_valid['day'], 
        plot_df_valid['rolling_residual_change'],
        c=plot_df_valid['scaled_depth'],
        cmap='RdYlBu',  
        s=50,  
        alpha=1,
        edgecolors='black',
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax3, orientation='horizontal', pad=0.15, aspect=40)
    cbar.set_label('Scaled Depth (0=dry, 1=max)', labelpad=10)
    
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.axvline(log_date, color='red', linestyle='-', linewidth=2, label='Logging Date')
    ax3.set_ylabel(f'3-Day Change in Residuals ({units} / d)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Format x-axis for dates
    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.show()

    return plot_df

def residual_change_vs_depth(residual_df: pd.DataFrame, log_date: pd.Timestamp):
    """ 
    Plots relationship between depth (scaled 0-1) and the rolling residual change from 
    the dmc curve. Colored by pre vs. post logging.
    """

    plot_df = residual_df.dropna(subset=['scaled_depth', 'rolling_residual_change']).copy()
    plot_df['pre_logging'] = plot_df['day'] < log_date
    pre_df = plot_df[plot_df['pre_logging']]
    post_df = plot_df[~plot_df['pre_logging']]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(pre_df['scaled_depth'], pre_df['rolling_residual_change'], 
                   alpha=0.6, s=50, color='blue', edgecolors='black', 
                   linewidth=0.5, label='Pre-logging')
    ax.scatter(post_df['scaled_depth'], post_df['rolling_residual_change'], 
                   alpha=0.6, s=50, color='red', edgecolors='black', 
                   linewidth=0.5, label='Post-logging')
    ax.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Scaled Depth (0=dry, 1=max)', fontsize=12)
    ax.set_ylabel('3-Day Change in Residuals (m / d)', fontsize=12)
    ax.set_title('DMC Residual Change vs Wetland Depth', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    plt.show()


def plot_storage_curves(logged_hypsometry: pd.DataFrame, reference_hypsometry: pd.DataFrame):

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(
        logged_hypsometry['wetland_depth'], 
        logged_hypsometry['volume_m3'].cumsum(), 
        label='Logged Basin', 
        color='tab:orange'
    )

    ax.plot(
        reference_hypsometry['wetland_depth'], 
        reference_hypsometry['volume_m3'].cumsum(), 
        label='Reference Basin', 
        color='tab:blue'
    )

    ax.set_xlabel('Well Depth (m)')
    ax.set_ylabel('Volume (m³)')
    ax.set_title('Volume vs Wetland Depth')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
