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

    # Split logged data into pre and post
    logged_pre = logged_df[logged_df['day'] < logged_date]
    logged_post = logged_df[logged_df['day'] >= logged_date]

    plt.figure(figsize=(10, 6))
    
    # Plot reference series
    plt.plot(reference_df['day'], reference_df[well_depth_col_ref], 
             label=f'Reference', marker='o', markersize=2.5, linestyle='None', color='blue')
    
    # Plot logged series - pre-logging in grey
    plt.plot(logged_pre['day'], logged_pre[well_depth_col_log], 
             label=f'Logged Pre', marker='o', markersize=2.5, linestyle='None', color='#333333')
    
    # Plot logged series - post-logging in gold
    plt.plot(logged_post['day'], logged_post[well_depth_col_log], 
             label=f'Logged Post', marker='o', markersize=2.5, linestyle='None', color='#E69F00')
    
    plt.axvline(logged_date, color='red', linestyle='--', label='Logged Date', linewidth=2.5)
    plt.xlabel('')
    plt.ylabel(f'{y_label}', fontsize=14)
    plt.title('Well Time Series', fontsize=16)
    plt.legend(fontsize=12)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='both', labelsize=12)
    
    plt.show()

def remove_flagged_buffer(ts_df, buffer_days=1):
    """
    Remove ±buffer_days from records where flag ==2
    """
    
    bottomed_flag_dates = ts_df[ts_df['flag'] == 2]['day']
    # Create set of dates to remove (±buffer_days around each bottomed-out date)
    dates_to_remove = set()
    for date in bottomed_flag_dates:
        for offset in range(-buffer_days, buffer_days + 1):  # -2, -1, 0, 1, 2 days
            dates_to_remove.add(date + pd.Timedelta(days=offset))
    
    filtered_df = ts_df[~ts_df['day'].isin(dates_to_remove)].copy()

    return filtered_df, dates_to_remove

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
        color= '#333333',
        label='Pre-logging',
        alpha=0.6,
        s=40
    )
    ax.scatter(
        post_df[x_series_name],
        post_df[y_series_name], 
        color='#E69F00',
        label='Post-logging',
        alpha=0.6,
        s=40
    )
    
    # Create smooth line range for plotting regression lines
    x_min, x_max = comparison_df[x_series_name].min(), comparison_df[x_series_name].max()
    x_smooth = np.linspace(x_min, x_max, 100)
    
    # Plot regression lines using model parameters
    ax.plot(x_smooth, pre_slope * x_smooth + pre_intercept, 
            '#333333', linewidth=2, linestyle='--',
            label=f'Pre: m={pre_slope:.2f}, b={pre_intercept:.2f}, R²={pre_r_sq:.2f}')
    ax.plot(x_smooth, post_slope * x_smooth + post_intercept, 
            '#E69F00', linewidth=2, linestyle='--',
            label=f'Post: m={post_slope:.2f}, b={post_intercept:.2f}, R²={post_r_sq:.2f}')
    
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
    # ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
    #         verticalalignment='top', bbox=props)
    
    # Formatting
    ax.set_xlabel("Reference Stage (m)", fontsize=14)
    ax.set_ylabel("Logged Stage (m)", fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc='lower right', fontsize=12, framealpha=1)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Model summary (n={model_results['model_fit']['n']}):")
    print(f"  Pre-logging:  y = {pre_slope:.3f}x + {pre_intercept:.3f}")
    print(f"  Post-logging: y = {post_slope:.3f}x + {post_intercept:.3f}")
    print(f"  R² = {model_results['model_fit']['joint_r2']:.3f}")
    
    if p_slope < 0.05:
        print(f"  *** Significant slope change: {slope_change:+.3f} (p={p_slope:.3f})")
    if p_intercept < 0.05:
        print(f"  *** Significant intercept change: {intercept_change:+.3f} (p={p_intercept:.3f})")

def _prep_arrays_for_regression(
        comparison_df: pd.DataFrame, 
        x_series_name: str,
        y_series_name: str,
        log_date: pd.Timestamp
    ):
    """
    Preps dataframe into arrays to meet the statsmodels api
    """

    df = comparison_df.copy()
    df['pre_logging'] = df['day'] <= log_date
    df['group'] = (~df['pre_logging']).astype(int)
    y = df[y_series_name].astype(float).to_numpy()
    x = df[x_series_name].astype(float).to_numpy()
    G = df['group'].to_numpy()
    X = np.column_stack([
        np.ones_like(x),  # For the dummy and interaction variable    
        x,                   
        G,                   
        x * G  # interaction
    ])

    return df, X, y

def _compute_pre_post_r2(model, X, y, G):
    """
    Used to compute pre-logging and post-logging r-squared values sepperately. Meant to asses
    the degree to which model fit degrades post-logging. 
    """

    y_hat = model.fittedvalues

    # Pre
    mask_pre = (G == 0)
    y_pre = y[mask_pre]
    y_hat_pre = y_hat[mask_pre]
    r2_pre = 1 - np.sum((y_pre - y_hat_pre)**2) / np.sum((y_pre - y_pre.mean())**2)

    # Post 
    mask_post = (G == 1)
    y_post = y[mask_post]
    y_hat_post = y_hat[mask_post]
    r2_post = 1 - np.sum((y_post - y_hat_post)**2) / np.sum((y_post - y_post.mean())**2)

    return float(r2_pre), float(r2_post)

def fit_interaction_model_ols(
    comparison_df: pd.DataFrame,
    x_series_name: str,
    y_series_name: str,
    log_date: pd.Timestamp,
    cov_type: str = "HC3"
    ):
    """
    Asses the significance and magnitude of logging change on the hydro variable
    """

    df, X, y = _prep_arrays_for_regression(comparison_df, x_series_name, y_series_name, log_date)

    model = sm.OLS(y, X).fit(cov_type=cov_type) 

    b0, m_pre, bg, m_g = model.params
    V = model.cov_params()
    post_intercept = b0 + bg
    post_slope = m_pre + m_g
    pre_r2, post_r2 = _compute_pre_post_r2(model, X, y, df['group'])

    se_pre_intercept = np.sqrt(V[0,0])
    se_pre_slope = np.sqrt(V[1,1])
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
            "slope": float(m_pre),
            "se_intercept": float(se_pre_intercept),
            "se_slope": float(se_pre_slope), 
            "pre_r2": float(pre_r2)
        },
        "post": {
            "intercept": float(post_intercept),
            "slope": float(post_slope),
            "se_intercept": float(se_post_intercept),
            "se_slope": float(se_post_slope), 
            "post_r2": float(post_r2)
        },
        "tests": {
            "p_intercept_diff": float(p_intercept_diff),
            "p_slope_diff": float(p_slope_diff),
            "joint_F": float(joint.fvalue),
        },
        "model_fit": {
            "joint_r2": float(model.rsquared),
            "n": int(len(df)),
            "type": "OLS",
            "cov_type": cov_type
        }
    }
    return results

def fit_interaction_model_huber(
        comparison_df: pd.DataFrame,
        x_series_name: str, 
        y_series_name: str,
        log_date: pd.Timestamp,
        ):
    
    df, X, y = _prep_arrays_for_regression(comparison_df, x_series_name, y_series_name, log_date)
    #NOTE: 1) Used default value for Huber's t function 2) should I add the option to change the covariance type?
    model = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.345)).fit()

    b0, m_pre, bg, m_g = model.params
    V = model.cov_params()
    post_intercept = b0 + bg
    post_slope = m_pre + m_g

    pre_r2, post_r2 = _compute_pre_post_r2(model, X, y, df['group'])

    se_pre_intercept = float(np.sqrt(V[0,0]))
    se_pre_slope = float(np.sqrt(V[1,1]))
    se_post_intercept = np.sqrt(V[0,0] + V[2,2] + 2*V[0,2])
    se_post_slope = np.sqrt(V[1,1] + V[3,3] + 2*V[1,3])

    # NOTE: this is r-squared is determined by sum of squared errors, and does not relate to the Huber model's optimization
    r2 = 1 - np.sum((y - model.fittedvalues) ** 2) / np.sum((y - y.mean())**2)

    results = {
        'pre': {
            "intercept": float(b0),
            "slope": float(m_pre),
            "se_intercept": float(se_pre_intercept),
            "se_slope": float(se_pre_slope),
            "pre_r2": float(pre_r2)
        },
        'post': {
            "intercept": float(post_intercept),
            "slope": float(post_slope),
            "se_intercept": float(se_post_intercept),
            "se_slope": float(se_post_slope),
            "post_r2": float(post_r2)
        },
        'tests': {
            'p_intercept_diff': float(np.nan),
            'p_slope_diff': float(np.nan), 
            'joint_F': float(np.nan)
        },
        'model_fit': {
            'type': 'HuberRLM',
            'joint_r2': float(r2),
            'n': int(len(df))
        }
    }
    return results

def compute_residuals(
        comparison_df: pd.DataFrame,
        log_date: pd.Timestamp,
        x_series_name: str,
        y_series_name: str,
        models: dict
    ):
    """
    Calculates the residuals for a given pair's pre and post logging models. 
    Returns a dataframe with the residuals pre and post logging.
    """

    comparison_df = comparison_df.copy()
    comparison_df['logged'] = comparison_df['day'] >= log_date

    # Extract the coefficients
    pre_model = models['pre']
    pre_slope = pre_model['slope']
    pre_int = pre_model['intercept']
    post_model = models['post']
    post_slope = post_model['slope']
    post_int = post_model['intercept']

    # Apply models to get predictions adn residuals. 
    pre_predicted = comparison_df[x_series_name] * pre_slope + pre_int
    post_predicted = comparison_df[x_series_name] * post_slope + post_int

    comparison_df['predicted'] = np.where(
        comparison_df['logged'],
        post_predicted,
        pre_predicted
    )

    comparison_df['residual'] = comparison_df[y_series_name] - comparison_df['predicted']

    return comparison_df[['day', x_series_name, 'predicted', 'residual']]

# def visualize_residuals(residuals_df: pd.DataFrame, log_date: pd.Timestamp):
#     """
#     Visualize model residuals with residuals vs predicted plot and Q-Q plot.
#     Separates pre and post logging periods for comparison.
#     """
    
#     # Add logging period indicator
#     residuals_df = residuals_df.copy()
#     residuals_df['logged'] = residuals_df['day'] >= log_date
    
#     # Split into pre and post
#     pre_df = residuals_df[~residuals_df['logged']]
#     post_df = residuals_df[residuals_df['logged']]
    
#     # Create 2x2 subplot figure
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))

#     # Pre;ogging (top left)
#     axes[0, 0].scatter(pre_df['predicted'], pre_df['residual'], 
#                        alpha=0.6, s=40, color='grey', edgecolors='black', linewidth=0.5)
#     axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
#     axes[0, 0].set_xlabel('Predicted Values [m]', fontsize=11)
#     axes[0, 0].set_ylabel('Residuals [m]', fontsize=11)
#     axes[0, 0].set_title(f'Pre-Logging: Residuals vs Predicted (n={len(pre_df)})', 
#                          fontsize=12, fontweight='bold')
#     axes[0, 0].grid(True, alpha=0.3)
    
#     # Post-logging (top right)
#     axes[0, 1].scatter(post_df['predicted'], post_df['residual'],
#                        alpha=0.6, s=40, color='coral', edgecolors='black', linewidth=0.5)
#     axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
#     axes[0, 1].set_xlabel('Predicted Values [m]', fontsize=11)
#     axes[0, 1].set_ylabel('Residuals [m]', fontsize=11)
#     axes[0, 1].set_title(f'Post-Logging: Residuals vs Predicted (n={len(post_df)})',
#                          fontsize=12, fontweight='bold')
#     axes[0, 1].grid(True, alpha=0.3)

#     # Pre-logging Q-Q (bottom left)
#     stats.probplot(pre_df['residual'], dist="norm", plot=axes[1, 0])
#     axes[1, 0].set_title('Pre-Logging: Normal Q-Q Plot', fontsize=12, fontweight='bold')
#     axes[1, 0].set_xlabel('Theoretical Quantiles', fontsize=11)
#     axes[1, 0].set_ylabel('Sample Quantiles', fontsize=11)
#     axes[1, 0].grid(True, alpha=0.3)
#     axes[1, 0].get_lines()[0].set_markerfacecolor('steelblue')
#     axes[1, 0].get_lines()[0].set_markeredgecolor('black')
#     axes[1, 0].get_lines()[0].set_markersize(5)
#     axes[1, 0].get_lines()[1].set_color('red')
#     axes[1, 0].get_lines()[1].set_linewidth(2)
    
#     # Post-logging Q-Q (bottom right)
#     stats.probplot(post_df['residual'], dist="norm", plot=axes[1, 1])
#     axes[1, 1].set_title('Post-Logging: Normal Q-Q Plot', fontsize=12, fontweight='bold')
#     axes[1, 1].set_xlabel('Theoretical Quantiles', fontsize=11)
#     axes[1, 1].set_ylabel('Sample Quantiles', fontsize=11)
#     axes[1, 1].grid(True, alpha=0.3)
#     axes[1, 1].get_lines()[0].set_markerfacecolor('coral')
#     axes[1, 1].get_lines()[0].set_markeredgecolor('black')
#     axes[1, 1].get_lines()[0].set_markersize(5)
#     axes[1, 1].get_lines()[1].set_color('red')
#     axes[1, 1].get_lines()[1].set_linewidth(2)
    
#     plt.suptitle('Residual Diagnostics: Pre vs Post-Logging', 
#                  fontsize=15, fontweight='bold', y=0.995)
#     plt.tight_layout()
#     plt.show()

def flatten_model_results(
        results: dict,
        log_id: str,
        log_date: pd.Timestamp,
        ref_id: str,
        data_set: str
    ):
    """
    Unnests the dictionaries from fit_interaction_model_huber() and fit_interaction_model_ols() 
    the purpose is to make dataframe/.csv storage more efficient without the nested data in rows. 
    """

    out = {}

    out["log_id"] = log_id
    out["log_date"] = log_date
    out["ref_id"] = ref_id
    out["data_set"] = data_set

    # Pre
    pre = results["pre"]
    out["pre_intercept"] = pre["intercept"]
    out["pre_slope"] = pre["slope"]
    out["pre_se_intercept"] = pre["se_intercept"]
    out["pre_se_slope"] = pre["se_slope"]
    out["pre_r2"] = pre["pre_r2"]

    # Post
    post = results["post"]
    out["post_intercept"] = post["intercept"]
    out["post_slope"] = post["slope"]
    out["post_se_intercept"] = post["se_intercept"]
    out["post_se_slope"] = post["se_slope"]
    out["post_r2"] = post["post_r2"]

    # Tests NOTE: some tests are not valid in Huber models
    tests = results.get("tests", {})
    out["p_intercept_diff"] = tests.get("p_intercept_diff", np.nan)
    out["p_slope_diff"] = tests.get("p_slope_diff", np.nan)
    out["joint_F"] = tests.get("joint_F", np.nan)

    mf = results.get("model_fit", {})
    out["r2_joint"] = mf.get("joint_r2", np.nan)
    out["n"] = mf.get("n", np.nan)
    out["model_type"] = mf.get("type", None)
    out["cov_type"] = mf.get("cov_type", None)

    return out

def sample_reference_ts(df: pd.DataFrame, column_name: str="wetland_depth_ref", n: int=1000):
    """
    Draw n samples from the empirical distribution F of the reference series.
    Expects columns: 'wetland_depth_ref' and 'pre_logging' (bool).
    """
    
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
    ax1.hist(pre_model_distributions, bins=bins, alpha=0.8, label='Modeled Pre-Logging Distribution', color='#333333', range=(x_min, x_max), density=True)
    ax1.hist(post_model_distributions, bins=bins, alpha=0.8, label='Modeled Post-Logging Distribution', color='#E69F00', range=(x_min, x_max), density=True)
    ax1.axvline(np.mean(pre_model_distributions), color='#333333', linestyle='--', linewidth=2, label='Pre Mean')
    ax1.axvline(np.mean(post_model_distributions), color='#E69F00', linestyle='--', linewidth=2, label='Post Mean')
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylabel('% of Days', fontsize=16)
    ax1.tick_params(labelsize=12)
    
    # Bottom panel: Reference distribution
    if f_dist is not None:
        ax2.hist(f_dist, bins=bins, alpha=0.8, color='blue', label='Reference Distribution', range=(x_min, x_max), density=True)
        #ax2.axvline(np.mean(f_dist), color='blue', linestyle='--', linewidth=2, label='Reference Mean')
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylabel('% of Days', fontsize=16)
    
    ax2.set_xlabel('Stage (m)', fontsize=16)
    ax1.set_title('Modeled Stage in Logged Wetland', fontsize=14)
    ax2.set_title('Observed Stage in Reference Wetland', fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Collect all legend handles and labels
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels() if f_dist is not None else ([], [])
    
    # Create single legend below subplots
    fig.legend(handles1 + handles2, labels1 + labels2, loc='lower center', 
               ncol=1, bbox_to_anchor=(0.5, -0.15), fontsize=12)
    
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

# def plot_dmc(
#         comparison_df: pd.DataFrame, 
#         x_series_name: str, 
#         y_series_name: str, 
#         log_date: pd.Timestamp, 
#     ):

#     pre_logged = comparison_df[comparison_df['day'] <= log_date]
#     if log_date in comparison_df['day'].values:
#         log_x_value = comparison_df.loc[comparison_df['day'] == log_date, x_series_name].values[0]
#     else:
#         # Find the nearest date and get its x value
#         nearest_idx = (comparison_df['day'] - log_date).abs().argsort()[0]
#         log_x_value = comparison_df.iloc[nearest_idx][x_series_name]
#         nearest_date = comparison_df.iloc[nearest_idx]['day']
#         print(f"Exact log date not found. Using nearest date: {nearest_date}")
 
#     x_pre = pre_logged[x_series_name].to_numpy()
#     y_pre = pre_logged[y_series_name].to_numpy()
#     x_full = comparison_df[x_series_name].to_numpy()

#     result = np.linalg.lstsq(x_pre[:, None], y_pre, rcond=None)
#     m = result[0][0]

#     plt.figure(figsize=(8, 8))
#     plt.scatter(comparison_df[x_series_name], comparison_df[y_series_name], label=f"DMC")
#     plt.plot(x_full, m * x_full, color='black', linestyle='--', label=f'Pre-logging fit')
#     plt.axvline(log_x_value, color='red', linestyle='-', label='logging date')
#     plt.xlabel('Cummulative Reference')
#     plt.ylabel('Cummulative Logged')
#     ax = plt.gca()
#     ax.text(0.02, 0.98, f"m = {m:.3f}", transform=ax.transAxes, ha='left', va='top')
#     plt.show()

#     return m

# def plot_dmc_residuals(
#         comparison_df: pd.DataFrame,
#         x_series_name: str,
#         y_series_name: str,
#         dmc_slope: float,
#         log_date: pd.Timestamp,
#         stage: bool
#     ):
    
#     if stage:
#         units = 'm'
#     else:
#         units = 'm3'

#     comparison_df = comparison_df.copy()
#     comparison_df['predicted'] = comparison_df[x_series_name] * dmc_slope
#     comparison_df['residual'] = comparison_df[y_series_name] - comparison_df['predicted']

#     comparison_df = comparison_df.sort_values('day')

#     date_range = pd.date_range(start=comparison_df['day'].min(),
#                                end=comparison_df['day'].max(),
#                                freq='D')
    
#     filled_df = comparison_df.set_index('day').reindex(date_range)
#     filled_df['rolling_residual_change'] = filled_df['residual'].diff(periods=3) / 3

#     def _scale_depth(df: pd.DataFrame):

#         df = df.copy()
#         df['mean_wetland_depth'] = (df['wetland_depth_ref'] + df['wetland_depth_log']) / 2
#         min_d = df['mean_wetland_depth'].min()
#         max_d = df['mean_wetland_depth'].max()
#         range_d = max_d - min_d

#         df['scaled_depth'] = df['mean_wetland_depth'] / range_d

#         return df

#     filled_df = _scale_depth(filled_df)
#     plot_df = filled_df.reset_index().rename(columns={'index': 'day'})

#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
#     # Top panel is cummulative residuals
#     ax1.plot(plot_df['day'], plot_df['residual'], linewidth=2.5, marker='o')
#     ax1.axvline(log_date, color='red', linestyle='-', label='Logged Date')
#     ax1.axhline(0, color='black', linestyle='--', linewidth=1)
#     ax1.set_ylabel(f'Cummulative Residual ({units})')
#     ax1.grid(True, alpha=0.3)
#     ax1.tick_params(labelbottom=False)

#     ax2.plot(plot_df['day'], plot_df['rolling_residual_change'], 'g-', linewidth=2.5, marker='o')
#     ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
#     ax2.axvline(log_date, color='red', linestyle='-', linewidth=2, label='Logging Date')
#     ax2.set_ylabel(f'3-Day Change in Residuals ({units} / d)')
#     ax2.grid(True, alpha=0.3)
#     ax2.tick_params(labelbottom=False)
    
#     # Bottom panel rolling change in residuals
#     plot_df_valid = plot_df.dropna(subset=['rolling_residual_change', 'scaled_depth'])
    
#     # Create scatter plot with color mapping
#     scatter = ax3.scatter(
#         plot_df_valid['day'], 
#         plot_df_valid['rolling_residual_change'],
#         c=plot_df_valid['scaled_depth'],
#         cmap='RdYlBu',  
#         s=50,  
#         alpha=1,
#         edgecolors='black',
#     )
    
#     # Add colorbar
#     cbar = plt.colorbar(scatter, ax=ax3, orientation='horizontal', pad=0.15, aspect=40)
#     cbar.set_label('Scaled Depth (0=dry, 1=max)', labelpad=10)
    
#     ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
#     ax3.axvline(log_date, color='red', linestyle='-', linewidth=2, label='Logging Date')
#     ax3.set_ylabel(f'3-Day Change in Residuals ({units} / d)')
#     ax3.grid(True, alpha=0.3)
#     ax3.legend()
    
#     # Format x-axis for dates
#     ax3.xaxis.set_major_locator(mdates.YearLocator())
#     ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
#     plt.tight_layout()
#     plt.show()

#     return plot_df

# def residual_change_vs_depth(residual_df: pd.DataFrame, log_date: pd.Timestamp):
#     """ 
#     Plots relationship between depth (scaled 0-1) and the rolling residual change from 
#     the dmc curve. Colored by pre vs. post logging.
#     """

#     plot_df = residual_df.dropna(subset=['scaled_depth', 'rolling_residual_change']).copy()
#     plot_df['pre_logging'] = plot_df['day'] < log_date
#     pre_df = plot_df[plot_df['pre_logging']]
#     post_df = plot_df[~plot_df['pre_logging']]

#     fig, ax = plt.subplots(figsize=(10, 8))

#     ax.scatter(pre_df['scaled_depth'], pre_df['rolling_residual_change'], 
#                    alpha=0.6, s=50, color='blue', edgecolors='black', 
#                    linewidth=0.5, label='Pre-logging')
#     ax.scatter(post_df['scaled_depth'], post_df['rolling_residual_change'], 
#                    alpha=0.6, s=50, color='red', edgecolors='black', 
#                    linewidth=0.5, label='Post-logging')
#     ax.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=1.5)
#     ax.set_xlabel('Scaled Depth (0=dry, 1=max)', fontsize=12)
#     ax.set_ylabel('3-Day Change in Residuals (m / d)', fontsize=12)
#     ax.set_title('DMC Residual Change vs Wetland Depth', fontsize=14, fontweight='bold')
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
#     plt.tight_layout()
#     plt.show()

# def plot_storage_curves(logged_hypsometry: pd.DataFrame, reference_hypsometry: pd.DataFrame):

#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))

#     ax.plot(
#         logged_hypsometry['wetland_depth'], 
#         logged_hypsometry['volume_m3'].cumsum(), 
#         label='Logged Basin', 
#         color='tab:orange'
#     )

#     ax.plot(
#         reference_hypsometry['wetland_depth'], 
#         reference_hypsometry['volume_m3'].cumsum(), 
#         label='Reference Basin', 
#         color='tab:blue'
#     )

#     ax.set_xlabel('Well Depth (m)')
#     ax.set_ylabel('Volume (m³)')
#     ax.set_title('Volume vs Wetland Depth')
#     ax.legend()
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()
