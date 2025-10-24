# %%

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

domain = (-3, 3)

def generate_logistic_func(domain: tuple, inflection_pt: float, k: float, plot: bool = False):

    start = domain[0]
    end = domain[1]
    x = np.linspace(start, end, 10_000)
    
    y = 1 / (1 + np.exp(-k * (x - inflection_pt)))
    y = y * 100

    if plot:
        plt.plot(x, y)
        plt.xlabel('Depth (m)')
        plt.ylabel('Area (%)')
        plt.title(f'Simulated Hypsometry (k={k}, inflection={inflection_pt})')
        plt.grid(True)
        plt.show()
    
    return x, y

def hypsometry_cdf_to_dAdh(cdf: pd.DataFrame, plot: bool = False):
    """
    Convert hypsometry CDF to dA/dh (rate of area change with depth).
    Parameters:
    - cdf: DataFrame with 'depth' and 'area' columns
    Returns:
    - DataFrame with added 'dAdh' column
    """
    
    dAdh_list = []
    
    for idx, row in cdf.iterrows():
        if idx == 0:  # Handle first row
            dAdh_list.append(np.nan)  # or 0, depending on your needs
        else:
            # Get current and previous values
            current_depth = row['depth']
            current_area = row['area']
            prev_depth = cdf.iloc[idx-1]['depth']
            prev_area = cdf.iloc[idx-1]['area']
            
            # Calculate differences
            dh = current_depth - prev_depth
            dA = current_area - prev_area
            
            # Calculate dA/dh (avoid division by zero)
            if dh != 0:
                dAdh = dA / dh  # Fixed: was dh/dA
            else:
                dAdh = np.nan  # Handle zero denominator
                
            dAdh_list.append(dAdh)
    
    cdf_copy = cdf.copy()  # Don't modify original DataFrame
    cdf_copy['dAdh'] = dAdh_list

    if plot:
        plt.plot(cdf_copy['depth'], cdf_copy['dAdh'], color='orange')
        plt.xlabel('Depth (m)')
        plt.ylabel('dAdh (% Area Change / meter)')
        plt.grid(True)
        plt.show()
    
    return cdf_copy

def generate_wtr_depth_normal(domain: tuple, mean: float, std: float, plot: bool = False):

    start, end = domain[0], domain[1]
    x = np.linspace(start, end, 1000)
    
    # Normal PDF, scaled to 0-100%
    y = np.exp(-0.5 * ((x - mean) / std) ** 2)
    y = (y / np.max(y)) 

    if plot:
        plt.plot(x, y)
        plt.xlabel('Depth (m)')
        plt.ylabel('Proportional Stage Occurence (Not Normalized)')
        plt.title(f'Proportional Stage Occurrence (mean={mean}, std={std})')
        plt.grid(True)
        plt.show()
    
    return x, y

def generate_wtr_depth_bimodal(
        domain: tuple, 
        peak1: float, 
        peak2: float, 
        peak_ratio_1_2: float, # NOTE: positive means peak 1 larger than peak 2
        std: float, 
        plot: bool = False
):
    
    """Generate bimodal distribution with two peaks."""
    # TODO: Give the peaks different variances?
    start, end = domain[0], domain[1]
    x = np.linspace(start, end, 10_000)
    
    # Two normal distributions combined
    y1 = np.exp(-0.5 * ((x - peak1) / std) ** 2)  * peak_ratio_1_2
    y2 = np.exp(-0.5 * ((x - peak2) / std) ** 2)
    y = y1 + y2
    y = (y / np.max(y)) 

    if plot:
        plt.plot(x, y)
        plt.xlabel('Depth (m)')
        plt.ylabel('Proportional Stage Occurence (Not Normalized)')
        plt.title(f'Stage Distribution (Bimiodal) (peaks at {peak1}, {peak2}, proportion={peak_ratio_1_2}, std={std})')
        plt.grid(True)
        plt.show()
    
    return x, y

def tai_curve_from_dataframes(
    hyps_df: pd.DataFrame,
    stage_df: pd.DataFrame,
    delta: float = 0.05,
    plot: bool = True
):
    """
    Calculate TAI probability density function.
    
    Parameters:
    - hyps_df: DataFrame with 'depth' and 'area' columns (hypsometric curve)
    - stage_df: DataFrame with 'depth' and 'weight' columns (stage distribution)
    - delta: TAI depth range (0 to delta meters water depth)
    - plot: Whether to plot the results
    
    Returns:
    - DataFrame with 'tai' and 'probability' columns
    """
    
    # Create interpolation function for the hypsometric curve
    hyps_df = hyps_df.sort_values('depth').reset_index(drop=True)

    # TODO: Normalize stage df

    def get_area_at_depth(depth):
        """
        Get interpolated area value at a given depth.
        Uses linear interpolation between the nearest points above and below.
        """
        # Handle case where depth is outside the range of the dataframe
        if depth <= hyps_df['depth'].min():
            return hyps_df.loc[hyps_df['depth'].idxmin(), 'area']
        elif depth >= hyps_df['depth'].max():
            return hyps_df.loc[hyps_df['depth'].idxmax(), 'area']
        
        # Find points below and above
        below_mask = hyps_df['depth'] <= depth
        above_mask = hyps_df['depth'] >= depth
        
        if not below_mask.any() or not above_mask.any():
            # Fallback to nearest if we can't find points on both sides
            return hyps_df.loc[(hyps_df['depth'] - depth).abs().idxmin(), 'area']
        
        # Get the closest point below
        below_idx = hyps_df[below_mask]['depth'].idxmax()
        below_depth = hyps_df.loc[below_idx, 'depth']
        below_area = hyps_df.loc[below_idx, 'area']
        
        # Get the closest point above
        above_idx = hyps_df[above_mask]['depth'].idxmin()
        above_depth = hyps_df.loc[above_idx, 'depth']
        above_area = hyps_df.loc[above_idx, 'area']
        
        # If we're exactly on a point, just return its value
        if below_depth == depth:
            return below_area
        if above_depth == depth:
            return above_area
        
        # Linear interpolation
        weight = (depth - below_depth) / (above_depth - below_depth)
        interpolated_area = below_area + weight * (above_area - below_area)
        
        return interpolated_area

    # Calculate TAI for each water depth
    tai_values = []

    for _, row in stage_df.iterrows():
        water_depth = row['depth']
        weight = row['weight']
        # Calculate area deeper than delta meters (not in TAI zone)
        below_depth = water_depth - delta
        below_area = get_area_at_depth(below_depth)
        above_depth = water_depth + delta
        above_area = get_area_at_depth(above_depth)
        
        # TAI is the area between 0-delta meters water depth
        tai_area = above_area - below_area
        
        tai_values.append({
            'water_depth': water_depth,
            'tai': max(0, tai_area),
            'weight': weight
        })
    
    # Convert to DataFrame
    result = pd.DataFrame(tai_values)
    result['probability'] = result['weight'] / result['weight'].sum()
    
    bins = np.linspace(0, result['tai'].max() * 1.05, 100)  # Add 5% margin
    result['tai_bin'] = pd.cut(result['tai'], bins)
    grouped = result.groupby('tai_bin')['probability'].sum().reset_index()
    grouped['tai'] = grouped['tai_bin'].apply(lambda x: x.mid)

    # Apply moving average smoothing
    window_size = 5
    grouped['smooth_probability'] = grouped['probability'].rolling(
        window=window_size, center=True, min_periods=1
    ).mean()

    grouped['smooth_probability'] = grouped['smooth_probability'] * 100

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['tai'], grouped['smooth_probability'], 'r-')
        plt.xlabel('TAI (% of Total Area)')
        plt.ylabel('% of Days')
        plt.title(f'Simulated TAI')
        plt.grid(True)
        plt.show()
    
    return grouped[['smooth_probability', 'tai']]

# %%

x, y = generate_logistic_func(
    domain=domain, 
    inflection_pt=0.9, 
    k=3, 
    plot=True
)

cdf = pd.DataFrame({'depth': x, 'area': y})

cdf_with_dAdh = hypsometry_cdf_to_dAdh(cdf)

x, y = generate_wtr_depth_bimodal(
    domain=domain,
    peak1=-0.75,
    peak2=0.3,
    peak_ratio_1_2=0.45,
    std=0.25,
    plot=True
)
stage_df = pd.DataFrame({'depth': x, 'weight': y})

tai_results = tai_curve_from_dataframes(
    hyps_df=cdf,          
    stage_df=stage_df,    
    delta=0.05,           
    plot=True
)

# %%

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

# Panel 1: Basin Elevation CDF (Spatial) - Hypsometry
# Normalize the CDF to 0-1 range
cdf_normalized = cdf['area'] / 100  # Convert percentage to fraction
ax1.plot(cdf['depth'], cdf_normalized, 'g-', linewidth=2, label='Basin Elevation CDF (Spatial)')
ax1.set_ylabel('CDF')
ax1.set_ylim(0, 1)
ax1.grid(True, alpha=0.3)
ax1.legend()

# Panel 2: Basin Elevation PDF (Spatial) - dA/dh from hypsometry
# Calculate the derivative (dA/dh) and normalize
dAdh_normalized = cdf_with_dAdh['dAdh'].fillna(0)  # Fill NaN with 0
dAdh_normalized = dAdh_normalized / np.nanmax(dAdh_normalized)  # Normalize to max = 1
ax2.fill_between(cdf_with_dAdh['depth'], dAdh_normalized, alpha=0.6, 
                color='orange', label='Basin Elevation PDF (Spatial)')
ax2.plot(cdf_with_dAdh['depth'], dAdh_normalized, 'orange', linewidth=1)
ax2.set_ylabel('PDF (Not Normalized)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Panel 3: Water Depth Distribution (Time)
# Normalize the stage distribution
stage_normalized = stage_df['weight'] / np.max(stage_df['weight'])
ax3.fill_between(stage_df['depth'], stage_normalized, alpha=0.6, 
                color='lightblue', label='Water Depth (Time)')
ax3.plot(stage_df['depth'], stage_normalized, 'blue', linewidth=1)
ax3.set_ylabel('PDF (Not Normalized)')
ax3.set_xlabel('water_depth')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Set consistent x-axis limits for all panels
xlim = (-1.5, 2.5)
ax1.set_xlim(xlim)
ax2.set_xlim(xlim)
ax3.set_xlim(xlim)

plt.tight_layout()
plt.show()

# %% Quick simulation varrying topographic slope

k_vals = [2, 3.5, 5]

plt.figure(figsize=(8, 5))
for k in k_vals:
    x, y = generate_logistic_func(
        domain=domain,
        inflection_pt=1,
        k=k,
        plot=False
    )
    cdf = pd.DataFrame({'depth': x, 'area': y})

    tai_results = tai_curve_from_dataframes(
        hyps_df=cdf,          
        stage_df=stage_df,    
        delta=0.05,           
        plot=False
    )

    plt.plot(tai_results['tai'], tai_results['smooth_probability'], label=f'k={k}')

plt.xlabel('TAI (% of Total Area)')
plt.ylabel('% of Days')
plt.title('Simulated TAI for Different Hypsometry k')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %% Quick simulation varrying the proportion of wet vs dry time

ratios = [0.3, 0.5, 0.7]

x, y = generate_logistic_func(
    domain=domain,
    inflection_pt=0.8,
    k=2.7,
    plot=False
)

cdf = pd.DataFrame({'depth': x, 'area': y})

plt.figure(figsize=(8, 5))
for r in ratios:
    x, y = generate_wtr_depth_bimodal(
        domain=domain,
        peak1=-0.75,
        peak2=0.8,
        peak_ratio_1_2=r,
        std=0.5,
        plot=False
    )
    stage_df = pd.DataFrame({'depth': x, 'weight': y})

    tai_results = tai_curve_from_dataframes(
        hyps_df=cdf,          
        stage_df=stage_df,    
        delta=0.05,           
        plot=False
    )
    plt.plot(tai_results['tai'], tai_results['smooth_probability'], label=f'r={r}')

plt.xlabel('TAI (% of Total Area)')
plt.ylabel('% of Days')
plt.title('Simulated TAI by Wetness')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
"""
NOTE: Illustrative code is below. Not really for analysis.
"""
# %%

k_vals = [2, 3.5, 5]

for k in k_vals:
    x, y = generate_logistic_func(
        domain=domain,
        inflection_pt=1,
        k=k
    )
    plt.plot(x, y, label=f'k={k}')

plt.xlabel('Depth (m)')
plt.ylabel('Area (%)')
plt.title('Simulated Hypsometry for different k values')
plt.legend()
plt.grid(True)
plt.xlim(-1, 3)
plt.show()

inflection_pts = [0.75, 1, 1.25]

for i in inflection_pts:
    x, y = generate_logistic_func(
        domain=domain,
        inflection_pt=i,
        k=3.5
    )
    plt.plot(x, y, label=f'mid pt={i}')

plt.xlabel('Depth (m)')
plt.ylabel('Area (%)')
plt.title('Simulated Hypsometry for different mid pts')
plt.legend()
plt.grid(True)
plt.xlim(-1, 3)
plt.show()


# %%
ratios = [0.3, 0.5, 0.7]

for r in ratios:
    x, y = generate_wtr_depth_bimodal(
        domain=domain,
        peak1=-0.75,
        peak2=0.8,
        peak_ratio_1_2=r,
        std=0.5,
        plot=False
    )
    plt.plot(x, y, label=f'ratio={r}')

plt.xlabel('Stage')
plt.ylabel('Density')
plt.title('Simulated Stage')
plt.legend()
plt.grid(True)
plt.xlim(-3, 3)
plt.show()
# %%
stds = [0.4, 0.5, 0.6]

for s in stds:
    x, y = generate_wtr_depth_bimodal(
        domain=domain,
        peak1=-1,
        peak2=1,
        peak_ratio_1_2=0.5,
        std=s,
    )
    plt.plot(x, y, label=f'std={s}')

plt.xlabel('Stage')
plt.ylabel('Density')
plt.title('Simulated Stage')
plt.legend()
plt.grid(True)
plt.xlim(-3, 3)
plt.show()
# %%
