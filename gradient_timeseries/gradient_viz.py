import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

from scipy import stats

def gradient_ts_plot(
    gradient_df: pd.DataFrame,
    well_pairs_list: list,
    y_var: str,
    color_arg: str,
    y_lim: tuple,
):
    """
    Plot time series for a given y variable for each well pair.
    """
    
    plt.figure(figsize=(12, 8))
    
    # Use single color if more than 15 well pairs, otherwise use color cycle
    if color_arg == 'same_color':
        colors = ['steelblue'] * len(well_pairs_list)  # All same color
        show_legend = False  # Too many series for meaningful legend
    elif color_arg == 'by_elevation_gradient':
        # Get elevation gradients for each well pair
        elevation_gradients = gradient_df.groupby('well_pair')['elevation_gradient_cm_m'].first()
        
        # Create color map based on elevation gradients
        min_grad = elevation_gradients.min()
        max_grad = elevation_gradients.max()
        norm = plt.Normalize(vmin=min_grad, vmax=max_grad)
        cmap = plt.cm.viridis
        
        # Map colors to well pairs based on their elevation gradients
        colors = {}
        for well_pair in well_pairs_list:
            if well_pair in elevation_gradients.index:
                colors[well_pair] = cmap(norm(elevation_gradients[well_pair]))
            else:
                colors[well_pair] = 'gray'  # Default for missing data
        
        show_legend = False  # Use colorbar instead
    else:
        if len(well_pairs_list) <= 8:
            colors = plt.cm.Dark2(np.linspace(0, 1, len(well_pairs_list)))
            show_legend = True
        else:
            colors = plt.cm.tab20(np.linspace(0, 1, len(well_pairs_list)))
            show_legend = True
    
    for idx, p in enumerate(well_pairs_list):
        # Filter for specific well pair first
        data = gradient_df[gradient_df['well_pair'] == p].copy()
        
        data = data.dropna(subset=[y_var]).sort_values('Date')
        
        if len(data) == 0:
            continue
        
        y_vals = data[y_var]
        # Find gaps larger than 5 days
        date_diff = data['Date'].diff()
        gap_mask = date_diff > pd.Timedelta(days=5)
        
        # Split data into segments where gaps > 5 days
        split_indices = np.where(gap_mask)[0]
        start_idx = 0
        
        # Get color for this well pair
        if color_arg == 'by_elevation_gradient':
            line_color = colors[p]
        else:
            line_color = colors[idx]
        
        for i, split_idx in enumerate(np.append(split_indices, len(data))):
            segment_data = data.iloc[start_idx:split_idx]
            segment_y = y_vals.iloc[start_idx:split_idx]
            
            # Only plot if segment has data
            if len(segment_data) > 0:
                plt.plot(segment_data['Date'], segment_y, 
                        color=line_color,
                        label=p if (i == 0 and show_legend) else "", # Only label if showing legend
                        linewidth=1.5,
                        alpha=0.7 if len(well_pairs_list) > 15 else 1.0)  # Add transparency for many series
            
            start_idx = split_idx
    
    plt.axhline(y=0, color='red', linestyle=':', linewidth=4)
    plt.title(f'{y_var} time series by well pair')
    plt.ylabel(y_var, fontsize=14)
    plt.xlabel('Date', fontsize=14)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    if show_legend:
        plt.legend()
    elif color_arg == 'by_elevation_gradient':
        # Add colorbar for elevation gradient coloring
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        # Get current axes and pass it to colorbar
        ax = plt.gca()
        cbar = plt.colorbar(sm, ax=ax)  # Add ax parameter here
        cbar.set_label('Elevation Gradient (cm/m)', fontsize=12)
        
    if y_lim:
        plt.ylim(y_lim)
    
    plt.tight_layout()
    plt.show()


def summary_correlations_plot(
    summary_df: pd.DataFrame,
    pair_type_filter: list,
):
    
    clean = summary_df.dropna(subset=['elevation_gradient', 'mean_head_gradient'])
    if pair_type_filter:
        clean = clean[clean['pair_type'].isin(pair_type_filter)]
    # Calculate Pearson correlation (linear relationship)
    pearson_corr, p_value = stats.pearsonr(
    clean['elevation_gradient'], 
    clean['mean_head_gradient']
    )

    # Calculate Spearman rank correlation (monotonic relationship)
    spearman_corr, sp_p_value = stats.spearmanr(
    clean['elevation_gradient'],
    clean['mean_head_gradient']
    )

    # Create a scatter plot with points colored by pair_type
    plt.figure(figsize=(12, 8))
    g = sns.scatterplot(x='elevation_gradient', y='mean_head_gradient', 
            hue='pair_type', data=clean, alpha=0.6)
    
    # Add error bars for standard deviation
    if 'std_head_gradient' in clean.columns:
        plt.errorbar(x=clean['elevation_gradient'], y=clean['mean_head_gradient'],
                yerr=clean['std_head_gradient'], fmt='none', ecolor='black', 
                elinewidth=1, capsize=3, alpha=0.5)

    # Add regression line for all points
    sns.regplot(x='elevation_gradient', y='mean_head_gradient', data=clean,
        scatter=False, color='red', line_kws={'linestyle': '--'})

    # Add labels for each point
    for idx, row in clean.iterrows():
        plt.annotate(row['well_pair'], 
                (row['elevation_gradient'], row['mean_head_gradient']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=8, alpha=0.7)

    plt.title(f'Elevation Gradient vs. Head Gradient\nPearson r: {pearson_corr:.3f} (p={p_value:.4f})\n' +
        f'Spearman ρ: {spearman_corr:.3f} (p={sp_p_value:.4f})', 
        fontsize=14)
    plt.xlabel('Elevation Gradient (cm/m)', fontsize=12)
    plt.ylabel('Mean Head Gradient (cm/m)', fontsize=12)
    plt.grid(alpha=0.3)

    # Add a horizontal line at y=0
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()