import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def gradient_ts_plot(
    gradient_df: pd.DataFrame,
    well_pairs_list: list,
    y_var: str,
    abs_vals: bool, 
    y_lim:tuple
):
    """
    Plot time series for a given y variable for each well pair.
    
    Parameters:
        gradient_df (pd.DataFrame): DataFrame containing the gradient timeseries.
        well_pairs_list (list): List of well pair names to plot.
        y_var (str): The name of the y variable column to plot.
        abs_vals (bool): Whether to apply abs() to the values.
    """

    plt.figure(figsize=(14, 7))
    
    # Get default color cycle to ensure consistent colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(well_pairs_list)))
    
    for idx, p in enumerate(well_pairs_list):
        data = gradient_df[gradient_df['well_pair'] == p].copy()

        # if y_lim is not None:
        #     data = data[(data[y_var] > y_lim[0]) & (data[y_var] < y_lim[1])]
            
        data = data.dropna(subset=[y_var]).sort_values('Date')
        
        if len(data) == 0:
            continue
            
        # Apply absolute values if needed
        y_vals = data[y_var].abs() if abs_vals else data[y_var]
        
        # Find gaps larger than 5 days
        date_diff = data['Date'].diff()
        gap_mask = date_diff > pd.Timedelta(days=5)
        
        # Split data into segments where gaps > 5 days
        split_indices = np.where(gap_mask)[0]
        start_idx = 0
        
        for i, split_idx in enumerate(np.append(split_indices, len(data))):
            segment_data = data.iloc[start_idx:split_idx]
            segment_y = y_vals.iloc[start_idx:split_idx]
            
            # Only plot if segment has data
            if len(segment_data) > 0:
                plt.plot(segment_data['Date'], segment_y, 
                        color=colors[idx],  # Use consistent color for this well pair
                        label=p if i == 0 else "", # Only label first segment
                        linewidth=1.5)
            
            start_idx = split_idx
        
    plt.title(f'{y_var} time series by well pair')
    plt.ylabel(y_var, fontsize=14)
    plt.xlabel('Date', fontsize=14)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    if y_lim:
        plt.ylim(y_lim)

    plt.legend()
    plt.tight_layout()
    plt.show()