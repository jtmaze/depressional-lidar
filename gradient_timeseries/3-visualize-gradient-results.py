# %%
import os
import sys
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gradient_timeseries import gradient_ts_viz

os.chdir('/Users/jmaze/Documents/projects/depressional_lidar/')
catchment = 'jl'
gradients_path = f'./delmarva/out_data/{catchment}_gradient_timeseries.csv'

# %% 
gradient_ts = pd.read_csv(gradients_path)
gradient_ts['Date'] = pd.to_datetime(gradient_ts['Date'])

# %%

well_pairs_to_plot = ['TS-UW1__to__TS-CH', 'TS-UW1__to__BD-CH', 
                      'DK-UW1__to__DK-UW2', 'DK-CH__to__DK-UW2']

gradient_ts_viz.gradient_ts_plot(
    gradient_ts, 
    well_pairs_to_plot, 
    y_var='head_gradient_cm_m', 
    abs_vals=True, 
    y_lim=(0.3, 6))

# %%

"""
Scratch function -- actually pretty useful for plotting
"""

def plot_well_timeseries(well_name, data):
    test_plot = data[data['Site_Name'] == well_name]
    
    # Define a custom color map for flags
    flag_colors = {0: 'blue', 1: 'pink', 2: 'green', 3: 'orange', 4: 'red'}
    colors = [flag_colors.get(flag, 'gray') for flag in test_plot['Flag']]
    
    plt.figure(figsize=(12,6))
    plt.scatter(test_plot['Date'], 
                test_plot['waterLevel'],
                c=colors,
                alpha=0.7)
    

    plt.title(f'Water Level Time Series for {well_name}')
    plt.ylabel('Water Level')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    
    # Create a custom legend for the flag colors
    legend_elements = [Line2D([0], [0], marker='o', color='w', 
                      label=f'Flag {flag}', markerfacecolor=color, markersize=10)
                      for flag, color in flag_colors.items()]
    plt.legend(handles=legend_elements, loc='best')
    
    plt.tight_layout()
    plt.show()



# %%
