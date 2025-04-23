# %% Libraries and file paths

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

stage_data = pd.read_csv('./stage_data/waterlevel_offsets_tracked.csv', low_memory=False)

# %% Trim the stage data for relevant sites

sites = ['6_93', '15_409', '6_629', '14_500', '5a_582', '13_267']
stage_data = stage_data[['Date', 'Site_ID', 'revised_depth']].copy()
stage_data['Date'] = pd.to_datetime(stage_data['Date'])
stage_data = stage_data[stage_data['Site_ID'].isin(sites)].copy()

# Quick timeseries of each site to remove bottomed out data
for site in sites:
    plt.figure(figsize=(10, 6))
    site_data = stage_data[stage_data['Site_ID'] == site].copy()
    
    # Sort by date
    site_data = site_data.sort_values('Date')
    
    # Get unique dates and remove first and last day
    unique_dates = site_data['Date'].dt.date.unique()
    if len(unique_dates) > 2:  # Make sure we have at least 3 days of data
        first_day = unique_dates[0]
        last_day = unique_dates[-1]
        
        # Filter out the first and last day
        site_data = site_data[~(site_data['Date'].dt.date == first_day) & 
                              ~(site_data['Date'].dt.date == last_day)]
    
    plt.plot(site_data['Date'], site_data['revised_depth'])
    plt.title(f'Site {site}')
    plt.xlabel('Date')
    plt.ylabel('Revised Depth')
    #plt.ylim(top=-0.35)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% Remove bottomed out data

# Define the bottoming out thresholds for each site
site_bottom_out_depths = {
    '6_93': -5,
    '15_409': -0.45,
    '6_629': -5,
    '14_500': -0.44,
    '5a_582': -0.35,
    '13_267': -0.95
}

# Create a new column that copies the revised_depth
stage_data['revised_depth_nobottom'] = stage_data['revised_depth']

# Set values to NaN where the depth is below the bottoming out threshold
for site, depth in site_bottom_out_depths.items():
    mask = (stage_data['Site_ID'] == site) & (stage_data['revised_depth'] < depth)
    stage_data.loc[mask, 'revised_depth_nobottom'] = np.nan

for site in sites:
    plt.figure(figsize=(10, 6))
    site_data = stage_data[stage_data['Site_ID'] == site]
    plt.plot(site_data['Date'], site_data['revised_depth_nobottom'])
    plt.title(f'Site {site}')
    plt.xlabel('Date')
    plt.ylabel('Revised Depth')
    plt.ylim(top=-0.35)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% Make PDFs of the stage data

color_dict = {
    '6_93': '#1f77b4',    # default blue
    '15_409': '#ff7f0e',  # default orange
    '6_629': '#2ca02c',   # default green
    '14_500': '#d62728',  # default red
    '5a_582': '#9467bd',  # default purple
    '13_267': '#8c564b'   # default brown
}

plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=stage_data, 
    x='revised_depth_nobottom', 
    hue='Site_ID',
    palette=color_dict
)

plt.xlabel('Stage (m) at Wetland Well')
plt.ylabel('Density')
plt.title('Stage PDFs for the Intensive Sites')
plt.xlim(left=-1)
plt.tight_layout()
plt.show()

# %% Map the relative well elevations to the stage data

well_elevations_original = {
    '6_93': 43.146157,
    '15_409': 43.365976,
    '6_629': 44.017723,
    '14_500': 41.568565,
    '5a_582': 44.167662,
    '13_267': 42.056070
}

well_elevations = {
    '6_93': 43.050, # NOTE: this has been changed
    '15_409': 43.1, # NOTE: this has been changed
    '6_629': 44.2, # NOTE: this has been changed
    '14_500': 41.568565,
    '5a_582': 44.167662,
    '13_267': 42.1 # NOTE: this has been changed
}

stage_data['well_elevation'] = stage_data['Site_ID'].map(well_elevations)
stage_data['stage_on_dem'] = stage_data['revised_depth'] + stage_data['well_elevation']

bathymetric_curves = pd.read_csv('./out_data/bathymetric_curves.csv')
bathymetric_curves['stage_m'] = bathymetric_curves['stage'] * 0.3048
bathymetric_curves = bathymetric_curves[['well_id', 'stage_m', 'area', 'volume']].copy()
bathymetric_curves.rename(columns={'well_id': 'Site_ID'}, inplace=True)

# %% Merge the stage data with the bathymetric curves

# Perform the merge for each site separately
result_dfs = []

for site in stage_data['Site_ID'].unique():
    site_data = stage_data[stage_data['Site_ID'] == site].copy()
    bathym_data = bathymetric_curves[bathymetric_curves['Site_ID'] == site].copy()

    bathym_data_sorted = bathym_data.sort_values(["Site_ID", "stage_m"]).reset_index(drop=True)
    site_data_sorted = site_data.sort_values(["Site_ID", "stage_on_dem"]).reset_index(drop=True)
    # Perform the merge_asof operation
    merged_df = pd.merge_asof(
        site_data_sorted,
        bathym_data_sorted,
        by='Site_ID',
        left_on='stage_on_dem',
        right_on='stage_m',
        direction='nearest'
    )

    result_dfs.append(merged_df)

merged_df = pd.concat(result_dfs, ignore_index=True)

# %% Plot the merged data
# test_site = '13_267'
# plot_data = merged_df[merged_df['Site_ID'] == test_site]
# plt.figure(figsize=(8, 6))
# plt.hist(plot_data['area'].dropna(), bins=30, color='skyblue', edgecolor='black')
# plt.xlabel('Area')
# plt.ylabel('Frequency')
# plt.title('Histogram of Area for merged_df')
# plt.tight_layout()
# plt.show()
merged_df['area'] = merged_df['area'] * 100
plt.figure(figsize=(10, 6))
sns.kdeplot(
    data=merged_df, 
    x='area', 
    hue='Site_ID',
    palette=color_dict,
    #clip=(0, 1)
)

plt.xlabel('Inundated Area (%) of Basin Maximum')
plt.ylabel('Density')
plt.title('Inundated Area PDFs for the Intensive Sites')
plt.tight_layout()
plt.show()


# %% 

# plt.figure(figsize=(10, 6))

# # Loop through each site and plot separately
# for site in merged_df['Site_ID'].unique():
#     site_data = merged_df[merged_df['Site_ID'] == site].copy()
    
#     # Make sure data is sorted by date
#     site_data = site_data.sort_values('Date')
    
#     # Plot using matplotlib directly
#     plt.plot(site_data['Date'], site_data['area'], 
#              label=site, 
#              color=color_dict.get(site))

# # Add labels and formatting
# plt.xlabel('Date')
# plt.ylabel('Inundated Area (%)')
# plt.title('Wetland Inundation Over Time')
# plt.legend(title='Site ID')
# plt.grid(True, linestyle='--', alpha=0.7)

# # Format date axis
# plt.gcf().autofmt_xdate()  # Auto-format the date labels
# plt.tight_layout()
# plt.show()

# %% Inundated area timeseries just for site 15_409
import matplotlib.dates as mdates
# Add water year column
def get_water_year(date):
    if date.month >= 10:  # October to December
        return date.year + 1
    else:  # January to September
        return date.year

# Add water year column to merged_df
merged_df['water_year'] = merged_df['Date'].apply(get_water_year)

# Filter for water year 2024
wy2024_data = merged_df[merged_df['water_year'] == 2025].copy()

# Create the plot
# Create the plot
plt.figure(figsize=(12, 6))

# Loop through each site and plot the inundated area
for site in sites:
    site_data = wy2024_data[wy2024_data['Site_ID'] == site].copy()
    
    if len(site_data) == 0:
        print(f"No data for site {site} in Water Year 2024")
        continue
    
    # Sort by date
    site_data = site_data.sort_values('Date')
    
    # Plot the data
    plt.plot(site_data['Date'], site_data['area'], 
             label=site, 
             color=color_dict.get(site))

# Add labels and formatting
plt.xlabel('Date')
plt.ylabel('Inundated Area (%) of Basin Maximum')
plt.title('Wetland Inundation for Water Year 2024')

# Move legend outside to the right
plt.legend(title='Site ID', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True, linestyle='--', alpha=0.7)

# Format x-axis to show month abbreviations
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))

# Add water year annotation
plt.figtext(0.5, 0.01, "Oct 1, 2023 - Sep 30, 2024", 
           ha='center', fontsize=10)

# Adjust layout to make room for the legend
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Left, bottom, right, top margins

plt.show()
# %%

site = '6_629'
site_data = merged_df[merged_df['Site_ID'] == site].copy()
site_data = site_data.sort_values('Date')

    
site_data['water_year'] = site_data['Date'].apply(get_water_year)

# Filter for water years 2022-2025
water_years = [2024]

# Plot each water year in a separate figure
for year in water_years:
    # Get data for this water year
    year_data = site_data[site_data['water_year'] == year]
    
    if len(year_data) == 0:
        print(f"No data for Water Year {year}")
        continue
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # Plot area on primary y-axis
    color = color_dict.get(site)
    ax1.scatter(year_data['Date'], year_data['area'], 
            color=color, label='Inundated Area', s=15, alpha=0.8)
    ax1.set_ylabel('Inundated Area (%) of Basin Maximum', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # Plot depth on secondary y-axis
    depth_color = 'darkblue'
    ax2.scatter(year_data['Date'], year_data['revised_depth_nobottom'], 
            color=depth_color, label='Stage (m)', s=15, alpha=0.8,
            marker='s')  # Use square markers to differentiate
    ax2.set_ylabel('Stage at Wetland Well', color=depth_color)
    ax2.tick_params(axis='y', labelcolor=depth_color)
    
    # Add labels and formatting
    ax1.set_title(f'Stage at Well vs. Inundated Area - Site {site} WY ({year})')
    ax1.set_xlabel('Date')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis to show month abbreviations
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    
    # Add water year span to subtitle
    plt.figtext(0.5, 0.01, f"Oct 1, {year-1} - Sep 30, {year}", 
               ha='center', fontsize=10)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()