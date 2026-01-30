# %% 1.0 Libraries and filepaths

import pandas as pd
data_dir = "D:/depressional_lidar/data/osbs/"
path = data_dir + "/in_data/sampling_elevations/FaithH_complete_osbs_soilsampling_dataset.xlsx"

df = pd.read_excel(path)
df = df.drop(columns=[
    "Site Location",
    "weigh_boat_num",
    "Soil sample depth increment (cm)",
    "soil sample average depth increment of the 5cm slice (cm)",
    "Sample depth-increment elevation relative to soil surface elevation at well (cm)"
])

df.rename(columns={
    "Water depth at time of sampling (cm)": "sample_wtr_depth", 
    "Soil surface elevation at soil coring point relative to soil surface elevation at well (cm)": "well_to_core_diff",
    "proximity to sensor within sampling transect (sensor 1 being lowest elevation in wetland with most inundation and sensor 5 being highest in elevation leading out of the wetland)": "sensor_proximity"
}, inplace=True)

print(df.head())

# %% 2.0 Group-by to Wetland Name and Sensor Proximity

grouped = df.groupby(["Wetland Name", "sensor_proximity"]).agg({
    "well_to_core_diff": "unique",
    "sample date" : "unique",
    "Soil Core ID": "count",
}).reset_index()

# %% 3.0 Check by each wetland name

for wetland in grouped["Wetland Name"].unique():
    print("\nWetland Name:", wetland)
    wetland_df = grouped[grouped["Wetland Name"] == wetland]
    print(wetland_df.to_string(index=False))
    

# %%
