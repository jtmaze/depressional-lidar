# %%

from pathlib import Path
import richdem as rd
from lidar.filling import ExtractSinks
from lidar import DelineateDepressions

base = Path(r"D:/depressional_lidar/data/osbs")
out_dir = base / "temp"
out_dir.mkdir(parents=True, exist_ok=True)

filtered_dem = base / "in_data" / "osbs_DEM_resampled_cleaned.tif"
assert filtered_dem.exists(), f"Filtered DEM not found: {filtered_dem}"

# %% 1.0 read & fill with richdem
dem = rd.LoadGDAL(str(filtered_dem))
filled = rd.FillDepressions(dem)
filled_path = out_dir / "dem_filled_richdem.tif"
rd.SaveGDAL(str(filled_path), filled)
print("Wrote filled DEM:", filled_path)

# %% 3.0 Call ExtractSinks using the filled_dem argument
sink_path = ExtractSinks(
    str(filtered_dem),     # input (original filtered DEM)
    min_size=70,
    out_dir=str(out_dir),
    filled_dem=str(filled_path),
    engine='richdem'
)
print("sink_path:", sink_path)

# %% Run 

min_size = 70
min_depth = 0.1
interval = 0.25
bool_shp = True

dep_id_path, dep_level_path = DelineateDepressions(sink_path,
                                                   min_size,
                                                   min_depth,
                                                   interval,
                                                   out_dir,
                                                   bool_shp)
print('Results are saved in: {}'.format(out_dir))


# %%
