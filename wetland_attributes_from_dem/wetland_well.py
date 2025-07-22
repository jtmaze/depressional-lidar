from pathlib import Path

import pandas as pd
import geopandas as gpd
from pyproj import CRS
import numpy as np
import rasterio as rio

#from wetland_attributes_from_dem import well_elevation_estimators

class WetlandWell:

    def __init__(
        self,
        well_id: str,
        well_pts_gdf: gpd.GeoDataFrame,
        dem_source: str,
        hydrograph_df: pd.DataFrame,
    ):
        
        self.well_id = well_id
        # 1.0 Get the projection info from the DEM
        self.dem_source = Path(dem_source)
        # NOTE: Putting everything in the DEM crs
        with rio.open(self.dem_source) as dem:
            dem_crs = CRS(dem.crs)
        if dem_crs is None:
            raise ValueError("DEM has no CRS: cannot align")
        
        self.crs = dem_crs
        
        if well_pts_gdf.crs is None:
            raise ValueError("well_pts_gdf must have a defined CRS.")

        # 2.0 Convert the well points to the DEM crs and extract points
        well_pts_gdf = well_pts_gdf.to_crs(dem_crs)
        self.well_coords = well_pts_gdf[well_pts_gdf['well_id'] == well_id].geometry.iloc[0]

        # 3.0 Get well's specific hydrograph
        self.hydrograph = hydrograph_df[hydrograph_df['well_id'] == well_id].copy()

    def get_well_dem_elevation(self) -> float:
        """
        Get the DEM elevation at the well coordinates.
        """
        



