
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point
from affine import Affine

#from wetland_attributes_from_dem import well_elevation_estimators

@dataclass
class ClippedDEM:
    """In-memory DEM subset clipped to the wetland footprint."""
    dem: np.ndarray              # 2D array, np.nan outside footprint
    transform: Affine            # affine transform for the clipped array
    crs: CRS                     # pyproj CRS (compatible with geopandas)
    nodata: Optional[float]      # original nodata value, if present

@dataclass
class DeepestPoint:
    elevation: float
    location: gpd.GeoSeries

@dataclass
class RadialTransect:
    # TODO: tweak this dataclass
    """A radial transect for the wetland basin."""
    start: gpd.GeoSeries
    end: gpd.GeoSeries
    elevation_profile: pd.Series

@dataclass
class RadialTransects:
    # TODO: tweak this dataclass
    """Collection of radial transects for the wetland basin."""
    transects: list[RadialTransect]
    center: gpd.GeoSeries
    basin_buffered: int

@dataclass
class WellPoint:
    elevation_dem: float # The well's elevatation - coords placed on DEM
    elevation_rtk: float # The well's elevation as measured by RTK GPS
    depth: float # The well's location relative to wetland bottom
    location: gpd.GeoSeries


@dataclass
class WetlandBasin:
    wetland_id: str
    source_dem_path: str
    footprint: gpd.GeoDataFrame
    well_point_info: gpd.GeoDataFrame

    @cached_property
    def deepest_point(self) -> DeepestPoint:
        return self.find_deepest_point()

    def visualize_shape(
            self, 
            show_deepest: bool = False,
            show_centroid: bool = False,
            show_well: bool = False
        ):

        # Create bounding box
        bounds = self.footprint.total_bounds
        buffer_bounds = [
            bounds[0] - 50,  # minx
            bounds[1] - 50,  # miny
            bounds[2] + 50,  # maxx
            bounds[3] + 50   # maxy
        ]
        
        with rio.open(self.source_dem_path) as dem:
            # Create window from bounds
            window = rio.windows.from_bounds(*buffer_bounds, dem.transform)
            dem_data = dem.read(1, window=window)
            dem_transform = rio.windows.transform(window, dem.transform)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        show(dem_data, transform=dem_transform, ax=ax, cmap='viridis')
        self.footprint.plot(ax=ax, facecolor='none', edgecolor='red')
        
        if show_deepest:
            deepest = self.deepest_point
            deepest.location.plot(ax=ax, color='blue', marker='*', markersize=100)
            ax.annotate(f"Deepest: {deepest.elevation:.2f}m", 
                        xy=(deepest.location.x.values[0], deepest.location.y.values[0]),
                        xytext=(10, 10), textcoords='offset points',
                        color='white', fontweight='bold')
            
        if show_centroid:
            centroid = self.footprint.geometry.centroid
            centroid_elevation = self._find_point_elevation(centroid)
            centroid.plot(ax=ax, color='orange', marker='*', markersize=100)
            ax.annotate(f"Centroid: {centroid_elevation:.2f}m", 
                        xy=(centroid.x.values[0], centroid.y.values[0]),
                        xytext=(10, 10), textcoords='offset points',
                        color='white', fontweight='bold')
            
        if show_well:
            WellPoint = self.establish_well_point(self.well_point_info)
            if WellPoint:
                WellPoint.location.plot(ax=ax, color='violet', marker='o', markersize=100)
                ax.annotate(f"DEM {WellPoint.elevation_dem:.2f}m -- RTK {WellPoint.elevation_rtk:.2f}m", 
                            xy=(WellPoint.location.x.values[0], WellPoint.location.y.values[0]),
                            xytext=(10, 10), textcoords='offset points',
                            color='white', fontweight='bold')

        plt.title(f"Wetland Basin: {self.wetland_id}")
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.show()

    def get_clipped_dem(self) -> ClippedDEM:

        shape = [self.footprint.geometry.values[0]]
        with rio.open(self.source_dem_path) as dem:
            data, out_transform = rio_mask(
                dem,
                shape,
                crop=True,
                filled=True,
                nodata=dem.nodata
            )

            band = data[0].astype("float64")
            nodata = dem.nodata
            
            # Replace nodata values with np.nan
            band = np.where(band == nodata, np.nan, band)

        return ClippedDEM(
                dem=band,
                transform=out_transform,
                crs=CRS.from_user_input(dem.crs),
                nodata=nodata
            )
    
    def find_deepest_point(self) -> DeepestPoint:
        """
        Find the minimum DEM value within the footprint and return its location.
        """
        clipped = self.get_clipped_dem()
        dem_data = clipped.dem

        # min value and its (row, col) index
        min_val = float(np.nanmin(dem_data))
        row, col = np.unravel_index(np.nanargmin(dem_data), dem_data.shape)

        # convert (row, col) to map x,y using the clipped transform
        x, y = rio.transform.xy(clipped.transform, row, col, offset="center")

        # return as GeoSeries in the DEM/footprint CRS
        pt = gpd.GeoSeries([Point(x, y)], crs=clipped.crs)

        return DeepestPoint(elevation=min_val, location=pt)

    def _find_point_elevation(self, point: gpd.GeoSeries) -> float:

        clipped = self.get_clipped_dem()
        dem_data = clipped.dem
        # Get the elevation value at the point location
        x, y = point.geometry.x.values[0], point.geometry.y.values[0]
        row, col = rio.transform.rowcol(clipped.transform, x, y)

        pt_elevation = dem_data[row, col]

        return float(pt_elevation) if pt_elevation != clipped.nodata else np.nan

    def establish_well_point(self, well_point_info: gpd.GeoSeries) -> WellPoint:
        """
        Uses well point (lat, long, rtk_elevation) to establish a WellPoint.
        """
        if well_point_info.crs != self.footprint.crs:
            print("Warning: Well point CRS does not match footprint CRS. Reprojecting...")
            well_point_info = well_point_info.to_crs(self.footprint.crs)

        elevation_dem = self._find_point_elevation(well_point_info.geometry)

        basin_low = self.deepest_point.elevation
        depth = elevation_dem - basin_low
        rtk = float(well_point_info['rtk_elevation'].values[0])

        return WellPoint(
            elevation_dem=elevation_dem,
            elevation_rtk=rtk,
            depth=depth,
            location=well_point_info.geometry
        )
    
    def calculate_hypsometry(self):
        pass

    def plot_basin_hypsometry(self):
        pass

    def establish_radial_transects(self):
        pass

    def map_radial_transects(self):
        pass

    def plot_radial_transects(self):
        pass
