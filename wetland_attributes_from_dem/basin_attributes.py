
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
from shapely.geometry import Point, LineString
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
    # Defaults cached for radial transects
    transect_method: str = 'deepest'  # 'centroid' or 'deepest'
    transect_n: int = 8
    transect_buffer: float = 0.0

    @cached_property
    def deepest_point(self) -> DeepestPoint:
        return self.find_deepest_point()
    @cached_property
    def clipped_dem(self) -> ClippedDEM:
        return self.get_clipped_dem()
    @cached_property
    def radial_transects(self) -> gpd.GeoDataFrame:
        return self.establish_radial_transects(
            method=self.transect_method,
            n=self.transect_n,
            buffer_distance=self.transect_buffer
        )

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
        ax = show(dem_data, transform=dem_transform, ax=ax, cmap='viridis')
        if ax.images:
            plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
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
        Find the minimum DEM value within the footprint based on a 3x3 cell average.
        Returns the location of the minimum 3x3 average with its actual elevation.
        """
        clipped = self.clipped_dem
        dem_data = clipped.dem
        
        # Create an output array for the 3x3 averages
        dem_avg = np.zeros_like(dem_data)
        dem_avg.fill(np.nan)
        
        # Calculate 3x3 averages
        rows, cols = dem_data.shape
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                # Extract 3x3 window
                window = dem_data[i-1:i+2, j-1:j+2]
                # Calculate average ignoring NaNs
                if not np.all(np.isnan(window)):
                    dem_avg[i, j] = np.nanmean(window)
        
        # Find the minimum value and its location in the averaged DEM
        row, col = np.unravel_index(np.nanargmin(dem_avg), dem_avg.shape)
        
        # Get the actual elevation from the original DEM at this location
        min_val = float(dem_data[row, col])
        
        # Convert (row, col) to map x,y using the clipped transform
        x, y = rio.transform.xy(clipped.transform, row, col, offset="center")
        
        # return as GeoSeries in the DEM/footprint CRS
        pt = gpd.GeoSeries([Point(x, y)], crs=clipped.crs)

        return DeepestPoint(elevation=min_val, location=pt)

    def _find_point_elevation(self, point: gpd.GeoSeries) -> float:
        """
        Find the elevation at a specific point using the clipped DEM.
        # BUG some of our wells are outside the basins
        """
        clipped = self.clipped_dem
        dem_data = clipped.dem
        # Get the point location
        x, y = point.geometry.x.values[0], point.geometry.y.values[0]
        row, col = rio.transform.rowcol(clipped.transform, x, y)

        # Check if the point is within the DEM bounds
        rows, cols = dem_data.shape
        if 0 <= row < rows and 0 <= col < cols:
            # Calculate window boundaries (handle edge cases)
            row_start = max(0, row-1)
            row_end = min(rows, row+2)
            col_start = max(0, col-1)
            col_end = min(cols, col+2)
            
            # Extract 3x3 window (or smaller if near edge)
            window = dem_data[row_start:row_end, col_start:col_end]
            
            # Calculate average ignoring NaNs
            if not np.all(np.isnan(window)):
                pt_elevation = np.nanmean(window)
            else:
                pt_elevation = np.nan
        else:
            # Point is outside DEM bounds
            pt_elevation = np.nan

        return float(pt_elevation) if not np.isnan(pt_elevation) and pt_elevation != clipped.nodata else np.nan

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

    def calculate_hypsometry(self, method: str = "total"):
        step = 0.02 # NOTE: Hardcoded this for now
        dem_data = self.clipped_dem.dem
        total_area = self.footprint.area.values[0]  # area in square meters
        print(total_area)
        min_elevation = np.nanmin(dem_data)
        max_elevation = np.nanmax(dem_data)

        if method == "total":

            flat_dem = dem_data.flatten()
            bins = np.arange(min_elevation, max_elevation + step, step)
            hist, bin_edges = np.histogram(flat_dem, bins=bins)
            cum_area_m2 = np.cumsum(hist) # NOTE: Assumes 1x1m cell size on DEM
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            return cum_area_m2, bin_centers
        
        else:
            print(f"Method '{method}' not implemented for hypsometry calculation.")
            return None, None

    def plot_basin_hypsometry(
            self,
            plot_points: bool = False
        ):

        cum_area_m2, bin_centers = self.calculate_hypsometry()
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, cum_area_m2, label="Cumulative Area", color="blue")

        if plot_points:
            # Add well point elevations if available
            well_point = self.establish_well_point(self.well_point_info)
            
            # Interpolate cumulative area at well point elevations
            dem_area = np.interp(well_point.elevation_dem, bin_centers, cum_area_m2)
            rtk_area = np.interp(well_point.elevation_rtk, bin_centers, cum_area_m2)

            plt.plot(well_point.elevation_dem, dem_area, 'ro', markersize=8,
                     label=f"Well DEM Elevation ({well_point.elevation_dem:.2f}m, {dem_area:.2f}m^2)")
            plt.plot(well_point.elevation_rtk, rtk_area, 'go', markersize=8,
                    label=f"Well RTK Elevation ({well_point.elevation_rtk:.2f}m, {rtk_area:.2f}m^2)")
        
        plt.xlabel("Elevation (m)")
        plt.ylabel("Cumulative Area (m^2)")
        plt.title(f"{self.wetland_id} Hypsometry")
        plt.grid()
        plt.legend()
        plt.show()

    def establish_radial_transects(
            self, 
            method: str = 'deepest',
            n: int = 8,
            buffer_distance: float = 0
        ) -> gpd.GeoDataFrame:

        poly = self.footprint.geometry.values[0]
        target_poly = poly if buffer_distance == 0 else poly.buffer(buffer_distance)

        if method == 'deepest':
            center_pt: Point = self.deepest_point.location.values[0]
        elif method == 'centroid':
            center_pt: Point = self.footprint.centroid.values[0]
        else:
            raise ValueError(f"Method '{method}' not recognized. Use 'centroid' or 'deepest'.")

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        records = []
        lines = []

        for i in angles:
            dx, dy = np.cos(i), np.sin(i)
            far_pt = Point(
                center_pt.x + dx * 1000,  # Extend 1000m in the direction of angle
                center_pt.y + dy * 1000
            )

            ray = LineString([center_pt, far_pt])
            inter = ray.intersection(target_poly.boundary)

            if inter.geom_type == 'Point':
                end_pt = inter
            elif inter.geom_type == 'MultiPoint':
                pts = [p for p in getattr(inter, 'geoms', []) if p.geom_type == 'Point']
                end_pt = max(pts, key=lambda p: p.distance(center_pt))
            else:
                print('Error: Radial Transect is not a Point or MultiPoint')
                continue

            line = LineString([center_pt, end_pt])
            lines.append(line)
            records.append({
                'angle_rad': i,
                'length_m': float(center_pt.distance(end_pt)),
                'geometry': line,
            })

        return gpd.GeoDataFrame(records, geometry=lines, crs=self.footprint.crs)

    def radial_transects_map(self):

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

        radial_transects = self.radial_transects
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = show(dem_data, transform=dem_transform, ax=ax, cmap='viridis')
        if ax.images:
            plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
        self.footprint.boundary.plot(ax=ax, color='blue', linewidth=2, label='Basin Boundary')

        # Plot each transect line
        for i, line in enumerate(radial_transects.geometry):
            x, y = line.xy
            ax.plot(x, y, color='red', linewidth=1.5)

        if len(radial_transects.geometry) > 0:
            center_x, center_y = radial_transects.geometry[0].xy[0][0], radial_transects.geometry[0].xy[1][0]
            ax.plot(center_x, center_y, 'yo', markersize=8)

        plt.title("Radial Transects")
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.legend()
        plt.show()

    def plot_radial_transects(self):
        pass
