
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
    @cached_property
    def transect_profiles(self) -> pd.DataFrame:
        return self.find_radial_transects_vals()
    @cached_property
    def aggregated_transect_profiles(self) -> pd.DataFrame:
        return self.aggregate_radial_transects_vals()

    def visualize_shape(
            self, 
            show_deepest: bool = False,
            show_centroid: bool = False,
            show_well: bool = False
        ):

        # Create bounding box
        bounds = self.footprint.total_bounds
        buffer_bounds = [
            bounds[0] - 75,  # minx
            bounds[1] - 75,  # miny
            bounds[2] + 75,  # maxx
            bounds[3] + 75   # maxy
        ]
        
        with rio.open(self.source_dem_path) as dem:
            # Create window from bounds
            window = rio.windows.from_bounds(*buffer_bounds, dem.transform)
            dem_data = dem.read(1, window=window)
            dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)
            dem_transform = rio.windows.transform(window, dem.transform)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = show(dem_data, transform=dem_transform, ax=ax, cmap='viridis')
        if ax.images:
            plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
        self.footprint.plot(ax=ax, facecolor='none', edgecolor='red')

        if self.transect_buffer != 0:
            self.footprint.geometry.buffer(self.transect_buffer).plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--')

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

        if self.transect_buffer != 0:
            shape = [self.footprint.geometry.buffer(self.transect_buffer).values[0]]
        else:   
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
        # BUG some of our wetland wells are outside the basins
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
            if abs(well_point.elevation_rtk - well_point.elevation_dem) > 2:
                print(f"Warning: RTK elevation ({well_point.elevation_rtk:.2f}m) differs from DEM elevation ({well_point.elevation_dem:.2f}m) by more than 2m.")
            else:
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
            bounds[0] - 75,  # minx
            bounds[1] - 75,  # miny
            bounds[2] + 75,  # maxx
            bounds[3] + 75   # maxy
        ]

        with rio.open(self.source_dem_path) as dem:
            # Create window from bounds
            window = rio.windows.from_bounds(*buffer_bounds, dem.transform)
            dem_data = dem.read(1, window=window)
            dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)  
            dem_transform = rio.windows.transform(window, dem.transform)

        radial_transects = self.radial_transects
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = show(dem_data, transform=dem_transform, ax=ax, cmap='gray')  # Changed colormap to 'gray'
        if ax.images:
            plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
        self.footprint.boundary.plot(ax=ax, color='red', linewidth=2, label='Basin Boundary')

        if self.transect_buffer != 0:
            self.footprint.geometry.buffer(self.transect_buffer).plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--', label='Buffered Boundary')

        # Create a colormap for the transects
        num_transects = len(radial_transects)
        cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)
        
        # Plot each transect line with a different color based on index
        for i, line in enumerate(radial_transects.geometry):
            x, y = line.xy
            color = cmap(i)
            ax.plot(x, y, color=color, linewidth=1.5)

        center_x, center_y = radial_transects.geometry[0].xy[0][0], radial_transects.geometry[0].xy[1][0]
        ax.plot(center_x, center_y, 'yo', markersize=8, label='Radial Reference Point')

        plt.title(f"Radial Transects for {self.wetland_id}")
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.legend()
        plt.show()

    def sample_transect_dem_vals(
        self,
        line: LineString,
        step: float
    ) -> pd.DataFrame:
        
        if step is None:
            print('tbd figure this out later')

        # Generate Points along the line
        total_length = line.length
        distances = np.arange(0, total_length + step, step)
        points = [line.interpolate(d) for d in distances]
        coords = [(p.x, p.y) for p in points]

        dem_data = self.clipped_dem.dem
        transform = self.clipped_dem.transform

        rows, cols = rio.transform.rowcol(
            transform, 
            [p[0] for p in coords], 
            [p[1] for p in coords]
        )

        dem_rows, dem_cols = dem_data.shape
        elevations = []
        for r, c in zip(rows, cols):
            if 0 <= r < dem_rows and 0 <= c < dem_cols:
                row_start = max(0, r-1)
                row_end = min(dem_rows, r+2)
                col_start = max(0, c-1)
                col_end = min(dem_cols, c+2)
                window = dem_data[row_start:row_end, col_start:col_end]
                
                if not np.all(np.isnan(window)):
                    elev = np.nanmean(window)
                else:
                    elev = np.nan
                
                # Store list of elevations
                elevations.append(elev)

        return pd.DataFrame({
            'distance_m': distances,
            'dem_elevation': elevations
        })
            
    def find_radial_transects_vals(self):
        """
        Find the elevation values along the radial transects.
        """
        transects = self.radial_transects
        dfs = []
        for idx, row in transects.iterrows():
            profile = self.sample_transect_dem_vals(
                row['geometry'], 
                step=1.0  # 1 meter step
            )
            profile['angle_rad'] = row['angle_rad']
            profile['trans_idx'] = idx
            profile['length_m'] = row['length_m']
            dfs.append(profile)

        return pd.concat(dfs, ignore_index=True)

    def plot_individual_radial_transects(self):
        
        transects = self.find_radial_transects_vals()

        indexes = transects['trans_idx'].unique()

        num_transects = len(indexes)
        cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)

        for i in indexes:
            subset = transects[transects['trans_idx'] == i]
            plt.plot(subset['distance_m'], subset['dem_elevation'], label=f'Transect {i}', color=cmap(i))

        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation (m)")
        plt.title("Radial Transects")
        plt.legend()
        plt.show()

    def hayashi_p_constants(self, r0: int, r1: int) -> float:
        """
        Based on Hayashi et. al (2000)
        """
        transects = self.transect_profiles
        unique_idx = transects['trans_idx'].unique()

        hayashi_ps = []
        for i in unique_idx:
            trans_idx = i
            trans = transects[transects['trans_idx'] == i].copy()
            min_elevation = trans['dem_elevation'].min()
            trans['depth_from_min'] = trans['dem_elevation'] - min_elevation
            z0 = trans[trans['distance_m'] == r0]['depth_from_min'].mean()
            z1 = trans[trans['distance_m'] == r1]['depth_from_min'].mean()

            if np.isnan(z0) or np.isnan(z1):
                p = np.nan
            else:
                print(z0/z1, r0/r1)
                p = np.log(z0/z1) / np.log(r0/r1)

            results = {
                'trans_idx': trans_idx,
                'p': p
            }
            hayashi_ps.append(results)

        return pd.DataFrame(hayashi_ps)
    

    def plot_hayashi_p(self, r0: int, r1: int):

        df = self.hayashi_p_constants(r0, r1)

        plt.figure(figsize=(10, 6))
                
        num_transects = len(df)
        cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)
        
        colors = [cmap(i) for i in range(len(df))]
        plt.bar(df['trans_idx'], df['p'], color=colors)
        
        plt.xlabel("Transect Index")
        plt.ylabel("Hayashi p")
        plt.title(f"Basin {self.wetland_id} Hayashi p Constants for Radial Transects\n Computed with r0={r0}, r1={r1}")
        plt.show()

    def aggregate_radial_transects(self) -> pd.DataFrame:
        """
        Aggregate transect profiles by distance from the center, averaging elevations
        across all transects at each distance.
        Returns a DataFrame with distance_m and summary stats.
        """
        profiles = self.transect_profiles  # triggers computation via cached_property
        df = profiles[['distance_m', 'dem_elevation']].dropna()

        agg = (
            df.groupby('distance_m', as_index=False)['dem_elevation']
                .agg(n='count', mean='mean', std='std')
        )

        agg = agg[agg['n'] > 4]  # Filter out distances with only four transects available

        return agg.sort_values('distance_m', ignore_index=True)
    
    def plot_aggregated_radial_transects(self):
        
        agg = self.aggregate_radial_transects()
        plt.errorbar(agg['distance_m'], agg['mean'], yerr=agg['std'], fmt='o')
        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation (m)")
        plt.title(f"Aggregated Radial Transects for {self.wetland_id}\nwith n={self.transect_n} with basin shape buffer={self.transect_buffer}")
        plt.show()

