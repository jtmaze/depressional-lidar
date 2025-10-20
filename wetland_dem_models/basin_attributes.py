
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
    def well_point(self) -> WellPoint:
        """
        Establishes the well point either with or without a prior basin geometry.
        """
        if self.well_point_info is None:
            raise ValueError("Well point info is required")
        
        if self.footprint is not None:
            # Use DEM clipped to the prior basin footprint
            return self.establish_well_point(self.well_point_info)
        else:
            # Use source DEM becuase there's no prior basin footprint
            return self._establish_well_point_from_source()

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
    def truncated_transect_profiles(self) -> pd.DataFrame:
        return self.truncate_radial_transects_by_zmin()
    
    @cached_property
    def aggregated_transect_profiles(self) -> pd.DataFrame:
        return self.aggregate_radial_transects_vals()

    def visualize_shape(
            self, 
            show_deepest: bool = False,
            show_centroid: bool = False,
            show_well: bool = False
        ):


        if self.footprint is not None:
            plot_shape = self.footprint
            bounds = self.footprint.total_bounds
        else:
            well_point = self.well_point.location
            buffer_dist = self.transect_buffer if self.transect_buffer > 0 else 100
            buffered = well_point.buffer(buffer_dist)
            plot_shape = buffered
            bounds = buffered.total_bounds

        buffer_bounds = [
            bounds[0] - 100,  # minx
            bounds[1] - 100,  # miny
            bounds[2] + 100,  # maxx
            bounds[3] + 100   # maxy
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
        plot_shape.plot(ax=ax, facecolor='none', edgecolor='red')

        if self.transect_buffer != 0 and self.footprint is not None:
            plot_shape.geometry.buffer(self.transect_buffer).plot(ax=ax, facecolor='none', edgecolor='red', linestyle='--')

        if show_deepest:
            deepest = self.deepest_point
            deepest.location.plot(ax=ax, color='blue', marker='*', markersize=100)
            ax.annotate(f"Deepest: {deepest.elevation:.2f}m", 
                xy=(deepest.location.x.values[0], deepest.location.y.values[0]),
                xytext=(10, 10), textcoords='offset points',
                color='white', fontweight='bold')
            
        if show_centroid and self.footprint is not None:
            centroid = plot_shape.geometry.centroid
            centroid_elevation = self._find_point_elevation(centroid)
            centroid.plot(ax=ax, color='orange', marker='*', markersize=100)
            ax.annotate(f"Centroid: {centroid_elevation:.2f}m", 
                xy=(centroid.x.values[0], centroid.y.values[0]),
                xytext=(10, 10), textcoords='offset points',
                color='white', fontweight='bold')
            
        if show_well:
            WellPoint = self.well_point
            if WellPoint:
                WellPoint.location.plot(ax=ax, color='violet', marker='o', markersize=100)
                ax.annotate(f"DEM {WellPoint.elevation_dem:.2f}m", 
                        xy=(WellPoint.location.x.values[0], WellPoint.location.y.values[0]),
                        xytext=(10, 10), textcoords='offset points',
                        color='white', fontweight='bold')

        plt.title(f"Wetland Basin: {self.wetland_id}")
        plt.xlabel("x (meters)")
        plt.ylabel("y (meters)")
        plt.show()

    def get_clipped_dem(self) -> ClippedDEM:

        """
        Two approaches to clipping the DEM:
            1) If a basin_footprint is passed to the WetlandBasin constructor, use that 
            geometry buffered by some distance to select DEM cells.
            2) Otherwise, select the DEM cells within some distance of the well.
        """

        if self.footprint is not None:
            if self.transect_buffer != 0:
                shape = [self.footprint.geometry.buffer(self.transect_buffer).values[0]]
            else:   
                shape = [self.footprint.geometry.values[0]]
        else:
            if self.well_point_info is None:
                raise ValueError("Either footprint or well_point_info must be provided")
            
            well_buffer_dist = self.transect_buffer if self.transect_buffer > 0 else 100
            well_point = self.well_point_info.geometry.values[0]
            shape = [well_point.buffer(well_buffer_dist)]

        # Clip DEM to either basin footprint or the well point
        with rio.open(self.source_dem_path) as dem:
            data, out_transform = rio_mask(
                dem, shape, crop=True, filled=True, nodata=dem.nodata
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
        Find the minimum DEM value within the footprint based on a 5x5 cell average.
        Returns the location of the minimum 5x5 average with its actual elevation.
        """

        #TODO: Find the percentile?
        clipped = self.clipped_dem
        dem_data = clipped.dem
        
        # Create an output array for the 5x5 averages
        dem_avg = np.zeros_like(dem_data)
        dem_avg.fill(np.nan)
        
        # Calculate 5x5 averages
        rows, cols = dem_data.shape
        for i in range(2, rows-2):
            for j in range(2, cols-2):
                # Extract 5x5 window
                window = dem_data[i-2:i+3, j-2:j+3]
                # Calculate IQR mean to get a better estimate
                flat_window = window.flatten()
                valid_vals = flat_window[~np.isnan(flat_window)]
                if len(valid_vals) >= 4:
                    q25, q75 = np.nanpercentile(valid_vals, [25, 75])
                    iqr_vals = valid_vals[(valid_vals >= q25) & (valid_vals <= q75)]
                    dem_avg[i, j] = np.mean(iqr_vals)
        
        # Find the minimum value and its location in the averaged DEM
        row, col = np.unravel_index(np.nanargmin(dem_avg), dem_avg.shape)
        
        # Get the actual elevation from the original DEM at this location
        def _get_low_elevation(row, col):
            """
            Calculate the 25th percentile elevation within a 5×5 window centered on the given cell.
            """
            row_start = max(0, row - 2)
            row_end = min(dem_data.shape[0], row + 3)
            col_start = max(0, col - 2)
            col_end = min(dem_data.shape[1], col + 3)

            min_window = dem_data[row_start:row_end, col_start:col_end]

            # Calculate 25th percentile, ignoring NaN values
            valid_vals = min_window[~np.isnan(min_window)]
            min_val = float(np.percentile(valid_vals, 25))

            return min_val

        min_val = _get_low_elevation(row, col)
        
        # Convert (row, col) to map x,y using the clipped transform
        x, y = rio.transform.xy(clipped.transform, row, col, offset="center")
        
        # return as GeoSeries in the DEM/footprint CRS
        pt = gpd.GeoSeries([Point(x, y)], crs=clipped.crs)

        return DeepestPoint(elevation=min_val, location=pt)

    def _find_point_elevation(self, point: gpd.GeoSeries) -> float:
        """
        Find the elevation at a specific point using the clipped DEM.
        Uses a 3x3 window to mitigate DEM noise
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

        rtk = float(well_point_info['rtk_elevation'].values[0])

        return WellPoint(
            elevation_dem=elevation_dem,
            elevation_rtk=rtk,
            location=well_point_info.geometry
        )
    
    def _establish_well_point_from_source(self) -> WellPoint:
        """
        Establishes the well point using the original source DEM (before clipping)
        """
        if self.well_point_info is None:
            raise ValueError("well_point_info is required becuase no basin footprint is required")
        
        with rio.open(self.source_dem_path) as dem:
            x = self.well_point_info.geometry.x.values[0]
            y = self.well_point_info.geometry.y.values[0]
            # Convert to row/col in source DEM
            row, col = rio.transform.rowcol(dem.transform, x, y)
            window = rio.windows.Window(col-1, row-1, 3, 3)
            dem_data = dem.read(1, window=window)
            dem_data = np.where(dem_data==dem.nodata, np.nan, dem_data)
            elevation_dem = float(np.nanmean(dem_data))

        rtk = float(self.well_point_info['rtk_elevation'].values[0])
        
        return WellPoint(
            elevation_dem=elevation_dem,
            elevation_rtk=rtk,
            location=self.well_point_info.geometry
        )

    def calculate_hypsometry(self, method: str = "total"):
        step = 0.01 # NOTE: Hardcoded this for now
        dem_data = self.clipped_dem.dem
        dem_scale = self.clipped_dem.transform.a 
        min_elevation = np.nanmin(dem_data)
        max_elevation = np.nanmax(dem_data)

        if method == "total":
            flat_dem = dem_data.flatten()
            bins = np.arange(min_elevation, max_elevation + step, step)
            hist, bin_edges = np.histogram(flat_dem, bins=bins)
            cum_area_m2 = np.cumsum(hist) * (dem_scale ** 2)  
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            return cum_area_m2, bin_centers
        
        if method == "pct_trim":
            flat_dem = dem_data.flatten()
            p_low, p_high = np.nanpercentile(flat_dem, [2, 98])
            flat_dem = flat_dem[(flat_dem >= p_low) & (flat_dem <= p_high)]
            bins = np.arange(flat_dem.min(), flat_dem.max() + step, step)
            hist, bin_edges = np.histogram(flat_dem, bins=bins)
            cum_area_m2 = np.cumsum(hist) * (dem_scale ** 2)  
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
        plt.plot(bin_centers, cum_area_m2, label="Inundated Area", color="blue")

        if plot_points:
            # Add well point elevations if available
            well_pt = self.well_point
            
            # Interpolate cumulative area at well point elevations
            dem_area = np.interp(well_pt.elevation_dem, bin_centers, cum_area_m2)
            rtk_area = np.interp(well_pt.elevation_rtk, bin_centers, cum_area_m2)

            plt.plot(well_pt.elevation_dem, dem_area, 'ro', markersize=8,
                     label=f"Well DEM Elevation ({well_pt.elevation_dem:.2f}m, {dem_area:.2f}m^2)")
            if abs(well_pt.elevation_rtk - well_pt.elevation_dem) > 0.25:
                print(f"Warning: RTK elevation ({well_pt.elevation_rtk:.2f}m) differs from DEM elevation ({well_pt.elevation_dem:.2f}m) by more than 0.25m.")
            else:
                plt.plot(well_pt.elevation_rtk, rtk_area, 'go', markersize=8,
                        label=f"Well RTK Elevation ({well_pt.elevation_rtk:.2f}m, {rtk_area:.2f}m^2)")
        
        plt.xlabel("Elevation (m)")
        plt.ylabel("Inundated Area (m^2)")
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

    def radial_transects_map(self, uniform: bool = False):

        # Don't compute transects without basin footprint
        if self.footprint is None:
            print('Will not compute radial transects without prior basin')
            return
        
        plot_shape = self.footprint
        bounds = self.footprint.total_bounds
        buffer_bounds = [
            bounds[0] - 100,  # minx
            bounds[1] - 100,  # miny
            bounds[2] + 100,  # maxx
            bounds[3] + 100   # maxy
        ]
        with rio.open(self.source_dem_path) as dem:
            # Create window from bounds
            window = rio.windows.from_bounds(*buffer_bounds, dem.transform)
            dem_data = dem.read(1, window=window)
            dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)  
            dem_transform = rio.windows.transform(window, dem.transform)

        if uniform:
            # A bit of a hacky strategy...
            # Clip the geometries in radial_transects (gdf) by distance values in truncated_profiles (df)
            # Render a new gdf with the truncated geometries.
            truncated_profiles = self.truncated_transect_profiles
            original_transects = self.radial_transects
            max_distances = truncated_profiles.groupby('angle_rad')['distance_m'].max().reset_index()
            truncated_geoms = []
            for _, row in max_distances.iterrows():
                angle = row['angle_rad']
                max_dist = row['distance_m']

                # Get the original transect for this angle
                original_transect = original_transects[original_transects['angle_rad'] == angle].iloc[0]
                original_line = original_transect['geometry']

                # Truncate the line at the max distance
                if max_dist < original_line.length:
                    truncated_line = LineString([
                        original_line.coords[0],  # Start point (center)
                        original_line.interpolate(max_dist)  # End point at max distance
                    ])
                else:
                    truncated_line = original_line  # Keep original if max_dist exceeds line length
                
                truncated_geoms.append({
                    'angle_rad': angle,
                    'length_m': max_dist,
                    'geometry': truncated_line
                })
            plot_transects = gpd.GeoDataFrame(truncated_geoms, crs=self.footprint.crs)

        else:
            plot_transects = self.radial_transects

        fig, ax = plt.subplots(figsize=(10, 8))
        ax = show(dem_data, transform=dem_transform, ax=ax, cmap='gray')  # Changed colormap to 'gray'
        if ax.images:
            plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
        plot_shape.boundary.plot(ax=ax, color='red', linewidth=2, label='Basin Boundary')

        if self.transect_buffer != 0 and self.footprint is not None:
            plot_shape.geometry.buffer(self.transect_buffer).plot(
                ax=ax, facecolor='none', edgecolor='red', linestyle='--', label='Buffered Boundary'
            )

        # Create a colormap for the transects
        num_transects = len(plot_transects)
        cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)
        
        # Plot each transect line with a different color based on index
        for i, line in enumerate(plot_transects.geometry):
            x, y = line.xy
            color = cmap(i)
            ax.plot(x, y, color=color, linewidth=1.5)

        center_x, center_y = plot_transects.geometry[0].xy[0][0], plot_transects.geometry[0].xy[1][0]
        ax.plot(center_x, center_y, 'yo', markersize=8, label='Radial Reference Point')

        plt.title(f"Radial Transects for {self.wetland_id} (Uniform: {uniform})")
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

    def plot_individual_radial_transects(self, uniform: bool = False):
        # skip theres no basin shape
        if self.footprint is None:
            print('Will not compute radial transects without prior basin')
            return
        
        if uniform:
            transects = self.truncated_transect_profiles
        else:
            transects = self.transect_profiles

        indexes = transects['trans_idx'].unique()

        num_transects = len(indexes)
        cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)

        for i in indexes:
            subset = transects[transects['trans_idx'] == i]
            plt.plot(subset['distance_m'], subset['dem_elevation'], label=f'Transect {i}', color=cmap(i))

        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation (m)")
        plt.title(f"Radial Transects -- Uniform z={uniform}")
        plt.legend()
        plt.show()

    def _hayashi_p_calculator(self, single_transect: pd.DataFrame, r0: int, r1: int) -> float:
        """
        Helper function to calculate Hayash P value on single transect
        used in calc_hayashi_p_defined_r() and calc_hayashi_p_uniform_z()
        """
        
        z0 = single_transect[single_transect['distance_m'] == r0]['depth_from_min'].mean()
        z1 = single_transect[single_transect['distance_m'] == r1]['depth_from_min'].mean()

        if np.isnan(z0) or np.isnan(z1):
            p = np.nan
        else:
            p = np.log(z1/z0) / np.log(r1/r0)

        result = {
            'trans_idx': single_transect['trans_idx'].iloc[0],
            'p': p
        }
        return result
    
    def calc_hayashi_p_defined_r(self, r0: int, r1: int) -> float:
        """
        Based on Hayashi et. al (2000)
        """
        transects = self.transect_profiles
        unique_idx = transects['trans_idx'].unique()

        hayashi_ps = []
        for i in unique_idx:
            trans = transects[transects['trans_idx'] == i].copy()
            min_elevation = trans['dem_elevation'].min()
            trans['depth_from_min'] = trans['dem_elevation'] - min_elevation
            
            result = self._hayashi_p_calculator(trans, r0, r1)

            hayashi_ps.append(result)

        return pd.DataFrame(hayashi_ps)
    
    def truncate_radial_transects_by_zmin(self):
        """
        Takes the original transects and adjust their radial distance to the spill elevation. 
        In this case, the spill elevation is the highest point on the lowest transect
        """

        transects = self.transect_profiles
        unique_idx = transects['trans_idx'].unique()
        transects_max_z = {}

        # Find the maximum z value relative to wetland bottom for each transect.
        for i in unique_idx:
            trans = transects[transects['trans_idx'] == i].copy()
            min_elevation = trans['dem_elevation'].min()
            trans['depth_from_min'] = trans['dem_elevation'] - min_elevation
            max_z = trans['depth_from_min'].max()
            transects_max_z[i] = max_z

        # Find the transect with the lowest z_max value
        min_idx = min(transects_max_z, key=transects_max_z.get)
        z_val = transects_max_z[min_idx]

        # Restrict each transect's radius (distance_m) whenever the z_val is first reached
        truncated_transects = []
        for i in unique_idx:
            trans = transects[transects['trans_idx'] == i].copy()
            min_elevation = trans['dem_elevation'].min()
            trans['depth_from_min'] = trans['dem_elevation'] - min_elevation

            # Ensure values sorted by distance from center
            trans = trans.sort_values(by='distance_m', ascending=True)

            # Find the first distance value where z_val is exceded
            msk = trans['depth_from_min'] >= z_val
            if msk.any():
                first_exceed_idx = msk.idxmax()
                truncated_trans = trans.loc[:first_exceed_idx]
                truncated_transects.append(truncated_trans)
            else:
                # if z_val isn't exceded, keep the entire transect
                truncated_transects.append(trans)

        return pd.concat(truncated_transects, ignore_index=True)
    
    def calc_hayashi_p_uniform_z(self, r0: int):

        transects = self.truncated_transect_profiles
        unique_idx = transects['trans_idx'].unique()

        hayashi_ps = []

        for i in unique_idx:
            trans = transects[transects['trans_idx'] == i]
            r_max = trans['distance_m'].max()
            p = self._hayashi_p_calculator(trans, r0=r0, r1=r_max)

            hayashi_ps.append(p)

        return pd.DataFrame(hayashi_ps)

    def plot_hayashi_p(self, r0: int, r1: int, uniform: bool = False):

        if self.footprint is None:
            print('Will not compute radial transects without prior basin')
            return  
        
        if uniform:
            df = self.calc_hayashi_p_uniform_z(r0=r0)
            r1 = 'max on transect'
        else:
            df = self.calc_hayashi_p_defined_r(r0, r1)

        plt.figure(figsize=(10, 6))
                
        num_transects = len(df)
        cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)
        
        colors = [cmap(i) for i in range(len(df))]
        plt.bar(df['trans_idx'], df['p'], color=colors)
        
        plt.xlabel("Transect Index")
        plt.ylabel("Hayashi p")
        plt.title(
            f"Basin {self.wetland_id} Hayashi p Constants for Radial Transects\n"
            f"Computed with r0={r0}, r1={r1}"
            f"Uniform z = {uniform}"
        )
        plt.show()

    def aggregate_radial_transects(self, uniform: bool = True) -> pd.DataFrame:
        """
        Aggregate transect profiles by distance from the center, averaging elevations
        across all transects at each distance.
        Returns a DataFrame with distance_m and summary stats.
        """
        if uniform:
            profiles = self.truncated_transect_profiles
        else:
            profiles = self.transect_profiles
            
        df = profiles[['distance_m', 'dem_elevation']].dropna()

        agg = (
            df.groupby('distance_m', as_index=False)['dem_elevation']
                .agg(n='count', mean='mean', std='std')
        )

        agg = agg[agg['n'] > 4]  # Filter out distances with only four transects available

        return agg.sort_values('distance_m', ignore_index=True)
    
    def plot_aggregated_radial_transects(self, uniform: bool = True):

        if self.footprint is None:
            print('Will not compute radial transects without prior basin')
            return

        if uniform:
            agg = self.aggregate_radial_transects(uniform=uniform)
        else:
            agg = self.aggregate_radial_transects(uniform=uniform)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(agg['distance_m'], agg['mean'], yerr=agg['std'], fmt='o')
        plt.xlabel("Distance (m)")
        plt.ylabel("Elevation (m)")
        plt.title(
            f"Aggregated Radial Transects for {self.wetland_id}\n"
            f"with n={self.transect_n} with basin shape buffer={self.transect_buffer}"
            f"z is uniform {uniform}"
        )
        plt.show()


