
from dataclasses import dataclass
from functools import cached_property

import richdem as rd
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS
import rasterio as rio
from rasterio.plot import show
from rasterio.mask import mask as rio_mask
from scipy.ndimage import gaussian_filter, label, binary_dilation
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
class SpillPoint:
    elevation: float
    location: gpd.GeoSeries

@dataclass
class SmoothedDEM:
    """Gaussian-smoothed DEM for robust spill-point detection."""
    dem: np.ndarray              # 2D smoothed array, np.nan outside footprint
    transform: Affine
    crs: CRS
    sigma: float                 # Gaussian sigma used (in pixels)

@dataclass
class WellPoint:
    elevation_dem: float # The well's elevatation - coords placed on DEM
    elevation_rtk: float # The well's elevation as measured by RTK GPS
    location: gpd.GeoSeries

@dataclass
class FilledDEM:
    """Filled DEM and per-cell fill depths."""
    filled: np.ndarray       # 2D filled DEM, np.nan outside footprint
    fill_depth: np.ndarray   # filled - original (>=0), np.nan outside footprint
    transform: Affine
    crs: CRS


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
    def spill_point(self) -> SpillPoint:
        return self.find_spill_point()
    @cached_property
    def spill_point_smoothed(self) -> SpillPoint:
        return self.find_spill_point_smoothed()
    @cached_property
    def spill_point_contiguous(self) -> float:
        return self.find_contiguous_spill_z()
    @cached_property
    def clipped_dem(self) -> ClippedDEM:
        return self.get_clipped_dem()
    @cached_property
    def smoothed_dem(self) -> SmoothedDEM:
        return self.get_smoothed_dem()
    @cached_property
    def local_fill(self) -> FilledDEM:
        return self.get_local_fill()
    
    # @cached_property
    # def radial_transects(self) -> gpd.GeoDataFrame:
    #     return self.establish_radial_transects(
    #         method=self.transect_method,
    #         n=self.transect_n,
    #         buffer_distance=self.transect_buffer
    #     )
    # @cached_property
    # def transect_profiles(self) -> pd.DataFrame:
    #     return self.find_radial_transects_vals()
    
    # @cached_property
    # def truncated_transect_profiles(self) -> pd.DataFrame:
    #     return self.truncate_radial_transects_by_zmin()
    
    # @cached_property
    # def aggregated_transect_profiles(self) -> pd.DataFrame:
    #     return self.aggregate_radial_transects_vals()

    def visualize_shape(
            self, 
            show_deepest: bool = False,
            show_centroid: bool = False,
            show_well: bool = False,
            show_spill: bool = False,
            show_smoothed_spill: bool = False,
            show_shape: bool = True
        ):


        if self.footprint is not None:
            plot_shape = self.footprint
            buffer_dist = self.transect_buffer
            bounds = self.footprint.buffer(buffer_dist).total_bounds
        else:
            well_point = self.well_point.location
            buffer_dist = self.transect_buffer if self.transect_buffer > 0 else 100
            buffered = well_point.buffer(buffer_dist)
            plot_shape = buffered
            bounds = buffered.total_bounds

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
            dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)
            dem_transform = rio.windows.transform(window, dem.transform)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        vmin, vmax = np.nanmin(dem_data), np.nanmax(dem_data)
        h, w = dem_data.shape
        extent = [dem_transform.c, dem_transform.c + w * dem_transform.a,
                  dem_transform.f + h * dem_transform.e, dem_transform.f]
        img = ax.imshow(np.ma.masked_invalid(dem_data), extent=extent, origin='upper',
                        cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        if ax.images:
                    cbar = plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
                    cbar.ax.tick_params(labelsize=12)
                    cbar.set_label('Elevation (m)', fontsize=16)
        if show_shape:
            plot_shape.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=2)

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
                print(len(WellPoint.location))
                print(WellPoint.location)
                WellPoint.location.plot(ax=ax, color='red', marker='X', markersize=250)
                # ax.annotate(f"DEM {WellPoint.elevation_dem:.2f}m", 
                #         xy=(WellPoint.location.x.values[0], WellPoint.location.y.values[0]),
                #         xytext=(10, 10), textcoords='offset points',
                #         color='white', fontweight='bold')

        if show_spill and self.footprint is not None:
            spill = self.spill_point
            spill.location.plot(ax=ax, color='magenta', marker='v', markersize=120)
            ax.annotate(f"Spill: {spill.elevation:.2f}m",
                xy=(spill.location.x.values[0], spill.location.y.values[0]),
                xytext=(10, 10), textcoords='offset points',
                color='white', fontweight='bold')

        if show_smoothed_spill and self.footprint is not None:
            spill_sm = self.spill_point_smoothed
            spill_sm.location.plot(ax=ax, color='cyan', marker='v', markersize=120)
            ax.annotate(f"Smoothed Spill: {spill_sm.elevation:.2f}m",
                xy=(spill_sm.location.x.values[0], spill_sm.location.y.values[0]),
                xytext=(10, -20), textcoords='offset points',
                color='cyan', fontweight='bold')

        plt.title(f"Wetland Basin: {self.wetland_id}")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
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
            src_crs = CRS.from_user_input(dem.crs)
            data, out_transform = rio_mask(
                dem, shape, crop=True, filled=False
            )
            band = data[0].astype("float64")
            nodata = dem.nodata

        # Convert out-of-shape mask to NaN even when source nodata is undefined.
        band = np.asarray(band.filled(np.nan), dtype="float64")
        if nodata is not None:
            band = np.where(band == nodata, np.nan, band)

        return ClippedDEM(
                dem=band,
                transform=out_transform,
                crs=src_crs,
                nodata=nodata
            )
    
    def find_deepest_point(self) -> DeepestPoint:
        """
        Find the minimum DEM value within the footprint based on a 5x5 cell average.
        Returns the location of the minimum 5x5 average with its actual elevation.
        """

        clipped = self.clipped_dem
        dem_data = clipped.dem.copy()

        # Mask DEM data to the original (unbuffered) basin footprint
        if self.footprint is not None:
            import rasterio.features
            footprint_geom = [self.footprint.geometry.values[0]]
            outside_mask = rasterio.features.geometry_mask(
                footprint_geom,
                out_shape=dem_data.shape,
                transform=clipped.transform,
                invert=False  # True where OUTSIDE the footprint
            )
            dem_data[outside_mask] = np.nan
        
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

            # Calculate 20th percentile, ignoring NaN values
            valid_vals = min_window[~np.isnan(min_window)]
            min_val = float(np.percentile(valid_vals, 25))

            return min_val

        min_val = _get_low_elevation(row, col)
        
        # Convert (row, col) to map x,y using the clipped transform
        x, y = rio.transform.xy(clipped.transform, row, col, offset="center")
        
        # return as GeoSeries in the DEM/footprint CRS
        pt = gpd.GeoSeries([Point(x, y)], crs=clipped.crs)

        return DeepestPoint(elevation=min_val, location=pt)

    def find_spill_point(self) -> SpillPoint:
        """
        Find the lowest point along the perimeter of the basin footprint.
        
        Traces the boundary at DEM-cell resolution, computes the 25th percentile
        elevation in a 5x5 window around each boundary cell, and returns the
        cell with the lowest such value.
        """
        if self.footprint is None:
            raise ValueError("Cannot find spill point without a basin footprint")

        clipped = self.clipped_dem
        dem_data = clipped.dem
        rows, cols = dem_data.shape

        # Get the boundary of the footprint as a LineString/MultiLineString
        boundary = self.footprint.geometry.values[0].boundary

        # Sample points along the boundary at the DEM cell resolution
        cell_size = abs(clipped.transform.a)
        step = cell_size  # sample at pixel spacing
        distances = np.arange(0, boundary.length, step)
        boundary_points = [boundary.interpolate(d) for d in distances]

        # Convert boundary points to row/col in the clipped DEM
        xs = [p.x for p in boundary_points]
        ys = [p.y for p in boundary_points]
        row_indices, col_indices = rio.transform.rowcol(clipped.transform, xs, ys)

        # Deduplicate to unique cells
        seen = set()
        unique_cells = []
        for r, c in zip(row_indices, col_indices):
            if (r, c) not in seen and 0 <= r < rows and 0 <= c < cols:
                seen.add((r, c))
                unique_cells.append((r, c))

        # For each boundary cell, compute the 25th percentile in a 5x5 window
        best_val = np.inf
        best_row, best_col = None, None
        for r, c in unique_cells:
            row_start = max(0, r - 2)
            row_end = min(rows, r + 3)
            col_start = max(0, c - 2)
            col_end = min(cols, c + 3)

            window = dem_data[row_start:row_end, col_start:col_end]
            valid_vals = window[~np.isnan(window)]
            if len(valid_vals) == 0:
                continue

            p25 = float(np.percentile(valid_vals, 25))
            if p25 < best_val:
                best_val = p25
                best_row, best_col = r, c

        if best_row is None:
            raise ValueError("No valid boundary cells found on the clipped DEM")

        x, y = rio.transform.xy(clipped.transform, best_row, best_col, offset="center")
        pt = gpd.GeoSeries([Point(x, y)], crs=clipped.crs)

        return SpillPoint(elevation=best_val, location=pt)

    def get_smoothed_dem(self, sigma: float = 5.0) -> SmoothedDEM:
        """
        Apply Gaussian smoothing to the clipped DEM.
        NaN cells are temporarily filled with the local nanmean so the
        filter doesn't propagate NaN, then re-masked afterward.
        """
        clipped = self.clipped_dem
        dem_data = clipped.dem.copy()
        nan_mask = np.isnan(dem_data)

        # Fill NaNs with the nanmean so gaussian_filter ignores gaps
        fill_value = np.nanmean(dem_data)
        dem_filled = np.where(nan_mask, fill_value, dem_data)

        smoothed = gaussian_filter(dem_filled, sigma=sigma)
        smoothed[nan_mask] = np.nan

        return SmoothedDEM(
            dem=smoothed,
            transform=clipped.transform,
            crs=clipped.crs,
            sigma=sigma,
        )

    def find_spill_point_smoothed(self) -> SpillPoint:
        """
        Find the spill point using a Gaussian-smoothed DEM for (x, y) location,
        then extract the z-value from the *unsmoothed* DEM using the 25th
        percentile in a 5x5 window.

        This avoids DEM noise / vegetation artifacts driving the spill location
        while preserving accurate elevations (e.g. ditches) for the z-value.
        """
        if self.footprint is None:
            raise ValueError("Cannot find spill point without a basin footprint")

        smoothed = self.smoothed_dem
        sm_data = smoothed.dem
        rows, cols = sm_data.shape

        # --- locate spill (x, y) on the smoothed surface ---
        boundary = self.footprint.geometry.values[0].boundary
        cell_size = abs(smoothed.transform.a)
        distances = np.arange(0, boundary.length, cell_size)
        boundary_points = [boundary.interpolate(d) for d in distances]

        xs = [p.x for p in boundary_points]
        ys = [p.y for p in boundary_points]
        row_indices, col_indices = rio.transform.rowcol(smoothed.transform, xs, ys)

        seen = set()
        unique_cells = []
        for r, c in zip(row_indices, col_indices):
            if (r, c) not in seen and 0 <= r < rows and 0 <= c < cols:
                seen.add((r, c))
                unique_cells.append((r, c))

        best_val = np.inf
        best_row, best_col = None, None
        for r, c in unique_cells:
            row_start = max(0, r - 2)
            row_end = min(rows, r + 3)
            col_start = max(0, c - 2)
            col_end = min(cols, c + 3)

            window = sm_data[row_start:row_end, col_start:col_end]
            valid_vals = window[~np.isnan(window)]
            if len(valid_vals) == 0:
                continue

            p25 = float(np.percentile(valid_vals, 25))
            if p25 < best_val:
                best_val = p25
                best_row, best_col = r, c

        if best_row is None:
            raise ValueError("No valid boundary cells found on the smoothed DEM")

        # --- extract z from the unsmoothed DEM ---
        raw_dem = self.clipped_dem.dem
        r_start = max(0, best_row - 2)
        r_end = min(rows, best_row + 3)
        c_start = max(0, best_col - 2)
        c_end = min(cols, best_col + 3)

        raw_window = raw_dem[r_start:r_end, c_start:c_end]
        raw_valid = raw_window[~np.isnan(raw_window)]
        if len(raw_valid) == 0:
            raise ValueError("No valid unsmoothed DEM cells in 5x5 window at spill location")
        spill_z = float(np.percentile(raw_valid, 25))

        x, y = rio.transform.xy(smoothed.transform, best_row, best_col, offset="center")
        pt = gpd.GeoSeries([Point(x, y)], crs=smoothed.crs)

        return SpillPoint(elevation=spill_z, location=pt)
    
    def find_contiguous_spill_z(self, min_flooded_area: float) -> float:
        """ 
        Finds the spill elevation such that a contiguous flooded surface reaches 
        from the deepest point to beyond the buffered basin footprint.
        Applies a min_flooded_area threshold to handle ditches on basin perimeters. 
        """
        if self.footprint is None:
            raise ValueError("footprint is required for find_contiguous_spill_z")

        clipped = self.clipped_dem
        dem_data = clipped.dem
        cell_size = abs(clipped.transform.a)
        pixel_area = cell_size ** 2
        nan_mask = np.isnan(dem_data)

        # Locate the deepest point in the clipped DEM grid
        deep_x = self.deepest_point.location.x.values[0]
        deep_y = self.deepest_point.location.y.values[0]
        deep_row, deep_col = rio.transform.rowcol(clipped.transform, deep_x, deep_y)

        z_min = self.deepest_point.elevation
        z_max = float(np.nanmax(dem_data))
        dz = 0.01

        while True:
            surface_z = z_min + dz

            if surface_z > z_max + 1.0:
                raise ValueError(
                    f"No spill elevation found for wetland '{self.wetland_id}': "
                    f"surface_z ({surface_z:.3f} m) exceeded DEM maximum ({z_max:.3f} m)."
                )

            # Binary flood mask: cells at or below surface_z (exclude nodata)
            flooded = (dem_data <= surface_z) & ~nan_mask

            # Label 4-connected components
            labeled, _ = label(flooded)

            # Component label at the deepest point
            comp_label = labeled[deep_row, deep_col]

            if comp_label != 0:
                component = labeled == comp_label
                area = float(np.sum(component)) * pixel_area

                # Connected to boundary = component is adjacent to a NaN cell
                # (NaN marks cells outside the clipped/buffered footprint extent)
                connected_to_boundary = bool(np.any(binary_dilation(component) & nan_mask))

                if area > min_flooded_area and connected_to_boundary:
                    return surface_z

            dz += 0.01

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
        Uses well point (lat, long, rtk_z) to establish a WellPoint.
        """

        if len(well_point_info) > 1:
            # Prefer rows with non-null rtk_z
            valid_rtk = well_point_info[well_point_info['rtk_z'].notna()]
            if len(valid_rtk) > 0:
                well_point_info = valid_rtk.iloc[[0]]
            else:
                well_point_info = well_point_info.iloc[[0]]

        if well_point_info.crs != self.footprint.crs:
            print("Warning: Well point CRS does not match footprint CRS. Reprojecting...")
            well_point_info = well_point_info.to_crs(self.footprint.crs)

        elevation_dem = self._find_point_elevation(well_point_info.geometry)

        if well_point_info['rtk_z'].values[0] is not None:
            rtk = float(well_point_info['rtk_z'].values[0])
        else:
            rtk = np.nan

        return WellPoint(
            elevation_dem=elevation_dem,
            elevation_rtk=rtk,
            location=well_point_info.geometry.iloc[[0]]
        )
    
    def _establish_well_point_from_source(self) -> WellPoint:
        """
        Establishes the well point using the original source DEM (before clipping)
        """
        if self.well_point_info is None:
            raise ValueError("well_point_info is required becuase no basin footprint is required")
        
        # Filter to single point if multiple rows exist
        well_point_info = self.well_point_info
        if len(well_point_info) > 1:
            # Prefer rows with non-null rtk_z
            valid_rtk = well_point_info[well_point_info['rtk_z'].notna()]
            if len(valid_rtk) > 0:
                well_point_info = valid_rtk.iloc[[0]]
            else:
                well_point_info = well_point_info.iloc[[0]]

        with rio.open(self.source_dem_path) as dem:
            x = self.well_point_info.geometry.x.values[0]
            y = self.well_point_info.geometry.y.values[0]
            # Convert to row/col in source DEM
            row, col = rio.transform.rowcol(dem.transform, x, y)
            window = rio.windows.Window(col-1, row-1, 3, 3)
            dem_data = dem.read(1, window=window)
            dem_data = np.where(dem_data==dem.nodata, np.nan, dem_data)
            elevation_dem = float(np.nanmean(dem_data))

        rtk_val = well_point_info['rtk_z'].values[0]
        if rtk_val is None or rtk_val == "Missing":
            rtk = np.nan
        else:
            rtk = float(rtk_val)
        
        return WellPoint(
            elevation_dem=elevation_dem,
            elevation_rtk=rtk,
            location=self.well_point_info.geometry.iloc[[0]]
        )
    
    def get_local_fill(self) -> FilledDEM:
        """
        Fill topographic depressions using the Wang & Liu algorithm (via richdem).
        Operates on the clipped DEM (buffered footprint domain).
        epsilon=False produces exact flat fills — spill elevation equals the lowest
        outlet cell, with no artificial gradient added inside the depression.
        """

        clipped = self.clipped_dem
        dem_data = clipped.dem.copy()
        nan_mask = np.isnan(dem_data)

        _NODATA = -9999.0
        dem_input = np.where(nan_mask, _NODATA, dem_data)

        rda = rd.rdarray(dem_input, no_data=_NODATA)
        filled_rda = rd.FillDepressions(rda, epsilon=False, in_place=False)

        filled_arr = np.array(filled_rda, dtype=np.float64)
        filled_arr[nan_mask] = np.nan

        fill_depth = filled_arr - dem_data
        fill_depth[nan_mask] = np.nan

        return FilledDEM(
            filled=filled_arr,
            fill_depth=fill_depth,
            transform=clipped.transform,
            crs=clipped.crs,
        )

    def well_fill_depth(self) -> float:
        """
        Return the mean fill depth (filled - original DEM) in a 5x5 window
        centred on the well point location.
        """
        fill = self.local_fill
        well = self.well_point

        x = well.location.x.values[0]
        y = well.location.y.values[0]
        row, col = rio.transform.rowcol(fill.transform, x, y)

        rows, cols = fill.fill_depth.shape
        r_start = max(0, row - 2)
        r_end   = min(rows, row + 3)
        c_start = max(0, col - 2)
        c_end   = min(cols, col + 3)

        window = fill.fill_depth[r_start:r_end, c_start:c_end]
        valid  = window[~np.isnan(window)]

        if len(valid) == 0:
            return np.nan

        return float(np.mean(valid))

    def max_fill_depth(self) -> tuple[float, float]:
        """
        Return the maximum fill depth on the FilledDEM and the corresponding
        raw DEM elevation at that location.

        Uses the same sliding-window IQR-mean approach as find_deepest_point()
        to locate the winning cell (highest IQR-mean fill depth across all 5x5
        windows).  For the winning window the reported values are:
          - fill_depth_p75 : 75th percentile of fill_depth in the 5x5 window
          - dem_elevation_p25 : 25th percentile of the original DEM in the same
                                5x5 window (mirrors _get_low_elevation())

        Returns
        -------
        (fill_depth_p75, dem_elevation_p25) : tuple[float, float]
        """
        fill = self.local_fill
        depth = fill.fill_depth.copy()
        raw_dem = self.clipped_dem.dem

        rows, cols = depth.shape
        scores = np.full_like(depth, np.nan)

        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                window = depth[i - 2:i + 3, j - 2:j + 3].flatten()
                valid = window[~np.isnan(window)]
                if len(valid) >= 4:
                    q25, q75 = np.nanpercentile(valid, [25, 75])
                    iqr_vals = valid[(valid >= q25) & (valid <= q75)]
                    scores[i, j] = np.mean(iqr_vals)

        row, col = np.unravel_index(np.nanargmax(scores), scores.shape)

        r_start = max(0, row - 2)
        r_end   = min(rows, row + 3)
        c_start = max(0, col - 2)
        c_end   = min(cols, col + 3)

        win_depth = depth[r_start:r_end, c_start:c_end].flatten()
        valid_depth = win_depth[~np.isnan(win_depth)]
        fill_depth_p75 = float(np.percentile(valid_depth, 75))

        win_dem = raw_dem[r_start:r_end, c_start:c_end].flatten()
        valid_dem = win_dem[~np.isnan(win_dem)]
        dem_elevation_p25 = float(np.percentile(valid_dem, 25))

        return fill_depth_p75, dem_elevation_p25

    def plot_local_fill(self):
        """
        Render a viridis map of depression fill depths (filled - original DEM).
        Cells outside depressions show 0; depressed cells show depth to spill.
        Well location marked as a red X.
        """
        fill = self.local_fill

        fig, ax = plt.subplots(figsize=(10, 8))
        vmin, vmax = np.nanmin(fill.fill_depth), np.nanmax(fill.fill_depth)
        h, w = fill.fill_depth.shape
        extent = [fill.transform.c, fill.transform.c + w * fill.transform.a,
                  fill.transform.f + h * fill.transform.e, fill.transform.f]
        ax.imshow(np.ma.masked_invalid(fill.fill_depth), extent=extent, origin='upper',
                  cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        if ax.images:
            cbar = plt.colorbar(ax.images[0], ax=ax)
            cbar.set_label('Fill Depth (m)', fontsize=14)

        well = self.well_point
        well.location.plot(ax=ax, color='red', marker='x', markersize=125,
                           label=f"Well ({well.elevation_dem:.2f}m)")

        ax.legend(loc='upper right')
        ax.set_title(f"{self.wetland_id} — Depression Fill Depths")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        plt.show()

    def calculate_hypsometry(self, method: str = "total_cdf"):
        step = 0.001 # NOTE: Hardcoded this for now
        dem_data = self.clipped_dem.dem
        flat_dem = dem_data.flatten()
        dem_scale = self.clipped_dem.transform.a 
        min_elevation = np.nanmin(dem_data)
        max_elevation = np.nanmax(dem_data)

        if method == "total_cdf":
            bins = np.arange(min_elevation, max_elevation + step, step)
            hist, bin_edges = np.histogram(flat_dem, bins=bins)
            cum_area_m2 = np.cumsum(hist) * (dem_scale ** 2)  
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            return cum_area_m2, bin_centers
        
        if method == "pct_trim_cdf":
            p_low, p_high = np.nanpercentile(flat_dem, [2, 98])
            flat_dem = flat_dem[(flat_dem >= p_low) & (flat_dem <= p_high)]
            bins = np.arange(flat_dem.min(), flat_dem.max() + step, step)
            hist, bin_edges = np.histogram(flat_dem, bins=bins)
            cum_area_m2 = np.cumsum(hist) * (dem_scale ** 2)  
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            return cum_area_m2, bin_centers
        
        if method == "pct_trim_pdf":
            p_low, p_high = np.nanpercentile(flat_dem, [2, 98])
            flat_dem = flat_dem[(flat_dem >= p_low) & (flat_dem <= p_high)]
            bins = np.arange(flat_dem.min(), flat_dem.max() + step, step)
            hist, bin_edges = np.histogram(flat_dem, bins=bins)
            area = hist * (dem_scale ** 2)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            return area, bin_centers
    
        else:
            print(f"Method '{method}' not implemented for hypsometry calculation.")
            return None, None

    def plot_basin_hypsometry(
            self,
            plot_points: bool = False,
            plot_spill: bool = False,
            plot_smoothed_spill: bool = False, 
            plot_deepest: bool = False,
            plot_contiguous_spill: bool = False,
            min_flooded_area: float = 0.0
        ):

        cum_area_m2, bin_centers = self.calculate_hypsometry()
        plt.figure(figsize=(10, 6))
        plt.plot(bin_centers, cum_area_m2, label="Inundated Area", color="blue")

        if plot_points:
            # Add well point elevations if available
            well_pt = self.well_point
            
            # Interpolate cumulative area at well point elevations
            dem_area = np.interp(well_pt.elevation_dem, bin_centers, cum_area_m2)
            # rtk_area = np.interp(well_pt.elevation_rtk, bin_centers, cum_area_m2)

            plt.plot(well_pt.elevation_dem, dem_area, 'ro', markersize=8,
                     label=f"Well DEM Elevation ({well_pt.elevation_dem:.2f}m, {dem_area:.2f}m^2)")
            # if abs(well_pt.elevation_rtk - well_pt.elevation_dem) > 0.25:
            #     print(f"Warning: RTK elevation ({well_pt.elevation_rtk:.2f}m) differs from DEM elevation ({well_pt.elevation_dem:.2f}m) by more than 0.25m.")
            # else:
            #     plt.plot(well_pt.elevation_rtk, rtk_area, 'go', markersize=8,
            #             label=f"Well RTK Elevation ({well_pt.elevation_rtk:.2f}m, {rtk_area:.2f}m^2)")

        if plot_spill and self.footprint is not None:
            spill = self.spill_point
            plt.axvline(x=spill.elevation, color='magenta', linestyle='--', linewidth=3.5,
                        label=f"Spill Elevation ({spill.elevation:.2f}m)")

        if plot_deepest:
            deepest = self.deepest_point
            deepest_area = np.interp(deepest.elevation, bin_centers, cum_area_m2)
            plt.plot(deepest.elevation, deepest_area, 'b*', markersize=12,
                     label=f"Deepest Point ({deepest.elevation:.2f}m, {deepest_area:.2f}m\u00b2)")
            
        if plot_smoothed_spill and self.footprint is not None:
            spill_smoothed = self.spill_point_smoothed
            plt.axvline(x=spill_smoothed.elevation, color='cyan', linestyle=':', linewidth=3.5,
                        label=f"Smoothed Spill Elevation ({spill_smoothed.elevation:.2f}m)")
            
        if plot_contiguous_spill and self.footprint is not None:
            contiguous_spill_z = self.find_contiguous_spill_z(min_flooded_area=min_flooded_area)
            plt.axvline(x=contiguous_spill_z, color='green', linestyle='-.', linewidth=3.5,
                        label=f"Contiguous Spill Elevation ({contiguous_spill_z:.2f}m)")

        plt.xlabel("Elevation (m)")
        plt.ylabel("Inundated Area (m^2)")
        plt.title(f"{self.wetland_id} Hypsometry")
        plt.grid()
        plt.legend()
        plt.show()

    def map_spill_inundation(
            self,
            use_smoothed: bool = True,
            plot_contiguous_spill: bool = False,
            min_flooded_area: float = 0.0
        ):
        """
        Map the inundation extent when water level equals the spill elevation.
        Renders a greyscale DEM with a semi-transparent blue overlay for inundated cells.

        Parameters
        ----------
        use_smoothed : bool
            If True (default), use the smoothed spill point and render the
            smoothed DEM as the background.
        plot_contiguous_spill : bool
            If True, overlay the contiguous flooded region at the contiguous spill
            elevation as a semi-transparent green layer.
        min_flooded_area : float
            Minimum flooded area (m²) passed to find_contiguous_spill_z().
        """
        if self.footprint is None:
            raise ValueError("Cannot map spill inundation without a basin footprint")

        if use_smoothed:
            spill = self.spill_point_smoothed
            display_dem = self.smoothed_dem.dem
            display_transform = self.smoothed_dem.transform
        else:
            spill = self.spill_point
            display_dem = self.clipped_dem.dem
            display_transform = self.clipped_dem.transform

        # Inundation mask uses the *unsmoothed* DEM for accuracy
        raw_dem = self.clipped_dem.dem
        inundation = np.where(
            ~np.isnan(raw_dem) & (raw_dem <= spill.elevation), 1.0, np.nan
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        # Greyscale DEM
        vmin, vmax = np.nanmin(display_dem), np.nanmax(display_dem)
        h_d, w_d = display_dem.shape
        t_d = display_transform
        extent_bg = [t_d.c, t_d.c + w_d * t_d.a, t_d.f + h_d * t_d.e, t_d.f]
        ax.imshow(np.ma.masked_invalid(display_dem), extent=extent_bg, origin='upper',
                  cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        if ax.images:
            plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')

        # Compute geographic extent for overlays (raw DEM grid).
        h, w = raw_dem.shape
        t = self.clipped_dem.transform
        overlay_extent = [t.c, t.c + w * t.a, t.f + h * t.e, t.f]  # [xmin, xmax, ymin, ymax]

        # Navy inundation overlay: transparent where NaN, navy where 1
        # navy_cmap = LinearSegmentedColormap.from_list(
        #     'transparent_navy', [(0, (0, 0, 0, 0)), (1, (0, 0, 0.502, 0.7))]
        # )
        # navy_cmap.set_bad(alpha=0)
        # ax.imshow(
        #     np.ma.masked_invalid(inundation),
        #     extent=overlay_extent, origin='upper',
        #     cmap=navy_cmap, vmin=0, vmax=1,
        #     interpolation='nearest', aspect='auto'
        # )

        # Contiguous spill overlay
        if plot_contiguous_spill:
            contiguous_z = self.find_contiguous_spill_z(min_flooded_area=min_flooded_area)
            from scipy.ndimage import label as nd_label

            deep_x = self.deepest_point.location.x.values[0]
            deep_y = self.deepest_point.location.y.values[0]
            deep_row, deep_col = rio.transform.rowcol(self.clipped_dem.transform, deep_x, deep_y)

            flooded = (raw_dem <= contiguous_z) & ~np.isnan(raw_dem)
            labeled, _ = nd_label(flooded)
            comp_label = labeled[deep_row, deep_col]
            contiguous_mask = np.where(labeled == comp_label, 1.0, np.nan) if comp_label != 0 else np.full_like(raw_dem, np.nan)

            green_cmap = LinearSegmentedColormap.from_list(
                'transparent_green', [(0, (0, 0, 0, 0)), (1, (0, 0.6, 0, 0.55))]
            )
            green_cmap.set_bad(alpha=0)
            ax.imshow(
                np.ma.masked_invalid(contiguous_mask),
                extent=overlay_extent, origin='upper',
                cmap=green_cmap, vmin=0, vmax=1,
                interpolation='nearest', aspect='auto'
            )

        # Mark spill point
        spill.location.plot(ax=ax, color='magenta', marker='v', markersize=120,
                            label=f"Spill Point ({spill.elevation:.2f}m)")

        # Mark deepest point
        deepest = self.deepest_point
        deepest.location.plot(ax=ax, color='blue', marker='*', markersize=100,
                              label=f"Deepest Point ({deepest.elevation:.2f}m)")
        
        # Mark the well location
        well_point = self.well_point
        well_point.location.plot(ax=ax, color='red', marker='x', markersize=125,
                                 label=f"Well location ({well_point.elevation_dem:.2f}")

        # if plot_contiguous_spill:
        #     ax.axhline(y=np.nan, color='green', linestyle='-.', linewidth=2,
        #                label=f"Contiguous Spill ({contiguous_z:.2f}m)")

        ax.legend(loc='upper right')
        ax.set_title(f"{self.wetland_id} Inundation at Spill Elevation ({spill.elevation:.2f}m)")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        plt.tight_layout()
        plt.show()

"""
NOTE: this is old code from back when we were interested in TAI curvature metrics. 
"""

    # def establish_radial_transects(
    #         self, 
    #         method: str = 'deepest',
    #         n: int = 8,
    #         buffer_distance: float = 0
    #     ) -> gpd.GeoDataFrame:

    #     poly = self.footprint.geometry.values[0]
        
    #     target_poly = poly if buffer_distance == 0 else poly.buffer(buffer_distance)

    #     if method == 'deepest':
    #         center_pt: Point = self.deepest_point.location.values[0]
    #     elif method == 'centroid':
    #         center_pt: Point = self.footprint.centroid.values[0]
    #     else:
    #         raise ValueError(f"Method '{method}' not recognized. Use 'centroid' or 'deepest'.")

    #     angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    #     records = []
    #     lines = []

    #     for i in angles:
    #         dx, dy = np.cos(i), np.sin(i)
    #         far_pt = Point(
    #             center_pt.x + dx * 1000,  # Extend 1000m in the direction of angle
    #             center_pt.y + dy * 1000
    #         )

    #         ray = LineString([center_pt, far_pt])
    #         inter = ray.intersection(target_poly.boundary)

    #         if inter.geom_type == 'Point':
    #             end_pt = inter
    #         elif inter.geom_type == 'MultiPoint':
    #             pts = [p for p in getattr(inter, 'geoms', []) if p.geom_type == 'Point']
    #             end_pt = max(pts, key=lambda p: p.distance(center_pt))
    #         else:
    #             print('Error: Radial Transect is not a Point or MultiPoint')
    #             continue

    #         line = LineString([center_pt, end_pt])
    #         lines.append(line)
    #         records.append({
    #             'angle_rad': i,
    #             'length_m': float(center_pt.distance(end_pt)),
    #             'geometry': line,
    #         })

    #     return gpd.GeoDataFrame(records, geometry=lines, crs=self.footprint.crs)

    # def radial_transects_map(self, uniform: bool = False):

    #     # Don't compute transects without basin footprint
    #     if self.footprint is None:
    #         print('Will not compute radial transects without prior basin')
    #         return
        
    #     plot_shape = self.footprint
    #     bounds = self.footprint.total_bounds
    #     buffer_bounds = [
    #         bounds[0] - 100,  # minx
    #         bounds[1] - 100,  # miny
    #         bounds[2] + 100,  # maxx
    #         bounds[3] + 100   # maxy
    #     ]
    #     with rio.open(self.source_dem_path) as dem:
    #         # Create window from bounds
    #         window = rio.windows.from_bounds(*buffer_bounds, dem.transform)
    #         dem_data = dem.read(1, window=window)
    #         dem_data = np.where(dem_data == dem.nodata, np.nan, dem_data)  
    #         dem_transform = rio.windows.transform(window, dem.transform)

    #     if uniform:
    #         # A bit of a hacky strategy...
    #         # Clip the geometries in radial_transects (gdf) by distance values in truncated_profiles (df)
    #         # Render a new gdf with the truncated geometries.
    #         truncated_profiles = self.truncated_transect_profiles
    #         original_transects = self.radial_transects
    #         max_distances = truncated_profiles.groupby('angle_rad')['distance_m'].max().reset_index()
    #         truncated_geoms = []
    #         for _, row in max_distances.iterrows():
    #             angle = row['angle_rad']
    #             max_dist = row['distance_m']

    #             # Get the original transect for this angle
    #             original_transect = original_transects[original_transects['angle_rad'] == angle].iloc[0]
    #             original_line = original_transect['geometry']

    #             # Truncate the line at the max distance
    #             if max_dist < original_line.length:
    #                 truncated_line = LineString([
    #                     original_line.coords[0],  # Start point (center)
    #                     original_line.interpolate(max_dist)  # End point at max distance
    #                 ])
    #             else:
    #                 truncated_line = original_line  # Keep original if max_dist exceeds line length
                
    #             truncated_geoms.append({
    #                 'angle_rad': angle,
    #                 'length_m': max_dist,
    #                 'geometry': truncated_line
    #             })
    #         plot_transects = gpd.GeoDataFrame(truncated_geoms, crs=self.footprint.crs)

    #     else:
    #         plot_transects = self.radial_transects

    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     ax = show(dem_data, transform=dem_transform, ax=ax, cmap='gray')  # Changed colormap to 'gray'
    #     if ax.images:
    #         plt.colorbar(ax.images[0], ax=ax, label='Elevation (m)')
    #     plot_shape.boundary.plot(ax=ax, color='red', linewidth=2, label='Basin Boundary')

    #     if self.transect_buffer != 0 and self.footprint is not None:
    #         plot_shape.geometry.buffer(self.transect_buffer).plot(
    #             ax=ax, facecolor='none', edgecolor='red', linestyle='--', label='Buffered Boundary'
    #         )

    #     # Create a colormap for the transects
    #     num_transects = len(plot_transects)
    #     cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)
        
    #     # Plot each transect line with a different color based on index
    #     for i, line in enumerate(plot_transects.geometry):
    #         x, y = line.xy
    #         color = cmap(i)
    #         ax.plot(x, y, color=color, linewidth=1.5)

    #     center_x, center_y = plot_transects.geometry[0].xy[0][0], plot_transects.geometry[0].xy[1][0]
    #     ax.plot(center_x, center_y, 'yo', markersize=8, label='Radial Reference Point')

    #     plt.title(f"Radial Transects for {self.wetland_id} (Uniform: {uniform})")
    #     plt.xlabel("x (meters)")
    #     plt.ylabel("y (meters)")
    #     plt.legend()
    #     plt.show()

    # def sample_transect_dem_vals(
    #     self,
    #     line: LineString,
    #     step: float
    # ) -> pd.DataFrame:
        
    #     if step is None:
    #         print('tbd figure this out later')

    #     # Generate Points along the line
    #     total_length = line.length
    #     distances = np.arange(0, total_length + step, step)
    #     points = [line.interpolate(d) for d in distances]
    #     coords = [(p.x, p.y) for p in points]

    #     dem_data = self.clipped_dem.dem
    #     transform = self.clipped_dem.transform

    #     rows, cols = rio.transform.rowcol(
    #         transform, 
    #         [p[0] for p in coords], 
    #         [p[1] for p in coords]
    #     )

    #     dem_rows, dem_cols = dem_data.shape
    #     elevations = []
    #     for r, c in zip(rows, cols):
    #         if 0 <= r < dem_rows and 0 <= c < dem_cols:
    #             row_start = max(0, r-1)
    #             row_end = min(dem_rows, r+2)
    #             col_start = max(0, c-1)
    #             col_end = min(dem_cols, c+2)
    #             window = dem_data[row_start:row_end, col_start:col_end]
                
    #             if not np.all(np.isnan(window)):
    #                 elev = np.nanmean(window)
    #             else:
    #                 elev = np.nan
                
    #             # Store list of elevations
    #             elevations.append(elev)

    #     return pd.DataFrame({
    #         'distance_m': distances,
    #         'dem_elevation': elevations
    #     })
            
    # def find_radial_transects_vals(self):
    #     """
    #     Find the elevation values along the radial transects.
    #     """
    #     transects = self.radial_transects
    #     dfs = []
    #     for idx, row in transects.iterrows():
    #         profile = self.sample_transect_dem_vals(
    #             row['geometry'], 
    #             step=1.0  # 1 meter step
    #         )
    #         profile['angle_rad'] = row['angle_rad']
    #         profile['trans_idx'] = idx
    #         profile['length_m'] = row['length_m']
    #         dfs.append(profile)

    #     return pd.concat(dfs, ignore_index=True)

    # def plot_individual_radial_transects(self, uniform: bool = False):
    #     # skip theres no basin shape
    #     if self.footprint is None:
    #         print('Will not compute radial transects without prior basin')
    #         return
        
    #     if uniform:
    #         transects = self.truncated_transect_profiles
    #     else:
    #         transects = self.transect_profiles

    #     indexes = transects['trans_idx'].unique()

    #     num_transects = len(indexes)
    #     cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)

    #     for i in indexes:
    #         subset = transects[transects['trans_idx'] == i]
    #         plt.plot(subset['distance_m'], subset['dem_elevation'], label=f'Transect {i}', color=cmap(i))

    #     plt.xlabel("Distance (m)")
    #     plt.ylabel("Elevation (m)")
    #     plt.title(f"Radial Transects -- Uniform z={uniform}")
    #     plt.legend()
    #     plt.show()

    # def _hayashi_p_calculator(self, single_transect: pd.DataFrame, r0: int, r1: int) -> float:
    #     """
    #     Helper function to calculate Hayash P value on single transect
    #     used in calc_hayashi_p_defined_r() and calc_hayashi_p_uniform_z()
    #     """
        
    #     z0 = single_transect[single_transect['distance_m'] == r0]['depth_from_min'].mean()
    #     z1 = single_transect[single_transect['distance_m'] == r1]['depth_from_min'].mean()

    #     if np.isnan(z0) or np.isnan(z1):
    #         p = np.nan
    #     else:
    #         p = np.log(z1/z0) / np.log(r1/r0)

    #     result = {
    #         'trans_idx': single_transect['trans_idx'].iloc[0],
    #         'p': p
    #     }
    #     return result
    
    # def calc_hayashi_p_defined_r(self, r0: int, r1: int) -> float:
    #     """
    #     Based on Hayashi et. al (2000)
    #     """
    #     transects = self.transect_profiles
    #     unique_idx = transects['trans_idx'].unique()

    #     hayashi_ps = []
    #     for i in unique_idx:
    #         trans = transects[transects['trans_idx'] == i].copy()
    #         min_elevation = trans['dem_elevation'].min()
    #         trans['depth_from_min'] = trans['dem_elevation'] - min_elevation
            
    #         result = self._hayashi_p_calculator(trans, r0, r1)

    #         hayashi_ps.append(result)

    #     return pd.DataFrame(hayashi_ps)
    
    # def truncate_radial_transects_by_zmin(self):
    #     """
    #     Takes the original transects and adjust their radial distance to the spill elevation. 
    #     In this case, the spill elevation is the highest point on the lowest transect
    #     """

    #     transects = self.transect_profiles
    #     unique_idx = transects['trans_idx'].unique()
    #     transects_max_z = {}

    #     # Find the maximum z value relative to wetland bottom for each transect.
    #     for i in unique_idx:
    #         trans = transects[transects['trans_idx'] == i].copy()
    #         min_elevation = trans['dem_elevation'].min()
    #         trans['depth_from_min'] = trans['dem_elevation'] - min_elevation
    #         max_z = trans['depth_from_min'].max()
    #         transects_max_z[i] = max_z

    #     # Find the transect with the lowest z_max value
    #     min_idx = min(transects_max_z, key=transects_max_z.get)
    #     z_val = transects_max_z[min_idx]

    #     # Restrict each transect's radius (distance_m) whenever the z_val is first reached
    #     truncated_transects = []
    #     for i in unique_idx:
    #         trans = transects[transects['trans_idx'] == i].copy()
    #         min_elevation = trans['dem_elevation'].min()
    #         trans['depth_from_min'] = trans['dem_elevation'] - min_elevation

    #         # Ensure values sorted by distance from center
    #         trans = trans.sort_values(by='distance_m', ascending=True)

    #         # Find the first distance value where z_val is exceded
    #         msk = trans['depth_from_min'] >= z_val
    #         if msk.any():
    #             first_exceed_idx = msk.idxmax()
    #             truncated_trans = trans.loc[:first_exceed_idx]
    #             truncated_transects.append(truncated_trans)
    #         else:
    #             # if z_val isn't exceded, keep the entire transect
    #             truncated_transects.append(trans)

    #     return pd.concat(truncated_transects, ignore_index=True)
    
    # def calc_hayashi_p_uniform_z(self, r0: int):

    #     transects = self.truncated_transect_profiles
    #     unique_idx = transects['trans_idx'].unique()

    #     hayashi_ps = []

    #     for i in unique_idx:
    #         trans = transects[transects['trans_idx'] == i]
    #         r_max = trans['distance_m'].max()
    #         p = self._hayashi_p_calculator(trans, r0=r0, r1=r_max)

    #         hayashi_ps.append(p)

    #     return pd.DataFrame(hayashi_ps)

    # def plot_hayashi_p(self, r0: int, r1: int, uniform: bool = False):

    #     if self.footprint is None:
    #         print('Will not compute radial transects without prior basin')
    #         return  
        
    #     if uniform:
    #         df = self.calc_hayashi_p_uniform_z(r0=r0)
    #         r1 = 'max on transect'
    #     else:
    #         df = self.calc_hayashi_p_defined_r(r0, r1)

    #     plt.figure(figsize=(10, 6))
                
    #     num_transects = len(df)
    #     cmap = plt.cm.get_cmap('tab20' if num_transects <= 20 else 'viridis', num_transects)
        
    #     colors = [cmap(i) for i in range(len(df))]
    #     plt.bar(df['trans_idx'], df['p'], color=colors)
        
    #     plt.xlabel("Transect Index")
    #     plt.ylabel("Hayashi p")
    #     plt.title(
    #         f"Basin {self.wetland_id} Hayashi p Constants for Radial Transects\n"
    #         f"Computed with r0={r0}, r1={r1}"
    #         f"Uniform z = {uniform}"
    #     )
    #     plt.show()

    # def aggregate_radial_transects(self, uniform: bool = True) -> pd.DataFrame:
    #     """
    #     Aggregate transect profiles by distance from the center, averaging elevations
    #     across all transects at each distance.
    #     Returns a DataFrame with distance_m and summary stats.
    #     """
    #     if uniform:
    #         profiles = self.truncated_transect_profiles
    #     else:
    #         profiles = self.transect_profiles
            
    #     df = profiles[['distance_m', 'dem_elevation']].dropna()

    #     agg = (
    #         df.groupby('distance_m', as_index=False)['dem_elevation']
    #             .agg(n='count', mean='mean', std='std')
    #     )

    #     agg = agg[agg['n'] > 4]  # Filter out distances with only four transects available

    #     return agg.sort_values('distance_m', ignore_index=True)
    
    # def plot_aggregated_radial_transects(self, uniform: bool = True):

    #     if self.footprint is None:
    #         print('Will not compute radial transects without prior basin')
    #         return

    #     if uniform:
    #         agg = self.aggregate_radial_transects(uniform=uniform)
    #     else:
    #         agg = self.aggregate_radial_transects(uniform=uniform)
        
    #     plt.figure(figsize=(10, 6))
    #     plt.errorbar(agg['distance_m'], agg['mean'], yerr=agg['std'], fmt='o')
    #     plt.xlabel("Distance (m)")
    #     plt.ylabel("Elevation (m)")
    #     plt.title(
    #         f"Aggregated Radial Transects for {self.wetland_id}\n"
    #         f"with n={self.transect_n} with basin shape buffer={self.transect_buffer}"
    #         f"z is uniform {uniform}"
    #     )
    #     plt.show()


