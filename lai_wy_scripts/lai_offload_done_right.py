# %% Libraries and file paths

import ee 
import geemap
import geopandas as gpd
import pandas as pd

ee.Authenticate()
ee.Initialize()

target_wetlands_path = f'D:/depressional_lidar/data/bradford/in_data/bradford_basins_assigned_wetland_ids_KG.shp'
target_wetlands = gpd.read_file(target_wetlands_path)
target_wetlands = target_wetlands[['wetland_id', 'geometry']]
target_wetlands['wetland_id'] = target_wetlands['wetland_id'].str.replace('/', '.')
print(target_wetlands.crs)

other_wetlands_path = f'D:/depressional_lidar/data/bradford/in_data/original_basins/bradford_original_depressions_all_wetlands.shp'
other_wetlands = gpd.read_file(other_wetlands_path)
all_wetlands = gpd.GeoDataFrame(
    pd.concat([target_wetlands.geometry, other_wetlands.geometry]),
    crs=target_wetlands.crs)

all_wetlands.to_crs('EPSG:4326', inplace=True)

well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
well_points = (gpd.read_file(well_points_path)[['wetland_id', 'type', 'site', 'geometry']]
               .query("type in ['core_well', 'wetland_well'] and site == 'bradford'")
)
well_points['wetland_id'] = well_points['wetland_id'].str.replace('/', '.')


# %% Functions for earth engine

def convert_gpd_geom_to_ee(geom, est_utm):
    """
    Takes a geopandas geom object and coverts it to an Earth Engine polygon
    """
    if est_utm is None:
        out_crs = 'EPSG:4326'
    else:
        out_crs = est_utm

    coords = list(geom.exterior.coords)
    coords_list = [[x, y] for x, y in coords]

    return ee.Geometry.Polygon(coords_list, proj=out_crs)

def make_upland_mask(wetland_shapes: gpd.GeoDataFrame):
    """
    Uses a GeoDataFrame to mask the wetlands from the upland LAI calculations
    Returns an ee.Image() from the wetlands shapes.
    """
    ee_geoms = []
    for idx, row in wetland_shapes.iterrows():
        try:
            ee_geom = convert_gpd_geom_to_ee(row['geometry'], est_utm='EPSG:4326')
            ee_geoms.append(ee_geom)
        except:
            print(f"could not resovle geometry {idx}")
            continue # skipping problematic geometries
    
    ee_features = [ee.Feature(geom) for geom in ee_geoms]
    wetlands_fc = ee.FeatureCollection(ee_features)

    wetland_mask = ee.Image(0).byte().paint(
        featureCollection=wetlands_fc,
        color=1  # Wetland pixels get value 1
    )    
    # Create upland mask (inverse of wetland mask)
    upland_mask = wetland_mask.Not()
    
    return upland_mask

def make_nlcd_mask(polygon):

    nlcd = (ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019")
            .clip(polygon)
            .select('landcover')
    )
    nlcd_keep_mask = (
        nlcd.eq(41) # Decidous Forest
        .Or(nlcd.eq(42)) # Evergreen Forest
        .Or(nlcd.eq(43)) # Mixed Forest
        .Or(nlcd.eq(52)) # Shrub/Scrub
        .Or(nlcd.eq(71)) # Grassland/Herbaceous
        .Or(nlcd.eq(90)) # Woody Wetlands
        .Or(nlcd.eq(95)) # Emergent Herbaceaous Wetlands NOTE: some contention about whether to keep this one. 
    )

    return nlcd_keep_mask

def apply_scale_factors(image):
    optical = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True)

def rename_ls8_bands(image):
    bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5']
    new_names = ['B', 'G', 'R', 'NIR']
    return image.select(bands).rename(new_names)

def mask_clouds(image):
    qa = image.select('QA_PIXEL')
    cloudsBitMask = (1 << 3)
    cloudShadowBitMask = (1 << 4)
    cirrusBitMask = (1 << 2)
    dilatedBitMask = (1 << 1)
    mask = (qa.bitwiseAnd(cloudsBitMask).eq(0)
            .And(qa.bitwiseAnd(cloudShadowBitMask).eq(0))
            .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
            .And(qa.bitwiseAnd(dilatedBitMask).eq(0)))
    return image.updateMask(mask)
    
def make_ls8_collection(polygon, start_date, end_date):

    ls8_full = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")

    ls8 = (ls8_full
        .filterDate(start_date, end_date)
        .filterMetadata('CLOUD_COVER', 'less_than', 90)
        .filterBounds(polygon)
        .map(lambda img: img.clip(polygon))
        .map(lambda img: img.set('year', img.date().get('year')))
        .map(lambda img: img.set('month', img.date().get('month')))
        .map(mask_clouds)
        .map(apply_scale_factors)
        .map(rename_ls8_bands)
    )

    return ls8

def calc_lai(image):
        SR = image.select('NIR').divide(image.select('R'))
        LAI = image.expression('(0.332915 * SR) - 0.00212', {'SR': SR}).rename('LAI').clamp(0, 10)
        return image.addBands(LAI)

def build_lai_monthly_composites(ls8_col, measure_mask):
    
    ls8_lai = (ls8_col
               .map(calc_lai)
               .select('LAI')
               .map(lambda img: img.set('year', img.date().get('year'))
                                     .set('month', img.date().get('month'))
                                     .set('ym', ee.String(img.date().get('year')).cat('_').cat(img.date().get('month'))))
    )

    ym_list = ee.List(ls8_lai.aggregate_array('ym')).distinct().sort()

    def make_composite(ym):
        ym = ee.String(ym)
        filtered = ls8_lai.filter(ee.Filter.eq('ym', ym))
        comp = ee.Image(filtered.reduce(ee.Reducer.mean()))
        first_band = ee.String(comp.bandNames().get(0))
        comp = comp.select([first_band]).rename('LAI')
        #comp = comp.updateMask(measure_mask)
        parts = ym.split('_')
        year = ee.Number.parse(parts.get(0))
        month = ee.Number.parse(parts.get(1))
        return comp.set('year', year).set('month', month).set('ym', ym)

    comps_list = ym_list.map(make_composite)  
    composites = ee.ImageCollection(comps_list)
    return composites

def calc_monthly_means(lai_composite, polygon, wetland_id):

    def mean_from_composite(img):
        mean_dict = img.reduceRegion(ee.Reducer.mean(), polygon, scale=30, maxPixels=1e10)
        month = ee.Number(img.get('month'))
        year = ee.Number(img.get('year'))
        mean_lai = mean_dict.get('LAI')
        feat = ee.Feature(None, {
            'year': year,
            'month': month,
            'mean_LAI': mean_lai,
            'wetland_id': wetland_id
        })

        return feat
    
    monthly_data = lai_composite.map(mean_from_composite)
    
    return ee.FeatureCollection(monthly_data)


# %% Run the functions to produce LAI timeseries for each wetland

def process_wetland_by_year_chunks(wetland_id, tgt_shape, start_year, end_year, years_per_chunk=1):
    """
    NOTE: Needed to chunk processing because Earth Engine computation size limtes
    Processing a wetland in smaller time chunks to reduce computation complexity
    """

    original_geom = tgt_shape.geometry.iloc[0]
    buffered_geom = original_geom.buffer(400)

    buffered_geom_4326 = gpd.GeoSeries([buffered_geom], crs=target_wetlands.crs).to_crs('EPSG:4326').iloc[0]

    #buffer_gdf = gpd.GeoDataFrame([1], geometry=[buffered_geom_4326], crs='EPSG:4326')
    #wetlands_around_buff_geom = gpd.clip(all_wetlands, buffer_gdf)

    poly = convert_gpd_geom_to_ee(buffered_geom_4326, est_utm='EPSG:4326')
    
    # Generate masks once
    nlcd_mask = make_nlcd_mask(polygon=poly)
    #upland_mask = make_upland_mask(wetland_shapes=wetlands_around_buff_geom)
    combined_mask = nlcd_mask #.And(upland_mask)
    
    # Process in year chunks
    for chunk_start in range(start_year, end_year + 1, years_per_chunk):
        chunk_end = min(chunk_start + years_per_chunk - 1, end_year)    
        chunk_start_date = f'{chunk_start}-01-01'
        if chunk_end >= 2025:
            chunk_end_date = f'{chunk_end}-10-01'
        else:
            chunk_end_date = f'{chunk_end}-12-31'
        
        try:
            ls8 = make_ls8_collection(polygon=poly, start_date=chunk_start_date, end_date=chunk_end_date)
            ls8_lai = build_lai_monthly_composites(ls8_col=ls8, measure_mask=combined_mask)
            monthly_data = calc_monthly_means(lai_composite=ls8_lai, polygon=poly, wetland_id=wetland_id)
            monthly_data_export = monthly_data.select(['year', 'month', 'mean_LAI', 'wetland_id'])
            
            # Export with year range in filename
            task = ee.batch.Export.table.toDrive(
                collection=monthly_data_export,
                description=f'well_buffer_400m_nomasking_{wetland_id}_{chunk_start}_{chunk_end}',
                folder='well_buffer_400m_nomasking',
                fileNamePrefix=f'well_buffer_400m_nomasking_{wetland_id}_{chunk_start}_{chunk_end}',
                fileFormat='CSV'
            )
            task.start()
            print(f'Started export for {wetland_id} years {chunk_start}-{chunk_end}')
            
        except Exception as e:
            print(f"Error processing {wetland_id} for years {chunk_start}-{chunk_end}: {e}")

# %% Use the chunked processing approach
target_shapes = well_points

for idx, i in enumerate(target_shapes['wetland_id'].unique()):
    #print(f"Processing wetland {idx+1}/{len(target_shapes['wetland_id'].unique())}: {i}")
    tgt_shape = target_shapes[target_shapes['wetland_id'] == i]
    process_wetland_by_year_chunks(i, tgt_shape=tgt_shape, start_year=2015, end_year=2025, years_per_chunk=3)


# %% Scratch code


# %% Quick visualization to check masking
def visualize_masking(wetland_id, year, month):
    """
    Create a quick visualization to check if masking is working properly
    """
    # Setup geometry for a specific wetland
    tgt_wetland = target_wetlands[target_wetlands['wetland_id'] == wetland_id]
    original_geom = tgt_wetland.geometry.iloc[0]
    buffered_geom = original_geom.buffer(250)
    buffered_geom_4326 = gpd.GeoSeries([buffered_geom], crs=target_wetlands.crs).to_crs('EPSG:4326').iloc[0]
    
    # Get clipped wetlands
    buffer_gdf = gpd.GeoDataFrame([1], geometry=[buffered_geom_4326], crs='EPSG:4326')
    wetlands_around_buff_geom = gpd.clip(all_wetlands, buffer_gdf)
    
    poly = convert_gpd_geom_to_ee(buffered_geom_4326, est_utm='EPSG:4326')
    
    # Create masks
    nlcd_mask = make_nlcd_mask(polygon=poly)
    upland_mask = make_upland_mask(wetland_shapes=wetlands_around_buff_geom)
    combined_mask = nlcd_mask.And(upland_mask)

    nlcd_raw = (ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019")
                .clip(poly)
                .select('landcover'))
    
    # Get a single month of data
    start_date = f'{year}-{month:02d}-01'
    end_date = f'{year}-{month:02d}-28'
    
    ls8 = make_ls8_collection(polygon=poly, start_date=start_date, end_date=end_date)
    
    # Create LAI without mask
    ls8_lai_unmasked = (ls8
                       .map(calc_lai)
                       .select('LAI')
                       .mean()
                       .rename('LAI_unmasked'))
    
    # Create LAI with mask
    ls8_lai_masked = (ls8
                     .map(calc_lai)
                     .select('LAI')
                     .mean()
                     .updateMask(combined_mask)
                     .rename('LAI_masked'))
    
    # Create a map
    m = geemap.Map()
    m.add_basemap('SATELLITE')
    m.centerObject(poly, 14)
    
    # Visualization parameters
    lai_vis = {
        'min': 0,
        'max': 6,
        'palette': ['red', 'yellow', 'green', 'darkgreen']
    }
    
    mask_vis = {
        'min': 0,
        'max': 1,
        'palette': ['blue']
    }

    # nlcd_vis = {
    #     'min': 0,
    #     'max': 95,
    #     'palette': [
    #         '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000',
    #         '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000', '000000',
    #         '476BA0', 'D1DDF8', '0565A6', 'E2EEF9', 'B50000', 'DD0000', 'FF6B6B', 'FDD7A5', 'DEB887', 'AA0000',
    #         'B2ADA3', '68AA63', '1C6330', 'B5C98E', 'A58C30', 'CCBA7C', 'E2E2C1', 'CCD198', '1C6330', '68AA63',
    #         'B5C98E', 'CCBA7C', 'E2E2C1', 'CCD198', 'A58C30', 'E29E8C', 'FF0000', 'FFD800', 'FF6B6B', 'FFEBAF',
    #         'FDD7A5', 'D2D2FF', 'AAA', 'FDD7A5', 'DCD939', 'AB0000', 'B5C98E', '68AA63', '1C6330', 'B5C98E',
    #         'CCBA7C', 'E2E2C1', 'CCD198', '1C6330', '68AA63', 'B5C98E', 'CCBA7C', 'E2E2C1', 'CCD198', 'A58C30',
    #         'E29E8C', 'FF0000', 'FFD800', 'FF6B6B', 'FFEBAF', 'FDD7A5', 'D2D2FF', 'AAA', 'FDD7A5', 'DCD939',
    #         'AB0000', 'B5C98E', '68AA63', '1C6330', 'B5C98E', 'CCBA7C', 'E2E2C1', 'CCD198', '1C6330', '68AA63',
    #         'B5C98E', 'CCBA7C', 'E2E2C1', 'CCD198', 'A58C30', 'E29E8C'
    #     ]
    # }
    
    # Add layers
    # m.addLayer(nlcd_raw, nlcd_vis, 'NLCD Landcover')
    m.addLayer(ls8_lai_unmasked, lai_vis, 'LAI Unmasked')
    m.addLayer(ls8_lai_masked, lai_vis, 'LAI Masked')
    # m.addLayer(nlcd_mask, mask_vis, 'NLCD Mask', opacity=0.5)
    m.addLayer(upland_mask, mask_vis, 'Upland Mask', opacity=0.5)
    m.addLayer(combined_mask, mask_vis, 'Combined Mask', opacity=0.5)

    # Add the original wetland and buffer boundaries
    m.addLayer(ee.Geometry.Point([buffered_geom_4326.centroid.x, buffered_geom_4326.centroid.y]), 
               {'color': 'red'}, 'Wetland Center')
    
    return m

# Test the visualization
test_map = visualize_masking(wetland_id='14_500', year=2024, month=10)
test_map





# %%
