# %% 1.0 Libraries and file paths

import ee 
import geopandas as gpd

ee.Authenticate()
ee.Initialize()

all_wetlands_path = f'D:/depressional_lidar/data/bradford/in_data/original_basins/bradford_original_depressions_all_wetlands.shp'
all_wetlands = gpd.read_file(all_wetlands_path)
all_wetlands.to_crs('EPSG:4326', inplace=True)

well_points_path = 'D:/depressional_lidar/data/rtk_pts_with_dem_elevations.shp'
well_points = (gpd.read_file(well_points_path)[['wetland_id', 'type', 'site', 'geometry']]
               .query("type in ['main_doe_well', 'aux_wetland_well'] and site == 'Bradford'")
)
well_points['wetland_id'] = well_points['wetland_id'].str.replace('/', '.')
print(len(well_points))
print(well_points['wetland_id'].unique())

# %% 2.0 Compartmentalized helper functions for Earth Engine

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
    """
    Produces a filtered image collection for LAI calculations. 
    """

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
        LAI = image.expression('(0.332915 * SR) - 0.00212', {'SR': SR}).rename('LAI').clamp(0, 5.6)

        return image.addBands(LAI)

def build_lai_monthly_composites(ls8_col):
    """
    Produces a stack of LAI images indexed by year and month
    """
    
    ls8_lai = (ls8_col
               .map(calc_lai)
               .select('LAI')
               .map(lambda img: img.set('year', img.date().get('year'))
                                     .set('month', img.date().get('month'))
                                     .set('ym', ee.String(img.date().get('year')).cat('_').cat(img.date().get('month'))))
    )

    ym_list = ee.List(ls8_lai.aggregate_array('ym')).distinct().sort()

    def _make_composite(ym):
        ym = ee.String(ym)
        filtered = ls8_lai.filter(ee.Filter.eq('ym', ym))
        comp = ee.Image(filtered.reduce(ee.Reducer.mean()))
        first_band = ee.String(comp.bandNames().get(0))
        comp = comp.select([first_band]).rename('LAI')
        parts = ym.split('_')
        year = ee.Number.parse(parts.get(0))
        month = ee.Number.parse(parts.get(1))
        return comp.set('year', year).set('month', month).set('ym', ym)

    comps_list = ym_list.map(_make_composite)  
    composites = ee.ImageCollection(comps_list)
    return composites

def calc_monthly_means(lai_composite, polygon, wetland_id):
    """
    Produces monthly timeseries around a given well from the LAI image stacks
    """

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


# %% 3.0 Main function to produce LAI timeseries for each wetland

def process_wetland_by_year_chunks(wetland_id, buffer_size, tgt_shape, start_year, end_year, years_per_chunk=1):
    """
    NOTE: I needed to chunk processing (by years) because of Earth Engine computation size limits
    Processing a wetland in smaller time chunks to reduce computation complexity
    """

    original_geom = tgt_shape.geometry.iloc[0]
    # Buffer radius around a well and convert to web mapping crs
    buffered_geom = original_geom.buffer(buffer_size)
    buffered_geom_4326 = gpd.GeoSeries([buffered_geom], crs=tgt_shape.crs).to_crs('EPSG:4326').iloc[0]
    # Convert to Earth Engine Polygon
    poly = convert_gpd_geom_to_ee(buffered_geom_4326, est_utm='EPSG:4326')
    
    # Process in year chunks
    for chunk_start in range(start_year, end_year + 1, years_per_chunk):
        chunk_end = min(chunk_start + years_per_chunk - 1, end_year)    
        chunk_start_date = f'{chunk_start}-01-01'
        if chunk_end >= 2025:
            chunk_end_date = f'{chunk_end}-12-01'
        else:
            chunk_end_date = f'{chunk_end}-12-31'
        
        try:
            ls8 = make_ls8_collection(polygon=poly, start_date=chunk_start_date, end_date=chunk_end_date)
            # NOTE: not using the combined mask in build_lai_monthly_composites()
            ls8_lai = build_lai_monthly_composites(ls8_col=ls8)
            monthly_data = calc_monthly_means(lai_composite=ls8_lai, polygon=poly, wetland_id=wetland_id)
            monthly_data_export = monthly_data.select(['year', 'month', 'mean_LAI', 'wetland_id'])
            
            # Export well's monthly timeseries
            task = ee.batch.Export.table.toDrive(
                collection=monthly_data_export,
                description=f'well_buffer_{buffer_size}m_nomasking_{wetland_id}_{chunk_start}_{chunk_end}',
                folder=f'well_buffer_{buffer_size}m_nomasking',
                fileNamePrefix=f'well_buffer_{buffer_size}m_nomasking_{wetland_id}_{chunk_start}_{chunk_end}',
                fileFormat='CSV'
            )
            task.start()
            print(f'Started export for {wetland_id} years {chunk_start}-{chunk_end}')
            
        except Exception as e:
            print(f"Error processing {wetland_id} for years {chunk_start}-{chunk_end}: {e}")

# %% 4.0 Calculate the LAI timeseries

target_shapes = well_points

for idx, i in enumerate(well_points['wetland_id'].unique()):
    #print(f"Processing wetland {idx+1}/{len(target_shapes['wetland_id'].unique())}: {i}")
    tgt_shape = well_points[well_points['wetland_id'] == i]
    process_wetland_by_year_chunks(
        i, 
        buffer_size=150,
        tgt_shape=tgt_shape, 
        start_year=2015, 
        end_year=2025, 
        years_per_chunk=3
    )

# %% Scratch functions

"""
Scratch function used to mask landcover types
"""
# def make_upland_mask(wetland_shapes: gpd.GeoDataFrame):
#     """
#     NOTE: This function was discarded
#     Uses a GeoDataFrame to mask the wetlands from the upland LAI calculations
#     Returns an ee.Image() from the wetlands shapes.
#     """
#     ee_geoms = []
#     for idx, row in wetland_shapes.iterrows():
#         try:
#             ee_geom = convert_gpd_geom_to_ee(row['geometry'], est_utm='EPSG:4326')
#             ee_geoms.append(ee_geom)
#         except:
#             print(f"could not resovle geometry {idx}")
#             continue # skipping problematic geometries
    
#     ee_features = [ee.Feature(geom) for geom in ee_geoms]
#     wetlands_fc = ee.FeatureCollection(ee_features)

#     wetland_mask = ee.Image(0).byte().paint(
#         featureCollection=wetlands_fc,
#         color=1  # Wetland pixels get value 1
#     )    
#     # Create upland mask (inverse of wetland mask)
#     upland_mask = wetland_mask.Not()
    
#     return upland_mask

# def make_nlcd_mask(polygon):

#     """
#     NOTE: Omitted this function.
#     Uses National Landcover to mask LAI calculations. 
#     """

#     nlcd = (ee.Image("USGS/NLCD_RELEASES/2019_REL/NLCD/2019")
#             .clip(polygon)
#             .select('landcover')
#     )
#     nlcd_keep_mask = (
#         nlcd.eq(41) # Decidous Forest
#         .Or(nlcd.eq(42)) # Evergreen Forest
#         .Or(nlcd.eq(43)) # Mixed Forest
#         .Or(nlcd.eq(52)) # Shrub/Scrub
#         .Or(nlcd.eq(71)) # Grassland/Herbaceous
#         .Or(nlcd.eq(90)) # Woody Wetlands
#         .Or(nlcd.eq(95)) # Emergent Herbaceaous Wetlands NOTE: some contention about whether to keep this one. 
#     )

#     return nlcd_keep_mask
