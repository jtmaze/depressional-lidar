# %% 1.0 Libraries and filepaths

import ee
import geopandas as gpd

ee.Initialize(project='wetland-ditching-prelim')

start = '2019-01-01'
end = '2025-12-31'

data_dir = 'D:/depressional_lidar/data/bradford/'
bradford_boundary_path = f'{data_dir}/bradford_boundary.shp'
wetlands_path = f'{data_dir}/out_data/bradford_wetland_basins_vf_clipped.shp'

bradford_boundary = gpd.read_file(bradford_boundary_path)
wetlands = gpd.read_file(wetlands_path)

# %% 2.0 Earth Engine Processing Functions

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

def make_wetlandscape_mask(wetland_shapes: gpd.GeoDataFrame, upland_wetland: str):
    """
    Uses a GeoDataFrame to mask the wetlands or upland LAI calculations
    Returns an ee.Image() for the mask shapes
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

    wetlandscape_mask = ee.Image(0).byte().paint(
        featureCollection=wetlands_fc,
        color=1  # Wetland pixels get value 1
    )    

    if upland_wetland == "upland":
        # Create upland mask (inverse of wetland mask)
        out_mask = wetlandscape_mask.Not()
    else: 
        out_mask = wetlandscape_mask
    
    return out_mask

def summarize_lai(image):
    image = ee.Image(image)
    date = ee.Date.fromYMD(
        ee.Number(image.get('year')),
        ee.Number(image.get('month')),
        1
    ).format('YYYY-MM-dd')

    lai_boundary = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=bradford_boundary_ee,
        scale=30,
        maxPixels=1e9
    ).get('LAI')

    lai_upland = image.updateMask(upland_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=bradford_boundary_ee,
        scale=30,
        maxPixels=1e9
    ).get('LAI')

    lai_wetland = image.updateMask(wetland_mask).reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=bradford_boundary_ee,
        scale=30,
        maxPixels=1e9
    ).get('LAI')

    return ee.Feature(None, {
        'date': date,
        'lai_bradford': lai_boundary,
        'lai_upland': lai_upland,
        'lai_wetland': lai_wetland
    })

# %% 3.0 Generate sepperate timeseries for entire bradford forest, weltands, and uplands

bradford_geom_4326 = bradford_boundary.to_crs('EPSG:4326').geometry.union_all()
bradford_boundary_ee = ee.Geometry(bradford_geom_4326.__geo_interface__)

wetlands_4326 = wetlands.to_crs('EPSG:4326')
wetlands_4326['geometry'] = wetlands_4326.simplify(tolerance=0.0001)  # ~10m
wetland_mask = make_wetlandscape_mask(wetlands_4326, upland_wetland='wetland')
upland_mask = make_wetlandscape_mask(wetlands_4326, upland_wetland='upland')

ls8_col = make_ls8_collection(
    polygon=bradford_boundary_ee,
    start_date=start,
    end_date=end
)

lai_monthly = build_lai_monthly_composites(ls8_col)
lai_features_all = ee.FeatureCollection(lai_monthly.map(summarize_lai)).sort('date')

drive_folder = 'depressional_lidar_bradford'
task_lai = ee.batch.Export.table.toDrive(
    collection=lai_features_all,
    description='landscape_lai_timeseries',
    folder=drive_folder,
    fileNamePrefix='landscape_lai_timeseries',
    fileFormat='CSV'
)

task_lai.start()

print('Started Earth Engine table export to Drive folder:', drive_folder)
print('Task: landscape_lai_timeseries')

# %% 4.0