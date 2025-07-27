
# 1.0 Libraries and directories -------------------------------------------

library(tidyverse)
library(readxl)
library(rlang)
library(glue)

soil_core_dir <- 'D:/depressional_lidar/data/osbs/in_data/sampling_elevations/faith_osbs_soilcore_elevations.xlsx'
soil_core <- read_excel(
  soil_core_dir,
  sheet='Sheet1'
)

stage_dir <- "D:/depressional_lidar/data/osbs/in_data/stage_data/osbs_core_wells_consistent_datum.csv"
stage <- read_csv(
  stage_dir
  # TODO: Obtain spring download for __
  # I filter out early data so we have consistent period of record across wetlands, safe comparison
) %>% filter(timestamp_utc >= as.POSIXct('2022-03-09 12:00:00', tz='UTC'))

daily_stage <- stage %>% 
  # Take daily mean water level for easier computation
  mutate(day = as.Date(timestamp_utc)) %>% 
  group_by(day, well_id) %>% 
  summarise(
    well_depth_m = mean(well_depth_m, na.rm = TRUE),
    #max_depth_m = mean(max_depth_m, na.rm = TRUE),
  )

# 2.0 Munge the soil core data into a summary table ----------------------------------------------

soil_core_summary <- soil_core %>% 
  group_by(soil_core_id) %>% 
  summarize(
    well_id = first(well_id),
    sample_date = first(sample_date), 
    soil_core_to_well_elevation_cm = first(soil_core_to_well_elevation_cm)
  ) %>% 
  filter(!is.na(soil_core_id))


# 3.0 Functions ----------------------------------------------

calculate_stage_comp <- function(elevation_temp, stage_temp, core_well_elevation_diff){
  # Helper function for fetch_core_info()
  #’ Calculate the discrepancy between a soil‑core measurment and the predicted well stage
  #’
  #’ this function computes:
  #’ 1. the core’s measured stage (in meters, negative = below ground),  
  #’ 2. the predicted core stage from the well reading plus elevation difference,  
  #’ 3. their difference (core_stage – predicted_core_stage).
  #’
  #’ Retuns a tibble with columns:
  #’   - `core_stage` (numeric, m)  
  #’   - `predicted_core_stage` (numeric, m)  
  #’   - `stage_discrepancy` (numeric, m)  

  sample_date <- unique(elevation_temp$sample_date)
  core_depths_cm <- elevation_temp$water_depth_on_sample_date_cm
  # Error handling for NA water depths in soil cores.
  if (all(is.na(core_depths_cm)) || !any(!is.na(suppressWarnings(as.numeric(core_depths_cm))))) {
    return(tibble::tibble(
      core_stage           = NA_real_,
      predicted_core_stage = NA_real_,
      stage_discrepancy    = NA_real_
    ))
  }
  
  # cm to meters and change convention (negative means below ground)
  core_stage <- as.numeric(unique(core_depths_cm)) / -100 
  
  well_stage_on_date <- stage_temp %>% 
    filter(day == sample_date) %>% 
    pull(well_depth_m)
  
  predicted_core_stage <- well_stage_on_date - core_well_elevation_diff
  
  # Check the well's adjusted stage vs. soil core measured stage for sample date
  # print(glue('Stage measured at soil core = {core_stage} meters'))
  # print(glue('Est. stage from well and elevation = {predicted_core_stage} meters'))
  
  stage_discrepancy = core_stage - predicted_core_stage
  
  stage_comp <- tibble(
    core_stage = core_stage,
    predicted_core_stage = predicted_core_stage,
    stage_discrepancy = stage_discrepancy
  )
  
  return(stage_comp)
  
}

fetch_core_info <- function(stage_df, elevation_df, target_well_id, core_id){
  
  #’ Fetches both the adjusted hydrograph and the stage comparison for a given soil core
  #’
  #’ returns list with two elements:  
  #’   - `hydrograph`: a tibble timeseries with stage adjuseted to soil core elevation
  #’   - `stage_comp`: a one‑row tibble (shows well vs. measured stage) from calculate_stage_comp()
  
  # Filter the stage and core data to match
  stage_temp <- stage_df %>% 
    filter(well_id == target_well_id) 
  
  elevation_temp <- elevation_df %>% 
    filter(soil_core_id == core_id,
           well_id == target_well_id)

  
  core_well_elevation_diff <- unique(elevation_temp$soil_core_to_well_elevation_cm)
  core_well_elevation_diff <- as.numeric(core_well_elevation_diff) / 100 # cm to meters
  
  stage_comp <- calculate_stage_comp(elevation_temp, stage_temp, core_well_elevation_diff)
  
  core_hydrograph <- stage_temp %>% 
    mutate(core_stage_m = well_depth_m - core_well_elevation_diff) %>% 
    select(c(day, core_stage_m))
    
  return(list(
    core_hydrograph = core_hydrograph,
    stage_comp = stage_comp
  ))
}
  


# 3.0 Testing ----------------------------------------------

soil_core_summary_test <- soil_core_summary %>% 
  rowwise() %>% 
  mutate(
    basic_core_info = list(fetch_core_info(daily_stage, soil_core, well_id, soil_core_id)),
    core_hydrograph = list(basic_core_info$core_hydrograph),
    stage_comp = list(basic_core_info$stage_comp)
  ) %>% 
  ungroup() %>% 
  select(-c(basic_core_info)) %>% 
  unnest_wider(stage_comp)




# compute_percent_days_inundated <- function(hydrograph, stage_var) {
#   
#   # Allow bare name or string
#   stage_sym <- rlang::ensym(stage_var)
#   
#   hydrograph %>%
#     filter(!is.na(!!stage_sym))
#   
#   valid_days <- length(hydrograph)
#   inundated_days <- length(hydrograph %>% filter(!!stage_sm > 0))
#   pct <- valid_days / inundated_days * 100
#   
#   return(pct)
# }



