
# 1.0 Libraries and directories -------------------------------------------

library(tidyverse)
library(readxl)
library(rlang)
library(glue)

rm(list=ls())

soil_core_dir <- 'D:/depressional_lidar/data/osbs/in_data/sampling_elevations/complete_faith_osbs_soilcore_elevations.xlsx'
soil_core <- read_excel(
  soil_core_dir,
  sheet='Sheet1'
)
print(unique(soil_core$well_id))

core_ids <- c('Devils Den', 'Ross Pond', 'West Ford', 'Brantley North', 'Fish Cove', 'Surprise')

stage_dir <- "D:/depressional_lidar/data/osbs/in_data/stage_data/daily_well_depth_Fall2025.csv"
daily_stage <- read_csv(stage_dir) %>%
  # I filter out early data so we have consistent period of record across wetlands, safe comparison
   filter(date >= as.POSIXct('2022-03-09 12:00:00', tz='UTC')) %>% 
   filter(well_id %in% core_ids)

well_id_key <- tibble(
  'Fish Cove' = 'Fishcove',
  'Surprise' = 'Surprise Pond',
)

daily_stage <- daily_stage %>% 
  mutate(
    well_id = recode(well_id, !!!well_id_key, .default=well_id)
  ) %>% 
  rename('day'='date')
  
print(str(daily_stage))

# 2.0 Munge the soil core data into a summary table ----------------------------------------------

soil_core_summary <- soil_core %>% 
  group_by(soil_core_id) %>% 
  summarize(
    well_id = first(well_id),
    sample_date = first(sample_date), 
    soil_core_to_well_elevation_cm = first(soil_core_to_well_elevation_cm)
  ) %>% 
  filter(!is.na(soil_core_id))

soil_core_slice_summary <- soil_core %>% 
  group_by(soil_core_id, depth_increment_midpt_cm) %>% 
  summarise(
    well_id = first(well_id),
    sample_date = first(sample_date),
    soil_core_to_well_elevation_cm = first(soil_core_to_well_elevation_cm)
  ) %>% 
  filter(!is.na(soil_core_id)) %>% 
  mutate(slice_to_well_elevation = soil_core_to_well_elevation_cm - depth_increment_midpt_cm)

# 3.0 Functions for basic core info ----------------------------------------------

calculate_stage_comp <- function(elevation_temp, stage_temp, core_well_elevation_diff){
  # Helper function for fetch_core_info()
  #’ Calculate the discrepancy between a soil‑core measurement and the predicted well stage
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
  
  # cm to meters and change sign convention (negative means below ground)
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
    mutate(sample_stage_m = well_depth_m - core_well_elevation_diff) %>% 
    select(c(day, sample_stage_m))
    
  return(list(
    core_hydrograph = core_hydrograph,
    stage_comp = stage_comp
  ))
}

fetch_slice_hydrograph <- function(stage_df, elevation_slice_df, target_well_id, core_id, tgt_depth_slice){
  # Similar to fetch core info; however, fetches hydrograph for a given elevation
  # within the soil core. 
  # Uses the mid-point of the depth increments to assign the hydrograph
  
  stage_temp <- stage_df %>% 
    filter(well_id == target_well_id)
  
  slice_elevation_temp <- elevation_slice_df %>% 
    filter(soil_core_id == core_id,
           well_id == target_well_id,
           depth_increment_midpt_cm == tgt_depth_slice)
  
  slice_elevation_diff_cm <- unique(slice_elevation_temp$slice_to_well_elevation)

  slice_elevation_diff <- as.numeric(slice_elevation_diff_cm) / 100

  
  slice_hydrograph <- stage_temp %>% 
    mutate(sample_stage_m = well_depth_m - slice_elevation_diff) %>% 
    select(c(day, sample_stage_m))
  
  return(slice_hydrograph)
}

# ...3.1 Get basic core info and hydrograph ----------------------------------------------

soil_core_summary <- soil_core_summary %>% 
  rowwise() %>% 
  mutate(
    basic_core_info = list(fetch_core_info(daily_stage, soil_core, well_id, soil_core_id)),
    core_hydrograph = list(basic_core_info$core_hydrograph),
    stage_comp = list(basic_core_info$stage_comp)
  ) %>% 
  ungroup() %>% 
  select(-c(basic_core_info)) %>% 
  unnest_wider(stage_comp)


soil_core_slice_summary <- soil_core_slice_summary %>% 
  rowwise() %>% 
  mutate(
    slice_hydrograph = list(
      fetch_slice_hydrograph(
        daily_stage, 
        soil_core_slice_summary, 
        well_id, 
        soil_core_id, 
        depth_increment_midpt_cm
        )
      )
  ) %>% 
  ungroup() 

# ...3.2 Ensure stage measurements match ----------------------------------------------

plot_df <- soil_core_summary 

p <- ggplot(plot_df,
            aes(x=soil_core_id,
                y=stage_discrepancy,
                fill=well_id)) +    
  geom_col(position="dodge") +   
  labs(
    x="Soil Core ID",
    y="Stage discrepancy (core check - well logger) (m)",
    fill="Well ID",
    title="Stage discrepancy on sample date between well logger (elevation adjusted) and field measurement @core"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

print(p)

# 4.0 Calculate the core's hydrograph metrics ----------------------------------------------

calc_hydrograph_stats <- function(hydrograph, var_name="sample_stage_m") {
  
  hg <- hydrograph[[var_name]] # Note var_name is dynamic for depth incrmentn midpt metrics later
  stats <- tibble(
    mean_stage_full_ts=mean(hg, na.rm=TRUE), 
    median_stage_full_ts=median(hg, na.rm=TRUE),
    sd_stage_full_ts=sd(hg, na.rm=TRUE),
    p20_stage_full_ts=quantile(hg, 0.2, na.rm=TRUE, names=FALSE),
    p80_stage_full_ts=quantile(hg, 0.8, na.rm=TRUE, names=FALSE)
  )
  return(stats)
}

soil_core_summary <- soil_core_summary %>% 
  mutate(
    hydrograph_stats = map(core_hydrograph,
                           ~ calc_hydrograph_stats(.x, var_name = "sample_stage_m"))
  ) %>% 
  unnest_wider(hydrograph_stats)

soil_core_slice_summary <- soil_core_slice_summary %>% 
  mutate(
    hydrograph_stats = map(slice_hydrograph, 
                            ~ calc_hydrograph_stats(.x, var_name = 'sample_stage_m'))
  ) %>% 
  unnest_wider(hydrograph_stats)

# 5.0 Functions for binary hydrometrics ----------------------------------------------

inundated_durations <- function(binary) {

  # run-length encoding function returns vector of lengths and values
  r <- rle(binary)
  # Subset to only the runs with value == 1
  flooded_lengths <- r$lengths[ r$values == 1L ]
  
  # bail early if always dry
  if (length(flooded_lengths) == 0L) {
    return(tibble(
      max_inundated    = 0,
      min_inundated    = 0,
      median_inundated = 0,
      mean_inundated   = 0
    ))
  }
  # otherwise, return metrics on inundation
  max_inundated <- max(flooded_lengths)
  min_inundated <- min(flooded_lengths)
  median_inundated <- median(flooded_lengths)
  mean_inundated <- mean(flooded_lengths)
  
  durrations <- tibble(
    max_inundated=max_inundated,
    min_inundated=min_inundated,
    median_inundated=median_inundated,
    mean_inundated=mean_inundated
  )
  
  return(durrations)
  
}

wet_dry_events <- function(binary, wet_dry) {
  if (wet_dry == 'wet') {
    events <- sum(diff(c(0L, binary)) == 1L, na.rm = TRUE)
  } else if (wet_dry == 'dry') {
    events <- sum(diff(c(0L, binary)) == -1L, na.rm = TRUE)
  } else {
    events <- -9999 # catch bad function call
  }
  return(events)
}

compute_binary_metrics <- function(hydrograph) {
 
 # Remove any NA observations to ensure they aren't counted
 binary <- hydrograph %>% 
   drop_na(sample_stage_m) %>% 
   distinct(day, .keep_all=TRUE) %>% 
   arrange(day) %>% 
   transmute(day, inundated=as.integer(sample_stage_m >= 0))
 
 hydroperiod_percent <- mean(binary$inundated) * 100
 
 durations <- inundated_durations(binary$inundated)
 wet_events <- wet_dry_events(binary$inundated, 'wet')
 dry_events <- wet_dry_events(binary$inundated, 'dry')
 
 binary_metrics = tibble(
   pti_full_ts = hydroperiod_percent,
   max_inundated_durration_full_ts = durations$max_inundated,
   min_inundated_durration_full_ts = durations$min_inundated,
   median_inundated_durration_full_ts = durations$median_inundated,
   mean_inundated_durration_full_ts = durations$mean_inundated,
   wetup_event_count_full_ts = wet_events,
   dry_event_count_full_ts = dry_events
 )

 return(binary_metrics)
 
}

# ... 5.1 Calculate binary metrics ----------------------------------------------

soil_core_summary <- soil_core_summary %>% 
  rowwise() %>% 
  mutate(
    binary_metrics = compute_binary_metrics(core_hydrograph)
  ) %>% 
  ungroup() %>% 
  unnest_wider(binary_metrics)

soil_core_slice_summary <- soil_core_slice_summary %>% 
  rowwise() %>% 
  mutate(
    binary_metrics = compute_binary_metrics(slice_hydrograph)
  ) %>% 
  ungroup() %>% 
  unnest_wider(binary_metrics)

# 6.0 Functions for sample date metrics ----------------------------------------

get_prior_hydrograph_mean <- function(hg, sample_date, look_back){
  
  # Help func for calc_sample_date_info()
  
  # returns the hydrograph mean for the prior number of days (look_back)
  window_start <- sample_date - days(look_back)
  
  prior_avg <- hg %>%
    filter(day >= window_start, day < sample_date) %>%
    pull(sample_stage_m) %>%
    mean(na.rm = TRUE)
  
  return(prior_avg)
  
}

get_prior_hydrograph_inundation <- function(hg, sample_date, look_back){
  # Helper function for calc_sample_date_info()
  
  window_start <- sample_date - days(look_back)
  
  hg_binary <- hg %>% 
    drop_na(sample_stage_m) %>% 
    distinct(day, .keep_all=TRUE) %>% 
    arrange(day) %>% 
    transmute(day, inundated=as.integer(sample_stage_m >= 0))
  
  hydroperiod_percent <- mean(hg_binary$inundated) * 100
  wet_events <- wet_dry_events(hg_binary$inundated, 'wet')
  dry_events <- wet_dry_events(hg_binary$inundated, 'dry')
  
  return(tibble(
    pti=hydroperiod_percent,
    wet_events=wet_events,
    dry_events=dry_events
  ))
  
}

calc_sample_date_info <- function(hydrograph, sample_date) {
  
  days <- as.Date(hydrograph$day)
  sd   <- as.Date(sample_date)
  
  # Bail-out early if sample date is beynd the hydrograph's last date
  if (is.na(sd) || !(sd %in% days)) {
    return(tibble(
      mean_30d_stage=NA_real_, 
      mean_60d_stage=NA_real_,
      mean_90d_stage=NA_real_,
      mean_1yr_stage=NA_real_,
      stage_60d_deviation=NA_real_
    ))
  }
  
  mean_30 <- get_prior_hydrograph_mean(hydrograph, sample_date, 30)
  mean_90 <- get_prior_hydrograph_mean(hydrograph, sample_date, 90)
  mean_1yr_stage <- get_prior_hydrograph_mean(hydrograph, sample_date, 365)
  
  inundation_1yr <- get_prior_hydrograph_inundation(hydrograph, sample_date, 365)
  inundation_90d <- get_prior_hydrograph_inundation(hydrograph, sample_date, 90)
  
  sample_date_stage <- hydrograph %>% 
    filter(day == sample_date) %>% 
    pull()
  
  sample_date_info = tibble(
    sample_date_stage = sample_date_stage,
    mean_30d_stage = mean_30,
    mean_90d_stage = mean_90,
    mean_1yr_stage = mean_1yr_stage, 
    pti_1yr = inundation_1yr %>% pull(pti),
    wet_event_1yr = inundation_1yr %>% pull(wet_events),
    dry_event_1yr = inundation_1yr %>% pull(dry_events),
    pti_90d = inundation_90d %>% pull(pti), 
    wet_event_90d = inundation_90d %>% pull(wet_events),
    dry_event_90d = inundation_90d %>% pull(dry_events)
  )
  
  return(sample_date_info)
}


# ... 6.1 Calculate sample date metrics ----------------------------------------

soil_core_summary <- soil_core_summary %>% 
  rowwise() %>% 
  mutate(
    date_metrics = calc_sample_date_info(core_hydrograph, sample_date)
  ) %>% 
  ungroup() %>% 
  unnest_wider(date_metrics)

soil_core_slice_summary <- soil_core_slice_summary %>% 
  rowwise() %>% 
  mutate(
    date_metrics = calc_sample_date_info(slice_hydrograph, sample_date)
  ) %>% 
  ungroup() %>% 
  unnest_wider(date_metrics)

# ...  6.2 Calculate sample devation from long-term average ----------------------------------------

soil_core_summary <- soil_core_summary %>% 
  select(-c('core_hydrograph', 'sample_date_stage')) %>% 
  mutate(difference_90d_from_full_mean = mean_90d_stage - mean_stage_full_ts) %>% 
  mutate(zscore_90d = (mean_90d_stage - mean_stage_full_ts) / sd_stage_full_ts)


soil_core_slice_summary <- soil_core_slice_summary %>% 
  select(-c('slice_hydrograph', 'sample_date_stage')) %>% 
  mutate(difference_90d_from_full_mean = mean_90d_stage - mean_stage_full_ts) %>% 
  mutate(zscore_90d = (mean_90d_stage - mean_stage_full_ts) / sd_stage_full_ts)
  
# 7.0 Write soil core summary --------------------------------------------

output_cores <- soil_core_summary 

out_cores_path <- "D:/depressional_lidar/data/osbs/out_data/updated_faith_osbs_hydrometrics_at_cores.csv"

write_csv(output_cores, out_cores_path)

output_slices <- soil_core_slice_summary

out_slices_path <- "D:/depressional_lidar/data/osbs/out_data/updated_faith_osbs_hydrometrics_at_slices.csv"

write_csv(output_slices, out_slices_path)

