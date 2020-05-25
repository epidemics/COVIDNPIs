library(tidyverse)
library(EpiEstim)
library(progress)

# the lag on which to compute the incidence vector
INCIDENCE_LAG <- 1

args <- commandArgs(trailingOnly = TRUE)
si_sample_file <- args[1]
input_file <- args[2]
output_file <- args[3]
print(paste("Estimating R form JH data", input_file, "and writing to", output_file))

#Code,Date,Recovered,Confirmed,Deaths,Active
cases <- read.csv(input_file, colClasses=c("character", "character", "numeric", "numeric", "numeric", "numeric"))
names(cases) <- names(cases) %>% tolower() %>% str_replace('[\\/]', '_')

country_codes <- function() {
  return(unique(cases$code))
}

load_country <- function(country_code) {
  country_cases <- filter(cases, code == country_code) %>%
    mutate(date = as.Date(date)) %>%
    arrange(date) %>%
    mutate(
      lag_cases = lag(confirmed, INCIDENCE_LAG), # This seems to be because of the lag between symtomps and being tested as positive
      new_cases = confirmed - lag_cases,
      new_cases = ifelse(is.na(new_cases), 1, new_cases),
      new_cases = ifelse(new_cases < 0, 0, new_cases)
    ) %>%
    rename(
      I = new_cases,
      dates = date
    ) %>%
    select(I, dates)
  return (country_cases)
}

estimate_r <- function(country_code, si_sample) {
  inc <- load_country(country_code)

  # R estimation with saved SI data
  R_mcmc_estimated_si <- estimate_R(
    inc,
    method = "si_from_sample",
    si_sample = si_sample
  )

  out <- data.frame(
    Date=R_mcmc_estimated_si[["dates"]][8:length(R_mcmc_estimated_si[["dates"]])],
    RMean=R_mcmc_estimated_si[["R"]][["Mean(R)"]],
    RStd=R_mcmc_estimated_si[["R"]][["Std(R)"]]
  )
  out$code = country_code
  return(out)
}

countries <- country_codes()
si_sample <- read_rds(si_sample_file)

pb <- progress_bar$new(total = length(countries))
export <- data.frame(code=NULL, date=NULL, RMean=NULL, RStd=NULL)
for (country_code in countries) {
  country_estimates <- estimate_r(country_code, si_sample)
  export <- rbind(export, country_estimates)
  pb$tick()
}
write.csv(export, output_file)
print(paste("Exported R estimates for", length(countries), "countries to", output_file))
