import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import numpy as np
import torch

from ascab.utils.generic import fill_gaps

def get_meteo(params, verbose=False):
    url = "https://archive-api.open-meteo.com/v1/archive"

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    responses = openmeteo.weather_api(url, params=params)
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    if verbose:
        print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
        print(f"Elevation {response.Elevation()} m asl")
        print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
        print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()

    hourly_data = {"date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )}

    for i, name in enumerate(params['hourly']):
        hourly_data[name] = hourly.Variables(i).ValuesAsNumpy()

    hourly_dataframe = pd.DataFrame(data=hourly_data)
    hourly_dataframe.set_index('date', inplace=True)
    hourly_dataframe.index = hourly_dataframe.index.tz_convert(response.Timezone().decode('utf-8'))

    return hourly_dataframe


def is_wet(precipitation, vapour_pressure_deficit):
    precipitation_threshold = 0.0
    vapour_pressure_deficit_threshold = 0.25  # 2.5 hPa = 0.25kPa
    result = np.logical_or(precipitation > precipitation_threshold,
                           vapour_pressure_deficit < vapour_pressure_deficit_threshold)
    return result


def compute_leaf_wetness_duration(df_weather_day):
    # Zandelin, P. (2021). Virtual weather data for apple scab monitoring and management.
    result = df_weather_day.apply(lambda row: is_wet(row['precipitation'], row['vapour_pressure_deficit']),
                                  axis=1).sum()
    return result


def summarize_weather(dates, df_weather):
    result_data = {'Date': [], 'LeafWetness': []}
    for day in dates:
        result_data['Date'].append(day)
        df_weather_day = df_weather.loc[day.strftime('%Y-%m-%d')]
        leaf_wetness = compute_leaf_wetness_duration(df_weather_day)
        result_data['LeafWetness'].append(leaf_wetness)
    return pd.DataFrame(result_data)


def is_rain_event(df_weather_day, threshold=0.2, max_gap=2):
    precipitation = torch.tensor(df_weather_day['precipitation'].to_numpy())
    rain = precipitation >= threshold
    result = fill_gaps(rain, max_gap=max_gap)
    return result