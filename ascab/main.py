import pandas as pd
import datetime

from ascab.utils.weather import get_meteo, summarize_weather, summarize_rain
from ascab.utils.plot import plot_results, plot_precipitation_with_rain_event
from ascab.model.maturation import PseudothecialDevelopment, AscosporeMaturation, LAI
from ascab.model.infection import InfectionRate, get_values_last_infections, get_discharge_date

import matplotlib

matplotlib.use('TkAgg')

params = {
    "latitude": 50.8,
    "longitude": 5.2,
    "start_date": "2011-02-01",
    "end_date": "2011-08-01",
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "vapour_pressure_deficit", "is_day"],
    "timezone": "auto"
}


def run_episode(dates, df_weather):
    infections = []

    pseudothecia = PseudothecialDevelopment()
    ascospore = AscosporeMaturation(pseudothecia)
    lai = LAI()
    models = [pseudothecia, ascospore, lai]
    result_data = {'Date': [], **{model.__class__.__name__: [] for model in models}, 'Discharge': []}
    for day in dates:
        result_data['Date'].append(day)
        df_weather_day = df_weather.loc[day.strftime('%Y-%m-%d')]
        for m in models:
            m.update_rate(df_weather_day)
        for m in models:
            m.integrate()
        for m in models:
            result_data[m.__class__.__name__].append(m.value.clone())

        ascospore_value = models[1].value.clone()
        time_previous, pat_previous = get_values_last_infections(infections)
        discharge_date = get_discharge_date(df_weather_day, pat_previous, ascospore_value, time_previous)

        result_data['Discharge'].append(discharge_date is not None)
        if discharge_date is not None:
            infections.append(InfectionRate(discharge_date, ascospore_value, lai))

        for infection in infections:
            infection.progress(df_weather_day)
    return pd.DataFrame(result_data)


start_date = datetime.datetime.strptime(params['start_date'], "%Y-%m-%d")
end_date = datetime.datetime.strptime(params['end_date'], "%Y-%m-%d")
dates = [start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)]


def simulate():
    df_weather = get_meteo(params, True)
    weather_summary = summarize_weather(dates, df_weather)
    results_df = run_episode(dates, df_weather)
    merged_df = pd.merge(results_df, weather_summary, on='Date', how='inner')
    plot_results(merged_df)


def test_weather():
    # import random
    # day_to_plot = random.choice(dates)
    df_weather = get_meteo(params, True)
    df_rain = summarize_rain(dates, df_weather)
    day_to_plot = pd.Timestamp('2011-06-20')
    plot_precipitation_with_rain_event(df_rain, day_to_plot)
    day_to_plot = pd.Timestamp('2011-06-21')
    plot_precipitation_with_rain_event(df_rain, day_to_plot)


if __name__ == "__main__":
    simulate()
