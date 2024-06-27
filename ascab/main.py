import pandas as pd
import datetime

from ascab.utils.weather import get_meteo, summarize_weather, summarize_rain
from ascab.utils.plot import plot_results, plot_precipitation_with_rain_event, plot_infection
from ascab.model.maturation import PseudothecialDevelopment, AscosporeMaturation, LAI
from ascab.model.infection import InfectionRate, get_values_last_infections, get_discharge_date, compute_leaf_development, will_infect

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
    result_data = {'Date': [], **{model.__class__.__name__: [] for model in models}, 'LDR': [], 'Discharge': [], 'Infections': []}
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
        lai_value = models[2].value.clone()
        ldr = compute_leaf_development(lai_value)
        result_data['LDR'].append(ldr)

        time_previous, pat_previous = get_values_last_infections(infections)
        discharge_date = get_discharge_date(df_weather_day, pat_previous, ascospore_value, time_previous)

        result_data['Discharge'].append(discharge_date is not None)

        if discharge_date is not None:
            end_day = day + pd.DateOffset(days=5)
            df_weather_infection = df_weather.loc[day.strftime("%Y-%m-%d"):end_day.strftime("%Y-%m-%d")]
            infect, infection_duration, infection_temperature = will_infect(df_weather_infection)
            if infect:
                infections.append(InfectionRate(discharge_date, ascospore_value, pat_previous, lai, infection_duration, infection_temperature))
            else:
                print(f'No infection {infection_duration} {infection_temperature}')
        result_data["Infections"].append(len(infections))

        for infection in infections:
            infection.progress(df_weather_day)
    return pd.DataFrame(result_data), infections


start_date = datetime.datetime.strptime(params['start_date'], "%Y-%m-%d")
end_date = datetime.datetime.strptime(params['end_date'], "%Y-%m-%d")
dates = [start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)]


def simulate():
    df_weather = get_meteo(params, True)
    weather_summary = summarize_weather(dates, df_weather)
    results_df, infections = run_episode(dates, df_weather)
    merged_df= pd.merge(results_df, weather_summary, on='Date', how='inner')
    plot_infection(infections[0])
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
