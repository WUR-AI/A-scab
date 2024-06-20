import pandas as pd
import datetime

from ascab.utils.weather import get_meteo, summarize_weather
from ascab.utils.plot import plot_results
from ascab.model.maturation import PseudothecialDevelopment, AscosporeMaturation, LAI

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
    result_data = {'Date': [], **{model.__class__.__name__: [] for model in models}}
    for day in dates:
        result_data['Date'].append(day)
        df_weather_day = df_weather.loc[day.strftime('%Y-%m-%d')]
        for m in models:
            m.update_rate(df_weather_day)
        for m in models:
            m.integrate()
        for m in models:
            result_data[m.__class__.__name__].append(m.value.clone())
    return pd.DataFrame(result_data)


def main():
    start_date = datetime.datetime.strptime(params['start_date'], "%Y-%m-%d")
    end_date = datetime.datetime.strptime(params['end_date'], "%Y-%m-%d")
    dates = [start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)]

    df_weather = get_meteo(params, True)
    weather_summary = summarize_weather(dates, df_weather)

    results_df = run_episode(dates, df_weather)

    merged_df = pd.merge(results_df, weather_summary, on='Date', how='inner')
    plot_results(merged_df)


if __name__ == "__main__":
    main()
