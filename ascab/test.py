import pandas as pd
import datetime
import matplotlib
matplotlib.use('TkAgg')
from gymnasium.wrappers import FlattenObservation, FilterObservation

from ascab.utils.weather import get_meteo, summarize_rain
from ascab.utils.plot import plot_precipitation_with_rain_event
from ascab.env.env import AScabEnv

try:
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    check_env = None




params = {
    "latitude": 42.16200214762477,
    "longitude": 3.092425100060547,
    "start_date": "2022-08-15",
    "end_date": "2022-08-18",
    "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation", "vapour_pressure_deficit", "is_day"],
    "timezone": "auto"
}

start_date = datetime.datetime.strptime(params['start_date'], "%Y-%m-%d")
end_date = datetime.datetime.strptime(params['end_date'], "%Y-%m-%d")
start_end = [start_date + datetime.timedelta(n) for n in range((end_date - start_date).days + 1)]


def test_weather():
    # import random
    # day_to_plot = random.choice(dates)
    df_weather = get_meteo(params, True)
    df_rain = summarize_rain(start_end, df_weather)
    day_to_plot = pd.Timestamp('2022-08-15')
    plot_precipitation_with_rain_event(df_rain, day_to_plot)
    day_to_plot = pd.Timestamp('2022-08-18')
    plot_precipitation_with_rain_event(df_rain, day_to_plot)


def test_environment():
    ascab = AScabEnv()
    ascab = FilterObservation(ascab, filter_keys=["weather", "tree"])
    ascab = FlattenObservation(ascab)
    if check_env is not None:
        check_env(ascab)
    else:
        print("Stable-baselines3 is not installed. Skipping environment check.")


if __name__ == "__main__":
    test_environment()
    test_weather()
