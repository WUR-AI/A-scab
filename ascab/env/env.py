import gymnasium as gym
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from gymnasium.wrappers import FlattenObservation


from ascab.utils.weather import get_meteo, summarize_weather, WeatherSummary
from ascab.utils.plot import plot_results, plot_infection
from ascab.model.maturation import PseudothecialDevelopment, AscosporeMaturation, LAI
from ascab.model.infection import InfectionRate, get_values_last_infections, get_discharge_date, will_infect, get_risk


def get_default_location():
    return 50.8, 5.2


def get_default_dates():
    return "2011-02-01", "2011-08-01"


def get_weather_params(location=None, dates=None):
    if location is None:
        location = get_default_location()
    if dates is None:
        dates = get_default_dates()
    latitude, longitude = location
    start_date, end_date = dates
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "vapour_pressure_deficit",
            "is_day",
        ],
        "timezone": "auto",
    }
    return params


class AScabEnv(gym.Env):

    def __init__(self, location=None, dates=None, seed=42):
        if location is None:
            location = get_default_location()
        if dates is None:
            dates = get_default_dates()

        self.action_space = gym.spaces.Box(0, np.inf, shape=(1,))

        super().reset(seed=seed)

        pseudothecia = PseudothecialDevelopment()
        ascospore = AscosporeMaturation(pseudothecia)
        lai = LAI()
        self.dates = tuple(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
        self.models = {type(model).__name__: model for model in [pseudothecia, ascospore, lai]}
        self.infections = []
        self.weather = get_meteo(get_weather_params(location, dates), True)
        self.date = datetime.strptime(dates[0], '%Y-%m-%d').date()
        self.result_data = {"Date": [],
                            **{name: [] for name, _ in self.models.items()},
                            "Ascospores": [], "Discharge": [], "Infections": [], "Risk": []}

        self.observation_space_model = gym.spaces.Dict({
            name: gym.spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
            for name, _ in self.result_data.items() if name != "Date"
        })

        self.observation_space_weather_summary = gym.spaces.Dict({
            name: gym.spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)
            for name in WeatherSummary.get_variable_names()
        })

        self.observation_space = gym.spaces.Dict(
            {
                "model": self.observation_space_model,
                "weather": self.observation_space_weather_summary,
            }
        )
        self.action_space = gym.spaces.Box(0, np.inf, shape=(1,), dtype=np.float32)

    def step(self, action):
        """
        Perform a single step in the Gym environment.
        """

        df_weather_day = self.weather.loc[self.date.strftime("%Y-%m-%d")]

        self.result_data["Date"].append(self.date)
        for model in self.models.values():
            model.update_rate(df_weather_day)
        for model in self.models.values():
            model.integrate()
        for model in self.models.values():
            self.result_data[model.__class__.__name__].append(model.value)

        lai_value = self.models['LAI'].value
        ascospore_value = self.models['AscosporeMaturation'].value
        time_previous, pat_previous = get_values_last_infections(self.infections)
        discharge_date = get_discharge_date(df_weather_day, pat_previous, ascospore_value, time_previous)

        self.result_data['Discharge'].append(discharge_date is not None)
        self.result_data['Ascospores'].append(ascospore_value - pat_previous)

        if discharge_date is not None:
            end_day = self.date + timedelta(days=5)
            df_weather_infection = self.weather.loc[self.date.strftime("%Y-%m-%d"):end_day.strftime("%Y-%m-%d")]
            infect, infection_duration, infection_temperature = will_infect(df_weather_infection)
            if infect:
                self.infections.append(InfectionRate(discharge_date, ascospore_value, pat_previous, lai_value, infection_duration, infection_temperature))
            else:
                print(f'No infection {infection_duration} {infection_temperature}')
        for infection in self.infections:
            infection.progress(df_weather_day, action)
        self.result_data["Infections"].append(len(self.infections))
        self.result_data["Risk"].append(get_risk(self.infections, self.date))
        o = self._get_observation()
        r = self._get_reward()
        i = self._get_info()

        self.date = self.date + timedelta(days=1)

        return o, r, self._terminated(), False, i
        
    def _get_observation(self) -> dict:

        model_observation = {
            name: [value[-1]] if value else [0.0] for name, value in self.result_data.items() if name != "Date"
        }
        df_summary_weather = summarize_weather([self.date], self.weather)
        varnames = list(self.observation_space_weather_summary.spaces.keys())
        weather_observation = df_summary_weather[varnames].to_dict(orient="list")
        result = {
            "model": model_observation,
            "weather": weather_observation,
        }
        return result

    def _get_info(self):
        result = dict()
        return result

    def _terminated(self):
        return self.date >= self.dates[1]

    def _get_reward(self):
        risk = self.result_data["Risk"][-1]
        return -risk

    def render(self):
        start_end = [self.dates[0] + timedelta(n) for n in range((self.dates[1] - self.dates[0]).days + 1)]
        weather_summary = summarize_weather(start_end, self.weather)
        df_result = pd.DataFrame(self.result_data).assign(Date=lambda x: pd.to_datetime(x["Date"]))
        merged_df = pd.merge(df_result, weather_summary, on="Date", how="inner")
        plot_results(merged_df)
        if self.infections:
            import random
            plot_infection(random.choice(self.infections))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.__init__()
        return self._get_observation(), self._get_info()


if __name__ == "__main__":
    ascab = FlattenObservation(AScabEnv())
    terminated = False
    total_reward = 0.0
    while not terminated:
        observation, reward, terminated, truncated, info = ascab.step(1)
        total_reward += reward
    print(f'reward: {total_reward}')
    ascab.render()