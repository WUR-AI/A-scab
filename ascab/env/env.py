import gymnasium as gym
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random

from ascab.utils.weather import get_meteo, summarize_weather, WeatherSummary
from ascab.utils.plot import plot_results, plot_infection
from ascab.model.maturation import PseudothecialDevelopment, AscosporeMaturation, LAI
from ascab.model.infection import InfectionRate, get_values_last_infections, get_discharge_date, will_infect, get_risk


def get_default_location():
    return 50.8, 5.2


def get_default_dates():
    return "2011-02-01", "2011-08-01"


def get_weather_params(location: tuple = None, dates: list[str] = None):
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
    # See figure 2 in Rossi et al. for an overview
    def __init__(self, location: tuple = None, dates: list[str] = None, seed: int = 42, verbose: bool = False):
        if location is None:
            location = get_default_location()
        if dates is None:
            dates = get_default_dates()

        super().reset(seed=seed)

        pseudothecia = PseudothecialDevelopment()
        ascospore = AscosporeMaturation(pseudothecia)
        lai = LAI()
        self.verbose = verbose
        self.dates = tuple(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
        self.models = {type(model).__name__: model for model in [pseudothecia, ascospore, lai]}
        self.infections = []
        self.weather = get_meteo(get_weather_params(location, dates), False)
        self.date = datetime.strptime(dates[0], '%Y-%m-%d').date()
        self.info = {"Date": [],
                     **{name: [] for name, _ in self.models.items()},
                     "Ascospores": [], "Discharge": [], "Infections": [], "Risk": [],
                     **{name: [] for name in WeatherSummary.get_variable_names()},
                     "Action": [], "Reward": []}

        self.observation_space_disease = gym.spaces.Dict({
            name: gym.spaces.Box(0, np.inf, shape=(), dtype=np.float32)
            for name, _ in self.info.items()
            if name in {"AscosporeMaturation", "PseudothecialDevelopment", "Ascospores", "Infections", "Risk"}
        })

        self.observation_space_tree = gym.spaces.Dict({
            name: gym.spaces.Box(0, np.inf, shape=(), dtype=np.float32)
            for name, _ in self.info.items() if name in {"LAI"}
        })

        self.observation_space_weather_summary = gym.spaces.Dict({
            name: gym.spaces.Box(0, np.inf, shape=(), dtype=np.float32)
            for name, _ in self.info.items() if name in WeatherSummary.get_variable_names()
        })

        self.observation_space = gym.spaces.Dict(
            {
                "disease": self.observation_space_disease,
                "tree": self.observation_space_tree,
                "weather": self.observation_space_weather_summary,
            }
        )
        self.action_space = gym.spaces.Box(0, 1.0, shape=(), dtype=np.float32)
        self.render_mode = 'human'

    def step(self, action):
        """
        Perform a single step in the Gym environment.
        """
        self.info["Date"].append(self.date)

        df_summary_weather = summarize_weather([self.date], self.weather)
        varnames = [col for col in self.info.keys() if col in df_summary_weather.columns]
        weather_observation = df_summary_weather[varnames].to_dict(orient="list")
        [self.info[key].extend(value) for key, value in weather_observation.items() if key != "Date"]

        df_weather_day = self.weather.loc[self.date.strftime("%Y-%m-%d")]
        for model in self.models.values():
            model.update_rate(df_weather_day)
        for model in self.models.values():
            model.integrate()
        for model in self.models.values():
            self.info[model.__class__.__name__].append(model.value)

        lai_value = self.models['LAI'].value
        ascospore_value = self.models['AscosporeMaturation'].value
        time_previous, pat_previous = get_values_last_infections(self.infections)
        discharge_date = get_discharge_date(df_weather_day, pat_previous, ascospore_value, time_previous)

        self.info['Discharge'].append(discharge_date is not None)
        self.info['Ascospores'].append(ascospore_value - pat_previous)
        self.info["Action"].append(action)

        if discharge_date is not None:
            end_day = self.date + timedelta(days=5)
            df_weather_infection = self.weather.loc[self.date.strftime("%Y-%m-%d"):end_day.strftime("%Y-%m-%d")]
            infect, infection_duration, infection_temperature = will_infect(df_weather_infection)
            if infect:
                self.infections.append(InfectionRate(discharge_date, ascospore_value, pat_previous, lai_value, infection_duration, infection_temperature))
            else:
                if self.verbose: print(f'No infection {infection_duration} {infection_temperature}')
        for infection in self.infections:
            infection.progress(df_weather_day, action)
        self.info["Infections"].append(len(self.infections))
        self.info["Risk"].append(get_risk(self.infections, self.date))
        o = self._get_observation()
        r = self._get_reward()
        i = self.get_info()
        self.info["Reward"].append(r)

        self.date = self.date + timedelta(days=1)

        return o, r, self._terminated(), False, i
        
    def _get_observation(self) -> dict:
        def generate_observation(info, observation_space_keys):
            o = {name: np.array(value[-1], dtype=np.float32) if value else np.array(0.0, dtype=np.float32)
                 for name, value in info.items() if name in observation_space_keys}
            return o

        disease_observation = generate_observation(self.info, self.observation_space_disease.keys())
        tree_observation = generate_observation(self.info, self.observation_space_tree.keys())
        weather_observation = generate_observation(self.info, self.observation_space_weather_summary.keys())
        result = {
            "disease": disease_observation,
            "tree": tree_observation,
            "weather": weather_observation,
        }
        return result

    def get_info(self, to_dataframe: bool = False):
        result = self.info
        if to_dataframe:
            result = pd.DataFrame(self.info).assign(Date=lambda x: pd.to_datetime(x["Date"]))
        return result

    def _terminated(self):
        return self.date >= self.dates[1]

    def _get_reward(self):
        risk = self.info["Risk"][-1]
        action = self.info["Action"][-1]
        result = -risk -(action * 0.025)
        return result

    def render(self):
        df_info = self.get_info(to_dataframe=True)
        plot_results(df_info)

        if self.infections:
            plot_infection(max(self.infections, key=lambda x: x.risk[-1][1]))
            plot_infection(random.choice(self.infections))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.__init__()
        return self._get_observation(), self.get_info()
