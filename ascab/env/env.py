import gymnasium as gym
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import random
from typing import List, Union
from collections import defaultdict

from ascab.utils.weather import (get_meteo, summarize_weather, WeatherSummary, get_default_days_of_forecast,
                                 WeatherDataLibrary, get_first_full_day, get_last_full_day, construct_forecast)
from ascab.utils.plot import plot_results, plot_infection
from ascab.utils.generic import get_dates
from ascab.model.maturation import PseudothecialDevelopment, AscosporeMaturation, LAI, Phenology, get_default_budbreak_date
from ascab.model.infection import InfectionRate, Discharge, Pesticide, get_values_last_discharge, get_discharge_date, will_infect, get_risk, get_pat_threshold


def get_default_location():
    return 51.98680, 5.66359


def get_default_start_of_season():
    return "02-01"


def get_default_end_of_season():
    return "10-31"


def get_default_dates():
    return get_dates(years=[2024], start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())


def generate_hourly_list(days_of_forecast: int = get_default_days_of_forecast()) -> list:
    base_metrics = ["temperature_2m", "relative_humidity_2m", "precipitation"]
    hourly = []
    for metric in base_metrics:
        hourly.append(metric)
        for day in range(1, days_of_forecast + 1):
            hourly.append(f"{metric}_previous_day{day}")
    return hourly


def get_weather_params(location: tuple[float, float] = None, dates: tuple[str, str] = None,
                       days_of_forecast: int = get_default_days_of_forecast()):
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
        "hourly": generate_hourly_list(days_of_forecast),
        "timezone": "auto",
        "models": "jma_gsm"
    }
    return params


def get_weather_library(
    locations: List[tuple[float, float]],
    dates: Union[List[tuple[str, str]], tuple[str, str]],
    days_of_forecast: int = get_default_days_of_forecast(),
):
    result = WeatherDataLibrary()

    # Ensure dates is always a list of tuples for uniform processing
    if isinstance(dates, tuple):
        dates = [dates]

    for location in locations:
        for date_range in dates:
            result.collect_weather(
                params=get_weather_params(location=location, dates=date_range, days_of_forecast=days_of_forecast)
            )
    return result

def get_weather_library_from_csv(
        csv_path: str,
        *,
        location_col: str = "location",
        index_col: str = "date",
        location_id: str | None = None,
        years: list[int] | None = None,
):
    df = pd.read_csv(csv_path, parse_dates=[index_col])
    if index_col not in df.columns:
        raise ValueError(f"CSV must contain a '{index_col}' column.")
    if location_col not in df.columns:
        raise ValueError(f"CSV must contain a '{location_col}' column.")

    df[index_col] = pd.to_datetime(df[index_col],  utc=True,  errors="coerce")
    if location_id is not None:
        df = df[df[location_col] == location_id].copy()
    df = df.set_index(index_col).sort_index()
    if years is not None:
        df = df[df.index.year.isin(years)].copy()
    if df.empty:
        raise ValueError(f"No weather data found in {csv_path}.")

    result = WeatherDataLibrary()

    for loc_id, slice_ in df.groupby(location_col):

        lat = slice_.get("lat")
        lon = slice_.get("lon")
        lat = float(lat.iloc[0]) if lat is not None else None
        lon = float(lon.iloc[0]) if lon is not None else None

        for year, year_df in slice_.groupby(slice_.index.year, sort=False):
            params = dict(
                latitude=lat,
                longitude=lon,
                start_date=year_df.index.min().strftime("%Y-%m-%d"),
                end_date=year_df.index.max().strftime("%Y-%m-%d"),
            )

            key = f"{loc_id}_{year}_{params['start_date']}_{params['end_date']}"

            weather_df = year_df.drop(columns=[location_col], errors="ignore")

            result.collect_weather(
                params,
                key=key,
                loaded_weather=weather_df,
            )

    return result


def get_default_observations() -> list[str]:
    result = ['PseudothecialDevelopment', 'InfectionWindow', 'AscosporeMaturation', 'Discharge', 'Infections', 'Risk',
              'LAI', 'Phenology', 'Pesticide']
    result.extend(WeatherSummary.get_variable_names())
    result.append("Forecast")
    return result


def get_truncated_observations() -> list[str]:
    result = [
              # 'PseudothecialDevelopment',
              'InfectionWindow',
              'Discharge',
              'LAI',
              # 'Pesticide',
              'ActionHistory',
              # 'SinceLastAction',
              'AppliedPesticide',
              'Beta',
              'RemainingSprays'
              ]
    result.extend(WeatherSummary.get_variable_names())
    result.append("Forecast")
    return result


def get_weather_only_observations() -> list[str]:
    result = []
    result.extend(WeatherSummary.get_variable_names())
    result.append("Forecast")
    return result


def get_observation_set(a):
    if a == "full":
        return get_default_observations()
    elif a == "truncated":
        return get_truncated_observations()
    elif a == "weather_only":
        return get_weather_only_observations()
    else:
        raise ValueError("Invalid observation set. Choose from full, truncated, weather_only")



class AScabEnv(gym.Env):
    """
    Gymnasium Environment for Apple Scab Model (A-scab)

    The environment is designed based on the A-scab model described in Rossi et al. (2007).
    The A-scab model simulates the development of pseudothecia, ascospore maturation, discharge, deposition,
    and infection throughout the season. The simulation uses hourly weather data to predict these processes.
    The model produces a risk index for each infection period.

    The observation space comprises three elements:
        1. The state of the tree (e.g. LAI)
        2. The state of the fungus (e.g. Progress of infection)
        3. The state of the weather (e.g. LeafWetness)

    The action space comprises the amount of pesticide to spray, affecting the mortality rate of the fungus.

    Reference:
    Rossi, V., Giosue, S., & Bugiani, R. (2007). A‐scab (Apple‐scab), a simulation model for estimating
    risk of Venturia inaequalis primary infections. EPPO Bulletin, 37(2), 300-308.
    """

    def __init__(self, location: tuple[float, float] = get_default_location(), dates: tuple[str, str] = get_default_dates(),
                 weather: pd.DataFrame = None, weather_forecast: dict[int, pd.DataFrame] = None,
                 days_of_forecast: int = get_default_days_of_forecast(),
                 biofix_date: str = "March 10", budbreak_date: str = "March 10",
                 seed: int = 42, verbose: bool = False, discrete_actions: bool = False,
                 truncated_observations: str = 'truncated', action_budget: int = 8, beta: float = 0.025,
                 dynamic_beta: bool = False, beta_range: tuple[float, float] | None = None,
                 beta_curriculum: bool = False, beta_curriculum_start_fraction: float = 0.4,
                 dynamic_beta_hold_episodes: int = 1,):
        super().reset(seed=seed)

        self.seed = seed
        self.verbose = verbose
        self.dates = tuple(datetime.strptime(date, "%Y-%m-%d").date() for date in dates)
        self.weather = weather if weather is not None else get_meteo(get_weather_params(location, dates, days_of_forecast), verbose=False)
        self.weather_forecast = weather_forecast if weather_forecast is not None else construct_forecast(self.weather)
        self._reset_internal(biofix_date=biofix_date, budbreak_date=budbreak_date)
        self.total_pesticide_applied: float = 0.0
        self.since_last_spray: int = 0
        self.total_spraying_frequency: int = 0
        self.spray_budget: int = max(6, int(action_budget))
        self.remaining_sprays: int = self.spray_budget
        self.beta: float = beta
        self.dynamic_beta = dynamic_beta
        self.beta_range = tuple(beta_range) if beta_range is not None else (0.0, beta)
        self.current_beta_range = self.beta_range
        self.beta_curriculum = beta_curriculum
        self.beta_curriculum_start_fraction = beta_curriculum_start_fraction
        self.dynamic_beta_hold_episodes = max(1, int(dynamic_beta_hold_episodes))
        self.dynamic_beta_episodes_since_sample = self.dynamic_beta_hold_episodes
        self._validate_beta_range()
        self.set_beta_curriculum_progress(0.0)

        observation_filter = get_observation_set(truncated_observations)
        print(f"Truncated observations is {truncated_observations}")
        self.observation_space = gym.spaces.Dict({
            name: gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32)
            for name, _ in self.info.items()
            if name in observation_filter
            or "Forecast" in observation_filter
               and name.startswith("Forecast_") and name.split('_', 2)[-1] in observation_filter
        })
        self.is_discrete = discrete_actions
        self.action_space = gym.spaces.Box(0, 1.0, shape=(), dtype=np.float32) \
                            if not self.is_discrete \
                            else gym.spaces.Discrete(6)  # Discretize to bins of 0.05
        self.render_mode = 'human'

    def _get_days_of_forecast(self):
        return len(self.weather_forecast)

    def _reset_internal(self, biofix_date: str, budbreak_date: str):
        pseudothecia = PseudothecialDevelopment()
        ascospore = AscosporeMaturation(pseudothecia, biofix_date=biofix_date)
        lai = LAI(start_date=budbreak_date)
        phenology = Phenology()
        self.pesticide = Pesticide(dilution_rate_per_hour=0.006)

        self.models = {type(model).__name__: model for model in [pseudothecia, ascospore, lai, phenology]}
        self.infections = []
        self.discharges = []

        self.date = self.dates[0]
        self.info = {"Date": [], "DayOfYear": [],
                     **{name: [] for name, _ in self.models.items()},
                     "InfectionWindow": [], "Discharge": [], "Infections": [], "Risk": [], "Pesticide": [],
                     "ActionHistory": [], "SinceLastAction": [], "AppliedPesticide": [], "RemainingSprays": [], "Beta": [],
                     **{name: [] for name in WeatherSummary.get_variable_names()},
                     **{
                         f"Forecast_day{day}_{name}": []
                         for name in WeatherSummary.get_variable_names()
                         for day in range(1, self._get_days_of_forecast()+1)
                     },
                     "Action": [], "Reward": []}

    def step(self, action):
        """
        Perform a single step in the Gym environment.
        """
        self.info["Date"].append(self.date)

        # encode day into observation with cyclical sin and cos encoding
        day_of_year = self.date.timetuple().tm_yday
        # self.info.setdefault("SinDay", []).append(np.sin(2 * np.pi * day_of_year / 365.0))
        # self.info.setdefault("CosDay", []).append(np.cos(2 * np.pi * day_of_year / 365.0))

        df_summary_weather = summarize_weather([self.date], self.weather)
        varnames = [col for col in self.info.keys() if col in df_summary_weather.columns]
        weather_observation = df_summary_weather[varnames].to_dict(orient="list")
        [self.info[key].extend(value) for key, value in weather_observation.items() if key != "Date"]

        for day in range(1, self._get_days_of_forecast()+1):
            df_summary_weather_forecast = summarize_weather([self.date + timedelta(days=day)], self.weather_forecast[day])
            varnames = [col for col in self.info.keys() if col in df_summary_weather_forecast.columns]
            weather_forecast = df_summary_weather_forecast[varnames].to_dict(orient="list")
            [self.info[f'Forecast_day{day}_{key}'].extend(value) for key, value in weather_forecast.items() if key != "Date"]

        df_weather_day = self.weather.loc[self.date.strftime("%Y-%m-%d")]
        for model in self.models.values():
            model.update_rate(df_weather_day)
        for model in self.models.values():
            model.integrate()
        for model in self.models.values():
            self.info[model.__class__.__name__].append(model.value)

        # apply pesticide
        if isinstance(self.action_space, gym.spaces.Discrete):  # scale from Discrete to 0 to 1
            action = action / (self.action_space.n - 1)

        self._record_action(action)

        self.pesticide.update(df_weather_day=df_weather_day, action=action)
        lai_value = self.models['LAI'].value
        ascospore_value = self.models['AscosporeMaturation'].value
        time_previous, pat_previous = get_values_last_discharge(self.discharges)
        discharge_date = get_discharge_date(df_weather_day, pat_previous, ascospore_value, time_previous)

        self.info['Discharge'].append((discharge_date is not None) * (ascospore_value - pat_previous))
        self.info['InfectionWindow'].append(int(get_pat_threshold() < ascospore_value < 0.99))
        self.info["Action"].append(action)
        self.info["Pesticide"].append(self.pesticide.effective_coverage[-1])

        if discharge_date is not None:
            self.discharges.append(Discharge(discharge_date, ascospore_value))
            end_day = self.date + timedelta(days=10)
            df_weather_infection = self.weather.loc[self.date.strftime("%Y-%m-%d"):end_day.strftime("%Y-%m-%d")]
            infect, infection_duration, infection_temperature = will_infect(df_weather_infection)
            if infect:
                self.infections.append(InfectionRate(discharge_date, ascospore_value, pat_previous, lai_value,
                                                     infection_duration, infection_temperature))
            else:
                if self.verbose: print(f'No infection {self.date} {infection_duration} {infection_temperature}')

        for infection in self.infections:
            infection.progress(df_weather_day, self.pesticide.effective_coverage)
        self.info['ActionHistory'].append(self.total_spraying_frequency)
        self.info['SinceLastAction'].append(self.since_last_spray)
        self.info['AppliedPesticide'].append(self.total_pesticide_applied)
        self.info['RemainingSprays'].append(self.remaining_sprays)
        self.info['Beta'].append(self.beta)
        self.info['DayOfYear'].append(day_of_year)
        self.info["Infections"].append(len(self.infections))
        self.info["Risk"].append(get_risk(self.infections, self.date))
        o = self._get_observation()
        r = self._get_reward(beta=self.beta)
        i = self.get_info()
        self.info["Reward"].append(r)

        self.date = self.date + timedelta(days=1)

        return o, r, self._terminated(), False, i
        
    def _get_observation(self) -> dict:
        o = {}
        for name, value in self.info.items():
            if name not in set(self.observation_space.keys()):
                continue
            if name == "Beta" and not value:
                o[name] = np.array(self.beta, dtype=np.float32)
            else:
                o[name] = np.array(value[-1], dtype=np.float32) if value else np.array(0.0, dtype=np.float32)
        return o

    def get_info(self, to_dataframe: bool = False):
        result = self.info
        if to_dataframe:
            result = pd.DataFrame(self.info).assign(Date=lambda x: pd.to_datetime(x["Date"]))
        return result

    def _terminated(self):
        return self.date >= self.dates[1]

    def _get_reward(self, beta: float = 0.025):
        risk = self.info["Risk"][-1]
        action = self.info["Action"][-1]
        result = -risk - (action * beta)
        return float(result)

    def render(self):
        df_info = self.get_info(to_dataframe=True)
        plot_results(df_info)

        if self.infections:
            for infection in self.infections:
                plot_infection(infection)
            plot_infection(max(self.infections, key=lambda x: x.risk[-1][1]))
            plot_infection(random.choice(self.infections))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_action_records()
        self._reset_internal(biofix_date=self.models['AscosporeMaturation'].biofix_date,
                             budbreak_date=self.models['LAI'].start_date)
        self._sample_beta()
        return self._get_observation(), self.get_info()

    def _validate_beta_range(self):
        if len(self.beta_range) != 2:
            raise ValueError("beta_range must contain exactly two values: (min_beta, max_beta)")
        min_beta, max_beta = self.beta_range
        if min_beta > max_beta:
            raise ValueError("beta_range minimum must be less than or equal to maximum")
        if not 0.0 <= self.beta_curriculum_start_fraction <= 1.0:
            raise ValueError("beta_curriculum_start_fraction must be between 0.0 and 1.0")

    def set_beta(self, beta: float, dynamic_beta: bool | None = None):
        self.beta = float(beta)
        if dynamic_beta is not None:
            self.dynamic_beta = dynamic_beta
        if not self.dynamic_beta:
            self.current_beta_range = (self.beta, self.beta)
        self.dynamic_beta_episodes_since_sample = 0

    def set_beta_curriculum_progress(self, progress: float):
        if not self.dynamic_beta or not self.beta_curriculum:
            self.current_beta_range = self.beta_range
            return

        progress = float(np.clip(progress, 0.0, 1.0))
        min_beta, max_beta = self.beta_range
        midpoint = (min_beta + max_beta) / 2.0
        full_half_width = (max_beta - min_beta) / 2.0
        half_width_fraction = self.beta_curriculum_start_fraction + (
            (1.0 - self.beta_curriculum_start_fraction) * progress
        )
        half_width = full_half_width * half_width_fraction
        self.current_beta_range = (
            max(min_beta, midpoint - half_width),
            min(max_beta, midpoint + half_width),
        )

    def _sample_beta(self):
        if self.dynamic_beta:
            if self.dynamic_beta_episodes_since_sample >= self.dynamic_beta_hold_episodes:
                min_beta, max_beta = self.current_beta_range
                self.beta = float(self.np_random.uniform(min_beta, max_beta))
                self.dynamic_beta_episodes_since_sample = 1
            else:
                self.dynamic_beta_episodes_since_sample += 1

    def _record_action(self, action):
        if action > 0.0:
            self.total_spraying_frequency += 1
            self.total_pesticide_applied +=action
            self.since_last_spray = 0
            self.remaining_sprays -= 1
            self.remaining_sprays = max(0, self.remaining_sprays)
        if action == 0.0:
            self.since_last_spray += 1

    def _reset_action_records(self):
        self.total_spraying_frequency = 0
        self.total_pesticide_applied = 0.0
        self.since_last_spray = 0
        self.remaining_sprays = self.spray_budget

    def remaining_sprays_masker(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=np.bool_)
        # if self.remaining_sprays <= 0:
        #     mask[1:] = False
        return mask

    def _is_current_infection_window(self) -> bool:
        infection_window = self.info.get("InfectionWindow", [])
        if infection_window:
            return bool(infection_window[-1] == 1)

        ascospore_value = self.models["AscosporeMaturation"].value
        return get_pat_threshold() < ascospore_value < 0.99

    def action_masks(self) -> np.ndarray:
        mask = self.remaining_sprays_masker()
        if not self._is_current_infection_window():
            mask[1:] = False
        return mask


class MultipleWeatherASCabEnv(AScabEnv):
    def __init__(self,
                 weather_data_library: WeatherDataLibrary,
                 mode: str = "random",
                 discrete_actions: bool = False,
                 *args,
                 **kwargs):
        self.mode = mode
        self.weather_data_library = weather_data_library
        self.weather_keys = list(self.weather_data_library.data.keys())
        self.processed_keys = set()  # Track completed keys
        self.current_weather_key = self.weather_keys[0]  # Initialize the current weather key
        self.histogram = defaultdict(int)
        self.set_weather(self.current_weather_key)
        self.is_discrete = discrete_actions
        super().__init__(dates=(self.dates[0].strftime("%Y-%m-%d"), self.dates[1].strftime("%Y-%m-%d")),
                         weather=self.weather, weather_forecast=self.weather_forecast, discrete_actions=self.is_discrete,
                         *args, **kwargs)

    def set_weather(self, weather_key):
        self.weather = self.weather_data_library.get_weather(weather_key)
        self.weather_forecast = self.weather_data_library.get_weather_forecast(weather_key)
        start_date = get_first_full_day(self.weather)
        end_date = get_last_full_day(self.weather) + timedelta(-self._get_days_of_forecast())
        self.dates = start_date, end_date
        self.current_weather_key = weather_key  # Track the current weather key

    def get_next_weather_key(self):
        if self.mode == "random":
            return self.np_random.choice(list(self.weather_keys))
        elif self.mode == "sequential":
            # Select the next key based on the mode
            remaining_keys = set(self.weather_keys) - self.processed_keys
            return sorted(remaining_keys)[0]
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def reset(self, seed=None, options=None):
        if len(self.processed_keys) >= len(self.weather_keys):
            self.reset_processed_keys()
        # Get the next weather key
        weather_key = self.get_next_weather_key()
        self.set_weather(weather_key)
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        if done or truncated:
            self.processed_keys.add(self.current_weather_key)  # Mark this weather as processed
            self.histogram[self.current_weather_key] += 1
        return obs, reward, done, truncated, info

    def reset_processed_keys(self):
        """Reset the processed keys dictionary to allow reprocessing."""
        self.processed_keys.clear()


class ActionConstrainer(gym.ActionWrapper):
    """
    An action constrainer wrapper to constrain agents timestep-wise or frequency-wise (i.e., action of > 0.0)
    Use :param risk_period to constrain it only to act in certain timesteps
    and use :param action_budget to constrain it to act only a limited amount of times per-episode
    """
    def __init__(self, env: AScabEnv,
                 risk_period: bool = True,
                 action_budget: int = 0,

    ):
        super(ActionConstrainer, self).__init__(env)
        self.risk_period = risk_period
        self.action_budget = max(0, int(action_budget))
        self.actions_left = self.action_budget

    def reset(self, seed=None, options=None):
        self.actions_left = self.action_budget
        return super().reset(seed=seed, options=options)

    def action(self, action):
        if self.risk_period:
            action = self._constrain_risk_period(action)
        if self.action_budget > 0:
            action = self.constrain_action_budget(action)
        return action

    def _constrain_risk_period(self, action):
        if self.unwrapped.models["AscosporeMaturation"].value < get_pat_threshold() or self.unwrapped.models["AscosporeMaturation"].value > 0.99:
            return action * 0.0
        return action

    def constrain_action_budget(self, action):
        if action > 0.0:
            if self.actions_left > 0:
                self.actions_left -= 1
            else:
                action = 0.0
        return action


class EarlyTerminationWrapper(gym.Wrapper):
    def __init__(self,
                 env: AScabEnv,
                 max_actions: int = 8,
                 penalty: float = 0.0,
    ):
        super(EarlyTerminationWrapper, self).__init__(env)
        self.max_actions = max_actions
        self.actions_done = 0
        self.penalty = penalty

    def step(self, action):
        if action > 0.0:
            self.actions_done += 1

        obs, reward, terminated, truncated, info = self.env.step(action)

        if not terminated and self.actions_done > self.max_actions:
            truncated = True
            if self.penalty > 0.0:
                reward -= self.penalty

        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.actions_done = 0
        return super().reset(*args, **kwargs)


class PenaltyWrapper(gym.RewardWrapper):
    def __init__(self,
                 env: AScabEnv,
                 penalty: float = 0.1,
    ):
        super(PenaltyWrapper, self).__init__(env)
        self.penalty = max(0.0, penalty)

    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        if not terminated and self.env.unwrapped.remaining_sprays <= 1:
            if self.penalty > 0.0:
                reward -= self.penalty

        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.actions_done = 0
        return super().reset(*args, **kwargs)



