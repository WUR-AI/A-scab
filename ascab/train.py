import datetime
import os
import abc
import pandas as pd
from typing import Optional
from gymnasium.wrappers import FlattenObservation, FilterObservation

from ascab.utils.plot import plot_results
from ascab.env.env import AScabEnv, MultipleWeatherASCabEnv, ActionConstrainer, get_weather_library


try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError:
    PPO = None


def get_default_observation_filter():
    return ["weather", "tree"]


class BaseAgent(abc.ABC):
    def __init__(self, ascab: Optional[AScabEnv] = None, render: bool = True):
        self.ascab = ascab or AScabEnv()
        self.render = render

    def run(self) -> pd.DataFrame:
        observation, _ = self.ascab.reset()
        total_reward = 0.0
        terminated = False
        while not terminated:
            action = self.get_action(observation)
            observation, reward, terminated, _, _ = self.ascab.step(action)
            total_reward += reward

        print(f"Reward: {total_reward}")
        if self.render:
            self.ascab.render()

        return self.ascab.unwrapped.get_info(to_dataframe=True)

    @abc.abstractmethod
    def get_action(self, observation: Optional[dict] = None) -> float:
        pass


class CheatingAgent(BaseAgent):
    def get_action(self, observation: dict = None) -> float:
        if (
            self.ascab.get_wrapper_attr("info")["Discharge"]
            and self.ascab.get_wrapper_attr("info")["Discharge"][-1] > 0.5
        ):
            return 1.0
        return 0.0


class ZeroAgent(BaseAgent):
    def get_action(self, observation: dict = None) -> float:
        return 0.0


class FillAgent(BaseAgent):
    def __init__(
        self,
        ascab: Optional[AScabEnv] = None,
        render: bool = True,
        pesticide_threshold: float = 0.1
    ):
        super().__init__(ascab=ascab, render=render)
        self.pesticide_threshold = pesticide_threshold

    def get_action(self, observation: dict = None) -> float:
        if self.ascab.get_wrapper_attr("info")["Pesticide"] and self.ascab.get_wrapper_attr("info")["Pesticide"][-1] < self.pesticide_threshold:
            return 1.0
        return 0.0


class ScheduleAgent(BaseAgent):
    def __init__(
        self,
        ascab: Optional[AScabEnv] = None,
        render: bool = True,
        dates: list[datetime.date] = None
    ):
        super().__init__(ascab=ascab, render=render)
        if dates is None:
            year = self.ascab.get_wrapper_attr("date").year
            dates = [datetime.date(year, 4, 1), datetime.date(year, 4, 8)]
        self.dates = dates

    def get_action(self, observation: dict = None) -> float:
        if self.ascab.get_wrapper_attr("info")["Date"] and self.ascab.get_wrapper_attr("info")["Date"][-1] in self.dates:
            return 1.0
        return 0.0


class UmbrellaAgent(BaseAgent):
    def __init__(
        self,
        ascab: Optional[AScabEnv] = None,
        render: bool = True,
        pesticide_threshold: float = 0.1
    ):
        super().__init__(ascab=ascab, render=render)
        self.pesticide_threshold = pesticide_threshold

    def get_action(self, observation: dict = None) -> float:
        if self.ascab.get_wrapper_attr("info")["Forecast_day1_HasRain"] and self.ascab.get_wrapper_attr("info")["Forecast_day1_HasRain"][-1]:
            if self.ascab.get_wrapper_attr("info")["Pesticide"] and self.ascab.get_wrapper_attr("info")["Pesticide"][-1] < self.pesticide_threshold:
                return 1.0
        return 0.0


class RLAgent(BaseAgent):
    def __init__(
        self,
        ascab_train: Optional[AScabEnv] = None,
        ascab_test: Optional[AScabEnv] = None,
        n_steps: int = 5000,
        observation_filter: Optional[list] = get_default_observation_filter(),
        render: bool = True,
        path_model: Optional[str] = None,
        path_log: Optional[str] = None
    ):
        super().__init__(ascab=ascab_train, render=render)
        self.ascab_train = ascab_train
        self.ascab = ascab_test
        self.n_steps = n_steps
        self.observation_filter = observation_filter
        self.path_model = path_model
        self.path_log = path_log
        self.model = None

        self.train()

    def train(self):
        if PPO is None:
            raise ImportError(
                "stable-baselines3 is not installed. Please install it to use the rl_agent."
            )
        if self.observation_filter:
            print(f"Filter observations: {self.observation_filter}")
            self.ascab_train = FilterObservation(self.ascab_train, filter_keys=self.observation_filter)
            self.ascab = FilterObservation(self.ascab, filter_keys=self.observation_filter)
        self.ascab_train = FlattenObservation(self.ascab_train)
        self.ascab = FlattenObservation(self.ascab)
        if self.path_model is not None and (os.path.exists(self.path_model) or os.path.exists(self.path_model + ".zip")):
            print(f'Load model from disk: {self.path_model}')
            self.model = PPO.load(env=self.ascab_train, path=self.path_model, print_system_info=False)
        else:
            eval_callback = EvalCallback(
                eval_env=Monitor(self.ascab),
                eval_freq=1500,
                deterministic=True,
                render=False,
            )
            self.model = PPO("MlpPolicy", self.ascab_train, verbose=1, seed=42, tensorboard_log=self.path_log)
            self.model.learn(total_timesteps=self.n_steps, callback=[eval_callback])
            if self.path_model is not None:
                self.model.save(self.path_model)

    def get_action(self, observation: Optional[dict] = None) -> float:
        return self.model.predict(observation, deterministic=True)[0]


if __name__ == "__main__":
    ascab_env = AScabEnv(
        location=(42.1620, 3.0924), dates=("2022-02-01", "2022-09-30"),
        biofix_date="March 10", budbreak_date="March 10")
    ascab_env = ActionConstrainer(ascab_env)

    print("zero agent")
    zero_agent = ZeroAgent(ascab=ascab_env, render=False)  # -0.634
    zero_results = zero_agent.run()

    print("filling agent")
    fill_agent = FillAgent(ascab=ascab_env, pesticide_threshold=0.01, render=False)
    filling_results = fill_agent.run()

    print("cheating agent")
    cheating_agent = CheatingAgent(ascab=ascab_env, render=False)
    cheating_results = cheating_agent.run()

    print("schedule agent")
    schedule_agent = ScheduleAgent(ascab=ascab_env, render=False)
    schedule_results = schedule_agent.run()

    print("umbrella agent")
    umbrella_agent = UmbrellaAgent(ascab=ascab_env, render=False)
    umbrella_results = umbrella_agent.run()

    if PPO is not None:
        print("rl agent")
        save_path = os.path.join(os.getcwd(), "rl_agent_2023_2024_long")
        log_path = os.path.join(os.getcwd(), "log")
        ascab_train = MultipleWeatherASCabEnv(
            weather_data_library=get_weather_library(
                locations=[(42.1620, 3.0924), (42.1620, 3.0), (42.5, 2.5), (41.5, 3.0924), (42.5, 3.0924)], dates=[("2023-02-01", "2023-09-30"), ("2024-02-01", "2024-09-30")]
            ),
            biofix_date="March 10",
            budbreak_date="March 10",
        )
        ascab_rl = RLAgent(ascab_train=ascab_train, ascab_test=ascab_env, observation_filter=["weather", "tree", "disease"], n_steps=500000,
                           render=False, path_model=save_path, path_log=log_path)
        ascab_rl_results = ascab_rl.run()
        print(ascab_train.histogram)
        plot_results({"zero": zero_results, "filler": filling_results, "ubrella": umbrella_results, "rl": ascab_rl_results},
                     variables=["Precipitation", "LeafWetness", "AscosporeMaturation", "Discharge", "Infections", "Pesticide", "Risk", "Action"])
    else:
        print("Stable-baselines3 is not installed. Skipping RL agent.")
