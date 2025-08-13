import datetime
import os
import abc
import pickle
import pandas as pd
from typing import Optional, Dict, Any, Union, Type
from scipy.optimize import basinhopping, Bounds
import numpy as np

from gymnasium.wrappers import FlattenObservation, FilterObservation

from ascab.utils.plot import plot_results
from ascab.utils.generic import get_dates
from ascab.env.env import AScabEnv, MultipleWeatherASCabEnv, ActionConstrainer, get_weather_library, get_default_start_of_season, get_default_end_of_season

import gymnasium as gym

try:
    from stable_baselines3 import PPO, SAC, TD3, DQN, HER
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
except ImportError:
    PPO = None

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None


def find_attr_in_wrapped_env(env_instance, attr_name, default=None):
    """
    Recursively searches for an attribute within a Gymnasium/SB3 wrapped environment stack.

    Args:
        env_instance: The top-level Gymnasium environment instance (can be wrapped).
        attr_name (str): The name of the attribute to find.
        default: The value to return if the attribute is not found.

    Returns:
        The value of the attribute if found, otherwise the default value.
    """
    current_env = env_instance
    visited_envs = set()  # To prevent infinite loops with circular references

    while current_env is not None:
        # Check if attribute exists on the current environment object
        if hasattr(current_env, attr_name):
            return getattr(current_env, attr_name)

        # Mark current env as visited
        env_id = id(current_env)
        if env_id in visited_envs:
            # We've seen this env before, preventing infinite loop (e.g., if .env points back)
            break
        visited_envs.add(env_id)

        # Try to go deeper
        next_env_candidate = None

        # 1. Standard Gym/SB3 Wrapper (e.g., Monitor, FlattenObservation)
        if hasattr(current_env, 'env') and isinstance(current_env.env, gym.Env):
            next_env_candidate = current_env.env
        # 2. Stable-Baselines3 VecEnv (e.g., DummyVecEnv, SubprocVecEnv)
        elif hasattr(current_env, 'envs') and isinstance(current_env.envs, list) and len(current_env.envs) > 0:
            # For VecEnvs, we assume all environments in the list are identical, and we search the first one.
            next_env_candidate = current_env.envs[0]
        # 3. Handle VecNormalize specifically, as its .env is often a VecEnv
        elif isinstance(current_env, (VecMonitor)) and hasattr(current_env, 'env') and isinstance(current_env.env,
                                                                                                  DummyVecEnv):
            next_env_candidate = current_env.env  # Go from VecMonitor to the underlying VecEnv

        if next_env_candidate is None:
            # No more standard ways to go deeper
            break

        current_env = next_env_candidate

    return default

def print_histogram_nicely(name, histogram):
    if histogram is None:
        print(f"{name} histogram: Not found.")
        return

    # Convert np.str_ keys to standard Python strings if they exist
    cleaned_histogram = {str(k): v for k, v in histogram.items()}

    print(f"\n--- {name} Histogram ---")
    for key, value in cleaned_histogram.items():
        print(f"  {key}: {value}")
    print("-----------------------\n")

class BaseAgent(abc.ABC):
    def __init__(self, ascab: Optional[AScabEnv] = None, render: bool = False):
        self.ascab = ascab or AScabEnv()
        self.render = render

    def run(self) -> pd.DataFrame:

        all_infos = []
        all_rewards = []
        n_eval_episodes = self.get_n_eval_episodes()
        for i in range(n_eval_episodes):
            observation = self.reset_ascab()
            total_reward = 0.0
            terminated = False
            while not terminated:
                action = self.get_action(observation)
                observation, reward, terminated, info = self.step_ascab(action)
                total_reward += reward
            all_rewards.append(total_reward)
            print(f"Reward: {total_reward}")
            all_infos.append(self.ascab.get_wrapper_attr('get_info')(to_dataframe=True)
                             if not isinstance(self.ascab, VecNormalize)
                             else self.filter_info(info))
            if self.render:
                self.ascab.render()
        return pd.concat(all_infos, ignore_index=True)

    @abc.abstractmethod
    def get_action(self, observation: Optional[dict] = None) -> float:
        pass

    def step_ascab(self, action):
        observation, reward, terminated, _, info = self.ascab.step(action)
        return observation, reward, terminated, info

    def reset_ascab(self):
        observation, _ = self.ascab.reset()
        return observation

    def get_n_eval_episodes(self):
        if find_attr_in_wrapped_env(self.ascab, 'weather_keys'):
            return len(find_attr_in_wrapped_env(self.ascab, 'weather_keys'))
        else:
            return 1

    @staticmethod
    def filter_info(info):
        info = {k: v for k, v in info[0].items() if
                k not in {'TimeLimit.truncated', 'episode', 'terminal_observation'}}
        info = pd.DataFrame(info).assign(Date=lambda x: pd.to_datetime(x["Date"]))
        return info


class CeresAgent(BaseAgent):
    def __init__(self, ascab: Optional[AScabEnv] = None, render: bool = False):
        super().__init__(ascab, render)
        self.full_action_sequence = None
        self.unmasked_indices = None
        self.current_step = 0

    def set_action_sequence(self, optimized_actions, unmasked_indices, action_length):
        # Create a full action sequence initialized with zeros
        self.full_action_sequence = np.zeros(action_length)
        self.unmasked_indices = unmasked_indices
        # Fill in the optimized actions only at unmasked indices
        self.full_action_sequence[unmasked_indices] = optimized_actions
        self.current_step = 0

    def get_action(self, observation: Optional[dict] = None) -> float:
        # Select action for current step
        action = self.full_action_sequence[self.current_step]
        self.current_step += 1
        return action


def objective(optimized_actions, ascab_env, unmasked_indices, action_length):
    # Create and set up the agent with masked and unmasked actions
    agent = CeresAgent(ascab=ascab_env, render=False)
    agent.set_action_sequence(optimized_actions, unmasked_indices, action_length)

    # Run the agent to get cumulative reward
    df_results = agent.run()
    cumulative_reward = df_results["Reward"].sum()

    return -cumulative_reward


class CeresOptimizer:
    def __init__(self, ascab, save_path):
        self.ascab = ascab
        self.save_path = save_path

        self.optimized_actions = None
        self.unmasked_indices = None
        self.action_length = None

    def check_existing_solution(self):
        """Check if a solution already exists on disk and load it if so."""
        if os.path.exists(self.save_path):
            print(f"Loading Ceres solution from {self.save_path}")
            self.optimized_actions = np.loadtxt(self.save_path)
            return True
        return False

    def run_optimizer(self):
        one_agent = OneAgent(ascab=self.ascab, render=False)
        one_agent_results = one_agent.run()
        mask = one_agent_results["Action"].to_numpy()
        self.unmasked_indices = np.where(mask == 1)[0]
        self.action_length = len(mask)

        """Run the optimizer to find the best action sequence."""
        if self.check_existing_solution():
            return

        # Initial actions, replace this with your logic as needed
        initial_actions = np.zeros(len(self.unmasked_indices))
        bounds = Bounds([0] * len(self.unmasked_indices), [1] * len(self.unmasked_indices))

        print("Starting ceres...")

        result = basinhopping(
            objective,
            initial_actions,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "args": (self.ascab, self.unmasked_indices, self.action_length),
                "bounds": bounds,
                "options": {"maxiter": 30},
            },
            niter=30,
        )

        self.optimized_actions = result.x
        np.savetxt(self.save_path, self.optimized_actions)

    def run_ceres_agent(self):
        """Run the Ceres agent with the optimized actions."""
        if self.optimized_actions is None:
            raise ValueError("Optimized actions have not been set. Please run the optimizer first.")

        ceres_agent = CeresAgent(ascab=self.ascab, render=False)
        ceres_agent.set_action_sequence(self.optimized_actions, self.unmasked_indices, self.action_length)

        # Run the agent
        results = ceres_agent.run()
        return results


class ZeroAgent(BaseAgent):
    def get_action(self, observation: dict = None) -> float:
        return 0.0


class OneAgent(BaseAgent):
    def get_action(self, observation: dict = None) -> float:
        return 1.0


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
        pesticide_threshold: float = 0.1,
        pesticide_filled_to: float = 0.5,
    ):
        super().__init__(ascab=ascab, render=render)
        self.pesticide_threshold = pesticide_threshold
        self.pesticide_filled_to = pesticide_filled_to

    def get_action(self, observation: dict = None) -> float:
        if self.ascab.get_wrapper_attr("info")["Forecast_day1_HasRain"] and self.ascab.get_wrapper_attr("info")["Forecast_day1_HasRain"][-1]:
            if self.ascab.get_wrapper_attr("info")["Pesticide"] and self.ascab.get_wrapper_attr("info")["Pesticide"][-1] < self.pesticide_threshold:
                return self.pesticide_filled_to - self.ascab.get_wrapper_attr("info")["Pesticide"][-1]
        return 0.0

class RandomAgent(BaseAgent):
    def __init__(
        self,
        ascab: Optional[AScabEnv] = None,
        render: bool = True,
        seed: Optional[int] = 42,
    ):
        super().__init__(ascab=ascab, render=render)
        self.random_generator = np.random.RandomState(seed)

    def get_action(self, observation: dict = None) -> float:
        return self.random_generator.uniform(0.0, 1.0)


class EvalLogger(BaseCallback):
    def __init__(self, tag: str = None):
        super(EvalLogger, self).__init__()
        self.tag = tag
    parent: EvalCallback

    def _on_step(self) -> bool:
        subdir = f"eval-{self.tag}" if self.tag is not None else "eval"
        info = self.parent.eval_env.buf_infos[0]
        for cum_var in ["Action", "Reward"]:
            self.logger.record(f"{subdir}/sum_{cum_var}", float(np.sum(info[cum_var])))
        self.logger.dump(self.parent.num_timesteps)
        return True


class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(CustomEvalCallback, self).__init__(*args, **kwargs)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        if locals_["done"]:
            info = locals_["info"]
            tag = info["Date"][0].year
            for cum_var in ["Action", "Reward"]:
                self.logger.record(f"eval/{tag}-sum_{cum_var}", float(np.sum(info[cum_var])))
            print(f'{tag}: {np.sum(info["Reward"])}')
            self.training_env.save(os.path.join(self.best_model_save_path+"_norm.pkl"))


class NewBestCallback(BaseCallback):
    def __init__(self, best_model_save_path: str):
        super(NewBestCallback, self).__init__()
        # Store the path provided by the parent EvalCallback
        self.best_model_save_path = best_model_save_path

    def _on_step(self) -> bool:
        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(os.path.join(self.best_model_save_path, "best_model_norm.pkl"))
        return True


class RLAgent(BaseAgent):
    def __init__(
        self,
        ascab_test: Optional[AScabEnv],
        ascab_train: Optional[AScabEnv] = None,
        n_steps: int = 5000,
        observation_filter: Optional[list] = None,
        render: bool = False,
        path_model: Optional[str] = None,
        path_log: Optional[str] = None,
        rl_algorithm: Union[Type[BaseAlgorithm]] = None,
        discrete_actions: bool = False,
        normalize: bool = True,
        seed: int = 42,
        continue_training: bool = False,
        hyperparameters: dict = {},
    ):
        super().__init__(ascab=ascab_train, render=render)
        self.ascab_train = ascab_train
        self.ascab = ascab_test
        self.n_steps = n_steps
        self.observation_filter = observation_filter
        self.path_model = path_model
        self.path_log = path_log
        self.model = None
        self.algo = rl_algorithm
        self.is_discrete = discrete_actions
        self.continue_training = continue_training
        self.normalize = normalize
        self.hyperparams = hyperparameters
        self.seed = seed

        if self.ascab_train is None:
            print('Training environment is not provided; going into evaluation mode!...')
            assert self.path_model is not None, "path_model must be set to go to evaluation mode!"
            self.evaluate(seed)
        else:
            print('Training environment is provided; going into training mode!...')
            self.train(seed)

    def train(self, seed: int = 42):

        if PPO is None:
            raise ImportError(
                "stable-baselines3 is not installed. Please install it to use the rl_agent."
            )

        callbacks = []

        if self.observation_filter:
            print(f"Filter observations: {self.observation_filter}")
            if set(self.observation_filter) != set(list(self.ascab.unwrapped.observation_space.keys())):
                print("Overriding observation space with defined observation filter!")
                self.ascab_train.unwrapped.override_observation_space(observation_keys=self.observation_filter)
                self.ascab.unwrapped.override_observation_space(observation_keys=self.observation_filter)

            self.ascab_train = FilterObservation(self.ascab_train, filter_keys=self.observation_filter)
            self.ascab = FilterObservation(self.ascab, filter_keys=self.observation_filter)
        self.ascab_train = FlattenObservation(self.ascab_train)
        self.ascab = FlattenObservation(self.ascab)

        self.ascab = Monitor(self.ascab)

        if self.normalize:
            assert not self.is_wrapped(self.ascab, DummyVecEnv), "Already wrapped with VecEnv!"
            assert not self.is_wrapped(self.ascab_train, DummyVecEnv), "Already wrapped with VecEnv!"

            self.ascab_train = VecNormalize(DummyVecEnv([lambda: self.ascab_train]), norm_obs=True,
                                            norm_reward=False)
            self.ascab = VecNormalize(DummyVecEnv([lambda: self.ascab]), norm_obs=True, norm_reward=False,
                                      training=False)
        eval_callback = CustomEvalCallback(
            eval_env=self.ascab,
            eval_freq=1500,
            deterministic=True,
            render=False,
            n_eval_episodes=self.get_n_eval_episodes(),
            best_model_save_path=self.path_model,
            callback_on_new_best=NewBestCallback(best_model_save_path=self.path_model)
        )
        callbacks.append(eval_callback)

        print(f"Training with {self.algo.__name__}!")
        print(f"Using the following hyperparameters to train:\n {self.hyperparams}") \
            if self.hyperparams else print("Using default hyperparameters!")

        policy = "MlpPolicy" if self.algo != RecurrentPPO else "MlpLstmPolicy"
        self.model = self.algo(policy, self.ascab_train, verbose=1, seed=seed, tensorboard_log=self.path_log,
                               **self.hyperparams)
        print(f"Training with seed {seed}...")
        self.model.learn(total_timesteps=self.n_steps, callback=callbacks)
        print(f"Training finished!")
        if self.path_model is not None:
            print(f"Attempting to save model...")
            self.model.save(self.path_model)
            print(f"Model saved to {self.path_model}!")
            if self.normalize:
                print(f"Attempting to save normalization object...")
                self.ascab_train.save(self.path_model + "_norm.pkl")
                print(f"Normalization object saved to {self.path_model}_norm.pkl!")
        print("Running trained model in A-scab environment...!")
        self.run()

    def evaluate(self, seed: int = 42):
        if self.observation_filter:
            print(f"Filter observations: {self.observation_filter}")
            if set(self.observation_filter) != set(list(self.ascab.unwrapped.observation_space.keys())):
                print("Overriding observation space with defined observation filter!")
                self.ascab.unwrapped.override_observation_space(observation_keys=self.observation_filter)
            self.ascab = FilterObservation(self.ascab, filter_keys=self.observation_filter)
        self.ascab = FlattenObservation(self.ascab)

        print("Checking if model exists in path_model...")

        if self.model_path_checks():
            print(f'Load model from disk: {self.path_model}')
            self.model = self.algo.load(env=self.ascab, path=self.path_model+".zip", print_system_info=False)
            if self.normalize:
                self.ascab = Monitor(self.ascab)
                self.ascab = DummyVecEnv([lambda: self.ascab])
                print(f'Load normalization parameters from disk: {self.path_model+"_norm.pkl"}')
                self.ascab = VecNormalize.load(self.path_model+"_norm.pkl", self.ascab)
                self.ascab.training = False
                self.ascab.norm_reward = False
            print("Running trained model in A-scab environment...!")
            self.run()
        else:
            print(f"No model found at path_model: {self.path_model}")
            print("Make sure path_model points to your Stable Baselines file! (without the .zip extension)")

    def model_path_checks(self):
        if os.path.isfile(self.path_model):
            print(f"Found it!")
            return True
        elif not os.path.isfile(self.path_model) and os.path.isfile(self.path_model + ".zip"):
            print(f"Found it!")
            return True
        else:
            return False

    @staticmethod
    def is_wrapped(env, wrapper_class):
        """Check if the env is wrapped with wrapper_class."""
        while hasattr(env, 'env'):
            if isinstance(env, wrapper_class):
                return True
            env = env.env
        return isinstance(env, wrapper_class)  # Also check the final unwrapped env


    def run(self) -> pd.DataFrame:
        all_infos = []
        all_rewards = []
        n_eval_episodes = self.get_n_eval_episodes()
        for i in range(n_eval_episodes):
            observation = self.reset_ascab()
            total_reward = 0.0
            terminated = False
            states = None
            episode_start = np.ones((1,), dtype=bool)
            while not terminated:
                action, states = self.get_action(observation, states=states, episode_start=episode_start)
                observation, reward, terminated, info = self.step_ascab(action)
                total_reward += reward
                episode_start = terminated
            all_rewards.append(total_reward)
            print(f'Reward: {total_reward}')
            all_infos.append(self.ascab.get_wrapper_attr('get_info')(to_dataframe=True)
                             if not isinstance(self.ascab, VecNormalize)
                             else self.filter_info(info))
            if self.render:
                self.ascab.render()
        return pd.concat(all_infos, ignore_index=True)

    def step_ascab(self, action):
        if not isinstance(self.ascab, VecNormalize):
            observation, reward, terminated, _, info = self.ascab.step(action)
        else:
            observation, reward, terminated, info = self.ascab.step(action)

        return observation, reward, terminated, info

    def reset_ascab(self):
        # check if
        if not isinstance(self.ascab, VecNormalize):
            observation, _ = self.ascab.reset()
        else:
            observation = self.ascab.reset()
        return observation

    def get_action(self, observation: Optional[dict] = None, states = None, episode_start = None) -> float:
            return self.model.predict(observation,
                                      state=states,
                                      episode_start=episode_start,
                                      deterministic=True)


if __name__ == "__main__":
    ascab_env = MultipleWeatherASCabEnv(
            weather_data_library=get_weather_library(
                locations=[(42.1620, 3.0924)],
                dates=get_dates([year for year in range(2016, 2025) if year % 2 != 0], start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
            biofix_date="March 10",
            budbreak_date="March 10",
            mode="sequential",
        )
    ascab_env_constrained = ActionConstrainer(ascab_env)

    print("zero agent")
    zero_agent = ZeroAgent(ascab=ascab_env_constrained, render=False)  # -0.634
    zero_results = zero_agent.run()

    print("filling agent")
    fill_agent = FillAgent(ascab=ascab_env_constrained, pesticide_threshold=0.1, render=False)
    filling_results = fill_agent.run()

    print("schedule agent")
    schedule_agent = ScheduleAgent(ascab=ascab_env_constrained, render=False)
    schedule_results = schedule_agent.run()

    print("umbrella agent")
    umbrella_agent = UmbrellaAgent(ascab=ascab_env_constrained, render=False)
    umbrella_results = umbrella_agent.run()

    use_random = False
    if use_random:
        print("random agent")
        rng = np.random.RandomState(seed=107)
        dict_rand = {}
        for i in range(1):
            random_agent = RandomAgent(ascab=ascab_env_constrained, render=False, seed=rng.randint(0, 100))
            random_results = random_agent.run()
            dict_rand[i] = random_results

    use_ceres = False
    if use_ceres:
        ceres_results = pd.DataFrame()
        for y in [year for year in range(2016, 2025) if year % 2 != 0]:
            ascab_env = MultipleWeatherASCabEnv(
                weather_data_library=get_weather_library(
                    locations=[(42.1620, 3.0924)],
                    dates=get_dates([y], start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
                biofix_date="March 10",
                budbreak_date="March 10",
                mode="sequential",
            )
            ascab_env_constrained = ActionConstrainer(ascab_env)
            optimizer = CeresOptimizer(ascab_env_constrained,
                                       os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                                    "ceres",
                                                    f"ceres_{y}.txt"))
            optimizer.run_optimizer()
            year_results = optimizer.run_ceres_agent()
            ceres_results = pd.concat([ceres_results, year_results], ignore_index=True)
        save_path = os.path.join(os.getcwd(), f"rl_agent_ceres")
        with open(save_path + ".pkl", "wb") as f:
            print(f"saved to {save_path + 'cer.pkl'}")
            pickle.dump(ceres_results, file=f)


    if PPO is not None:
        print("rl agent")
        discrete_algos = ["PPO", "DQN", "RecurrentPPO"]
        box_algos = ["TD3", "SAC", "A2C", "RecurrentPPO"]
        algo = PPO
        log_path = os.path.join(os.getcwd(), "log")
        save_path = os.path.join(os.getcwd(), f"rl_agent_train_odd_{algo.__name__}")
        ascab_train = MultipleWeatherASCabEnv(
            weather_data_library=get_weather_library(
                locations=[(42.1620, 3.0924), (42.1620, 3.0), (42.5, 2.5), (41.5, 3.0924), (42.5, 3.0924)],
                dates=get_dates([year for year in range(2016, 2025) if year % 2 == 0], start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
            biofix_date="March 10", budbreak_date="March 10", discrete_actions=True if algo.__name__ in discrete_algos else False,
        )
        ascab_test = MultipleWeatherASCabEnv(
            weather_data_library=get_weather_library(
                locations=[(42.1620, 3.0924)],
                dates=get_dates([year for year in range(2016, 2025) if year % 2 != 0], start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
            biofix_date="March 10", budbreak_date="March 10", mode="sequential", discrete_actions=True if algo.__name__ in discrete_algos else False
        )

        ascab_train = ActionConstrainer(ascab_train, action_budget=8)
        ascab_test = ActionConstrainer(ascab_test, action_budget=8)

        observation_filter = list(ascab_train.observation_space.keys())

        ascab_rl = RLAgent(ascab_train=ascab_train, ascab_test=ascab_test, observation_filter=observation_filter, n_steps=100,
                           render=False, path_model=save_path, path_log=log_path, rl_algorithm=algo)

        print_histogram_nicely("Training Environment", find_attr_in_wrapped_env(ascab_train, 'histogram'))
        print_histogram_nicely("Test Environment", find_attr_in_wrapped_env(ascab_test, 'histogram'))

        ascab_rl_results = ascab_rl.run()

    else:
        print("Stable-baselines3 is not installed. Skipping RL agent.")

    all_results_dict = {"zero": zero_results, "umbrella": umbrella_results, }
    if use_random:
        all_results_dict["random"] = list(dict_rand.keys())[0]
    if use_ceres:
        all_results_dict["ceres"] = ceres_results
    if PPO:
        all_results_dict["rl"] = ascab_rl_results

    plot_results(all_results_dict,
                 save_path=os.path.join(os.getcwd(), "results.png"),
                 variables=["Precipitation", "LeafWetness", "AscosporeMaturation", "Discharge", "Pesticide", "Risk",
                            "Action", "Phenology"])