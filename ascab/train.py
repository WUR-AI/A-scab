import datetime
import os
import abc
import pickle
import pandas as pd
from typing import Optional, Dict, Any, Union, Type
from scipy.optimize import basinhopping, Bounds
import numpy as np

import gymnasium
from gymnasium.wrappers import FlattenObservation, FilterObservation
from gymnasium import Wrapper
from stable_baselines3.common.env_util import make_vec_env

from ascab.utils.plot import plot_results
from ascab.utils.generic import get_dates
from ascab.env.env import AScabEnv, MultipleWeatherASCabEnv, ActionConstrainer, get_weather_library, get_default_start_of_season, get_default_end_of_season, PenaltyWrapper

from ascab.agent.ppo_lagrangian import LagrangianPPO, CostActorCriticPolicy, max_action_constraint

try:
    from comet_ml import Experiment
    from comet_ml.integration.gymnasium import CometLogger
    use_comet = True
except ImportError:
    use_comet = False

try:
    import torch as th
    import tensorboard
except ImportError:
    use_tensorboard = False


try:
    from stable_baselines3 import PPO, SAC, TD3, DQN, HER
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
except ImportError:
    PPO = None

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

class BaseAgent(abc.ABC):
    def __init__(self, ascab: Optional[AScabEnv] = None, render: bool = True):
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

    def get_n_eval_episodes(self):
        if isinstance(self.ascab, VecNormalize):
            n_eval_episodes = len(self.ascab.get_attr('weather_keys')[0]) if hasattr(self.ascab.unwrapped.envs[0], "weather_keys") else 1
        else:
            n_eval_episodes = len(self.ascab.unwrapped.weather_keys) if hasattr(self.ascab.unwrapped, "weather_keys") else 1
        return n_eval_episodes

    @staticmethod
    def filter_info(info):
        info = {k: v for k, v in info[0].items() if
                k not in {'TimeLimit.truncated', 'episode', 'terminal_observation'}}
        info = pd.DataFrame(info).assign(Date=lambda x: pd.to_datetime(x["Date"]))
        return info


class CeresAgent(BaseAgent):
    def __init__(self, ascab: Optional[AScabEnv] = None, render: bool = True):
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

def is_wrapped(env, wrapper_cls) -> bool:
    """Return True iff *env* (possibly deep‑inside) is an instance of *wrapper_cls*."""
    while isinstance(env, Wrapper):
        if isinstance(env, wrapper_cls):
            return True
        env = env.env                      # peel one wrapper layer
    return False

class RLAgent(BaseAgent):
    def __init__(
        self,
        ascab_train: Optional[AScabEnv] = None,
        ascab_test: Optional[AScabEnv] = None,
        n_steps: int = 5000,
        observation_filter: Optional[list] = None,
        render: bool = True,
        path_model: Optional[str] = None,
        path_log: Optional[str] = None,
        rl_algorithm: Union[Type[PPO],Type[TD3],Type[SAC],Type[DQN]] = None,
        discrete_actions: bool = False,
        normalize: bool = True,
        seed: int = 42,
        continue_training: bool = False,
        hyperparams: dict = {},
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
        self.hyperparams = hyperparams
        self.seed = seed

        if use_comet:
            self.comet = None


        self.train(seed)

    def train(self, seed: int = 42):
        if PPO is None:
            raise ImportError(
                "stable-baselines3 is not installed. Please install it to use the rl_agent."
            )

        callbacks = []

        if self.observation_filter:
            print(f"Filter observations: {self.observation_filter}")
            if self.ascab_train:
                self.ascab_train = FilterObservation(self.ascab_train, filter_keys=self.observation_filter)
                self.ascab_train = FlattenObservation(self.ascab_train)
            self.ascab = FilterObservation(self.ascab, filter_keys=self.observation_filter)
        self.ascab = FlattenObservation(self.ascab)
        if self.path_model is not None and (os.path.exists(self.path_model) or os.path.exists(self.path_model + ".zip")):
            print(f'Load model from disk: {self.path_model}')
            self.model = PPO.load(env=self.ascab_train if self.continue_training else self.ascab, path=self.path_model+".zip", print_system_info=False)
            if self.normalize and not self.ascab_train:
                self.ascab = Monitor(self.ascab)
                self.ascab = DummyVecEnv([lambda: self.ascab])
                self.ascab = VecNormalize.load(self.path_model+"_norm.pkl", self.ascab)
                self.ascab.training = False
                self.ascab.norm_reward = False
            if not self.continue_training:
                return
        else:

            self.ascab = Monitor(self.ascab)

            use_comet = True
            if use_comet:
                self.comet_logging()

            if self.normalize:
                self.ascab_train = VecNormalize(DummyVecEnv([lambda: self.ascab_train]), norm_obs=True,
                                                norm_reward=False if not is_wrapped(self.ascab_train, PenaltyWrapper) else True)
                self.ascab = VecNormalize(DummyVecEnv([lambda: self.ascab]), norm_obs=True, norm_reward=False if not is_wrapped(self.ascab_train, PenaltyWrapper) else True,
                                          training=False)
            eval_callback = CustomEvalCallback(
                eval_env=self.ascab,
                eval_freq=1500,
                deterministic=True,
                render=False,
                n_eval_episodes=len(self.ascab.weather_keys) if hasattr(self.ascab, "weather_keys") else 1,
                best_model_save_path=self.path_model,
            )
            callbacks.append(eval_callback)

        policy = "MlpPolicy" if self.algo != RecurrentPPO else "MlpLstmPolicy"
        policy = CostActorCriticPolicy if self.algo == LagrangianPPO else policy
        self.model = self.algo(policy, self.ascab_train, verbose=1, seed=seed, tensorboard_log=self.path_log,
                               **self.algo_hyperparams(self.algo), **self.lag_ppo())
        print(f"Training with seed {seed}...")
        self.model.learn(total_timesteps=self.n_steps, callback=callbacks)
        if self.path_model is not None:
            self.model.save(self.path_model)

    def get_action(self, observation: Optional[dict] = None) -> float:
        return self.model.predict(observation, deterministic=True)[0]

    def lag_ppo(self):
        return {"constraint_fn": max_action_constraint} if self.algo == LagrangianPPO else {}

    @staticmethod
    def algo_hyperparams(alg):
        # include algorithm specific hyperparams here!
        return {
            "gamma": 0.99,
            # "batch_size": 271,
            # "n_steps": 2168,
            "learning_rate": 0.001,
            # "ent_coef": 0.01,
            "policy_kwargs": {
                "ortho_init": False,
                # "net_arch":
                #     {"pi": [128, 128],
                #      "vf": [128, 128],
                #      "cf": [128, 128],}
            },
        }

    def comet_logging(self):
        rootdir = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(rootdir, 'comet_key', 'comet_key'), 'r') as f:
            api_key = f.readline()
        comet_log = Experiment(
            api_key=api_key,
            project_name="paper_experiments",
            workspace="ascabgym",
            log_code=True,
            log_graph=True,
            auto_metric_logging=True,
            auto_histogram_tensorboard_logging=True
        )
        comet_log.log_code(folder=os.path.join(rootdir, 'ascab'))
        comet_log.log_parameters(self.hyperparams)

        obs_space = self.ascab_train.unwrapped.observation_space
        act_space = self.ascab_train.unwrapped.action_space
        exp_params = {
            "obs": obs_space,
            "act": act_space,
        }
        comet_log.log_parameters(exp_params)

        comet_log.set_name(f'{self.algo.__name__}-{self.seed}')

        self.ascab_train = CometLogger(self.ascab_train, comet_log)
        self.comet = comet_log
        print("Using Comet!")

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
        # with open(save_path + "cer.pkl", "wb") as f:
        #     print(f"saved to {save_path+'cer.pkl'}")
        #     pickle.dump(ceres_results, file=f)
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
        print(ascab_train.histogram)
        print(ascab_test.histogram)
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

