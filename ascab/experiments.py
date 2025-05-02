import os
import numpy as np
import pandas as pd
import pickle
import argparse
import torch as th

import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


from ascab.env.env import (
    MultipleWeatherASCabEnv,
    get_weather_library,
    get_default_start_of_season,
    get_default_end_of_season,
    ActionConstrainer, get_weather_library_from_csv,
)
from ascab.utils.generic import get_dates
from ascab.train import RLAgent

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

def unique_path(path: str) -> str:
    """
    If `path` doesn’t exist, return it.
    Otherwise, append _1, _2, … before the extension (if any)
    until we find a free name.
    """
    base, ext = os.path.splitext(path)
    candidate = path
    counter = 1

    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1

    return candidate

def run_seed(seed: int) -> str:
    print("rl agent")
    print("seed:", seed)

    discrete_algos = ["PPO", "DQN", "RecurrentPPO"]
    algo = PPO
    constrain = False
    normalize = True
    truncated_observations='truncated'
    log_path = os.path.join(os.getcwd(), "log")
    name_agent = f"rl_agent_{algo.__name__}_weather_obs_seed{seed}"
    save_path = os.path.join(os.getcwd(), "log", name_agent)
    # os.makedirs(save_path, exist_ok=True)
    save_path = unique_path(save_path)

    ascab_train = MultipleWeatherASCabEnv(
        # weather_data_library=get_weather_library(
        #     locations=[(42.1620, 3.0924), (42.1620, 3.0), (42.5, 2.5), (41.5, 3.0924), (42.5, 3.0924)],
        #     dates=get_dates([year for year in range(2016, 2025) if year % 2 == 0],
        #                     start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
            weather_data_library=get_weather_library_from_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", 'train.csv')),
            biofix_date="March 10", budbreak_date="March 10",
            discrete_actions=True if algo.__name__ in discrete_algos else False,
            truncated_observations=truncated_observations
        )
    ascab_test = MultipleWeatherASCabEnv(
        # weather_data_library=get_weather_library(
        #     locations=[(42.1620, 3.0924)],
        #     dates=get_dates([year for year in range(2016, 2025) if year % 2 != 0],
        #                     start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
        weather_data_library=get_weather_library_from_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", 'val.csv')),
        biofix_date="March 10", budbreak_date="March 10", mode="sequential",
        discrete_actions=True if algo.__name__ in discrete_algos else False,
        truncated_observations=truncated_observations
        )

    observation_filter = list(ascab_train.observation_space.keys())

    if constrain:
        ascab_train = ActionConstrainer(ascab_train)
        ascab_test = ActionConstrainer(ascab_test)

    ascab_rl = RLAgent(ascab_train=ascab_train, ascab_test=ascab_test, observation_filter=observation_filter,
                       n_steps=1_000_000, render=False, path_model=save_path, path_log=log_path, rl_algorithm=algo,
                       seed=seed, normalize=normalize)
    print(ascab_train.histogram)
    print(ascab_test.histogram)
    if normalize:
        ascab_rl.ascab_train.save(os.path.join(save_path+"_norm.pkl"))
    results = ascab_rl.run()

    with open(save_path+".pkl", "wb") as f:
        pickle.dump(results, file=f)



    return save_path

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--multiprocess", type=bool, default=False)
    args = argparser.parse_args()
    rng= np.random.default_rng()
    random_int = rng.integers(low=0, high=1_000_000, size=1)[0]
    run_seed(int(args.seed))