import os
import numpy as np
import pandas as pd
import pickle
import argparse

try:
    import comet_ml
    use_comet = True
except ImportError:
    use_comet = False

import gymnasium as gym

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


from ascab.env.env import (
    MultipleWeatherASCabEnv,
    get_weather_library,
    get_default_start_of_season,
    get_default_end_of_season,
    ActionConstrainer,
    EarlyTerminationWrapper,
    PenaltyWrapper
)
from ascab.utils.generic import get_dates
from ascab.train import RLAgent
from ascab.utils.plot import plot_results


from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO, CrossQ, MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from ascab.agent.ppo_lagrangian import LagrangianPPO

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

def run_seed(seed: int, n_steps: int, algo = PPO) -> str:
    print("rl agent")
    print("seed:", seed)

    print(f"Using {algo.__name__}")

    discrete_algos = ["PPO", "DQN", "RecurrentPPO", "LagrangianPPO", "MaskablePPO"]
    # algo = PPO
    constrain = False
    terminate_early = False
    penalty_wrap = False
    normalize = True
    truncated_observations='truncated'
    log_path = os.path.join(os.getcwd(), "log")
    name_agent = f"rl_agent_{algo.__name__}_constrained_seed{seed}"
    save_path = os.path.join(os.getcwd(), "log", name_agent)
    # os.makedirs(save_path, exist_ok=True)
    save_path = unique_path(save_path)

    ascab_train = MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library(
            locations=[(42.1620, 3.0924), (42.1620, 3.0), (42.5, 2.5), (41.5, 3.0924), (42.5, 3.0924)],
            dates=get_dates([year for year in range(2016, 2025) if year % 2 == 0],
                            start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
        biofix_date="March 10", budbreak_date="March 10", discrete_actions=True if algo.__name__ in discrete_algos else False,
        )
    ascab_test = MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library(
            locations=[(42.1620, 3.0924)],
            dates=get_dates([year for year in range(2016, 2025) if year % 2 != 0],
                            start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
        biofix_date="March 10", budbreak_date="March 10", discrete_actions=True if algo.__name__ in discrete_algos else False, mode='sequential'
        )

    observation_filter = list(ascab_train.observation_space.keys())

    if constrain:
        ascab_train = ActionConstrainer(ascab_train, risk_period=False, action_budget=8)
        ascab_test = ActionConstrainer(ascab_test, risk_period=False, action_budget=8)

    if terminate_early:
        ascab_train = EarlyTerminationWrapper(ascab_train, penalty=1.0)
        ascab_test = EarlyTerminationWrapper(ascab_test, penalty=1.0)

    if penalty_wrap:
        ascab_train = PenaltyWrapper(ascab_train, penalty=0.05)
        ascab_test = PenaltyWrapper(ascab_test, penalty=0.05)

    if algo == MaskablePPO:
        ascab_train = ActionMasker(ascab_train, lambda e: e.remaining_sprays_masker())
        ascab_test = ActionMasker(ascab_test, lambda e: e.remaining_sprays_masker())

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

    plot_results(
        {"rl_agent": results},
        variables=[
            "Precipitation",
            "AscosporeMaturation",
            "Discharge",
            "Pesticide",
            "Risk",
            "Action",
        ],
        save_path=os.path.join(save_path),
        per_year=True,
    )

    if use_comet:
        ascab_rl.comet.log_asset(file_data=os.path.join(save_path+".pkl"),
                                 file_name=f'{seed}-results')
        ascab_rl.comet.log_asset(file_data=os.path.join(save_path + "_norm.pkl"),
                                 file_name=f'{seed}-norm_stats')
        ascab_rl.comet.log_asset(file_data=os.path.join(save_path + ".zip"),
                                 file_name=f'{seed}-model')
        for year in ["2017", "2019", "2021", "2023"]:
            name_plot = f"plot_{year}.png"
            ascab_rl.comet.log_asset(file_data=os.path.join(save_path, name_plot),
                                     file_name=name_plot)

    return save_path

def agent_picker(agent):
    if agent == "PPO":
        return PPO
    elif agent == "LagrangianPPO":
        return LagrangianPPO
    elif agent == "RecurrentPPO":
        return RecurrentPPO
    elif agent == "DQN":
        return DQN
    elif agent == "CrossQ":
        return CrossQ
    elif agent == "MaskablePPO":
        return MaskablePPO
    else:
        raise ValueError("Unknown agent! Please input supported algorithm")

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--multiprocess", type=bool, default=False)
    argparser.add_argument("--agent", type=str, default="PPO")
    argparser.add_argument("--n_steps", type=int, default=1_000_000)
    args = argparser.parse_args()
    rng= np.random.default_rng()
    random_int = rng.integers(low=0, high=1_000_000, size=1)[0]
    run_seed(int(args.seed), args.n_steps, agent_picker(args.agent))
