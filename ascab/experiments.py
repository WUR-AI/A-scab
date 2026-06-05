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
from gymnasium.wrappers import FlattenObservation, FilterObservation

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.monitor import Monitor


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
from ascab.agent.maskable_recurrent_ppo import MaskableRecurrentPPO

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

DYNAMIC_BETA_FINAL_EVAL_BETAS = [0.01, 0.03, 0.05]
FR_TEST_LOCATION = (44.0986, 1.1628)


def run_seed(
        seed: int,
        n_steps: int,
        algo=PPO,
        beta: float = 0.025,
        dynamic_beta: bool = False,
        beta_min: float = 0.0,
        beta_max: float = 0.05,
        beta_curriculum_start_fraction: float = 0.4,
        dynamic_beta_hold_episodes: int = 5,
        omit_infection_window: bool = False,
        risk_period_constrainer: bool = False,
        eval_fr: bool = True,
) -> str:
    print("rl agent")
    print("seed:", seed)

    print(f"Using {algo.__name__}")
    print(f"beta: {beta}")
    if dynamic_beta:
        print(f"dynamic beta range: [{beta_min}, {beta_max}]")
        print(f"beta curriculum start fraction: {beta_curriculum_start_fraction}")
        print(f"dynamic beta hold episodes: {dynamic_beta_hold_episodes}")
    if omit_infection_window:
        print("Omitting InfectionWindow from observations")
    if risk_period_constrainer:
        print("Using risk-period ActionConstrainer")
    if eval_fr:
        print("Running final FR evaluation after training")

    discrete_algos = ["PPO", "DQN", "RecurrentPPO", "LagrangianPPO", "MaskablePPO", "MaskableRecurrentPPO"]
    # algo = PPO
    constrain = False
    terminate_early = False
    penalty_wrap = False
    normalize = True
    truncated_observations='truncated'
    log_path = os.path.join(os.getcwd(), "log")
    beta_label = f"dynamic_beta{beta_min}-{beta_max}_hold{dynamic_beta_hold_episodes}" if dynamic_beta else f"beta{beta}"
    observation_label = "_no_infection_window" if omit_infection_window else ""
    constrainer_label = "_risk_period_constrained" if risk_period_constrainer else ""
    name_agent = f"rl_agent_{algo.__name__}_seed{seed}_{beta_label}{observation_label}{constrainer_label}"
    save_path = os.path.join(os.getcwd(), "log", name_agent)
    # os.makedirs(save_path, exist_ok=True)
    save_path = unique_path(save_path)

    ascab_train = MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library(
            locations=[(42.1620, 3.0924), (42.1620, 3.0), (42.5, 2.5), (41.5, 3.0924), (42.5, 3.0924)],
            dates=get_dates([year for year in range(2016, 2025) if year % 2 == 0],
                            start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
        biofix_date="March 10", budbreak_date="March 10", discrete_actions=True if algo.__name__ in discrete_algos else False,
        beta=beta,
        dynamic_beta=dynamic_beta,
        beta_range=(beta_min, beta_max),
        beta_curriculum=dynamic_beta,
        beta_curriculum_start_fraction=beta_curriculum_start_fraction,
        dynamic_beta_hold_episodes=dynamic_beta_hold_episodes,
        )
    ascab_test = MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library(
            locations=[(42.1620, 3.0924)],
            dates=get_dates([year for year in range(2016, 2025) if year % 2 != 0],
                            start_of_season=get_default_start_of_season(), end_of_season=get_default_end_of_season())),
        biofix_date="March 10", budbreak_date="March 10", discrete_actions=True if algo.__name__ in discrete_algos else False, mode='sequential',
        beta=beta
        )

    observation_filter = list(ascab_train.observation_space.keys())
    if omit_infection_window and "InfectionWindow" in observation_filter:
        observation_filter.remove("InfectionWindow")

    if constrain:
        ascab_train = ActionConstrainer(ascab_train, risk_period=False, action_budget=8)
        ascab_test = ActionConstrainer(ascab_test, risk_period=False, action_budget=8)
    if risk_period_constrainer:
        ascab_train = ActionConstrainer(ascab_train, risk_period=True)
        ascab_test = ActionConstrainer(ascab_test, risk_period=True)

    if terminate_early:
        ascab_train = EarlyTerminationWrapper(ascab_train, penalty=1.0)
        ascab_test = EarlyTerminationWrapper(ascab_test, penalty=1.0)

    if penalty_wrap:
        ascab_train = PenaltyWrapper(ascab_train, penalty=0.05)
        ascab_test = PenaltyWrapper(ascab_test, penalty=0.05)

    if algo in (MaskablePPO, MaskableRecurrentPPO):
        ascab_train = ActionMasker(ascab_train, lambda e: e.action_masks())
        ascab_test = ActionMasker(ascab_test, lambda e: e.action_masks())

    ascab_rl = RLAgent(ascab_train=ascab_train, ascab_test=ascab_test, observation_filter=observation_filter,
                       n_steps=n_steps, render=False, path_model=save_path, path_log=log_path, rl_algorithm=algo,
                       seed=seed, normalize=normalize, irs=False, eval_beta=0.025)
    try:
        print(ascab_train.histogram)
        print(ascab_test.histogram)
    except Exception:
        pass
    if normalize:
        ascab_rl.ascab_train.save(os.path.join(save_path+"_norm.pkl"))
    os.makedirs(save_path, exist_ok=True)
    if dynamic_beta:
        eval_results = {}
        for eval_beta in DYNAMIC_BETA_FINAL_EVAL_BETAS:
            ascab_test.unwrapped.set_beta(eval_beta, dynamic_beta=False)
            ascab_test.unwrapped.reset_processed_keys()
            beta_results = ascab_rl.run()
            beta_results["EvalBeta"] = eval_beta
            eval_results[f"rl_agent_beta{eval_beta}"] = beta_results
            with open(f"{save_path}_beta{eval_beta}.pkl", "wb") as f:
                pickle.dump(beta_results, file=f)
        results = pd.concat(eval_results.values(), ignore_index=True)
    else:
        eval_results = {"rl_agent": ascab_rl.run()}
        results = eval_results["rl_agent"]

    with open(save_path+".pkl", "wb") as f:
        pickle.dump(results, file=f)

    fr_results_path = None
    fr_beta_result_paths = []
    if eval_fr:
        ascab_fr_base = MultipleWeatherASCabEnv(
            weather_data_library=get_weather_library(
                locations=[FR_TEST_LOCATION],
                dates=get_dates([year for year in range(2016, 2025) if year % 2 != 0],
                                start_of_season=get_default_start_of_season(),
                                end_of_season=get_default_end_of_season())),
            biofix_date="March 10",
            budbreak_date="March 10",
            discrete_actions=True if algo.__name__ in discrete_algos else False,
            mode='sequential',
            beta=beta
        )
        ascab_fr_test = ascab_fr_base
        if risk_period_constrainer:
            ascab_fr_test = ActionConstrainer(ascab_fr_test, risk_period=True)
        if algo in (MaskablePPO, MaskableRecurrentPPO):
            ascab_fr_test = ActionMasker(ascab_fr_test, lambda e: e.action_masks())
        if observation_filter:
            ascab_fr_test = FilterObservation(ascab_fr_test, filter_keys=observation_filter)
        ascab_fr_test = FlattenObservation(ascab_fr_test)
        if normalize:
            ascab_fr_test = Monitor(ascab_fr_test)
            ascab_fr_test = DummyVecEnv([lambda: ascab_fr_test])
            ascab_fr_test = VecNormalize.load(save_path + "_norm.pkl", ascab_fr_test)
            ascab_fr_test.training = False
            ascab_fr_test.norm_reward = False

        es_eval_env = ascab_rl.ascab
        ascab_rl.ascab = ascab_fr_test
        if dynamic_beta:
            fr_eval_results = {}
            for eval_beta in DYNAMIC_BETA_FINAL_EVAL_BETAS:
                ascab_fr_base.set_beta(eval_beta, dynamic_beta=False)
                ascab_fr_base.reset_processed_keys()
                beta_results = ascab_rl.run()
                beta_results["EvalBeta"] = eval_beta
                beta_results["EvalLocation"] = "FR"
                fr_eval_results[f"rl_agent_fr_beta{eval_beta}"] = beta_results
                fr_beta_path = f"{save_path}_FR_beta{eval_beta}.pkl"
                with open(fr_beta_path, "wb") as f:
                    pickle.dump(beta_results, file=f)
                fr_beta_result_paths.append(fr_beta_path)
            fr_results = pd.concat(fr_eval_results.values(), ignore_index=True)
        else:
            fr_results = ascab_rl.run()
            fr_results["EvalLocation"] = "FR"

        ascab_rl.ascab = es_eval_env
        fr_results_path = save_path + "_FR.pkl"
        with open(fr_results_path, "wb") as f:
            pickle.dump(fr_results, file=f)

    plot_results(
        eval_results,
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
        if fr_results_path is not None:
            ascab_rl.comet.log_asset(file_data=fr_results_path,
                                     file_name=f'{seed}-results-FR')
        for fr_beta_path in fr_beta_result_paths:
            ascab_rl.comet.log_asset(file_data=fr_beta_path,
                                     file_name=os.path.basename(fr_beta_path))
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
    argparser.add_argument("--beta", type=float, default=0.025)
    argparser.add_argument("--dynamic_beta", action="store_true")
    argparser.add_argument("--beta_min", type=float, default=0.0)
    argparser.add_argument("--beta_max", type=float, default=0.1)
    argparser.add_argument("--beta_curriculum_start_fraction", type=float, default=0.4)
    argparser.add_argument("--dynamic_beta_hold_episodes", type=int, default=5)
    argparser.add_argument("--omit_infection_window", action="store_true")
    argparser.add_argument("--risk_period_constrainer", action="store_true")
    argparser.add_argument("--skip_fr_eval", action="store_true")
    args = argparser.parse_args()
    rng= np.random.default_rng()
    random_int = rng.integers(low=0, high=1_000_000, size=1)[0]
    run_seed(
        int(args.seed),
        args.n_steps,
        agent_picker(args.agent),
        beta=args.beta,
        dynamic_beta=args.dynamic_beta,
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        beta_curriculum_start_fraction=args.beta_curriculum_start_fraction,
        dynamic_beta_hold_episodes=args.dynamic_beta_hold_episodes,
        omit_infection_window=args.omit_infection_window,
        risk_period_constrainer=args.risk_period_constrainer,
        eval_fr=not args.skip_fr_eval,
    )
