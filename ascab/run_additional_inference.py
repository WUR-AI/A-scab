import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.wrappers import FlattenObservation
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from ascab.env.env import (
    ActionConstrainer,
    MultipleWeatherASCabEnv,
    get_default_end_of_season,
    get_default_start_of_season,
    get_weather_library,
)
from ascab.utils.generic import get_dates


DEFAULT_LOCATION_LABEL = "ES"
DEFAULT_LOCATIONS = {
    "ES": (42.1620, 3.0924),
    "FR": (44.0986, 1.1628),
    "ES1": (37.8880, -4.7790),
}
TEST_YEARS = [year for year in range(2016, 2025) if year % 2 != 0]
SEED_PATTERN = re.compile(r"seed(?P<seed>\d+)")


def parse_bool(value: str) -> bool:
    if value.lower() in {"1", "true", "t", "yes", "y"}:
        return True
    if value.lower() in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def extract_seed(path: Path) -> str:
    match = SEED_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(f"Could not extract seed from model filename: {path.name}")
    return match.group("seed")


def get_location(label: str, location: list[float] | None) -> tuple[float, float]:
    if location is not None:
        if len(location) != 2:
            raise ValueError("--location requires two values: latitude longitude")
        return float(location[0]), float(location[1])
    return DEFAULT_LOCATIONS.get(label.upper(), DEFAULT_LOCATIONS[DEFAULT_LOCATION_LABEL])


def make_test_env(
    *,
    location: tuple[float, float],
    beta: float,
    risk_period: bool,
) -> MultipleWeatherASCabEnv | ActionConstrainer:
    env = MultipleWeatherASCabEnv(
        weather_data_library=get_weather_library(
            locations=[location],
            dates=get_dates(
                TEST_YEARS,
                start_of_season=get_default_start_of_season(),
                end_of_season=get_default_end_of_season(),
            ),
        ),
        biofix_date="March 10",
        budbreak_date="March 10",
        mode="sequential",
        discrete_actions=True,
        beta=beta,
    )
    return ActionConstrainer(env, risk_period=risk_period) if risk_period else env


def make_vec_env(
    *,
    norm_path: Path,
    location: tuple[float, float],
    beta: float,
    risk_period: bool,
) -> tuple[VecNormalize, int]:
    env = make_test_env(location=location, beta=beta, risk_period=risk_period)
    n_eval_episodes = len(env.unwrapped.weather_keys)
    env = FlattenObservation(env)
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load(str(norm_path), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env, n_eval_episodes


def run_model(model_path: Path, norm_path: Path, env: VecNormalize, n_eval_episodes: int) -> pd.DataFrame:
    model = RecurrentPPO.load(str(model_path), env=env, print_system_info=False)
    episode_results = []

    for _ in range(n_eval_episodes):
        observation = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)
        done = np.array([False])
        info = None

        while not done[0]:
            action, lstm_states = model.predict(
                observation,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            observation, _, done, infos = env.step(action)
            episode_starts = done
            info = infos[0]

        episode_results.append(info_to_dataframe(info))

    return pd.concat(episode_results, ignore_index=True)


def info_to_dataframe(info: dict) -> pd.DataFrame:
    ignored_keys = {"TimeLimit.truncated", "episode", "terminal_observation"}
    result = {key: value for key, value in info.items() if key not in ignored_keys}
    return pd.DataFrame(result).assign(Date=lambda x: pd.to_datetime(x["Date"]))


def output_path(output_dir: Path, seed: str, risk_period: bool, location_label: str) -> Path:
    constraint_label = "CONSTRAINED" if risk_period else "UNCONSTRAINED"
    return output_dir / f"{seed}-result-{location_label.upper()}.pkl"


def print_cumulative_reward_by_year(result: pd.DataFrame, seed: str) -> None:
    rewards_by_year = (
        result.assign(Year=lambda df: pd.to_datetime(df["Date"]).dt.year)
        .groupby("Year", sort=True)["Reward"]
        .sum()
    )
    print(f"Cumulative reward by year for seed {seed}:")
    for year, reward in rewards_by_year.items():
        print(f"  {year}: {reward:.6f}")


def find_model_paths(models_dir: Path) -> list[Path]:
    return sorted(models_dir.glob("rl_agent_RecurrentPPO_seed*.zip"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference for additional saved RecurrentPPO models with optional risk-period action constraints."
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("additional_runs"),
        help="Directory containing model .zip files and matching *_norm.pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("additional_runs"),
        help="Directory where inference result pickles will be saved.",
    )
    parser.add_argument(
        "--location-label",
        default=DEFAULT_LOCATION_LABEL,
        help="Short label used in output filenames. Defaults to ES.",
    )
    parser.add_argument(
        "--location",
        type=float,
        nargs=2,
        default=None,
        metavar=("LAT", "LON"),
        help="Optional latitude longitude override. If omitted, uses the label default.",
    )
    parser.add_argument(
        "--risk-period",
        type=parse_bool,
        default=False,
        help="Whether ActionConstrainer should zero actions outside the risk period. Defaults to true.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.025,
        help="Beta used in the evaluation environment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    location_label = args.location_label.upper()
    location = get_location(location_label, args.location)
    model_paths = find_model_paths(args.models_dir)
    if not model_paths:
        raise FileNotFoundError(f"No RecurrentPPO model .zip files found in {args.models_dir}")

    for model_path in model_paths:
        seed = extract_seed(model_path)
        norm_path = model_path.with_name(f"{model_path.stem}_norm.pkl")
        if not norm_path.exists():
            raise FileNotFoundError(f"Missing normalization file for {model_path.name}: {norm_path}")

        print(
            f"Running seed {seed} at {location_label} "
            f"with risk_period={args.risk_period}, beta={args.beta}"
        )
        env, n_eval_episodes = make_vec_env(
            norm_path=norm_path,
            location=location,
            beta=args.beta,
            risk_period=args.risk_period,
        )
        result = run_model(model_path, norm_path, env, n_eval_episodes)
        result["Seed"] = int(seed)
        result["ModelPath"] = str(model_path)
        result["NormPath"] = str(norm_path)
        result["RiskPeriodConstrained"] = args.risk_period
        result["LocationLabel"] = location_label
        result["Latitude"] = location[0]
        result["Longitude"] = location[1]
        result["Beta"] = args.beta

        path = output_path(args.output_dir, seed, args.risk_period, location_label)
        print_cumulative_reward_by_year(result, seed)
        with open(path, "wb") as f:
            pickle.dump(result, f)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
