import datetime
import os
from gymnasium.wrappers import FlattenObservation, FilterObservation

from ascab.utils.plot import plot_results
from ascab.env.env import AScabEnv


try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


def get_default_observation_filter():
    return ["weather", "tree"]


def rl_agent(
    ascab: AScabEnv = None,
    n_steps=5000,
    observation_filter=get_default_observation_filter(),
    render=True,
    path_save: str = None
):
    if PPO is None:
        raise ImportError(
            "stable-baselines3 is not installed. Please install it to use the rl_agent."
        )

    if ascab is None:
        ascab = AScabEnv()
    if observation_filter:
        print(f"filter observations: {observation_filter}")
        ascab = FilterObservation(ascab, filter_keys=observation_filter)
    ascab = FlattenObservation(ascab)
    if path_save is not None and (os.path.exists(path_save) or os.path.exists(path_save + ".zip")):
        print(f'load model from disk: {path_save}')
        model = PPO.load(env=ascab, path=path_save, print_system_info=False)
    else:
        model = PPO("MlpPolicy", ascab, verbose=1, seed=42)
        model.learn(total_timesteps=n_steps)
        if path_save is not None:
            model.save(path_save)
    terminated = False
    total_reward = 0.0
    observation, _ = ascab.reset()
    while not terminated:
        action_ = model.predict(observation, deterministic=True)[0]
        observation, reward, terminated, _, _ = ascab.step(action_)
        total_reward += reward
    print(f"reward: {total_reward:.3f}")
    if render:
        ascab.render()
    return ascab.get_info(to_dataframe=True)


def cheating_agent(ascab: AScabEnv = None, render=True):
    if ascab is None:
        ascab = AScabEnv()
    terminated = False
    total_reward = 0.0
    ascab.reset()
    while not terminated:
        action = 0.0
        if (
            ascab.get_wrapper_attr("info")["Discharge"]
            and ascab.get_wrapper_attr("info")["Discharge"][-1] > 0.5
        ):
            action = 1.0
        _, reward, terminated, _, _ = ascab.step(action)
        total_reward += reward
    print(f"reward: {total_reward}")
    if render:
        ascab.render()
    return ascab.get_info(to_dataframe=True)


def zero_agent(ascab: AScabEnv = None, render=True):
    if ascab is None:
        ascab = AScabEnv()
    terminated = False
    total_reward = 0.0
    ascab.reset()
    while not terminated:
        _, reward, terminated, _, _ = ascab.step(0.0)
        total_reward += reward
    print(f"reward: {total_reward}")
    if render:
        ascab.render()
    return ascab.get_info(to_dataframe=True)


def fill_it_up_agent(ascab: AScabEnv = None, pesticide_threshold: float = 0.1, render=True):
    if ascab is None:
        ascab = AScabEnv()
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = 0.0
        if (ascab.get_wrapper_attr("info")["Pesticide"] and ascab.get_wrapper_attr("info")["Pesticide"][-1] < pesticide_threshold):
            action = 1.0
        _, reward, terminated, _, _ = ascab.step(action)
        total_reward += reward
    print(f"reward: {total_reward}")
    if render:
        ascab.render()
    return ascab.get_info(to_dataframe=True)


def fixed_schedule_agent(
    ascab: AScabEnv = None, dates: list[datetime.date] = None, render=True
):
    if ascab is None:
        ascab = AScabEnv()
    if dates is None:
        year = ascab.date.year
        dates = [datetime.date(year, 4, 1), datetime.date(year, 4, 8)]
    terminated = False
    total_reward = 0.0
    ascab.reset()
    while not terminated:
        action = 0.0
        if (
            ascab.get_wrapper_attr("info")["Date"]
            and ascab.get_wrapper_attr("info")["Date"][-1] in dates
        ):
            action = 1.0
        _, reward, terminated, _, _ = ascab.step(action)
        total_reward += reward
    print(f"reward: {total_reward}")
    if render:
        ascab.render()
    return ascab.get_info(to_dataframe=True)


if __name__ == "__main__":
    ascab_env = AScabEnv(
        location=(42.1620, 3.0924), dates=("2022-02-15", "2022-08-15"),
        biofix_date="March 10", budbreak_date="March 10")
    print("zero agent")
    ascab_zero = zero_agent(ascab_env, render=False)  # -0.634
    print("cheating agent")
    ascab_cheating = cheating_agent(ascab_env, render=False)
    if PPO is not None:
        print("rl agent")
        ascab_rl = rl_agent(ascab=ascab_env, observation_filter=["weather", "tree", "disease"], n_steps=50000, render=False)
        plot_results({"zero": ascab_zero, "cheater": ascab_cheating, "rl_agent": ascab_rl}, variables=["HasRain", "LeafWetness", "PseudothecialDevelopment", "AscosporeMaturation", "Discharge", "Infections", "Risk", "Action"])
    else:
        print("Stable-baselines3 is not installed. Skipping RL agent.")
