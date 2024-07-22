from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation, FilterObservation
import pandas as pd
from ascab.utils.plot import plot_results
from ascab.env.env import AScabEnv


def rl_agent(n_steps=5000):
    ascab = FilterObservation(AScabEnv(), filter_keys=["weather", "tree"])
    ascab = FlattenObservation(ascab)
    model = PPO("MlpPolicy", ascab, verbose=1, seed=42)
    model.learn(total_timesteps=n_steps)
    terminated = False
    total_reward = 0.0
    observation, _ = ascab.reset()
    while not terminated:
        action_ = model.predict(observation, deterministic=True)[0]
        observation, reward, terminated, _, _ = ascab.step(action_)
        total_reward += reward
    print(f"reward: {total_reward:.3f}")
    #ascab.render()
    return ascab


def cheating_agent():
    ascab = FlattenObservation(AScabEnv())
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = 0.0
        if ascab.get_wrapper_attr('info')['Risk'] and ascab.get_wrapper_attr('info')['Risk'][-1] > 0.05: action = 1.0
        _, reward, terminated, _, _ = ascab.step(action)
        total_reward += reward
    print(f"reward: {total_reward}")
    #ascab.render()
    return ascab


def zero_agent():
    ascab = FlattenObservation(AScabEnv())
    terminated = False
    total_reward = 0.0
    while not terminated:
        _, reward, terminated, _, _ = ascab.step(0.0)
        total_reward += reward
    print(f"reward: {total_reward}")
    ascab.render()
    return ascab


if __name__ == "__main__":
    print('zero agent')
    ascab_zero = zero_agent() #-0.634
    #print("cheating agent")
    #ascab_cheating = cheating_agent()
    #ascab_rl = rl_agent()

    #plot_results({"zero": ascab_zero.get_info(to_dataframe=True),
    #              "cheater": ascab_cheating.get_info(to_dataframe=True)})

    #plot_results([pd.DataFrame(ascab_zero.info).assign(Date=lambda x: pd.to_datetime(x["Date"])),
    #              pd.DataFrame(ascab_cheating.info).assign(Date=lambda x: pd.to_datetime(x["Date"])),
    #              pd.DataFrame(ascab_rl.info).assign(Date=lambda x: pd.to_datetime(x["Date"]))])


