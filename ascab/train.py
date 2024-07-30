from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation, FilterObservation
from ascab.utils.plot import plot_results
from ascab.env.env import AScabEnv


def get_default_observation_filter():
    return ["weather", "tree"]


def rl_agent(ascab: AScabEnv = AScabEnv(), n_steps=5000, observation_filter=get_default_observation_filter()):
    if observation_filter:
        print(f'filter observations: {observation_filter}')
        ascab = FilterObservation(ascab, filter_keys=observation_filter)
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
    ascab.render()
    return ascab.get_info(to_dataframe=True)


def cheating_agent(ascab: AScabEnv = AScabEnv()):
    terminated = False
    total_reward = 0.0
    ascab.reset()
    while not terminated:
        action = 0.0
        if ascab.get_wrapper_attr('info')['Risk'] and ascab.get_wrapper_attr('info')['Risk'][-1] > 0.05: action = 1.0
        _, reward, terminated, _, _ = ascab.step(action)
        total_reward += reward
    print(f"reward: {total_reward}")
    ascab.render()
    return ascab.get_info(to_dataframe=True)


def zero_agent(ascab: AScabEnv = AScabEnv()):
    terminated = False
    total_reward = 0.0
    ascab.reset()
    while not terminated:
        _, reward, terminated, _, _ = ascab.step(0.0)
        total_reward += reward
    print(f"reward: {total_reward}")
    ascab.render()
    return ascab.get_info(to_dataframe=True)


if __name__ == "__main__":
    ascab = AScabEnv()
    print('zero agent')
    ascab_zero = zero_agent(ascab)  # -0.634
    print("cheating agent")
    ascab_cheating = cheating_agent(ascab)
    print("rl agent")
    ascab_rl = rl_agent(ascab, 5)
    plot_results({"zero": ascab_zero,
                  "cheater": ascab_cheating,
                  "rl_agent": ascab_rl})

