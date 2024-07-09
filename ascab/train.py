from stable_baselines3 import PPO
from gymnasium.wrappers import FlattenObservation, FilterObservation

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
    ascab.render()


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
    ascab.render()


def zero_agent():
    ascab = FlattenObservation(AScabEnv())
    terminated = False
    total_reward = 0.0
    while not terminated:
        _, reward, terminated, _, _ = ascab.step(0.0)
        total_reward += reward
    print(f"reward: {total_reward}")
    ascab.render()


if __name__ == "__main__":
    zero_agent() #-0.634
    cheating_agent()
    rl_agent()