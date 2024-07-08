from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation

from ascab.env.env import AScabEnv

import numpy as np
np.set_printoptions(precision=2, suppress=True)


def rl_agent():
    ascab = FlattenObservation(AScabEnv())
    model = PPO("MlpPolicy", ascab, verbose=1, seed=42)
    model.learn(total_timesteps=1000)
    reward_per_episode, _ = evaluate_policy(model, ascab, n_eval_episodes=1)
    print(f'{reward_per_episode}')
    
    terminated = False
    total_reward = 0.0
    observation = model.env.reset()
    ascab_ = FlattenObservation(AScabEnv())
    ascab_.reset()
    while not terminated:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, terminated, info = model.env.step(action)
        ascab_.step(action)
        total_reward += reward
    print(f"reward: {total_reward}")
    ascab_.render()


def cheating_agent():
    ascab = FlattenObservation(AScabEnv())
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = 0.0
        if ascab.result_data['Risk'] and ascab.result_data['Risk'][-1] > 0.05: action = 1.0
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