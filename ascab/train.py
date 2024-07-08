from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.wrappers import FlattenObservation

from ascab.env.env import AScabEnv

import numpy as np
np.set_printoptions(precision=2, suppress=True)


def rl_agent():
    ascab = FlattenObservation(AScabEnv())
    model = PPO("MlpPolicy", ascab, verbose=1, seed=42)
    model.learn(total_timesteps=50000)
    evaluate_with_sb = True
    if evaluate_with_sb:
        reward_per_episode, _ = evaluate_policy(model, ascab, n_eval_episodes=1)
        print(f'{reward_per_episode}')

    terminated = False
    total_reward = 0.0
    observation, _ = ascab.reset()
    while not terminated:
        action_ = model.predict(observation, deterministic=True)[0]
        observation, reward, terminated, _, _ = ascab.step(action_)
        total_reward += reward
    print(f"reward: {total_reward}")
    ascab.render()


def cheating_agent():
    ascab = FlattenObservation(AScabEnv())
    terminated = False
    total_reward = 0.0
    while not terminated:
        action = 0.0
        if ascab.get_wrapper_attr('result_data')['Risk'] and ascab.get_wrapper_attr('result_data')['Risk'][-1] > 0.05: action = 1.0
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
    #zero_agent() #-0.634
    #cheating_agent()
    rl_agent()