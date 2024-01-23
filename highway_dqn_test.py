import gymnasium as gym
from stable_baselines3 import DQN
import pprint
import numpy as np

env = gym.make('highway-fast-v0', render_mode='rgb_array')
env.reset()
pprint.pprint(env.config)
model = DQN.load("highway_dqn/model_1")

rewards = []
for round in range(10):
    total_reward = 0
    for step in range(10):
        (obs,info) = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            env.render()
            action, _ = model.predict(obs,deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward
    rewards.append(int(total_reward))
print(rewards)

    

