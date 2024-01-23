import gymnasium as gym
from stable_baselines3 import DQN
import pprint
import numpy as np

env = gym.make('highway-fast-v0', render_mode='rgb_array')
env.reset()
pprint.pprint(env.config)
model = DQN('MlpPolicy', env,
                policy_kwargs=dict(net_arch=[256, 256]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log='highway_dqn/')
model.learn(int(2e4))
model.save("highway_dqn/model_1")
