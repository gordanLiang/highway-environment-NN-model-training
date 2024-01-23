import os
import tensorflow as tf
from tensorflow import keras
import gymnasium as gym
import pprint
import numpy as np
from highway_dataset import highway_dataset

def main():
  total_reward = 0
  #Loading model
  model = keras.models.load_model('high_each1w_50k_300' + '.h5')
  
  env = gym.make('highway-fast-v0', render_mode='rgb_array')

  for _ in range(10):
    (obs,info) = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
      env.render()
      obs = obs.reshape(1,25)
      action = model.predict(obs)
      action = np.argmax(action, axis=1)
      obs, reward, done, truncated, info = env.step(int(action))
      total_reward += reward
      
  return int(total_reward)


if __name__ == "__main__":
  rewards = np.array([])
  for round in range(0,10):
    reward = main()
    rewards = np.append(rewards,reward)   
  print(rewards)