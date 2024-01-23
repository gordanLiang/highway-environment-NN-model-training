import gymnasium as gym
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import pprint
import numpy as np
import csv
import os


def main():
    env = gym.make('highway-fast-v0', render_mode='rgb_array')
    pprint.pprint(env.config)

    (obs,info) = env.reset()
    print(obs)

    model = DQN.load("highway_dqn/model_1")

    state_size = env.observation_space.shape[0] * 5
    dataset_data = np.array([])
    dataset_label = np.array([])
    count = 0
    reward_sum = []
    print(env.observation_type.features_range)
    print(env.observation_type.features)
    print(env.observation_type.absolute)
    print(env.observation_type.normalize)

    print(env.observation_type.space().shape)

    round = 1
    finished = False
    while not finished:
        print('Round:', round)
        print('Count:', count)
        env.reset()
        (obs, info), done ,truncated= env.reset(), False, False
        while not (done or truncated):
            #env.render()
            action, _ = model.predict(obs,deterministic=True)
            obs = np.reshape(obs,[1,state_size])
            dataset_data = np.append(dataset_data,obs)
            dataset_label = np.append(dataset_label,int(action))
            obs, reward, done, truncated, info = env.step(int(action))

            reward_sum.append(reward)
            count = count + 1
            unique,counter=np.unique(dataset_label,return_counts=True)
            data_count=dict(zip(unique,counter))
            
            index = np.array(np.where(counter > 1e4))
            if index.shape[1] == 5:
                finished = True
                break
        print(data_count)
        round = round + 1
    print(dataset_data)
    print(dataset_data.shape)
    print(count)
    print(state_size)
    dataset_data = np.reshape(dataset_data,[count,state_size])
    dataset_label = np.reshape(dataset_label,[count,1])

    one_index,_ = np.where(dataset_label == 0.0)
    two_index,_ = np.where(dataset_label == 1.0)
    three_index,_ = np.where(dataset_label == 2.0)
    four_index,_ = np.where(dataset_label == 3.0)
    five_index,_ = np.where(dataset_label == 4.0)
    sample_one_index = np.random.choice(one_index, size=10000, replace=False)
    sample_two_index = np.random.choice(two_index, size=10000, replace=False)
    sample_three_index = np.random.choice(three_index, size=10000, replace=False)
    sample_four_index = np.random.choice(four_index, size=10000, replace=False)
    sample_five_index = np.random.choice(five_index, size=10000, replace=False)
    one_data = dataset_data[sample_one_index,:]
    one_label = dataset_label[sample_one_index]
    two_data = dataset_data[sample_two_index,:]
    two_label = dataset_label[sample_two_index]
    three_data = dataset_data[sample_three_index,:]
    three_label = dataset_label[sample_three_index]
    four_data = dataset_data[sample_four_index,:]
    four_label = dataset_label[sample_four_index]
    five_data = dataset_data[sample_five_index,:]
    five_label = dataset_label[sample_five_index]

    data = np.array([])
    label = np.array([])
    data = np.append(data,one_data)
    label = np.append(label,one_label)
    data = np.append(data,two_data)
    label = np.append(label,two_label)
    data = np.append(data,three_data)
    label = np.append(label,three_label)
    data = np.append(data,four_data)
    label = np.append(label,four_label)
    data = np.append(data,five_data)
    label = np.append(label,five_label)
    data = np.reshape(data,[50000,25])
    root_path = os.path.abspath(os.path.dirname(__file__))
    print(root_path)
    np.savez(root_path + '/data_fast_each1w.npz',data=data,label=label)
    dataset = np.load(root_path + '/data_fast_each1w.npz')
    print(dataset['data'].shape[0])
    print(dataset['data'].shape)
    print(dataset['label'].shape)


if __name__=="__main__":
    main()