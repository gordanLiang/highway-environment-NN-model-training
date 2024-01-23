import os
import numpy as np

root_path = os.path.abspath(os.path.dirname(__file__))

class highway_dataset:
    def __init__(self):
        self.dataset = np.load(root_path + '/data_fast_each1w.npz')
    def load_data(self):
        data = self.dataset['data']
        label = self.dataset['label']
        return np.array(data),np.array(label)
    