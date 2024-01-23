import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard
from highway_dataset import highway_dataset
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import datetime
import json


#hyperparameter
epochs = 300
batch_size = 32
total_data_size = 50000
data_ratio = 0.0
split_ratio = 0.8
initial_learning_rate = 0.0001


def main():
  #tensorboard logger setting
  log_dir = "logs/high_each1w_" + str(int(total_data_size/1000)) + "k_" + str(epochs) + "/model_1"
  tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)

  #Creating 25*256*256*5 neural network
  #last layer is doing softmax to five results
  model = tf.keras.Sequential([
     layers.Input(shape=(25,)),
     layers.Dense(256, activation='relu', kernel_initializer=glorot_uniform()),
     layers.Dense(256, activation='relu', kernel_initializer=glorot_uniform()),
     layers.Dense(5, activation='softmax', kernel_initializer=glorot_uniform()),
  ])

  initial_learning_rate = 5e-4
  lr_schedule = ExponentialDecay(initial_learning_rate, decay_steps=10000, decay_rate=0.9)

  optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
  model.compile(optimizer=optimizer, loss='huber_loss', metrics=['accuracy'],)
  model.summary()

  #processing dataset  
  dataset = highway_dataset()
  high_data,high_label = dataset.load_data()
  high_num = high_data.shape[0]

  #shuffle whole dataset
  high_index = np.random.choice(high_num, size=int(total_data_size), replace=False)

  data = high_data[high_index,:]
  label = high_label[high_index]

  #processing label
  unique,label_count=np.unique(label,return_counts=True)
  count=dict(zip(unique,label_count))
  total = sum(count.values())
  ratio_count = {key: value / total * 100 for key, value in count.items()}
  print(ratio_count)
  label = to_categorical(label, num_classes=5)


  #splitting to training set and testing set
  splitting_index = int(total_data_size*split_ratio)
  train_data = data[0:splitting_index,:]
  test_data = data[splitting_index:total_data_size,:]
  train_label = label[0:splitting_index]
  test_label = label[splitting_index:total_data_size]

  #Training model
  model.fit(train_data,train_label, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])

  #Saving model
  model.save('high_each1w_' + str(int(total_data_size/1000)) + 'k_' + str(epochs) + '.h5')

  #Testing model
  predictions = model.predict(test_data)
  predicted_classes = np.argmax(predictions, axis=1)
  # 計算準確性
  true_labels = np.argmax(test_label, axis=1)
  print(predicted_classes)
  print(true_labels)
  accuracy = np.mean(true_labels == predicted_classes)
  print(f'Accuracy: {accuracy}')



if __name__ == "__main__":
  main() 

