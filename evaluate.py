# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
from datetime import datetime
from time import time
import os
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2, activity_l2
from keras.models import model_from_json
import pandas as pd
from tqdm import tqdm

def read_data(filename, is_display = False):
  data = pd.read_csv(filename, header=None).as_matrix()
  data_list = []
  for i in tqdm(range(len(data)-1), desc="reading data"):
    row = data[i]
    next_row = data[i+1]

    # x, x_dot, theta, theta_dot
    state = row[0:4].astype(float)
    next_state = next_row[0:4].astype(float)
    next_diff = next_state - state

    action = row[4]
    reward = row[5]
    done = row[6]

    if is_display:
      #print('state', state)
      #print('next_state', next_state)
      #print('next_diff', next_diff)

      #raw_input('waiting...')
      pass

    if not done:
      data_list.append((state, next_diff, action))
  
  action_translation = [-1, 1]

  data_size = len(data_list)
  input_array = np.zeros([len(data_list), 5])
  output_array = np.zeros([len(data_list), 4])

  for i in tqdm(range(data_size), desc="creating numpy array"):
    input_array[i][0:4] = data_list[i][0]
    input_array[i][4] = action_translation[data_list[i][2]]
    output_array[i][0:4] = data_list[i][1]

    if is_display:
      print (input_array[i])
      print (output_array[i])
      raw_input('waiting...')


  print('data_size : ' + str(data_size))

  return (input_array, output_array)


if __name__ == "__main__":
  if len(sys.argv) > 1:
    train_id = sys.argv[1]

  # setting the directory and filename for train and test data files
  data_dir = "data"
  model_dir = "model"
  postfix = train_id
  train_filename = os.path.join(data_dir, 'train_' + postfix +'.csv')
  test_filename = os.path.join(data_dir, 'test_' + postfix +'.csv')
  model_filename = os.path.join(model_dir, 'model_' + postfix)
  model_filename = os.path.join(model_dir, 'model_' + postfix)

  # default training parameters
  epochs = 2
  batch_size = 32

  test_data = read_data(test_filename)

  json_file = open(model_filename + '.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights(model_filename + ".h5")
  print("Loaded model from disk")

  loaded_model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
  score = loaded_model.evaluate(test_data[0], test_data[1], verbose=0, batch_size = batch_size)
  print "%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

  print(loaded_model.predict(np.array([[0, 0, 0, 0, 1]])))
  print(loaded_model.predict(np.array([[0, 0, 0, 0, -1]])))

  
