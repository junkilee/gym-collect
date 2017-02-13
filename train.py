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
import pandas as pd

def get_transition_model():
  model = Sequential()
  # input (4 states + 1 action)

  regularize = 0.00 
  model.add(Dense(20, input_dim=5, init='uniform', activation='tanh', W_regularizer=l2(regularize)))
  model.add(Dense(20, init='uniform', activation='tanh', W_regularizer=l2(regularize)))
  model.add(Dense(20, init='uniform', activation='tanh', W_regularizer=l2(regularize)))
  model.add(Dense(4, init='uniform', activation='linear'))
  # output difference

  model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])

  return model

def get_reward_model():
  model = Sequential()
  model.add(Dense(20, input_dim=5, init='uniform', activation='tanh'))
  model.add(Dense(20, init='uniform', activation='tanh'))
  model.add(Dense(20, init='uniform', activation='tanh'))
  model.add(Dense(4, init='uniform', activation='linear'))

  model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

  return model

def read_data(filename, is_display = False):
  data = pd.read_csv(filename, header=None)
  data_list = []
  for i in range(len(data)-1):
    row = data.iloc[i]

    # x, x_dot, theta, theta_dot
    state = row[0:4].as_matrix().astype(float)
    next_state = data.iloc[i+1][0:4].as_matrix().astype(float)
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

  for i in range(data_size):
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
  postfix = train_id
  train_filename = os.path.join(data_dir, 'train_' + postfix +'.csv')
  test_filename = os.path.join(data_dir, 'test_' + postfix +'.csv')

  # default training parameters
  epochs = 100
  batch_size = 32

  # initialize numpy
  #seed = 7
  #np.random.seed(seed)

  train_data = read_data(train_filename)
  test_data = read_data(test_filename)

  model = get_transition_model()

  model.fit(train_data[0], train_data[1], nb_epoch = epochs, batch_size = batch_size, validation_data=(test_data[0], test_data[1]))
  #print(model.evaluate(test_data[0], test_data[1], batch_size = batch_size))





