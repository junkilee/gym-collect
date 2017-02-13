# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
import sys
import gym
import time
import os
import tqdm

def collect(env, steps, f, is_render=True):
    for _ in tqdm.tqdm(range(steps)):
      if is_render:
        env.render()
      action = env.action_space.sample()
      new_observation, reward, done, _ = env.step(action)
      f.write("{},{},{},{}\n".format(','.join(map(str,new_observation)), action, reward, done))
      f.flush()
      if done:
        env.reset()

if __name__ == "__main__":
  data_dir = "data"
  is_display = False
  post_fix = ""
  if len(sys.argv) > 1:
    post_fix = sys.argv[1]
  train_filename = os.path.join(data_dir, 'train_' + post_fix +'.csv')
  test_filename = os.path.join(data_dir, 'test_' + post_fix +'.csv')

  env = gym.make('CartPole-v0')
  env.seed(int(time.time()))
  env.reset()

  train_size = 500000
  test_size = 5000
  with open(train_filename, 'w') as f:
    collect(env, train_size, f, is_display)
  with open(test_filename, 'w') as f:
    collect(env, test_size, f, is_display)


