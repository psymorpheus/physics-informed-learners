import gym
import mujoco_py
import numpy as np

env = gym.make('FetchSlide-v1')
env.reset()
iter = 0

while (iter<4000):
  env.render(mode="human")
  obs, reward, done, info = env.step(np.array([0,0,1,0],dtype=np.float32))
  # seethis = obs["obj_vel"]
  # print(seethis)
  iter += 1

env.close()