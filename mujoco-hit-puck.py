import gym
import mujoco_py
import numpy as np

env = gym.make('FetchSlide-v1')
env.reset()
iter = 0

defStep = np.array([-1,-1,1,0],dtype=np.float32)
pre_obs, reward, done, info = env.step(defStep)

while (iter<4000):
  env.render(mode="human")
  obs, reward, done, info = env.step(defStep)
  print(obs["observation"][:3] - pre_obs["observation"][:3], "-----", obs["observation"][-6:-3])
  if iter>100: defStep[:3] = 10*obs["observation"][6:9]
  pre_obs = obs
  iter += 1

env.close()