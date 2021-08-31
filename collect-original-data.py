import gym
import mujoco_py
import numpy as np
import scipy.io as sc

env = gym.make('FetchSlide-v1')
env.reset()
iter = 0

defStep = np.array([-1,-1,1,0],dtype=np.float32)
pre_obs, reward, done, info = env.step(defStep)
data = np.array([np.zeros(8, dtype=np.float32)])

while (iter<500):
    env.render(mode="human")
    obs, reward, done, info = env.step(defStep)
    # print(obs["observation"][:3] - pre_obs["observation"][:3], "-----", obs["observation"][-6:-3])
    if iter>50 and iter<80: defStep[:3] = 50*obs["observation"][6:9]


    obj_pos = obs["observation"][3:6]
    obj_vel = obs["observation"][14:17]

    pre_obs = obs
    iter += 1

    newrow = np.array([np.hstack([[iter],obj_pos,obj_vel,[obj_vel[1]/obj_vel[0]]])])
    data = np.vstack([data, newrow])

np.savetxt("data1.csv", data, delimiter=",")
sc.savemat("data1.mat", {'data':data})
env.close()