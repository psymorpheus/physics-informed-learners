import gym
import mujoco_py
import numpy as np
import scipy.io as sc
import time

env = gym.make('FetchSlide-v1')
env.reset()
iter = 0

defStep = np.array([-1,-1,1,0],dtype=np.float32)
obs = env.reset()

obj_pos = obs["observation"][3:6]
obj_vel = obs["observation"][14:17]
data = np.array([np.hstack([[0],obj_pos,obj_vel,[obj_vel[1]/obj_vel[0]]])])

while (iter<200):
    env.render(mode="human")
    obs, reward, done, info = env.step(defStep)
    # print(obs["observation"][:3] - pre_obs["observation"][:3], "-----", obs["observation"][-6:-3])
    # if iter>50 and iter<80: defStep[:3] = 50*obs["observation"][6:9]

    obj_pos = obs["observation"][3:6]
    obj_vel = obs["observation"][14:17]

    iter += 1

    newrow = np.array([np.hstack([[iter],obj_pos,obj_vel,[obj_vel[1]/obj_vel[0]]])])
    data = np.vstack([data, newrow])

filename = "data-" + time.strftime("%Y%m%d-%H%M%S")
toSave = input('Save file? [Y/n]')
if toSave[0].lower() == 'y':
    np.savetxt(filename + ".csv", data, delimiter=",")
    sc.savemat(filename + ".mat", {'data':data})
env.close()