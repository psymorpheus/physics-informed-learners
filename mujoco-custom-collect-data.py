import gym
import mujoco_py
import numpy as np
import scipy.io as sc
import time
import glfw

render = True
obj_init = np.array([1,0], dtype=np.float32)
env = gym.make('mujoco_collection_1:mujoco-slide-v0', obj_init=obj_init)
iter = 0

defStep = np.array([-1,-1,1,0],dtype=np.float32)
obs = env.reset()

initial_obj_pos = obs["observation"][3:6]
initial_obj_vel = obs["observation"][14:17]
debug_data = np.array([np.hstack([[0],initial_obj_pos,initial_obj_vel,[initial_obj_vel[1]/initial_obj_vel[0]]])])

while (iter<200):
    if render:
      env.render(mode="human")
    obs, reward, done, info = env.step(defStep)

    obj_pos = obs["observation"][3:6]
    obj_vel = obs["observation"][14:17]

    iter += 1

    new_observation = np.array([np.hstack([[iter],obj_pos,obj_vel,[obj_vel[1]/obj_vel[0]]])])
    debug_data = np.vstack([debug_data, new_observation])

# Needed so that windows closes without becoming unresponsive
env.close()
if render: glfw.terminate()

filename = "debug_data-" + time.strftime("%Y%m%d-%H%M%S")
toSave = input('Save file?[Y/n] ')
if toSave[0].lower() == 'y':
    np.savetxt(filename + ".csv", debug_data, delimiter=",")
    sc.savemat(filename + ".mat", {'debug_data':debug_data})

env.close()