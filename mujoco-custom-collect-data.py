import gym
import mujoco_py
import numpy as np
import scipy.io as sc
import time
import glfw
from tqdm import tqdm
import mujoco_collection_constants as mcc

# Final shape of collected data is (mcc.TOTAL_ITERATIONS, V_VALUES)
collected_data = np.zeros(shape=(mcc.TOTAL_ITERATIONS,1), dtype=np.float32)

for vx in tqdm(mcc.vx_range, desc = 'tqdm() Progress Bar'):
  obj_init = np.array([vx,0.0], dtype=np.float32)
  env = gym.make('mujoco_collection_1:mujoco-slide-v0', obj_init=obj_init)
  iter = 0

  defStep = np.array([-1,-1,1,0],dtype=np.float32)
  obs = env.reset()

  initial_obj_pos = obs["observation"][3:6]
  initial_obj_vel = obs["observation"][14:17]
  if mcc.save_debug:
    debug_data = np.array([np.hstack([[0],initial_obj_pos,initial_obj_vel,[initial_obj_vel[1]/initial_obj_vel[0]]])])

  data = np.zeros(shape=(mcc.TOTAL_ITERATIONS,1), dtype=np.float32)

  iter += 1

  while (iter<mcc.TOTAL_ITERATIONS):
      if mcc.render:
        env.render(mode="human")
      obs, reward, done, info = env.step(defStep)

      obj_pos = obs["observation"][3:6]
      obj_vel = obs["observation"][14:17]
      data[iter][0] = obj_pos[0] - initial_obj_pos[0]

      if mcc.save_debug:
        new_observation = np.array([np.hstack([[iter],obj_pos,obj_vel,[obj_vel[1]/obj_vel[0]]])])
        debug_data = np.vstack([debug_data, new_observation])

      iter += 1

  collected_data = np.hstack([collected_data, data])
  # Needed so that windows closes without becoming unresponsive
  env.close()
  if mcc.render: glfw.terminate()
  
  if mcc.save_debug:
    filename = "debug_data-" + time.strftime("%Y%m%d-%H%M%S")
    toSave = input('Save file?[Y/n] ')
    if toSave[0].lower() == 'y':
        np.savetxt(filename + ".csv", debug_data, delimiter=",")
        sc.savemat(filename + ".mat", {'debug_data':debug_data})

# To remove starting pad of zeros
collected_data = collected_data[:, 1:]

if mcc.save_collected:
  filename = "collected_data-" + time.strftime("%Y%m%d-%H%M%S")
  filename = "collected_data"
  np.savetxt(filename + ".csv", collected_data, delimiter=",")
  sc.savemat(filename + ".mat", {'collected_data':collected_data})