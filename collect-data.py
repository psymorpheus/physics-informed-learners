#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
import numpy as np
#Setting MountainCar-v0 as the environment
obj_init = np.array([2, 2, 0.41, 0, 0, 0], dtype=np.float32)
env = gym.make('mujoco_collection_1:mujoco-slide-v0', obj_init=obj_init)
# env = gym.make('FetchSlide-v1')
#Sets an initial state
env.reset()
# env1.reset()
done = False
iter = 0

while (iter<4000):
  #renders the environment
  if iter%200 == 0:
    env.render(mode="human")
  #Takes a random action from its action space 
  # aka the number of unique actions an agent can perform
  obs, reward, done, info = env.step(env.action_space.sample())
  iter += 1

displacement = obs["obj_pos"][:2] - obj_init[:2]
print(f"Completed in {iter} steps, displacement = {displacement} ")
env.close()