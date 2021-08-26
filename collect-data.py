import gym
import mujoco_py
import numpy as np

obj_init = np.array([1, 1, 0.42, 1.5, 0.75, 0], dtype=np.float32)
# [initial object coordinates, initial object velocity]

env = gym.make('mujoco_collection_1:mujoco-slide-v0', obj_init=obj_init)
env.reset()
done = False
obs = None
iter = 0

while not done:
  # if iter>18000:
  #   env.render(mode="human")
  obs, reward, done, info = env.step(env.action_space.sample())
  if iter%20 == 0:
    seethis = obs["obj_vel"]
    print(seethis)
  iter += 1
  # if(done and iter>100):
  #   break

displacement = obs["obj_pos"][:2] - obj_init[:2]
final_velocity = obs["obj_vel"]
print(f"Completed in {iter} steps, displacement = {displacement}, velocity = {final_velocity}")
env.close()