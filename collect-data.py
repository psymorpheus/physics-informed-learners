#Importing OpenAI gym package and MuJoCo engine
import gym
import mujoco_py
#Setting MountainCar-v0 as the environment
env = gym.make('mujoco_collection_1:mujoco-slide-v0')
# env1 = gym.make('FetchSlide-v1')
#Sets an initial state
env.reset()
# env1.reset()
print('here')
# Rendering our instance 300 times
for _ in range(300):
  #renders the environment
  env.render()
  #Takes a random action from its action space 
  # aka the number of unique actions an agent can perform
  env.step(env.action_space.sample())
env.close()