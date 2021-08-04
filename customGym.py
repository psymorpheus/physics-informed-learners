import numpy as np
import gym
import gym_slide
env = gym.make("slide-v0")
observation = env.reset()
for _ in range(1000):
  env.render()
  # action = env.action_space.sample() # your agent here (this takes random actions)
  action = np.array([100,0])
  observation, reward, done, info = env.step(action)

  if done:
    observation = env.reset()
env.close()