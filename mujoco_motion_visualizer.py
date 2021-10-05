import gym
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt

with open("mujoco_config.yaml", "r") as f:
    all_config = yaml.safe_load(f)
    common_config = all_config['COMMON'].copy()

vx = float(input('Enter initial velocity: '))
tsteps = int(input('Enter number of timesteps: '))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_configs = ['ARTIFICIAL_NOSTOP', 'ARTIFICIAL_STOP', 'SIMULATION_NOSTOP', 'SIMULATION_STOP']
model_configs = ['BORDER_O1', 'INTERNAL_O3', 'STRONG_FF', 'WEAK_FF']
active_data_config_name = data_configs[3]
active_model_config_name = model_configs[1]

active_data_config = all_config[active_data_config_name].copy()
active_data_config.update(common_config)
active_model_config = all_config[active_model_config_name].copy()
active_model_config.update(active_data_config)
config = active_model_config
config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*tsteps, step = config['TIMESTEP'])

model = torch.load(config['dirname'] + config['model_name'] + '.pt')
model.eval()

history = []

if not config['artificial_data']:
	obj_init = np.array([vx,0.0], dtype=np.float32)
	env = gym.make('mujoco_collection_1:mujoco-slide-v0', obj_init=obj_init)
	iter = 0
	defStep = np.array([-1,-1,1,0],dtype=np.float32)
	obs = env.reset()
	initial_obj_pos = obs["observation"][3:6]
	iter += 1
	while (iter<tsteps):
		obs, reward, done, info = env.step(defStep)
		simulated_obj_pos = obs["observation"][3:6][0] - initial_obj_pos[0]
		with torch.no_grad():
			predicted_obj_pos = model.forward(np.array([[vx, iter * config['TIMESTEP']]])).item()
		
		history.append([iter * config['TIMESTEP'], simulated_obj_pos, predicted_obj_pos])
		iter += 1
	# Needed so that windows closes without becoming unresponsive
	env.close()

else:
	stopping_time = vx/(-1*config['acc'])
	stopping_distance = (vx**2)/(-2*config['acc'])

	for t in range(tsteps):
		if (config['t_range'][t]>=stopping_time):
			simulated_obj_pos = (stopping_distance)
		else:
			simulated_obj_pos = (vx*config['t_range'][t] + 0.5*config['acc']*(config['t_range'][t]**2))
		
		with torch.no_grad():
			predicted_obj_pos = model.forward(np.array([[vx, config['t_range'][t]]])).item()
		
		# print(simulated_obj_pos, predicted_obj_pos)
		history.append([config['t_range'][t], simulated_obj_pos, predicted_obj_pos])

history = np.array(history)
epochs = history[:, 0].ravel()
simulated_positions = history[:, 1].ravel()
predicted_positions = history[:, 2].ravel()
plt.clf()
plt.plot(epochs, simulated_positions, color = (63/255, 97/255, 143/255), label='Simulated Position')
plt.plot(epochs, predicted_positions, color = (179/255, 89/255, 92/255), label='Predicted Position')
plt.title('Simulated and Predicted positions\n')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.legend()
plt.show()