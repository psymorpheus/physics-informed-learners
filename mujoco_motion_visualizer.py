import gym
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from mujoco_ff import FF_Baseline
from mujoco_pidnn import PINN

with open("mujoco_config.yaml", "r") as f:
    all_configs = yaml.safe_load(f)
    common_config = all_configs['COMMON'].copy()
    # Filling in models from model templates
    for instance in common_config['ALL_MODEL_CONFIGS']:
        template_name = instance[:instance.rfind('_')]
        training_points = int(instance[(instance.rfind('_')+1):])
        template_config = all_configs[template_name].copy()
        template_config['num_datadriven'] = training_points
        template_config['model_name'] = template_name.lower() + '_' + str(training_points)
        all_configs[template_name + '_' + str(training_points)] = template_config

# vx = float(input('Enter initial velocity: '))
# tsteps = int(input('Enter number of timesteps: '))
vx = 29
tsteps = 1900

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

active_data_config_name = 'WILDCARD'
active_model_config_name = 'FF_800'
noise = 0.05

active_data_config = all_configs[active_data_config_name].copy()
active_data_config.update(common_config)
active_model_config = all_configs[active_model_config_name].copy()
active_model_config.update(active_data_config)
config = active_model_config
config['t_range'] = np.arange(start=0.0, stop = config['TIMESTEP']*tsteps, step = config['TIMESTEP'])

model = torch.load(f'Models/Noise_{int(noise*100)}/{active_data_config_name.lower()}/{active_model_config_name.lower()}.pt')
model.eval()
device = 'cuda'
model.to(device)
history = []

if config['WILDCARD']:
	# -1 acc for 400m, free fall for 5 sec, -1 acc for remaining duration
	a1 = config['a1']
	d1 = config['d1']
	t1 = config['t1']
	a2 = config['a2']
	if vx < (np.sqrt(-2 * d1 * a1)):
		t_stop = -1 * vx/a1
		s_stop = (vx**2)/(-2 * a1)
		for t in range(tsteps):
			if (config['t_range'][t]>=t_stop):
				simulated_obj_pos = (s_stop)
			else:
				simulated_obj_pos = (vx*config['t_range'][t] + 0.5*(a1)*(config['t_range'][t]**2))
			with torch.no_grad():
				predicted_obj_pos = model.forward(torch.tensor([[vx, config['t_range'][t]]]).to(device)).item()
			history.append([config['t_range'][t], simulated_obj_pos, predicted_obj_pos])

	else:
		v_launch = np.sqrt((vx**2) + 2*a1*d1)
		t_up = (v_launch - vx)/a1
		t_stop = -1 * v_launch/a2
		linear_from, linear_to = np.inf, 0
		for t in range(tsteps):
			if config['t_range'][t] < t_up:
				simulated_obj_pos = vx*config['t_range'][t] + 0.5*a1*(config['t_range'][t]**2)
			elif config['t_range'][t] < (t_up + t1):
				linear_from = min(linear_from, config['t_range'][t])
				linear_to = max(linear_to, config['t_range'][t])
				simulated_obj_pos = d1 + v_launch*(config['t_range'][t] - t_up)
			elif (config['t_range'][t] - (t_up + t1))< t_stop or a2>0:
				simulated_obj_pos = d1 + t1*v_launch + v_launch*(config['t_range'][t] - (t_up+t1)) + 0.5*a2*((config['t_range'][t] - (t_up+t1))**2)
			else:
				simulated_obj_pos = (v_launch**2 - vx**2)/(2*a1) + t1*v_launch + (-v_launch**2)/(2*a2)
			with torch.no_grad():
				predicted_obj_pos = model.forward(torch.tensor([[vx, config['t_range'][t]]]).to(device)).item()	
			history.append([config['t_range'][t], simulated_obj_pos, predicted_obj_pos])
		print(f'Linear from {linear_from} to {linear_to}')

elif not config['toy_data']:
	obj_init = np.array([vx,0.0], dtype=np.float32)
	env = gym.make('mujoco_collection_1:mujoco-slide-v0', obj_init=obj_init)
	iter = 0
	defStep = np.array([-1,-1,1,0],dtype=np.float32)
	obs = env.reset()
	initial_obj_pos = None
	initial_obj_vel = None
	while (iter<tsteps):
		obs, reward, done, info = env.step(defStep)
		obj_pos = obs["observation"][3:6]
		obj_vel = obs["observation"][14:17]

		if iter==0:
			''' Changing initial velocity value '''
			initial_obj_vel = obj_vel[0]/config['TIMESTEP'] # Because simulator returns this multiplied by dt
			initial_obj_pos = obj_pos

		simulated_obj_pos = obs["observation"][3:6][0] - initial_obj_pos[0]
		with torch.no_grad():
			predicted_obj_pos = model.forward(np.array([[initial_obj_vel, iter * config['TIMESTEP']]])).item()
		
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

history = np.array(history[:])
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
# plt.show()
plt.savefig('dummy.png')