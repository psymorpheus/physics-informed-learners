import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
import scipy.io as sc
import time
from tqdm import tqdm
from mujoco_configloader import config

def artificial_datagen():
	# Final shape of collected data is (config['TOTAL_ITERATIONS'], V_VALUES)
	collected_data = np.zeros(shape=(len(config['t_range']),1), dtype=np.float32)

	for vx in tqdm(config['vx_range'], desc = 'Artificial data generation progress'):
		data = np.zeros((len(config['t_range']),1), dtype=np.float32)
		stopping_time = vx/(-1*config['acc'])
		stopping_distance = (vx**2)/(-2*config['acc'])

		for t in range(len(config['t_range'])):
			if (config['t_range'][t]>=stopping_time):
				data[t][0] = (stopping_distance)
			else:
				data[t][0] = (vx*config['t_range'][t] + 0.5*config['acc']*(config['t_range'][t]**2))

		collected_data = np.hstack([collected_data, data])

	# To remove starting pad of zeros
	collected_data = collected_data[:, 1:]

	if config['save_collected']:
		np.savetxt(config['filename'], collected_data, delimiter=",")

if __name__=="__main__":
  print("Please call this from collection constants file.")