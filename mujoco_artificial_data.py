import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
import scipy.io as sc
import time
from tqdm import tqdm
import mujoco_configloader as mcl

def artificial_data():
	# Final shape of collected data is (mcl.TOTAL_ITERATIONS, V_VALUES)
	collected_data = np.zeros(shape=(len(mcl.t_range),1), dtype=np.float32)

	for vx in tqdm(mcl.vx_range, desc = 'tqdm() Progress Bar'):
		data = np.zeros((len(mcl.t_range),1), dtype=np.float32)
		stopping_time = vx/(-1*mcl.acc)
		stopping_distance = (vx**2)/(-2*mcl.acc)

		for t in range(len(mcl.t_range)):
			if (mcl.t_range[t]>=stopping_time):
				data[t][0] = (stopping_distance)
			else:
				data[t][0] = (vx*mcl.t_range[t] + 0.5*mcl.acc*(mcl.t_range[t]**2))

		collected_data = np.hstack([collected_data, data])

	# To remove starting pad of zeros
	collected_data = collected_data[:, 1:]

	if mcl.save_collected:
		np.savetxt(mcl.filename, collected_data, delimiter=",")

if __name__=="__main__":
  print("Please call this from collection constants file.")