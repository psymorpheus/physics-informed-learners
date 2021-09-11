import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.lib.function_base import append
import scipy.io as sc
import time
from tqdm import tqdm
import mujoco_collection_constants as mcc

def artificial_collect_data():
	# Final shape of collected data is (mcc.TOTAL_ITERATIONS, V_VALUES)
	collected_data = np.zeros(shape=(len(mcc.t_range),1), dtype=np.float32)

	for vx in tqdm(mcc.vx_range, desc = 'tqdm() Progress Bar'):
		data = np.zeros((len(mcc.t_range),1), dtype=np.float32)
		stopping_time = vx/(-1*mcc.acc)
		stopping_distance = (vx**2)/(-2*mcc.acc)

		for t in range(len(mcc.t_range)):
			if (mcc.t_range[t]>=stopping_time):
				data[t][0] = (stopping_distance)
			else:
				data[t][0] = (vx*mcc.t_range[t] + 0.5*mcc.acc*(mcc.t_range[t]**2))

		collected_data = np.hstack([collected_data, data])

	# To remove starting pad of zeros
	collected_data = collected_data[:, 1:]

	if mcc.save_collected:
		np.savetxt(mcc.filename, collected_data, delimiter=",")

if __name__=="__main__":
  print("Please call this from collection constants file.")