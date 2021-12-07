import numpy as np
from numpy.core.fromnumeric import reshape, shape
from numpy.lib.function_base import append
import scipy.io as sc
import time
from tqdm import tqdm

def toy_datagen(config):
    # Final shape of collected data is (config['TOTAL_ITERATIONS'], V_VALUES)
    # collected_data = np.zeros(shape=(len(config['t_range']),1), dtype=np.float32)
    collected_data = np.array(config['t_range']).reshape((len(config['t_range']),1))
    vx_range = np.array(config['vx_range'])
    for vx in tqdm(vx_range, desc = 'Toy data generation progress'):
        data = np.zeros((len(config['t_range']),1), dtype=np.float32)
        if not config['WILDCARD']:
            stopping_time = vx/(-1*config['acc'])
            stopping_distance = (vx**2)/(-2*config['acc'])

            for t in range(len(config['t_range'])):
                if (config['t_range'][t]>=stopping_time):
                    data[t][0] = (stopping_distance)
                else:
                    data[t][0] = (vx*config['t_range'][t] + 0.5*config['acc']*(config['t_range'][t]**2))

            collected_data = np.hstack([collected_data, data])
        else:
            # -1 acc for 400m, free fall for 5 sec, -1 acc for remaining duration
            a1 = config['a1']
            d1 = config['d1']
            t1 = config['t1']
            a2 = config['a2']
            tsteps = len(config['t_range'])
            if vx < (np.sqrt(-2 * d1 * a1)):
                t_stop = -1 * vx/a1
                s_stop = (vx**2)/(-2 * a1)
                for t in range(tsteps):
                    if (config['t_range'][t]>=t_stop):
                        simulated_obj_pos = (s_stop)
                    else:
                        simulated_obj_pos = (vx*config['t_range'][t] + 0.5*(a1)*(config['t_range'][t]**2))
                    data[t][0] = simulated_obj_pos
                collected_data = np.hstack([collected_data, data])
            else:
                v_launch = np.sqrt((vx**2) + 2*a1*d1)
                t_up = (v_launch - vx)/a1
                t_stop = -1 * v_launch/a2
                for t in range(tsteps):
                    if config['t_range'][t] < t_up:
                        simulated_obj_pos = vx*config['t_range'][t] + 0.5*a1*(config['t_range'][t]**2)
                    elif config['t_range'][t] < (t_up + t1):
                        simulated_obj_pos = d1 + v_launch*(config['t_range'][t] - t_up)
                    elif (config['t_range'][t] - (t_up + t1))< t_stop or a2>0:
                            simulated_obj_pos = d1 + t1*v_launch + v_launch*(config['t_range'][t] - (t_up+t1)) + 0.5*a2*((config['t_range'][t] - (t_up+t1))**2)
                    else:
                        simulated_obj_pos = (v_launch**2 - vx**2)/(2*a1) + t1*v_launch + (-v_launch**2)/(2*a2)
                    data[t][0] = simulated_obj_pos
                collected_data = np.hstack([collected_data, data])

    collected_data = np.vstack([np.zeros(shape=(1,1+len(config['vx_range']))), collected_data])
    collected_data[0, 0] = -np.inf
    collected_data[0, 1:] = config['vx_range']

    if config['save_collected']:
        np.savetxt(config['datadir'] + config['datafile'], collected_data, delimiter=",")

if __name__=="__main__":
  print("Please call this from collection constants file.")