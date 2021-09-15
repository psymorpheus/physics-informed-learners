from enum import Flag
import numpy as np

data_config = ['artificial','nostop']
filename = 'data'
for dc in data_config: filename += '_' + dc
filename += '.csv'

save_collected = True
model_config = None

training_is_border = True
take_differential_points = True
num_datadriven = 200
num_collocation = 1000
num_layers = 8
neurons_per_layer = 20
num_validation = 1000
batch_size = 64

""" Tells the type of the differential equation used
1: x_t = v + acc*t
2: x_tt = acc
3. x_ttt = 0
"""
differential_order = 3

def invalid_config():
    print("Not a valid data/model configuration.")
    import sys
    sys.exit(1)

if 'artificial' in data_config:
    artificial_data = True
    if 'stop' in data_config:
        TOTAL_ITERATIONS = 1000
        V_VALUES = 20
        vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
        t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
        acc = np.float32(-0.5)
    elif 'nostop' in data_config:
        TOTAL_ITERATIONS = 1000
        V_VALUES = 20
        vx_range = np.linspace(0.0, 2.0, num=V_VALUES, dtype=np.float32)
        if not training_is_border: vx_range += 0.1
        t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
        acc = np.float32(-0.005)
    else:
        invalid_config()
elif 'mujoco' in data_config:
    render = False
    save_debug = False
    artificial_data = False
    acc = np.float32(-0.4)
    if 'stop' in data_config:
        TOTAL_ITERATIONS = 1000
        V_VALUES = 20
        vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
        t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
    elif 'nostop' in data_config:
        TOTAL_ITERATIONS = 40
        V_VALUES = 200
        vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
        if not training_is_border: vx_range += 1.5
        t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
    else:
        invalid_config()
else:
    invalid_config()

if __name__=="__main__":
    if artificial_data:
        from mujoco_artificial_collect_data import artificial_collect_data
        artificial_collect_data()
    else:
        from mujoco_custom_collect_data import custom_collect_data
        custom_collect_data()
