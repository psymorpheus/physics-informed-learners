from enum import Flag
import numpy as np

data_config = 'mujoco_stop'
filename = 'data_' + data_config + '.csv'

save_collected = True
model_config = None

collocation_is_border = False
take_differential_points = True
num_collocation = 200
num_differential = 10000
num_layers = 8
neurons_per_layer = 20

""" Tells the type of the differential equation used
1: x_t = v + acc*t
2: x_tt = acc
3. x_ttt = 0
"""
differential_order = 3

if data_config == 'artificial_nostop':
    artificial_data = True
    TOTAL_ITERATIONS = 1000
    V_VALUES = 20
    vx_range = np.linspace(0.0, 2.0, num=V_VALUES, dtype=np.float32)
    if not collocation_is_border: vx_range += 0.1
    t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
    acc = np.float32(-0.005)
    
elif data_config == 'artificial_stop':
    artificial_data = True
    TOTAL_ITERATIONS = 1000
    V_VALUES = 20
    vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
    t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
    acc = np.float32(-0.5)
    
elif data_config == 'mujoco_nostop':
    render = False
    save_debug = False
    artificial_data = False
    TOTAL_ITERATIONS = 40
    V_VALUES = 200
    vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
    if not collocation_is_border: vx_range += 1.5
    t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
    acc = np.float32(-0.4)
    
elif data_config == 'mujoco_stop':
    render = False
    save_debug = False
    artificial_data = False
    TOTAL_ITERATIONS = 1000
    V_VALUES = 20
    vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
    t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)

else:
    print("Not a valid data configuration.")
    import sys
    sys.exit(1)

if __name__=="__main__":
    if artificial_data:
        from mujoco_artificial_collect_data import artificial_collect_data
        artificial_collect_data()
    else:
        from mujoco_custom_collect_data import custom_collect_data
        custom_collect_data()
