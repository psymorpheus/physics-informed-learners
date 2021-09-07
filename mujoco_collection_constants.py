import numpy as np

render = False
save_debug = True
save_collected = True
TOTAL_ITERATIONS = 100
V_VALUES = 1

# vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
vx_range = np.array([2.0], dtype=np.float32)
t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)
isPIDNN = False