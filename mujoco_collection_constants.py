import numpy as np

render = False
save_debug = False
save_collected = True
TOTAL_ITERATIONS = 1000
V_VALUES = 20

vx_range = np.linspace(0.0, 1.0, num=V_VALUES, dtype=np.float32)
t_range = np.arange(start=0.0, stop=0.002*TOTAL_ITERATIONS, step=0.002)

isPIDNN = True
acc = np.float32(-0.005)    # For use with artificial data collection