import numpy as np
# import mujoco_configloader as mcl

EPSILON = 1e-16
data = np.genfromtxt('/Users/morpheus/Desktop/physics-informed-learners/data_mujoco_stop.csv', delimiter=',')
data = np.array(data, dtype=np.float32)

m = 1000.0
n = 20.0

# percentage of time it is moving
percentage = []

for i in range(int(n)):
    count = 1.0
    for j in range(1,int(m)):
        if data[j][i] - data[j-1][i]<EPSILON:
            percentage.append((count*100.0)/m)
            break
        else:
            count = count + 1
    if j==m:
        percentage.append(100.0)
    
print(percentage)

