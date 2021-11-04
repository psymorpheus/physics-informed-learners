from os import terminal_size
import sys
import numpy as np

if len(sys.argv)<2:
    print("Please enter datafile to analyse")
    sys.exit(1)

for filename in sys.argv[1:]:
    data = np.genfromtxt(filename, dtype=np.float32, delimiter=',')
    vx_range = data[0, 1:]
    t_range = data[1:, 0]
    data = data[1:, 1:]

    num_vx = vx_range.shape[0]
    num_t = t_range.shape[0]

    print(filename)
    print("VX_RANGE:", vx_range)
    print("T_RANGE:", t_range)
    print("NUM_VX:", num_vx)
    print("NUM_T:", num_t)
    print("DATASIZE:", num_vx*num_t)
    
    stopped_points = np.zeros(shape=num_vx, dtype=np.int32)
    total_points = (num_t-1) * np.ones(shape=num_vx, dtype=np.int32)
    for i in range(num_vx):
        for j in range(1, num_t):
            if(data[j][i]==data[j-1][i]):
                stopped_points[i] = num_t-j
                break
    print("NUM_STOPPED:", stopped_points)
    print("PERCENTAGE_STOPPED:", 100*stopped_points/total_points)

    print("-----------------------------------------")

# x = input()

# EPSILON = 1e-16
# data = np.genfromtxt('/Users/morpheus/Desktop/physics-informed-learners/data_mujoco_stop.csv', delimiter=',')
# data = np.array(data, dtype=np.float32)

# m = 1000.0
# n = 20.0

# # percentage of time it is moving
# percentage = []

# for i in range(int(n)):
#     count = 1.0
#     for j in range(1,int(m)):
#         if data[j][i] - data[j-1][i]<EPSILON:
#             percentage.append((count*100.0)/m)
#             break
#         else:
#             count = count + 1
#     if j==m:
#         percentage.append(100.0)
    
# print(percentage)

