import torch
import torch.autograd as autograd         # computation graph TODO see course about this
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1 import make_axes_locatable         # 3D plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker

import numpy as np
import time
from pyDOE import lhs         # Latin Hypercube Sampling
import scipy.io               # Loading .mat matlab data 

import mujoco_collection_constants as mcc

device = None

class PINN(nn.Module):
	
	def __init__(self, X_u, u, X_f, layers, lb, ub, mu):
		""" For better comments refer to Burgers-PINN/myBurgers.py"""

		super().__init__() #call __init__ from parent class 
				
		# Try to specify types like float as much as you can
		self.u_b = torch.from_numpy(ub).float().to(device)
		self.l_b = torch.from_numpy(lb).float().to(device)
	   
		self.xt_u = X_u
		
		self.xt_f = X_f
		
		self.u = u
		
		self.layers = layers
		self.mu = mu

		# activation function
		self.activation = nn.Tanh()

		# loss function
		self.loss_function = nn.MSELoss(reduction ='mean')
	
		# Initialise neural network as a list using nn.Modulelist
		# nn.Linear applies a linear transformation xA^T + b to the incoming data (acts like fc layer)
		# Number of layers will be len(layers)-1
		# https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463/3

		self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
		
		self.iter = 0
	
		# Xavier Normal Initialization
		# Xavier initialization method tries to initialize weights with a smarter value, such that neurons won't start training in saturation
		# std = gain * sqrt(2/(input_dim+output_dim))
		# Done similar to TensorFlow variation 
		# Can access weights and data of an nn.Module object viw .weight.data and .bias.data

		for i in range(len(layers)-1):
			
			# weights from a normal distribution with 
			# Recommended gain value for tanh = 5/3? TODO
			nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
			
			# set biases to zero
			nn.init.zeros_(self.linears[i].bias.data)

	def fill_meta(self, X_test_tensor, u_test, f_hat):
		self.X_test_tensor = X_test_tensor
		self.u_test = u_test
		self.f_hat = f_hat
	
	def forward(self,x):
		
		# To get from numpy to torch tensor
		if torch.is_tensor(x) != True:         
			x = torch.from_numpy(x)             
					  
		# Preprocessing input - sclaed from 0 to 1
		x = (x - self.l_b)/(self.u_b - self.l_b)
		
		# Convert to float
		a = x.float()
		
		for i in range(len(self.layers)-2):
			z = self.linears[i](a)
			a = self.activation(z)

		# Activation is not applied to last layer
		a = self.linears[-1](a)
		
		return a

	def loss_BC(self,x,y):

		# Loss at boundary and inital conditions
		loss_u = self.loss_function(self.forward(x), y)
		return loss_u

	def loss_PDE(self, x_to_train_f):
		
		# Loss at collocation points, calculated from Partial Differential Equation
						
		g = x_to_train_f.clone()    
		g.requires_grad = True
		
		u = self.forward(g)
		
		if mcc.differential_order==1:
			x_v_t = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			f = x_t - g[:,0:1] - mcc.acc*g[:,1:]
		elif mcc.differential_order==2:
			x_v_t = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(x_to_train_f.shape).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			f = x_tt - mcc.acc
		elif mcc.differential_order==3:
			x_v_t = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(x_to_train_f.shape).to(device), retain_graph=True, create_graph=True)[0]
			x_vvv_ttt = autograd.grad(x_vv_tt,g,torch.ones(x_to_train_f.shape).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			x_ttt = x_vvv_ttt[:,[1]]
			f = x_ttt
															
		# x_v = x_v_t[:,[0]]
		# x_t = x_v_t[:,[1]]

		loss_f = self.loss_function(f,self.f_hat)
				
		return loss_f

	def loss(self,x,y,x_to_train_f):

		loss_u = self.loss_BC(x,y)
		if mcc.take_differential_points:
			loss_f = self.loss_PDE(x_to_train_f)
		else:
			loss_f = 0
		loss_val = loss_u + loss_f
		
		return loss_val

	# Some optimization algorithms such as Conjugate Gradient and LBFGS need to re-evaluate the function multiple times,
	# so you have to pass in a closure that allows them to recompute your model.
	# The closure should clear the gradients, compute the loss, and return it.

	def closure(self):
		
		optimizer.zero_grad()
		loss = self.loss(self.xt_u, self.u, self.xt_f)
		loss.backward()		# To get gradients
				
		self.iter += 1
		if self.iter % 100 == 0:
			error_vec, _ = self.test()
			print(loss, error_vec)

		return loss
	
	def test(self):
				
		u_pred = self.forward(self.X_test_tensor)
		error_vec = torch.linalg.norm((self.u_test-u_pred),2)/torch.linalg.norm(self.u_test,2)		# Relative L2 Norm of the error (Vector)
		u_pred = u_pred.cpu().detach().numpy()
		u_pred = np.reshape(u_pred,(mcc.vx_range.shape[0],mcc.t_range.shape[0]),order='F')
		return error_vec, u_pred

def main_loop(N_u, N_f, num_layers, num_neurons):
	torch.set_default_dtype(torch.float)
	torch.manual_seed(1234)
	np.random.seed(1234)

	global device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("Running this on", device)
	if device == 'cuda': 
		print(torch.cuda.get_device_name())

	mu = np.float32(0.1)

	# layers is a list not an ndarray
	layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

	data = np.genfromtxt(mcc.filename, delimiter=',')
	data = np.array(data, dtype=np.float32)

	assert data.shape[0] == mcc.TOTAL_ITERATIONS
	assert data.shape[1] == mcc.vx_range.shape[0]
	
	# All these are numpy.ndarray
	# t.shape = (100,1), x.shape = (256,1), Exact.shape = (100,256)
	# usol means u for solution to those points
	# [:,None] is used to make a vertical array out of a horizontal one

	v = mcc.vx_range
	t = mcc.t_range
	xsol = data
	
	# X.shape = (100,256), Y.shape = (100, 256)
	# X has x repeated 100 times and T has t repeated 256 times

	V, T = np.meshgrid(v,t)
	
	# Horizontally stack them, X_test.shape = (25600,2) = u_star.shape

	VT_test = np.hstack((V.flatten()[:,None], T.flatten()[:,None]))
	x_true = xsol.flatten('C')[:,None]
	# Whether fortran-style flatten (column-major) to be used or normal: Transposing and .flatten('F') are the same thing

	# Domain bounds
	# numpy.ndarray.min(axis) returns the min along a given axes
	# lb = [-1,0], ub = [1,0.99]

	lb = VT_test[0]  # [0. 0.]
	ub = VT_test[-1] # [1.  1.998]
		
	# Python splicing a:b does not include b
	# xx1 has all values of x at time t=0, uu1 has corresponding u values
	# xx2 has min x at all t, xx3 has max x at all t

	#Initial Condition -1 =< x =<1 and t = 0  
	leftedge_vt = np.hstack((V[:,0][:,None], T[:,0][:,None])) #L1
	leftedge_x = xsol[:,0][:,None]

	#Boundary Condition x = -1 and 0 =< t =<1
	# bottomedge_vt = np.hstack((V[:,0][:,None], T[:,0][:,None])) #L2
	# bottomedge_x = xsol[-1,:][:,None]

	#Boundary Condition x = 1 and 0 =< t =<1
	topedge_vt = np.hstack((V[0,:][:,None], T[0,:][:,None])) #L3
	topedge_x = xsol[0,:][:,None]

	# xx1 forms initial condition, xx2 & xx3 form boundary conditions
	# vstack for vertically stacking (_,2) ndarrays
	# N_u is training data and N_f is collocation points
	# lhs(n, [samples, criterion, iterations])
	# n - number of factors, samples - number of samples to generate for each factor
	# Gives an samples*n output
	# X_f_train contains all collocation as well as ALL boundary and initial points (without sampling)
	
	VT_u_basecases = np.vstack([leftedge_vt, topedge_vt])
	x_basecases = np.vstack([leftedge_x, topedge_x])

	# idx tells which indices to pick finally by randomly sampling without replacement

	if mcc.collocation_is_border:
		idx = np.random.choice(VT_u_basecases.shape[0], N_u, replace=False)
		VT_u_train = VT_u_basecases[idx, :]
		u_train = x_basecases[idx, :]
	else:
		idx = np.random.choice(VT_test.shape[0], N_u, replace=False)
		VT_u_train = VT_test[idx, :]
		u_train = x_true[idx, :]

	# TODO correct this in initial paper, X_f_train should not contain X_u_train, or it should contain them properly sampled

	# These f points are the ones where it forces differential value to become 0
	VT_f_train = lb + (ub-lb)*lhs(2, N_f)
	VT_f_train = np.vstack((VT_f_train, VT_u_train))                  # TODO Don't know why this should be here

	# Convert all to tensors

	VT_u_train = torch.from_numpy(VT_u_train).float().to(device)
	u_train = torch.from_numpy(u_train).float().to(device)
	VT_f_train = torch.from_numpy(VT_f_train).float().to(device)

	VT_test_tensor = torch.from_numpy(VT_test).float().to(device)
	u = torch.from_numpy(x_true).float().to(device)
	f_hat = torch.zeros(VT_f_train.shape[0],1).to(device)
		
	model = PINN(VT_u_train, u_train, VT_f_train, layers, lb, ub, mu)
	model.fill_meta(VT_test_tensor, u, f_hat)
	model.to(device)

	print(model)

	# L-BFGS Optimizer

	global optimizer
	optimizer = torch.optim.LBFGS(model.parameters(), lr=0.01, 
								max_iter = 50000,
								tolerance_grad = 1.0 * np.finfo(float).eps, 
								tolerance_change = 1.0 * np.finfo(float).eps, 
								history_size = 100)
	
	start_time = time.time()
	optimizer.step(model.closure)		# Does not need any loop like Adam
	elapsed = time.time() - start_time                
	print('Training time: %.2f' % (elapsed))


	""" Model Accuracy """ 
	error_vec, u_pred = model.test()

	print('Test Error: %.5f'  % (error_vec))

if __name__ == "__main__": 
	main_loop(mcc.num_collocation, mcc.num_differential, mcc.num_layers, mcc.neurons_per_layer)
