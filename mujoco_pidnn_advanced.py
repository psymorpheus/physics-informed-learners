import torch
from collections import OrderedDict

from pyDOE import lhs
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time

import mujoco_dataloader as mdl

optimizer = None
training_history = []		# Has iter, training loss, validation loss
last_training_loss = None

# the deep neural network
class DNN(torch.nn.Module):
	def __init__(self, layers):
		super(DNN, self).__init__()
		
		# parameters
		self.depth = len(layers) - 1
		
		# set up layer order dict
		self.activation = torch.nn.Tanh
		
		layer_list = list()
		for i in range(self.depth - 1): 
			layer_list.append(
				('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
			)
			layer_list.append(('activation_%d' % i, self.activation()))
			
		layer_list.append(
			('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
		)
		layerDict = OrderedDict(layer_list)
		
		# deploy layers
		self.layers = torch.nn.Sequential(layerDict)
		
	def forward(self, x):
		out = self.layers(x)
		return out

# the physics-guided neural network
class PhysicsInformedNN():
	def __init__(self, X_u, u, X_f, layers, lb, ub, device):
		
		# boundary conditions
		self.lb = lb
		self.ub = ub
		
		# data
		self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True).float().to(device)
		self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True).float().to(device)
		self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
		self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
		self.u = torch.tensor(u).float().to(device)
		
		self.layers = layers
		self.device = device
		
		# deep neural networks
		self.dnn = DNN(layers).to(device)
		
		# optimizers: using the same settings
		self.optimizer = torch.optim.LBFGS(
			self.dnn.parameters(), 
			lr=0.1, 
			max_iter=50000, 
			max_eval=50000, 
			history_size=50,
			tolerance_grad=1e-5, 
			tolerance_change=1.0 * np.finfo(float).eps,
			line_search_fn="strong_wolfe"       # can be "strong_wolfe"
		)

		self.iter = 0
		
	def net_u(self, x, t):  
		u = self.dnn(torch.cat([x, t], dim=1))
		return u
	
	def net_f(self, x, t):
		""" The pytorch autograd version of calculating residual """
		u = self.net_u(x, t)

		u_t = torch.autograd.grad(
			u, t, 
			grad_outputs=torch.ones_like(u),
			retain_graph=True,
			create_graph=True
		)[0]
		u_tt = torch.autograd.grad(
			u_t, t, 
			grad_outputs=torch.ones_like(u_t),
			retain_graph=True,
			create_graph=True
		)[0]
		u_ttt = torch.autograd.grad(
			u_tt, t, 
			grad_outputs=torch.ones_like(u_tt),
			retain_graph=True,
			create_graph=True
		)[0]
		f = u_ttt
		return f

		u_x = torch.autograd.grad(
			u, x, 
			grad_outputs=torch.ones_like(u),
			retain_graph=True,
			create_graph=True
		)[0]
		u_xx = torch.autograd.grad(
			u_x, x, 
			grad_outputs=torch.ones_like(u_x),
			retain_graph=True,
			create_graph=True
		)[0]
		
		f = u_t + u * u_x - self.nu * u_xx
		return f
	
	def loss_func(self):
		self.optimizer.zero_grad()
		
		u_pred = self.net_u(self.x_u, self.t_u)
		f_pred = self.net_f(self.x_f, self.t_f)
		loss_u = torch.mean((self.u - u_pred) ** 2)
		loss_f = torch.mean(f_pred ** 2)
		
		loss = loss_u + loss_f
		
		loss.backward()
		self.iter += 1
		if self.iter % 100 == 0:
			print(
				'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (self.iter, loss.item(), loss_u.item(), loss_f.item())
			)
		return loss
	
	def train(self):
		self.dnn.train()
				
		# Backward and optimize
		self.optimizer.step(self.loss_func)

			
	def predict(self, X):
		x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
		t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)

		self.dnn.eval()
		u = self.net_u(x, t)
		f = self.net_f(x, t)
		u = u.detach().cpu().numpy()
		f = f.detach().cpu().numpy()
		return u, f

def pidnn_driver_advanced(config):
	global training_history
	training_history = []

	N_u = config['num_datadriven']
	N_f = config['num_collocation']
	num_layers = config['num_layers']
	num_neurons = config['neurons_per_layer']
	N_validation = config['num_validation']

	torch.set_default_dtype(torch.float)
	torch.manual_seed(1234)
	np.random.seed(1234)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("Running this on", device)
	if device == 'cuda': 
		print(torch.cuda.get_device_name())

	# layers is a list, not an ndarray
	layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

	VT_u_train, u_train, VT_f_train, lb, ub = mdl.dataloader(config, device)

	model = PhysicsInformedNN(VT_u_train, u_train, VT_f_train, layers, lb, ub, device)

	model.train()

	""" Model Accuracy """ 
	error_test = mdl.testset_loss(model.dnn, device).item()
	print('Test Error: %.5f'  % (error_test))                   

	# U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
	# Error = np.abs(Exact - U_pred)

