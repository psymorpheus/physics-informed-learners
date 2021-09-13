import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np
import time

import mujoco_collection_constants as mcc
from mujoco_pidnn_dataloader import dataloader

device = None

class PINN(nn.Module):
	
	def __init__(self, VT_u, X_u, VT_f, layers, lb, ub):
		""" For better comments refer to Burgers-PINN/myBurgers.py """

		super().__init__()
		
		self.u_b = torch.from_numpy(ub).float().to(device)
		self.l_b = torch.from_numpy(lb).float().to(device)
	   
		self.VT_u = VT_u
		self.X_u = X_u
		self.XT_f = VT_f
		self.layers = layers

		self.activation = nn.Tanh()
		self.loss_function = nn.MSELoss(reduction ='mean')
		self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
		self.iter = 0

		for i in range(len(layers)-1):
			# Recommended gain value for tanh = 5/3? TODO
			nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
			nn.init.zeros_(self.linears[i].bias.data)

	def fill_meta(self, X_test_tensor, u_test, f_hat):
		self.X_test_tensor = X_test_tensor
		self.u_test = u_test
		self.f_hat = f_hat
	
	def forward(self,x):
		if torch.is_tensor(x) != True:
			x = torch.from_numpy(x)             
					  
		# Preprocessing input - sclaed from 0 to 1
		x = (x - self.l_b)/(self.u_b - self.l_b)
		a = x.float()
		
		for i in range(len(self.layers)-2):
			z = self.linears[i](a)
			a = self.activation(z)

		# Activation is not applied to last layer
		a = self.linears[-1](a)
		
		return a

	def loss_BC(self,x,y):
		""" Loss at boundary and initial conditions """
		
		loss_u = self.loss_function(self.forward(x), y)
		return loss_u

	def loss_PDE(self, VT_f_train):
		""" Loss at collocation points, calculated from Partial Differential Equation
		Note: x_v = x_v_t[:,[0]], x_t = x_v_t[:,[1]]
		"""
						
		g = VT_f_train.clone()  
		g.requires_grad = True
		
		u = self.forward(g)
		
		if mcc.differential_order==1:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			f = x_t - g[:,0:1] - mcc.acc*g[:,1:]
		elif mcc.differential_order==2:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(VT_f_train.shape).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			f = x_tt - mcc.acc
		elif mcc.differential_order==3:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(VT_f_train.shape).to(device), retain_graph=True, create_graph=True)[0]
			x_vvv_ttt = autograd.grad(x_vv_tt,g,torch.ones(VT_f_train.shape).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			x_ttt = x_vvv_ttt[:,[1]]
			f = x_ttt

		loss_f = self.loss_function(f,self.f_hat)
				
		return loss_f

	def loss(self,VT_u_train,X_u_train,VT_f_train):

		loss_u = self.loss_BC(VT_u_train,X_u_train)
		if mcc.take_differential_points:
			loss_f = self.loss_PDE(VT_f_train)
		else:
			loss_f = 0
		loss_val = loss_u + loss_f
		
		return loss_val

	def closure(self):
		""" Called multiple times by optimizers like Conjugate Gradient and LBFGS.
		Clears gradients, compute and return the loss.
		"""
		
		optimizer.zero_grad()
		loss = self.loss(self.VT_u, self.X_u, self.XT_f)
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

	# layers is a list not an ndarray
	layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

	VT_u_train, u_train, VT_f_train, lb, ub, VT_test_tensor, u, f_hat = dataloader(N_u, N_f, device)
		
	model = PINN(VT_u_train, u_train, VT_f_train, layers, lb, ub)
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
