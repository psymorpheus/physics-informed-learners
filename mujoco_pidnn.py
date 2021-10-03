import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np
import time
import matplotlib.pyplot as plt

from mujoco_configloader import config
import mujoco_dataloader as mpd

device = None
training_history = []		# Has iter, training loss, validation loss, testing loss

def plot_history():
	global training_history
	training_history = np.array(training_history)
	epochs = training_history[:, 0].ravel()
	training_loss = training_history[:, 1].ravel()
	validation_loss = training_history[:, 2].ravel()
	testing_loss = training_history[:, 3].ravel()
	plt.plot(epochs, training_loss, color = (63/255, 97/255, 143/255), label='Training loss')
	plt.plot(epochs, validation_loss, color = (179/255, 89/255, 92/255), label='Validation loss')
	plt.plot(epochs, testing_loss, color = (107/255, 153/255, 84/255), label='Testing loss')
	plt.title('Training, Validation and Testing loss (MSE, relative L2, relative L2)')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	# plt.show()
	savefile_name = 'plot_' + config['filename'][5:-4]
	if config['training_is_border']: savefile_name += '_border'
	else: savefile_name += '_internal'
	savefile_name += '_order_' + str(config['differential_order'])
	savefile_name += '.png'
	plt.savefig(savefile_name)

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

		self.f_hat = torch.zeros(VT_f.shape[0],1).to(device)

		self.activation = nn.Tanh()
		self.loss_function = nn.MSELoss(reduction ='mean')
		self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
		self.iter = 0

		for i in range(len(layers)-1):
			# Recommended gain value for tanh = 5/3? TODO
			nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
			nn.init.zeros_(self.linears[i].bias.data)
	
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
		
		if config['differential_order']==1:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			f = x_t - g[:,0:1] - config['acc']*g[:,1:]
		elif config['differential_order']==2:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(VT_f_train.shape).to(device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			f = x_tt - config['acc']
		elif config['differential_order']==3:
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
		if config['take_differential_points']:
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
			training_loss = loss.item()
			validation_loss = mpd.testset_loss(self, device).item()
			testing_loss = mpd.testset_loss(self, device, validation=False).item()
			training_history.append([self.iter, training_loss, validation_loss, testing_loss])
			print(training_loss, validation_loss, testing_loss)

		return loss

def main_loop(N_u, N_f, num_layers, num_neurons, N_validation):
	torch.set_default_dtype(torch.float)
	torch.manual_seed(1234)
	np.random.seed(1234)

	global device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print("Running this on", device)
	if device == 'cuda': 
		print(torch.cuda.get_device_name())

	# layers is a list, not an ndarray
	layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

	VT_u_train, u_train, VT_f_train, lb, ub = mpd.dataloader(N_u, N_f, device, N_validation)
		
	model = PINN(VT_u_train, u_train, VT_f_train, layers, lb, ub)
	model.to(device)

	print(model)
	print("[Training] [Validation] [Testing]")

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
	error_test = mpd.testset_loss(model, device, validation=False).item()
	error_validation = mpd.testset_loss(model, device).item()

	print('Validation Error: %.5f'  % (error_validation))
	print('Test Error: %.5f'  % (error_test))

	"""" For plotting model train, validation, test errors """
	plot_history()

if __name__ == "__main__": 
	main_loop(config['num_datadriven'], config['num_collocation'], config['num_layers'], config['neurons_per_layer'], config['num_validation'])
