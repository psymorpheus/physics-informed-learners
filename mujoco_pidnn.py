from numpy.core.fromnumeric import argmin
import torch
import torch.autograd as autograd
import torch.nn as nn

import numpy as np
import time
import matplotlib.pyplot as plt

import mujoco_dataloader as mdl

optimizer = None
training_history = []		# Has iter, training loss, validation loss
last_training_loss = None

def plot_history(id, config, elapsed, error_validation):
	valid_history = np.array(training_history[id])
	epochs = valid_history[:, 0].ravel()
	loss = {}
	loss['Training'] = valid_history[:, 1].ravel()
	loss['Validation'] = valid_history[:, 2].ravel()

	for loss_type in loss.keys():
		plt.clf()
		plt.plot(epochs, loss[loss_type], color = (63/255, 97/255, 143/255), label=f'{loss_type} loss')
		if (loss_type == 'training'): title = 'Training loss (MSE)\n'
		else : title = 'Validation loss (Relative MSE)\n'
		if config['take_differential_points']: alpha = config['ALPHA'][id]
		else: alpha = 0

		plt.title(title + f'Elapsed: {elapsed:.2f}, Validation Error: {error_validation:.2f}, Train Error: {last_training_loss[id]:.2f}, Alpha: {alpha}')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		savefile_name = 'plot_' + config['model_name'] + '_' + loss_type
		savefile_name += '.png'
		plt.savefig(config['modeldir'] + savefile_name)

class PINN(nn.Module):
	
	def __init__(self, id, VT_u, X_u, VT_f, layers, lb, ub, device, config, alpha):
		""" For better comments refer to Burgers-PINN/myBurgers.py """

		super().__init__()
		
		self.id = id
		self.device = device
		self.config = config

		self.u_b = ub
		self.l_b = lb
		self.alpha = alpha
	   
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
		
		if self.config['differential_order']==1:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(self.device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			f = x_t - g[:,0:1] - self.config['acc']*g[:,1:]
		elif self.config['differential_order']==2:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(VT_f_train.shape).to(self.device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			f = x_tt - self.config['acc']
		elif self.config['differential_order']==3:
			x_v_t = autograd.grad(u,g,torch.ones([VT_f_train.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
			x_vv_tt = autograd.grad(x_v_t,g,torch.ones(VT_f_train.shape).to(self.device), retain_graph=True, create_graph=True)[0]
			x_vvv_ttt = autograd.grad(x_vv_tt,g,torch.ones(VT_f_train.shape).to(self.device), create_graph=True)[0]
			x_t = x_v_t[:,[1]]
			x_tt = x_vv_tt[:,[1]]
			x_ttt = x_vvv_ttt[:,[1]]
			f = x_ttt

		loss_f = self.loss_function(f,self.f_hat)
				
		return loss_f

	def loss(self,VT_u_train,X_u_train,VT_f_train):

		loss_u = self.loss_BC(VT_u_train,X_u_train)
		if self.config['take_differential_points']:
			loss_f = self.loss_PDE(VT_f_train)
		else:
			loss_f = 0
		loss_val = loss_u + self.alpha * loss_f
		
		return loss_val

	def closure(self):
		""" Called multiple times by optimizers like Conjugate Gradient and LBFGS.
		Clears gradients, compute and return the loss.
		"""
		global last_training_loss
		optimizer.zero_grad()
		loss = self.loss(self.VT_u, self.X_u, self.XT_f)
		loss.backward()		# To get gradients

		self.iter += 1

		if self.iter % 100 == 0:
			training_loss = loss.item()
			validation_loss = mdl.set_loss(self, self.device).item()
			training_history[self.id].append([self.iter, training_loss, validation_loss])
			print(
				'Iter %d, Training: %.5e, Validation: %.5e' % (self.iter, training_loss, validation_loss)
			)
			last_training_loss[self.id] = training_loss

		return loss


def pidnn_driver(config):
	global training_history, last_training_loss
	training_history = [[] for i in config['ALPHA']]
	last_training_loss = [None for i in config['ALPHA']]

	N_u = config['num_datadriven']
	N_f = config['num_collocation']
	num_layers = config['num_layers']
	num_neurons = config['neurons_per_layer']

	torch.set_default_dtype(torch.float)
	torch.manual_seed(1234)
	np.random.seed(1234)

	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'	# set as cpu for more parallel training

	print("Running this on", device)
	if device == 'cuda': 
		print(torch.cuda.get_device_name())

	# layers is a list, not an ndarray
	layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()

	VT_u_train, u_train, VT_f_train, lb, ub = mdl.dataloader(config, device)

	models = []
	validation_losses = []
	
	if not config['take_differential_points']:
		num_alpha = 1
	else:
		num_alpha = len(config['ALPHA'])

	for i in range(num_alpha):
		alpha = config['ALPHA'][i]
		model = PINN(i, VT_u_train, u_train, VT_f_train, layers, lb, ub, device, config, alpha)
		model.to(device)

		# print(model)

		# L-BFGS Optimizer
		global optimizer
		optimizer = torch.optim.LBFGS(model.parameters(), lr=0.001, 
									max_iter = 35000,
									tolerance_grad = 1.0 * np.finfo(float).eps, 
									tolerance_change = 1.0 * np.finfo(float).eps, 
									history_size = 100)
		# optimizer = torch.optim.LBFGS(
		# 	model.parameters(), 
		# 	lr=0.01, 
		# 	max_iter=50000, 
		# 	max_eval=50000, 
		# 	history_size=50,
		# 	tolerance_grad=1.0 * np.finfo(float).eps, 
		# 	tolerance_change=1.0 * np.finfo(float).eps,
		# 	line_search_fn="strong_wolfe"       # can be "strong_wolfe"
		# )
		
		start_time = time.time()
		optimizer.step(model.closure)		# Does not need any loop like Adam
		elapsed = time.time() - start_time                
		print('Training time: %.2f' % (elapsed))

		validation_losses.append(mdl.set_loss(model, device).item())

		model.to('cpu')
		models.append(model)

	model = models[argmin(validation_losses)] # choosing best model out of the bunch

	""" Model Accuracy """ 
	error_validation = validation_losses[model.id]
	print('Validation Error: %.5f'  % (error_validation))

	"""" For plotting model train and validation errors """
	if config['SAVE_PLOT']: plot_history(model.id, config, elapsed, error_validation)

	""" Saving model for reloading later """
	if config['SAVE_MODEL']: torch.save(model, config['modeldir'] + config['model_name'] + '.pt')

	if device == 'cuda':
		torch.cuda.empty_cache()

# if __name__ == "__main__": 
# 	main_loop(config['num_datadriven'], config['num_collocation'], config['num_layers'], config['neurons_per_layer'], config['num_validation'])
