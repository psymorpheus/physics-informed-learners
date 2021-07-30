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

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Running this on", device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

def solutionplot(u_pred, X_u_train, u_train, X, T, x, t, usol):
    
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow', 
                extent=[T.min(), T.max(), X.min(), X.max()], 
                origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 4, clip_on = False)

    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(x,t)$', fontsize = 10)
    
    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''
    
    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,usol.T[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')    
    ax.set_title('$t = 0.25s$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,usol.T[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[50,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50s$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,usol.T[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,u_pred.T[75,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75s$', fontsize = 10)
    
    plt.savefig('myBurgers.png',dpi = 500)   
    plt.show()

class PINN(nn.Module):
    
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu):
        super().__init__() #call __init__ from parent class 
                
        # Try to specify types like float as much as you can
        self.u_b = torch.from_numpy(ub).float().to(device)
        self.l_b = torch.from_numpy(lb).float().to(device)
       
        self.xt_u = X_u
        
        self.xt_f = X_f
        
        self.u = u
        
        self.layers = layers
        self.nu = nu

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
                
        x_1_f = x_to_train_f[:,[0]]
        x_2_f = x_to_train_f[:,[1]]
                        
        g = x_to_train_f.clone()    
        g.requires_grad = True
        
        u = self.forward(g)
        # TODO see autograd
        u_x_t = autograd.grad(u,g,torch.ones([x_to_train_f.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]                 
        u_xx_tt = autograd.grad(u_x_t,g,torch.ones(x_to_train_f.shape).to(device), create_graph=True)[0]
                                                            
        u_x = u_x_t[:,[0]]
        u_t = u_x_t[:,[1]]
        u_xx = u_xx_tt[:,[0]]
                                        
        f = u_t + (self.forward(g))*(u_x) - (self.nu)*u_xx
        loss_f = self.loss_function(f,self.f_hat)
                
        return loss_f

    def loss(self,x,y,x_to_train_f):

        loss_u = self.loss_BC(x,y)
        loss_f = self.loss_PDE(x_to_train_f)
        loss_val = loss_u + loss_f
        
        return loss_val

    # Some optimization algorithms such as Conjugate Gradient and LBFGS need to reevaluate the function multiple times,
    # so you have to pass in a closure that allows them to recompute your model.
    # The closure should clear the gradients, compute the loss, and return it.

    def closure(self):
        
        optimizer.zero_grad()
        loss = self.loss(self.xt_u, self.u, self.xt_f)
        loss.backward()                                                             # To get gradients
                
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec, _ = self.test()
            print(loss,error_vec)

        return loss
    
    def test(self):
                
        u_pred = self.forward(self.X_test_tensor)
        error_vec = torch.linalg.norm((self.u_test-u_pred),2)/torch.linalg.norm(self.u_test,2)        # Relative L2 Norm of the error (Vector)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred,(256,100),order='F')
        return error_vec, u_pred

def main_loop(N_u, N_f, num_layers, num_neurons):
     
    nu = 0.01/np.pi

    # layers is a list not an ndarray

    layers = np.concatenate([[2], num_neurons*np.ones(num_layers), [1]]).astype(int).tolist()
    
    data = scipy.io.loadmat('Data/burgers_shock_mu_01_pi.mat')
    
    # All these are numpy.ndarray
    # t.shape = (100,1), x.shape = (256,1), Exact.shape = (100,256)
    # usol means u for solution to those points
    # [:,None] is used to make a vertical array out of a horizontal one

    x = data['x']
    t = data['t']
    usol = data['usol']
    
    # X.shape = (100,256), Y.shape = (100, 256)
    # X has x repeated 100 times and T has t repeated 256 times

    X, T = np.meshgrid(x,t)
    
    # Horizontally stack them, X_test.shape = (25600,2) = u_star.shape

    X_test = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_true = usol.flatten('F')[:,None] 
    # Whether fortran-style flatten (column-major) to be used or normal: Transposing and .flatten('F') are the same thing

    # Domain bounds
    # numpy.ndarray.min(axis) returns the min along a given axes
    # lb = [-1,0], ub = [1,0.99]

    lb = X_test[0]  # [-1. 0.]
    ub = X_test[-1] # [1.  0.99]   
        
    # Python splicing a:b does not include b
    # xx1 has all values of x at time t=0, uu1 has corresponding u values
    # xx2 has min x at all t, xx3 has max x at all t

    #Initial Condition -1 =< x =<1 and t = 0  
    leftedge_x = np.hstack((X[0,:][:,None], T[0,:][:,None])) #L1
    leftedge_u = usol[:,0][:,None]

    #Boundary Condition x = -1 and 0 =< t =<1
    bottomedge_x = np.hstack((X[:,0][:,None], T[:,0][:,None])) #L2
    bottomedge_u = usol[-1,:][:,None]

    #Boundary Condition x = 1 and 0 =< t =<1
    topedge_x = np.hstack((X[:,-1][:,None], T[:,0][:,None])) #L3
    topedge_u = usol[0,:][:,None]

    # xx1 forms initial condition, xx2 & xx3 form boundary conditions
    # vstack for vertically stacking (_,2) ndarrays
    # N_u is training data and N_f is collocation points
    # lhs(n, [samples, criterion, iterations])
    # n - number of factors, samples - number of samples to generate for each factor
    # Gives an samples*n output
    # X_f_train contains all collocation as well as ALL boundary and initial points (without sampling)
    
    XT_u_basecases = np.vstack([leftedge_x, bottomedge_x, topedge_x])
    u_basecases = np.vstack([leftedge_u, bottomedge_u, topedge_u])

    # idx tells which indices to pick finally by randomly sampling without replacement

    idx = np.random.choice(XT_u_basecases.shape[0], N_u, replace=False)
    XT_u_train = XT_u_basecases[idx, :]
    u_train = u_basecases[idx, :]

    # TODO correct this in initial paper, X_f_train should not contain X_u_train, or it should contain them properly sampled

    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, XT_u_train))                  # TODO Don't know why this should be here

    # Convert all to tensors

    XT_u_train = torch.from_numpy(XT_u_train).float().to(device)
    u_train = torch.from_numpy(u_train).float().to(device)
    X_f_train = torch.from_numpy(X_f_train).float().to(device)

    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    u = torch.from_numpy(u_true).float().to(device)
    f_hat = torch.zeros(X_f_train.shape[0],1).to(device)
        
    model = PINN(XT_u_train, u_train, X_f_train, layers, lb, ub, nu)
    model.fill_meta(X_test_tensor, u, f_hat)
    model.to(device)

    print(model)

    # L-BFGS Optimizer

    global optimizer
    # optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, 
    #                             max_iter = 250, 
    #                             max_eval = None, 
    #                             tolerance_grad = 1e-05, 
    #                             tolerance_change = 1e-09, 
    #                             history_size = 100, 
    #                             line_search_fn = 'strong_wolfe')

    # self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
    #                                                             method = 'L-BFGS-B', 
    #                                                             options = {'maxiter': 50000,
    #                                                                        'maxfun': 50000,
    #                                                                        'maxcor': 50,
    #                                                                        'maxls': 50,
    #                                                                        'ftol' : 1.0 * np.finfo(float).eps})

    # Gets from 37% error to 8% error
    # For 0.02 error
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, 
                                max_iter = 50000,
                                tolerance_grad = 1.0 * np.finfo(float).eps, 
                                tolerance_change = 1.0 * np.finfo(float).eps, 
                                history_size = 100, 
                                line_search_fn = 'strong_wolfe')
    
    start_time = time.time()
    optimizer.step(model.closure)                   # Does not need any loop like Adam
    elapsed = time.time() - start_time                
    print('Training time: %.2f' % (elapsed))


    ''' Model Accuracy ''' 
    error_vec, u_pred = model.test()

    print('Test Error: %.5f'  % (error_vec))


    ''' Solution Plot '''
    # x_plot = data['x']
    # t_plot = data['t']
    # usol_plot = data['usol']
    # X_plot, T_plot = np.meshgrid(x_plot, t_plot)
    
    solutionplot(u_pred, XT_u_train.cpu().detach().numpy(), u_train.cpu().detach().numpy(), X, T, x, t, usol)


if __name__ == "__main__": 
    # main_loop(20, 2000, 8, 40)
    main_loop(100, 10000, 8, 20)
