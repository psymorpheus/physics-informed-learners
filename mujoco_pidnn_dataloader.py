import numpy as np
import mujoco_collection_constants as mcc
import torch
from pyDOE import lhs
from torch.utils.data import Dataset, DataLoader

# VT_test = None
# X_test = None
# VT_validation = None
# X_validation = None

test_dataset = None
validation_dataset = None
test_dataloader = None
validation_dataloader = None

class VTDataset(Dataset):
	def __init__(self, VT, X):
		self.VT = VT
		self.X = X
	def __len__(self):
		return self.X.shape[0]
	def __getitem__(self, idx):
		VT = self.VT[idx]
		X = self.X[idx]
		sample = {"VT": VT, "X": X}
		return sample

def testset_loss(model, device, validation=True):
	total_error = 0
	total = 0
	if validation:
		dataloader = validation_dataloader
	else:
		dataloader = test_dataloader
	with torch.no_grad():
		for (idx,batch) in enumerate(dataloader):
			VT = batch['VT'].float().to(device)
			X = batch['X'].float().to(device)

			X_pred = model.forward(VT)
			error_vec = torch.linalg.norm((X-X_pred),2)
			total_error += error_vec
			total += X.size(0)
		total_error = total_error/total
	return total_error

# def testset_loss(model, validation=True):
# 	if validation:
# 		with torch.no_grad():
# 			X_pred = model.forward(VT_validation)
# 		error_vec = torch.linalg.norm((X_validation-X_pred),2)/torch.linalg.norm(X_validation,2)		# Relative L2 Norm of the error (Vector)
# 	else:
# 		with torch.no_grad():
# 			X_pred = model.forward(VT_test)
# 		error_vec = torch.linalg.norm((X_test-X_pred),2)/torch.linalg.norm(X_test,2)		# Relative L2 Norm of the error (Vector)

# 	X_pred = X_pred.cpu().detach().numpy()
# 	# X_pred = np.reshape(X_pred,(mcc.vx_range.shape[0],mcc.t_range.shape[0]),order='F')
# 	# return error_vec, X_pred
# 	return error_vec

def dataloader(N_u, N_f, device, N_validation=0):
	""" N_u = training data / boundary points for data driven training
	N_f = collocation points for differential for differential driven training
	"""

	data = np.genfromtxt(mcc.filename, delimiter=',')
	data = np.array(data, dtype=np.float32)

	assert data.shape[0] == mcc.TOTAL_ITERATIONS
	assert data.shape[1] == mcc.vx_range.shape[0]

	vrange = mcc.vx_range
	trange = mcc.t_range
	xrange = data

	V, T = np.meshgrid(vrange,trange)
	VT_true = np.hstack((V.flatten()[:,None], T.flatten()[:,None]))
	X_true = np.array(xrange.flatten('C')[:,None])
	# Now ith column of VT corresponds to ith column of X

	lb = np.min(VT_true, axis=0)
	ub = np.max(VT_true, axis=0)

	if mcc.training_is_border:
		VT_indices = np.arange(VT_true.shape[0])
		VT_basecase_indices = VT_indices[np.logical_or(VT_true[:,0] == 0, VT_true[:,1] == 0)]

		idx_train = VT_basecase_indices[np.random.choice(VT_basecase_indices.shape[0], N_u, replace=False)]
		VT_u_train = VT_true[idx_train, :]
		X_u_train = X_true[idx_train, :]
	else:
		idx_train = np.random.choice(VT_true.shape[0], N_u, replace=False)
		VT_u_train = VT_true[idx_train, :]
		X_u_train = X_true[idx_train, :]

	
	VT_f_train = lb + (ub-lb)*lhs(2, N_f)
	VT_f_train = np.vstack((VT_f_train, VT_u_train))

	VT_u_train = torch.from_numpy(VT_u_train).float().to(device)
	X_u_train = torch.from_numpy(X_u_train).float().to(device)
	VT_f_train = torch.from_numpy(VT_f_train).float().to(device)

	# global VT_test, X_test, VT_validation, X_validation
	VT_test = np.delete(VT_true, idx_train, axis=0)
	X_test = np.delete(X_true, idx_train, axis=0)

	idx_validation = np.random.choice(VT_test.shape[0], N_validation, replace=False)
	VT_validation = VT_test[idx_validation, :]
	X_validation = X_test[idx_validation, :]
	VT_test = np.delete(VT_test, idx_validation, axis=0)
	X_test = np.delete(X_test, idx_validation, axis=0)

	# Convert all to tensors
	# VT_validation = torch.from_numpy(VT_validation).float().to(device)
	# X_validation = torch.from_numpy(X_validation).float().to(device)
	# VT_test = torch.from_numpy(VT_test).float().to(device)
	# X_test = torch.from_numpy(X_test).float().to(device)

	global test_dataset, validation_dataset, test_dataloader, validation_dataloader
	test_dataset = VTDataset(VT_test, X_test)
	validation_dataset = VTDataset(VT_validation, X_validation)
	test_dataloader = DataLoader(test_dataset, batch_size=mcc.batch_size, shuffle=True, drop_last=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=mcc.batch_size, shuffle=True, drop_last=True)

	return VT_u_train, X_u_train, VT_f_train, lb, ub

if __name__=="__main__":
  print("Please call this from pidnn file.")