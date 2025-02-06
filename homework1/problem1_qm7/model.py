import numpy as np

class Module:
	def update(self,lr): pass
	def average(self,nn,a): pass
	def backward(self,DY): pass
	def forward(self,X): pass

class Sequential(Module):

	def __init__(self,modules):
		self.modules = modules
	
	def forward(self,X):
		'''Given input X, perform a full forward pass through the MLP'''
		#TODO: fill in this function
		for m in self.modules:
			X = m.forward(X)
		return X

	def backward(self,DY):
		'''Perform a full backward pass through the MLP. 
			DY is gradient of the loss w.r.t the final output'''
		#TODO: fill in this function
		for m in reversed(self.modules):
			DY = m.backward(DY)

		return DY

	def update(self,lr):
		for m in self.modules: 
			X = m.update(lr)
		
	def average(self,nn,a):
		for m,n in zip(self.modules,nn.modules): 
			m.average(n,a)

class Input(Module):
	def __init__(self, inp):
		R, Z = inp
		sample_in = np.concatenate([R, np.expand_dims(Z, -1)], axis = -1)
		self.nbout = sample_in.shape[-2] * sample_in.shape[-1]

	def forward(self,inp): 
		R, Z = inp
		rz = np.concatenate([R, np.expand_dims(Z, -1)], axis = -1)
		return rz.reshape(rz.shape[0], -1)

class Output(Module):

	def __init__(self,T):
		self.tmean = T.mean()
		self.tstd  = T.std()
		self.nbinp = 1

	def forward(self,X):
		# un-normalize the final prediction 
		self.X = X.flatten()
		return self.X*self.tstd+self.tmean

	def backward(self,DY):
		#TODO: fill in this function
		return DY * self.tstd

class Linear(Module):

	def __init__(self,m,n):
		self.lr = 1 / np.sqrt(m)
		self.W = np.random.normal(0,1 / m**.5,[m,n]).astype('float32')
		self.B = np.zeros([n]).astype('float32')

	def forward(self,X):
		#TODO: fill in this function
		self.input = X
		out = np.dot(X, self.W) + self.B
		if np.isnan(out).any():
			print("NaN detected in Linear.forward")
		return np.dot(X, self.W) + self.B

	def backward(self,DY):
		#TODO: fill in this function
		DY = DY.reshape(-1, self.W.shape[1])
		self.DW = np.dot(self.input.T, DY)
		self.DB = np.sum(DY, axis=0)
		return np.dot(DY, self.W.T)

	def update(self,lr):
		self.W -= lr*self.lr*self.DW
		self.B -= lr*self.lr*self.DB

	def average(self,nn,a):
		self.W = a*nn.W + (1-a)*self.W
		self.B = a*nn.B + (1-a)*self.B

class Tanh(Module):

	def forward(self,X):
		#TODO: fill in this function
		self.input = X
		return np.tanh(X)
	def backward(self,DY):
		#TODO: fill in this function
		return DY * (1-np.tanh(self.input)**2)

