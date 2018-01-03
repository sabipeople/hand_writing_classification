import numpy as np

class batch_norm(object):
	def __init__(self, gamma, beta, momentum, learning_rate, is_training=True):
		self.gamma=gamma
		self.beta=beta
		self.momentum=momentum
		self.learning_rate=learning_rate
		#to store mini batch mean var std
		self.mean=None
		self.var=None
		self.std=None
		#to store mean shited x
		self.mean_shift_x=None
		self.y=None
		#to store mean, var,std to be used inference procedure
		self.inference_mean=None
		self.inference_var=None
		self.inference_std=None
		#backward propagation
		self.dgamma=None
		self.dbeta=None
		self.dx=None

	def forward(self, x, is_training):
		if self.inference_mean is None:
			N,D=x.shape
			self.inference_mean = np.zeros(D)
			self.inference_var =np.zeros(D)
			self.inference_std=np.zeros(D)
		
		if is_training:	
			self.mean = x.mean(axis=0)
			self.mean_shift_x = x-self.mean
			self.var = np.mean(self.mean_shift_x**2,axis=0)
			self.std = np.sqrt(self.var+10e-7)
			
			norm_x=self.mean_shift_x/self.std

			self.inference_mean=self.momentum*self.inference_mean +(1-self.momentum)*self.mean
			self.inference_var=self.momentum*self.inference_var + (1-self.momentum)*self.var
		else:
			norm_x=x-self.inference_mean
			y=norm_x/np.sqrt(self.inference_var+10e-7)
		
		self.y=np.multiply(norm_x,self.gamma)+self.beta

		return self.y

	def backward(self,dout):
		self.dbeta=np.sum(dout,axis=0)
		self.dgamma =np.sum(np.multiply(dout,self.y),axis=0)
		m=dout.shape[0]
		#dL/dy*dy/dxhat
		dL_dxhat=np.multiply(dout,self.gamma)
		
		#dL/dvar
		dL_dvar=-1/(2*self.var*self.std)
		dL_dvar=np.sum(np.multiply(np.multiply(dL_dxhat,self.mean_shift_x),dL_dvar),axis=0)

		#dl/dmean
		dl_dmean=np.sum(np.multiply(dL_dxhat,-1/self.std),axis=0)

		#dl/dx=dl/dxhat*1/std + dl/dvar*dvar/dx+dl/dmean*dmean/dx
		dmean_dx=(2*self.mean_shift_x/m)
		self.dx=np.multiply(dL_dxhat,1/self.std)+np.multiply(dmean_dx,dL_dvar)+dl_dmean/m
		return self.dx

	def update_param(self):
		self.gamma -=self.learning_rate*self.dgamma
		self.beta -=self.learning_rate*self.dbeta
