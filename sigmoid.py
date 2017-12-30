import numpy as np
class sigmoid(object):
	def __init__(self):
		self.x=None
		self.y=None
		self.dy=None
		self.learning_rate=0.01
	def sigmoid(self,x):
		return 1/(1+np.exp(-x))
	
	def forward(self, din):
		self.x=din
		self.y=self.sigmoid(din)
		return self.y

	def backword(self, dout):
		self.dy=np.multiply(dout,self.np.multiply(y,1-self.y))
		return self.dy
	
