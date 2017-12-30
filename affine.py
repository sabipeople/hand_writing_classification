import numpy as np
import pdb
class affine:
	def __init__(self, rows,cols,batch_size):
		self.w=np.random.rand(rows, cols)
		self.B=np.random.rand(batch_size, cols)
		self.x=None
		self.y=None
		self.dw=None
		self.dB=None
		self.dy=None
		self.learning_rate=0.01
	def forward(self,din):
		self.x=din
		self.y=np.dot(din,self.w)+self.B
		return self.y

	def backward(self, dout):
		self.dw=np.dot(self.x.getT(),dout)
		self.dB=dout
		self.dy=np.dot(dout,self.w.getT())

	def update(self):
		self.w-=self.learning_rate*self.dw
		self.B-=self.learning_rate*self.dB
	
	def update_learningrate(self, ratio):
		self.learning_rate *=ratio
