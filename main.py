#!/home/sabi/anaconda3/envs/DL_py/bin/python
import numpy as np
import sys
import os
import pdb
import affine
import sigmoid
import softmax

sys.path.append("/home/sabi/workspace/reference_code/deep-learning-from-scratch/")
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist

if __name__=="__main__":
    (x_train, t_train),(x_test, t_test)=load_mnist(normalize=True, one_hot_label=True)
    y={}
    dy={}
    bat_size=100
    affine_L1=affine.affine(rows=x_train.shape[1],cols=50,batch_size=bat_size)
    sigmoid_L2=sigmoid.sigmoid()
    affine_L3=affine.affine(50,100,batch_size=bat_size)
    sigmoid_L4=sigmoid.sigmoid()
    affine_L5=affine.affine(100,10,batch_size=bat_size)
    softmax_L6=softmax.softmax()
    loss=[]
    acculacy=[]
    per_epoch=max(x_train.shape[0]/bat_size,1)
    
    for i in range(x_train.shape[0]/bat_size):
        idx_list=np.random.choice(x_train.shape[0],bat_size)
    
    #predict process
        y['y1']=affine_L1.forward(x_train[idx_list,:])
        y['y2']=sigmoid_L2.forward(y['y1'])
        y['y3']=affine_L3.forward(y['y2'])
        y['y4']=sigmoid_L4.forward(y['y3'])
        y['y5']=affine_L5.forward(y['y4'])
        y['y6']=softmax_L6.forward(y['y5'],t_train[idx_list,:])
        


        dy['softmaxWithLoss']=softmax_L6.backward(t_train[idx_list,:])
        dy['affine_L5']=affine_L5.backward(dy['softmaxWithLoss'])
        dy['sigmoid_L4']=sigmoid_L4.backward(dy['affine_L5'])
        dy['affine_L3']=affine_L3.backward(dy['sigmoid_L4'])
        dy['sigmoid_L2']=sigmoid_L2.backward(dy['affine_L3'])
        dy['affine_L1']=affine_L1.backward(dy['sigmoid_L2'])
        
        if i % per_epoch == 0:
            #accumulate loss
            loss.append(np.sum(softmax.error)/bat_size)
            #compute acculacy
            predict_t=y['y6'].argmax()
            if t_train.ndim != 1: t=t_train.argmax()
            else: t=t_train.copy()
            acculacy.append(t[np.where(predict_t==t)].shape[0]/bat_size)
            
        pdb.set_trace()
