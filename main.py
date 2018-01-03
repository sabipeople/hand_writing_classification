#!/home/sabi/anaconda3/envs/DL_py/bin/python
import numpy as np
import sys
import os
import pdb
import affine
import sigmoid_array
import softmax
import batch_norm

sys.path.append("/home/sabi/workspace/reference_code/deep-learning-from-scratch/")
sys.path.append(os.getcwd())
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt

def drawHist(params):
    idx=1
    for i, a in params.items():
        if i=='y2' or i=='y4':
            plt.subplot(1,2,idx)
            plt.title(i+"-layer")
            plt.hist(params[i].flatten(),30)
            idx+=1
    plt.show()



if __name__=="__main__":
    (x_train, t_train),(x_test, t_test)=load_mnist(normalize=True, one_hot_label=True)
    y={}
    dy={}
    bat_size=100
    affine_L1=affine.affine(rows=x_train.shape[1],cols=50,batch_size=bat_size,weight_std=1/np.sqrt(x_train.shape[1]))
    layer_batchnorm_1=batch_norm.batch_norm(1,0,0.9,0.01)
    sigmoid_L2=sigmoid_array.sigmoid()
    affine_L3=affine.affine(50,100,batch_size=bat_size,weight_std=1/np.sqrt(50))
    layer_batchnorm_2=batch_norm.batch_norm(1,0,0.9,0.01)
    sigmoid_L4=sigmoid_array.sigmoid()
    affine_L5=affine.affine(100,10,batch_size=bat_size,weight_std=1/np.sqrt(100))
    softmax_L6=softmax.softmax()
    loss=[]
    acculacy=[]
    per_epoch=int(max(x_train.shape[0]/bat_size,1))
    for i in range(10000):
        idx_list=np.random.choice(x_train.shape[0],bat_size)
    
    #predict process
        y['y1']=affine_L1.forward(x_train[idx_list,:])
        y['bat1']=layer_batchnorm_1.forward(y['y1'],True)
        y['y2']=sigmoid_L2.forward(y['bat1'])
        y['y3']=affine_L3.forward(y['y2'])
        y['bat2']=layer_batchnorm_2.forward(y['y3'],True)
        y['y4']=sigmoid_L4.forward(y['bat2'])
        y['y5']=affine_L5.forward(y['y4'])
        y['y6']=softmax_L6.forward(y['y5'],t_train[idx_list,:])
#        drawHist(y)
    #back propagation
        dy['softmaxWithLoss']=softmax_L6.backward(t_train[idx_list,:])
        dy['affine_L5']=affine_L5.backward(dy['softmaxWithLoss'])
        dy['sigmoid_L4']=sigmoid_L4.backward(dy['affine_L5'])
        dy['bat2']=layer_batchnorm_2.backward(dy['sigmoid_L4'])
        dy['affine_L3']=affine_L3.backward(dy['bat2'])
        dy['sigmoid_L2']=sigmoid_L2.backward(dy['affine_L3'])
        dy['bat1']=layer_batchnorm_1.backward(dy['sigmoid_L2'])
        dy['affine_L1']=affine_L1.backward(dy['bat1'])
    #update parameter
        affine_L5.update()
        layer_batchnorm_2.update_param()
        affine_L3.update()
        layer_batchnorm_1.update_param()
        affine_L1.update()
        
        y['y1']=affine_L1.forward(x_train[idx_list,:])
        y['bat1']=layer_batchnorm_1.forward(y['y1'],True)
        y['y2']=sigmoid_L2.forward(y['bat1'])
        y['y3']=affine_L3.forward(y['y2'])
        y['bat2']=layer_batchnorm_2.forward(y['y3'],True)
        y['y4']=sigmoid_L4.forward(y['bat2'])
        y['y5']=affine_L5.forward(y['y4'])
        y['y6']=softmax_L6.forward(y['y5'],t_train[idx_list,:])

       # y['y1']=affine_L1.forward(x_train[idx_list,:])
       # y['y2']=sigmoid_L2.forward(y['y1'])
       # y['y3']=affine_L3.forward(y['y2'])
       # y['y4']=sigmoid_L4.forward(y['y3'])
       # y['y5']=affine_L5.forward(y['y4'])
       # y['y6']=softmax_L6.forward(y['y5'],t_train[idx_list,:])

        if i % per_epoch == 0:
            #accumulate loss
            loss.append(np.sum(softmax_L6.error))
            #compute acculacy
            predict_t=y['y6'].argmax(axis=1)
            if t_train.ndim != 1: t=t_train[idx_list,:].argmax(axis=1)
            else: t=t_train[idx_list,:].copy()
            acculacy.append(t[np.where(predict_t==t)].shape[0]/bat_size)
            print("loss: %f, accuracy: %f"%(loss[-1],acculacy[-1])) 
    x=np.arange(0,len(acculacy))
    plt.figure(1)
    plt.plot(x,acculacy)
    plt.title('accuracy')
    plt.figure(2)
    plt.plot(x,loss)
    plt.title('loss')
    plt.show()
    pdb.set_trace()
    dout=affine_L1.forward(x_test)
    dout=sigmoid_L2.forward(dout)
    dout=affine_L3.forward(dout)
    dout=sigmoid_L4.forward(dout)
    dout=affine_L5.forward(dout)
    dout=softmax_L6.forward(dout, t_test)

    predict_t=softmax_L6.y.argmax(axis=1)
    if t_test.ndim !=1: t=t_test.argmax(1)
    else: t=t_test.copy()

    print("accuracy: %f" %(t[np.where(predict_t==t)].shape[0]/t.shape[0]))
    

