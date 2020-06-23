import numpy as np
from sklearn.preprocessing import OneHotEncoder

traindata=np.loadtxt('train.txt', delimiter=',')
testdata=np.loadtxt('test.txt', delimiter=',')

traindata=np.mat(traindata)
X=traindata[:, :-1]
testdata=np.mat(testdata)
X_test=testdata[:, :-1]

d=X.shape[1]
n=len(X)

O=traindata[:, 9]
onehot_encoder = OneHotEncoder(sparse=False)
O = O.reshape(len(O), 1)
onehot_O = onehot_encoder.fit_transform(O)
onehot_O=np.mat(onehot_O) #onehot coding for y

p=onehot_O.shape[1]
m=5

bo=np.random.uniform(-0.01, 0.01,size=(p,1))
bh=np.random.uniform(-0.01, 0.01,size=(m,1))
wh=np.random.uniform(-0.01, 0.01,size=(d, m))
wo=np.random.uniform(-0.01, 0.01,size=(m, p))

lr=0.00001
maxiter=40

def ReLU_activation(net):
    for i in range(m):
        if net[i]<=0:
            net[i]=0           
    return net

def Softmax(net):
    denomiter=np.sum(list(map(lambda x:np.exp(x), net)))
    for i in range(p):
        net[i]=np.exp(net[i])/denomiter
    return net

def feed_forward_z(X_i, wh, bh):
    z_i=ReLU_activation(bh+np.dot(wh.T, X_i.T))
    return z_i

def feed_forward_o(z_i, wo, bo):
    o_i=Softmax(bo+np.dot(wo.T, z_i))
    return o_i

def back_phase_new(o, z, y_i, wo):
    allones_z=np.ones(shape=(m,1))
    So=o-y_i.T
    for i in range(m): 
        if z[i]<=0:
            allones_z[i]=0
    Sh=np.multiply(allones_z, np.dot(wo, So))
    return So, Sh

def gradient_b(bo, bh, lr, so, sh):
    bo=bo-lr*so
    bh=bh-lr*sh
    return bo, bh

def gradient_w(wo, wh, so, sh, lr, z, o, x):
    wo=wo-lr*np.dot(z, so.T)
    wh=wh-lr*np.dot(x.T, sh.T)
    return wo, wh

import random
def model(wo, wh, bo, bh, lr, X, y, maxiter):
    order_x=list(range(X.shape[0]))
    t=0
    while True:
        random.shuffle(order_x)
        for i in order_x:
            #forward
            z_i=feed_forward_z(X[i], wh, bh)
            o_i=feed_forward_o(z_i, wo, bo)
            #backward
            So_i, Sh_i=back_phase_new(o_i, z_i, y[i], wo)
            #update bo, bh
            bo, bh=gradient_b(bo, bh, lr, So_i, Sh_i)
            #update wo, wh
            wo, wh=gradient_w(wo, wh, So_i, Sh_i, lr, z_i, o_i, X[i])
        t=t+1
        if t >= maxiter:
            break
    return bo, bh, wo, wh

Bo, Bh, Wo, Wh=model(wo, wh, bo, bh, lr, X, onehot_O, maxiter)

print('the weight matrix for hidden layer is:', Wh)
print('the bias vector for hidden layer is:', Bh)
print('the weight matrix for output layer is:', Wo)
print('the bias vector for output layer is:', Bo)
def pred_y(bh, bo, wh, wo, x):
    o=np.zeros(p*len(x)).reshape(len(x),p)
    for i in range(len(x)):
        z_i=feed_forward_z(x[i], wh, bh)
        o[i]=feed_forward_o(z_i, wo, bo).T
    return o

predict_y=pred_y(Bh, Bo, Wh, Wo, X)

def adjust_yi(a):
    y=np.zeros(shape=(len(a), p))
    for k in range(len(a)):
        for i in range(p):
            for j in range(p):
                if a[k][i]==np.max(a[k]):
                    y[k][i]=1
    return y

pred_y_train=adjust_yi(predict_y)

def accuracy(y_pred,y_true):
    total_num = y_true.shape[0]
    count_good = 0
    for i in range(len(y_true)):
        if (y_pred[i] ==y_true[i]).sum()==7:
            count_good +=1
    return count_good/total_num

accuracy_train=accuracy(pred_y_train, onehot_O )
print('accuracy_train:', accuracy_train)

#test dataset
Output=testdata[:, 9]
onehot_encoder = OneHotEncoder(sparse=False)
Output = Output.reshape(len(Output), 1)
Y_test = onehot_encoder.fit_transform(Output)
Y_test=np.mat(Y_test) #onehot coding for y

predict_test=pred_y(Bh, Bo, Wh, Wo, X_test)

adjust_y_test=adjust_yi(predict_test)

accuracy_test=accuracy(adjust_y_test, Y_test)

print('accuracy_test:', accuracy_test)

