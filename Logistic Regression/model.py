import numpy as np
import random
path = r'/Users/fuxia/Desktop/train.txt'
path2 = r'/Users/fuxia/Desktop/test.txt'

traindata=np.loadtxt(path, delimiter=',')
testdata=np.loadtxt(path2, delimiter=',')

row=traindata.shape[0]

column=traindata.shape[1]

train=traindata[:, :-1]
test=testdata[:, :-1]


aug_train=np.c_[(np.ones(len(train))), train]
aug_test=np.c_[(np.ones(len(test))), test]


y_train=traindata[:, column-1]
y_test=testdata[:, column-1]

def sigmoid(z):
    result=np.exp(z)/(1+np.exp(z))
    return result

def norm(vector):
    sum_sq=0
    for i in vector:
        sum_sq+=i**2
    return(np.sqrt(sum_sq))

def SGA_function(x, y, learning_rate, err):
    order_x=list(range(x.shape[0]))
    w_c=np.zeros(column)
    t=0
    while True:
        random.shuffle(order_x)
        w_l=w_c
        w_t=w_c
        t+=1
        for i in order_x:
            gradient=(y[i]-sigmoid(np.dot(w_t, x[i])))* x[i]
            w_t=w_t+learning_rate*gradient
        w_c=w_t
        if (norm(w_c-w_l)<=err):
            break                              
    return (w_t)

w1=SGA_function(aug_train, y_train, 0.0001, 0.0001)
print('when eps=0.0001, eta=0.0001, the w1 is:', w1)
w2=SGA_function(aug_train, y_train, 0.01, 0.01)
print('when eps=0.01, eta=0.01, the w2 is:', w2)

def prediction(w, z):
    y=np.zeros(len(z))
    for i in range(len(z)):
        if sigmoid(np.dot(w, z[i]))>=0.5:
            y[i]=1                             
    return(y)

pred_y_test=prediction(w1, aug_test)
pred_y_test
pred_y_train=prediction(w1, aug_train)
pred_y_test2=prediction(w2, aug_test)
pred_y_train2=prediction(w2, aug_train)

def accuracy(pred_y, realdata):
    count=0
    for i in range(len(realdata)):
        if pred_y[i]==realdata[i]:
            count+=1
    accuracy=count/len(realdata)
    return accuracy

accuracy_w1_test=accuracy(pred_y_test, y_test)
print('accuracy_w1_test', accuracy_w1_test)

accuracy_w1_train=accuracy(pred_y_train, y_train)
print('accuracy_w1_train', accuracy_w1_train)

accuracy_w2_test=accuracy(pred_y_test2, y_test)
print('accuracy_w2_test', accuracy_w2_test)

accuracy_w2_train=accuracy(pred_y_train2, y_train)
print('accuracy_w2_train', accuracy_w2_train)

print('when eps=0.0001, eta=0.0001, the model has the best accuracy')
