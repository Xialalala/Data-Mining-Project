import numpy as np
#traindata=np.loadtxt('train.txt', delimiter=',')
#`testdata=np.loadtxt('test.txt', delimiter=',')

path = r'/Users/fuxia/Desktop/train.txt'
path2 = r'/Users/fuxia/Desktop/test.txt'

traindata=np.loadtxt(path, delimiter=',')
testdata=np.loadtxt(path2, delimiter=',')

train=traindata[:, :-1]
test=testdata[:, :-1]

row=traindata.shape[0]

column=traindata.shape[1]

y_train=traindata[:, column-1]
y_test=testdata[:, column-1]

def linear_kernel(train):
    K=np.zeros((row, row))
    for i in range(len(train)):
        for j in range(len(train)):
            K[i][j]=np.dot(train[i], train[j])
    return K

def quadratic_kernal(train):
    K=np.zeros((row, row))
    for i in range(len(train)):
        for j in range(len(train)):
            K[i][j]=(np.dot(train[i], train[j])+1)**2
    return K

def norm(vector):
    sum_sq=0
    for i in vector:
        sum_sq+=i**2
    return(np.sqrt(sum_sq))
    
def gaussian_kernel(train, s):
    K=np.zeros((row, row))
    for i in range(len(train)):
        for j in range(len(train)):
            K[i][j]=np.exp(-(norm(train[i]-train[j])**2)/(2*s))
    return K

sh=(row, row)
allone=np.ones(sh)
I=np.diag(np.diag(allone))

def aug_kernel(K):
    aug_K=K+1
    return aug_K

def c_compute(aug_K, alpha, Y):
    c=np.dot((np.linalg.inv(aug_K+alpha*I)), Y)
    return c

def linear_testing_kernel(Z, X):
    K=np.zeros((len(Z), len(X)))
    for i in range(len(Z)):
        for j in range(len(X)):
            K[i][j]=np.dot(Z[i], train[j])
    K=K+1
    return K

def quadratic_test_kernal(Z, X):
    K=np.zeros((len(Z), len(X)))
    for i in range(len(Z)):
        for j in range(len(X)):
            K[i][j]=(np.dot(Z[i], train[j])+1)**2
    K=K+1
    return K

def gaussian_test_kernel(Z, X, s):
    K=np.zeros((len(Z), len(X)))
    for i in range(len(Z)):
        for j in range(len(X)):
            K[i][j]=np.exp(-(norm(Z[i]-train[j])**2)/(2*s))
    K=K+1    
    return K

'''predict y'''

def pred_test_y(C, K_Z):
    test_y=np.dot(C.T, K_Z)
    return test_y

def binary(pred_y):
    for i in range(len(pred_y)):
        if pred_y[i]>=0.5:
            pred_y[i]=1
        else:
            pred_y[i]=0
    return pred_y

def accuracy(pred_y, test_y):
    count=0
    for i in range(len(test_y)):
        if pred_y[i]==test_y[i]:
            count+=1
    accuracy=count/len(test_y)
    return accuracy

alpha=0.01

'''linear regression'''
Linear_kernel=linear_kernel(train)
linear_c=c_compute(aug_kernel(Linear_kernel), alpha, y_train)
L_K_Z=linear_testing_kernel(test, train)
L_pred_Y=pred_test_y(linear_c, L_K_Z.T )

print('The linear_kernel regression prediction accuracy is :', accuracy(binary(L_pred_Y), y_test))

Q_kernel=quadratic_kernal(train)
Q_c=c_compute(aug_kernel(Q_kernel), alpha, y_train)
Q_K_Z=quadratic_test_kernal(test, train)
Q_pred_Y=pred_test_y(Q_c, Q_K_Z.T )

print('The quadratic_kernel regression prediction accuracy is :', accuracy(binary(Q_pred_Y), y_test))

def gaussian_prediction(train, alpha, y_train, Z, s):
    augK=gaussian_kernel(train, s)
    c=c_compute(aug_kernel(augK), alpha, y_train)
    pred_y=binary(pred_test_y(c, gaussian_test_kernel(Z, train, s).T))
    accu=accuracy(pred_y, y_test)
    return accu

print('when spread=0.1, The gaussian_kernel regression prediction accuracy is :', gaussian_prediction(train, alpha, y_train, test, 0.1))

print('when spread=0.5, The gaussian_kernel regression prediction accuracy is :', gaussian_prediction(train, alpha, y_train, test, 0.5))

print('when spread=1, The gaussian_kernel regression prediction accuracy is :', gaussian_prediction(train, alpha, y_train, test, 1))

print('when spread=10, The gaussian_kernel regression prediction accuracy is :', gaussian_prediction(train, alpha, y_train, test, 10))

print('when spread=100, The gaussian_kernel regression prediction accuracy is :', gaussian_prediction(train, alpha, y_train, test, 100))

print('when spread=0.5, the accuracy is the best')
