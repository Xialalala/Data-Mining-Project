
import numpy as np

train=np.loadtxt('train.txt', delimiter=',' )
test=np.loadtxt('test.txt', delimiter=',' )


train.shape

X_train=train[:, 0:(train.shape[1]-1)]
X_test=train[:, 0:(test.shape[1]-1)]


Aug_X_train=np.c_[X_train, (np.ones(len(train)))]
Aug_X_test=np.c_[X_test, (np.ones(len(train)))]


Y=np.mat(train[:, -1]).T
Y_test=np.mat(test[:, -1]).T


def distance(vector):
    sum_sq=0
    for i in vector:
        sum_sq+=i**2
    return(np.sqrt(sum_sq))


def learning_rate(K):
    eta=np.zeros(shape=(K.shape[0],1))
    for k in range(len(eta)):
        eta[k]=1/(K[k][k])
    return eta


def gradient_ascent(K,Y,eta,C,eps):
    t=0
    a=np.zeros(shape=(K.shape[0],1))
    while True:
        a_t = np.copy(a)
        for k in range(K.shape[0]):
            y_k = Y[k, 0]
            s_k = eta[k, 0]
            summa = 0
            for m in range(K.shape[0]):
                a_m = a[m, 0]
                y_m = Y[m, 0]
                k_m = K[m, k]
                summa = summa + a_m * y_m * k_m
            a[k, 0] = a[k, 0] + s_k * (1 - y_k*summa)
            if a[k][0] <0:
                a[k][0] = 0
            if a[k][0] >C:
                a[k][0] =C
        t = t+1   
        if distance(a-a_t)<=eps:
            break
    return a, t


def Accu(y_pred,y_true):
    total_num = y_true.shape[0]
    count = 0
    for i in range(len(y_true)):
        if y_pred[i]== y_true[i]:
            count +=1
    return count/total_num


# Linear


def linear_Kernel_for_Gradient(X):
    K=np.dot(X, X.T)
    return K


def L_SVM(X, Y, C, eps):
    K=linear_Kernel_for_Gradient(X)
    eta=learning_rate(K)
    alpha, t=gradient_ascent(K, Y, eta, C, eps)
    return alpha, t


def Linear_Kernel_test(X, test_X):
    K = np.dot(X, test_X.T)
    return K


def linear_classify(alpha,X,Y,test):
    pred = []
    
    for j in range(test.shape[0]):
        s_t = 0
        for i in range(X.shape[0]):
            if alpha[i,0] >0 :
                s_t += alpha[i,0]*Y[i,0]*Linear_Kernel_test(X,test[j])[i]
        if s_t > 0:
            s_t = 1
        else:
            s_t = -1
        pred.append(s_t)
    return pred



L_a, t=L_SVM(Aug_X_train, Y, C=50, eps=0.1)
print(t)

print ('the support vectors for linear kernel are:')
for i in range(L_a.shape[0]):
    if L_a[i]>0:
        print (i, L_a[i]) 


L_Y=linear_classify(L_a, Aug_X_train, Y, Aug_X_test)

print('The accuracy for linear kernel SVM is:', Accu(L_Y,Y_test))

def w_linear(alpha,Y,X):
    w = np.zeros(shape = (X.shape[1],1))
    for i in range(X.shape[0]):
        if alpha[i,0]>0:
            w += alpha[i,0]*Y[i,0]*(X[i].reshape(-1, 1))
    return w



print('the weight for linear kernel regression is:', w_linear(L_a,Y,Aug_X_train))


# Quadratic


def quadratic_kernel_for_gradient(X):
    K = np.square(np.dot(X, X.T))
    return K


def Q_SVM(X, Y, C, eps):
    K=quadratic_kernel_for_gradient(X)
    eta=learning_rate(K)
    alpha, t=gradient_ascent(K, Y, eta, C, eps)
    return alpha, t

def Quadratic_Kernel_test(X, test_X):
    K = np.square(np.dot(X, test_X.T))
    return K


def Quadratic_classify(alpha,X,Y,test_X):
    pred = []
    
    for j in range(test_X.shape[0]):
        s_t = 0
        for i in range(X.shape[0]):
            if alpha[i,0] >0 :
                s_t += alpha[i,0]*Y[i,0]*Quadratic_Kernel_test(X,test_X[j])[i]
        if s_t > 0:
            s_t = 1
        else:
            s_t = -1
        pred.append(s_t)
    return pred



Q_a, t= Q_SVM(Aug_X_train, Y, C=50, eps=1)

print ('the support vectors in Quadratic kernel are:')
for i in range(Q_a.shape[0]):
    if Q_a[i]>0:
        print ( i, Q_a[i]) 

Q_Y=Quadratic_classify(Q_a, Aug_X_train, Y, Aug_X_test)


print('Accuracy for quadratic kernel SVM is:', Accu(Q_Y,Y_test))


def phi(X_record):
    
    list_1=[]
    for i in range(X_record.shape[0]):
        list_1.append(X_record[i]**2)
    
    list2 = []
    for i in range(X_record.shape[0]):
        for j in range(X_record.shape[0]):
            if i < j :
                list2.append(X_record[i]*X_record[j]*np.sqrt(2))
    new_list=np.array(list_1+list2+[1]).reshape(-1,1)            
    return new_list

def wcq(alpha,Y,X):
    w = np.zeros(shape = [phi(X_train[0]).shape[0],1])
    for i in range(X.shape[0]):
        if alpha[i,0]>0:
            w += alpha[i,0]*Y[i,0]*phi(X[i])
    return w


print('the weight for quadratic kernel regression is :', wcq(Q_a,Y,X_train))


# Gussian

def Gau_kernel(X, s):
    K=np.zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(len(X)):
        for j in range(len(X)):
            K[i][j]=np.exp(-(distance(X[i]-X[j])**2)/(2*s))
    return K


def G_SVM(X, Y, C, eps, s):
    K=Gau_kernel(X, s)
    eta=learning_rate(K)
    alpha, t=gradient_ascent(K, Y, eta, C, eps)
    return alpha, t


def Gaussian_Kernel_test(X, test_z, s):
    K=np.zeros(shape=(X.shape[0], 1))
    for i in range(X.shape[0]):
        K[i]=np.exp(-(distance(X[i]-test_z)**2)/(2*s))
    return K

def Gaussian_classify(alpha,X,Y,test, s):
    pred = []
    
    for j in range(test.shape[0]):
        s_t = 0
        for i in range(X.shape[0]):
            if alpha[i,0] > 0 :
                s_t += alpha[i,0]*Y[i,0]*Gaussian_Kernel_test(X,test[j], s)[i]
        y=np.sign(s_t)
        pred.append(y)
    return pred

G_a, t=G_SVM(Aug_X_train, Y, C=20, eps=1, s=1)

print ('the support vectors in Gaussian kernel are:')
for i in range(G_a.shape[0]):
    if G_a[i]>0:
        print (i, G_a[i]) 

G_Y=Gaussian_classify(G_a, Aug_X_train, Y, Aug_X_test, s=1)


print('The accuracy for Gaussian kernel SVM is:', Accu(G_Y,Y_test))





