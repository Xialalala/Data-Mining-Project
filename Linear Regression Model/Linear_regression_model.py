import numpy as np

train=np.loadtxt('train.csv', delimiter=',')
Y_train=train[:, -1]
D_train=train[:, :-1]
A0=np.ones(len(train))
Aug_D_train=np.c_[A0, D_train]
dim=Aug_D_train.shape[1]

def gram_schmidt(A):
    Q=np.zeros_like(A)
    cnt = 0
    for a in A.T:
        u = np.copy(a)
        for i in range(0, cnt):
            u -= np.dot((np.dot(Q[:, i].T, a))/np.dot(Q[:, i], Q[:, i]), Q[:, i])
        Q[:, cnt] =u
        cnt += 1
    return (Q)
Q=gram_schmidt(Aug_D_train)

def compute_R(A, some_Q):
    R=np.zeros(shape=(dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            R[i][j]=(np.dot(some_Q[:, i].T, A[:, j]))/np.dot(some_Q[:, i], some_Q[:, i])
    return(R)
R=compute_R(Aug_D_train, Q)

def diag(some_Q):
    diag_matrix=np.zeros(shape=(dim,dim))
    for i in range(dim):
        diag_matrix[i, i]=1/np.dot(some_Q[:,i].T,some_Q[:, i])
    return diag_matrix
diag_matrix=diag(Q)

def scalar(diag_matrix, Q, Y_train):
    Scalar_proj=np.dot(np.dot(diag_matrix, Q.T), Y_train)
    return(Scalar_proj)
Scalar_proj=scalar(diag_matrix, Q, Y_train)
Scalar_proj

def back_substitution(A, b):
    n = b.size
    w = np.zeros_like(b)
    w[n-1] = b[n-1]/A[n-1, n-1]
    C = np.zeros((n,n))
    for i in range(n-2, -1, -1):
        bb = 0
        for j in range (i+1, n):
            bb += A[i, j]*w[j]
        C[i, i] = b[i] - bb
        w[i] = C[i, i]/A[i, i]
    return w
W=back_substitution(R, Scalar_proj)
print('The weight vector w is:\n', W)

l2_norm=np.sqrt(np.dot(W,W))
print('the l2_norm of w is:', l2_norm)

y_train_mean=Y_train.mean()
x_mean=np.mean(Aug_D_train,axis = 0)

SSE=0
for i in range(len(train)):
    SSE+=np.square(Y_train[i]-np.dot(W.T, Aug_D_train[i]))
print('SSE for train dataset', SSE)

TSS=np.dot((Y_train.T-y_train_mean), (Y_train.T-y_train_mean))

R_squre=(TSS-SSE)/TSS
print('The R square for training data:',  R_squre)

'''test data'''
test=np.loadtxt('test.csv', delimiter=',')

Y_test=test[:, -1]
Y_test_mean=np.mean(Y_test)
Aug_D_test=np.c_[A0, test[:, :-1]]

test_SSE=0
for i in range(len(test)):
    test_SSE+=np.square(Y_test[i]-np.dot(W.T, Aug_D_test[i]))
print('SSE for testing dataset', test_SSE)

test_TSS=np.dot((Y_test.T-Y_test_mean), (Y_test.T-Y_test_mean))

test_R_squre=(test_TSS-test_SSE)/test_TSS
print('The R square for testing data:',  test_R_squre)

'''ridge regression begins'''

sh=(dim, dim)
allone=np.ones(sh)
I=np.diag(np.diag(allone))

new_Y=np.concatenate((Y_train, np.zeros(dim)),axis=0)

alpha=[1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
new_D=[]
for i in range(len(alpha)):
    D_i=np.concatenate((Aug_D_train,np.sqrt(alpha[i]*I)),axis=0)
    new_D.append(D_i)

new_Q=[]
for k in range(len(alpha)):
    Q_k=gram_schmidt(new_D[k])
    new_Q.append(Q_k)

new_R=[]
for k in range(len(alpha)):
    R_k=compute_R(new_D[k], new_Q[k])
    new_R.append(R_k)

new_diag_matrix=[]
for k in range(len(alpha)):
    matrix=diag(new_Q[k])
    new_diag_matrix.append(matrix)


new_Scalar_proj=[]
for l in range(len(alpha)):
    proj_l=scalar(new_diag_matrix[l], new_Q[l], new_Y)
    new_Scalar_proj.append(proj_l)

new_W=[]
for l in range(len(alpha)):
    W_l=back_substitution(new_R[l], new_Scalar_proj[l])
    new_W.append(W_l)

import matplotlib.pyplot as plt
ax = plt.gca()

ax.plot(alpha, new_W)
ax.set_xscale('log')
plt.grid()
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

max_weight_alpha1000= max(np.absolute(new_W[3]))
print("Ridge regression prediction: the most important factor are 'TEMP','PM10','DEWP'etc, when I set ridge value equals to 1000.")
