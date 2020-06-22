import numpy as np
import pandas as pd
import sys

data=pd.read_csv(sys.argv[1], delimiter=',', header=None)
K=int(sys.argv[2])
eps=float(sys.argv[3])

data=np.array(data)


X=data[:, :-1]
dim=X.shape[1]
dim

Y=data[:, -1]
Y.shape

def initialization(X, dim, K):
    ini_mean={}
    ini_cov={}
    prior_p={}
    for i in range (K):
        inter_mean=np.zeros(shape=(1, dim))
        for j in range(dim):
            max_mean=np.max(X[:, j])
            min_mean=np.min(X[:, j])
            inter_mean[0,j]=np.random.uniform(min_mean, max_mean)
        ini_mean[i]=inter_mean
        ini_cov[i]=np.identity(dim)
        prior_p[i]=1/K
    return ini_mean, ini_cov, prior_p

def density_function(x, mean, var):
    inv_var=np.linalg.inv(var)
    exp_p=np.exp(-(np.dot(np.dot((x-mean), inv_var), (x-mean).T)*0.5)[0, 0])
    f=(1/(((np.sqrt(np.pi))**dim)*(np.linalg.det(var)**(0.5))))*exp_p
    return f

def expection(X, mean, cov_i, prior_p, K):
    #w=np.zeros(shape=(len(X), K))
    fenmu=np.zeros(shape=(len(X), 1))
    fenzi=np.zeros(shape=(len(X), K))
    small_value=np.identity(dim)*0.0001
    
    for i in range(K):
        for j in range(len(X)):
            fenzi[j, i]= density_function(X[j], mean[i], (cov_i[i]+small_value))*prior_p[i]
            fenmu=np.sum(fenzi, axis=1)

    w=np.zeros(shape=(len(X), K))
    for j in range(len(X)):
            w[j]=fenzi[j]/fenmu[j]
        
    return w


def maximization(X, w, K):
    dim=X.shape[1]
    n=len(X)
    new_mean={}
    new_cov={}
    new_P={}
    #update mean
    for i in range(K):
        inter_mean=np.zeros(shape=(1, dim))
        for j in range(n):
            inter_mean=inter_mean+(w[j, i]*X[j])
        new_mean[i]=np.array(inter_mean/np.sum(w[:, i]),dtype='float')
    
    #update covariance
    for i in range(K):
        inter_cov=np.zeros(shape=(dim, dim))
        for j in range(n):
            inter_cov=inter_cov+w[j, i]*np.outer((X[j]-new_mean[i]), (X[j]-new_mean[i]))
        new_cov[i]=np.array(inter_cov/np.sum(w[:, i]), dtype='float')
            
    # update p
    for i in range(K):
        new_P[i]=np.sum(w[:, i])/n
        
    return new_mean, new_cov, new_P


def norm(vector):
    vector=np.reshape(vector, (dim, 1))
    sum_sq=0
    for i in vector:
        sum_sq=sum_sq+i**2
    #print(sum_sq)
    return(np.sqrt(sum_sq[0]))


def iteration(X, K, eps, dim):
    t=0
    p_mean, cov, p=initialization(X, dim, K)
    while True:
        t=t+1
        t_mean=p_mean.copy()
        
        w=expection(X, p_mean, cov, p, K)

        p_mean, cov, P=maximization(X, w, K)
        #p_mean
        sum_mean=0
        
        for i in p_mean.keys():
            
            sum_mean=sum_mean+norm(p_mean[i]-t_mean[i])**2
            #print(type(norm(mean[1]-mean[2])  ))          
            #print(type(sum_mean))

        if sum_mean<=eps:
            break
    return w


def classify(w):
    new_matrix=np.argmax(w, axis=1)
    return new_matrix


w=iteration(X, K, 0.001, dim)
class_w=classify(w)



def purify_score(class_w, Y, K):
    LIST=[]
    for i in np.unique(class_w):
        max_match=np.zeros(shape=(K, 1))
        
        for j in np.unique(Y):
            match=0
            for k in range(Y.shape[0]):
                
                if (class_w[k]==i) & (Y[k]==j):
                    match=match+1
                    
            if match>max_match[i,0]:
                max_match[i,0]=match
        LIST.append(max_match[i,0])
    
    score=np.sum(LIST)/len(Y)
    return score
    

print('score:', purify_score(class_w, Y, K))





