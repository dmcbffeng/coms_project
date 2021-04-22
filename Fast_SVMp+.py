import time
import numpy as np
import pandas as pd
import sklearn.datasets
import matplotlib.pyplot as plt
from random import randrange
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import metrics
from copy import deepcopy

def l1_normalization(X):
    d = X.shape[1]
    Xnorm = np.sum(abs(X),axis = 1).reshape((-1,1))
    Xnorm[Xnorm==0] = 1
    Xnorm   = 1./Xnorm
    X = np.multiply(X,np.tile(Xnorm,(1,d)))
    return X
    
def linear_kernel(x,y):
    return x.dot(y.T)

def coordinate_descendt(train_X_F,train_X_PF,train_label,gamma,C,pi):
    start = time.time()
    # append bias
    train_X_F = np.hstack([train_X_F, np.ones(train_X_F.shape[0]).reshape(-1,1)])
    train_X_PF = np.hstack([train_X_PF, np.ones(train_X_PF.shape[0]).reshape(-1,1)])
    
    n_sample = train_X_F.shape[0]
    n_F_samples = train_X_PF.shape[0]
    # Calculate Kernel
    K = linear_kernel(train_X_F,train_X_F) # for trained features
    K_tilda = linear_kernel(train_X_PF,train_X_PF) # for privilegded features
    K_tilda = np.pad(K_tilda, ((0,n_sample - n_F_samples),(0,n_sample - n_F_samples)), 'constant')
    # Calculate Q, E and beta
    Q_12 = 1/gamma * K_tilda
    H = np.multiply(K, train_label.dot(train_label.T))
    Q_11 = H + Q_12.copy()
    Q_21 = Q_12.copy()
    Q_22 = Q_12.copy()
    Q = np.r_[np.c_[Q_11,Q_12],np.c_[Q_21,Q_22]]
    
    one_vec = np.repeat(1,n_sample).reshape(n_sample,1)
    E_part = 1 / gamma * np.dot(K_tilda,C * one_vec)
    E = np.c_[(one_vec + E_part).T,E_part.T].reshape((-1,1))
    
    # coordinate descent
    # w
    w = np.zeros(train_X_F.shape[1])
    w_tilde = -C/gamma * train_X_PF.sum(0)
    beta = np.zeros(n_sample+n_F_samples)
    
    max_iter = 10000
    tol = 1e-5
    order = np.arange(n_sample+n_F_samples)
    for it in range(max_iter):
        last_beta = deepcopy(beta)
        last_w = deepcopy(w)
        last_w_tilde = deepcopy(w_tilde)
        np.random.shuffle(order)
        for i in order:
            if i<n_sample:
                if i<n_F_samples:
                    df = train_label[i]*w.dot(train_X_F[i]) - 1 + w_tilde.dot(train_X_PF[i])
                else:
                    df = train_label[i]*w.dot(train_X_F[i]) - 1
                d = min(max(-beta[i], -df/K[i,i]), pi[i]*C-beta[i])
                w = w + d*train_label[i]*train_X_F[i]
                if i<n_F_samples:
                    w_tilde = + 1/gamma * d*train_X_PF[i]
            else:
                df = w_tilde.dot(train_X_PF[i-n_sample])
                d = max(-beta[i], -df/K_tilda[i-n_sample,i-n_sample])
                w_tilde = + 1/gamma * d*train_X_PF[i-n_sample]
            beta[i] += d
        if np.linalg.norm(beta-last_beta) <= tol:
            break
    end = time.time()
    print('time:', end-start, 's')
    print(it, 'iterations')
    
    return w, w_tilde, beta, end-start
    
def load_mnistp(path):
    dataset = {}
    dataset['test_features'] = pd.read_csv(os.path.join(path, 'test_features.txt')).to_numpy()
    dataset['test_labels'] = pd.read_csv(os.path.join(path, 'test_labels.txt')).to_numpy()
    dataset['train_features'] = pd.read_csv(os.path.join(path, 'train_features.txt')).to_numpy()
    dataset['train_labels'] = pd.read_csv(os.path.join(path, 'train_labels.txt')).to_numpy()
    dataset['train_prividege'] = np.hstack([pd.read_csv(os.path.join(path, 'train_PFfeatures.txt')).to_numpy(), pd.read_csv(os.path.join(path, 'train_YYfeatures.txt')).to_numpy()])
    dataset['validation_features'] = pd.read_csv(os.path.join(path, 'val_features.txt')).to_numpy()
    dataset['validation_labels'] = pd.read_csv(os.path.join(path, 'val_labels.txt')).to_numpy()
    return dataset

if __name__ == '__main__':
    # load data and preprocessing
    data = load_mnistp('data/MNIST+')
    elements = np.unique(data['train_labels'])
    data['train_labels'] = data['train_labels'].flatten()
    data['train_labels'][data['train_labels']==elements[0]] = -1
    data['train_labels'][data['train_labels']==elements[1]] = 1
    data['test_labels'] = data['test_labels'].flatten()
    data['test_labels'][data['test_labels']==elements[0]] = -1
    data['test_labels'][data['test_labels']==elements[1]] = 1
    data['train_features'] = data['train_features']/data['train_features'].max(1).reshape(-1,1)
    data['test_features'] = data['test_features']/data['test_features'].max(1).reshape(-1,1)
    data['train_prividege'] = data['train_prividege']/data['train_prividege'].max(1).reshape(-1,1)
    pi = np.ones(data['train_features'].shape[0])
    print("shape of training set:")
    print(data['train_features'].shape)
    print("shape of test set:")
    print(data['test_features'].shape)
    print("shape of training set with privileged information:")
    print(data['train_prividege'].shape)
    print(len(data['train_labels']))
    print(len(data['test_labels']))
    
    # train data
    w, w_tilde, beta = coordinate_descendt(data['train_features'],data['train_prividege'],data['train_labels'],1,1,pi)
    # train set
    train_features2 = np.hstack([data['train_features'], np.ones(data['train_features'].shape[0]).reshape(-1,1)])
    pred = np.sign(train_features2.dot(w))
    print('train accuracy:', len(np.where(pred==data['train_labels'])[0])/len(data['train_labels']))
    test_emb = PCA().fit_transform(data['train_features'])
    plt.scatter(test_emb[data['train_labels']==-1,0], test_emb[data['train_labels']==-1,1], color='red', s=3, label='label=-1')
    plt.scatter(test_emb[data['train_labels']==1,0], test_emb[data['train_labels']==1,1], color='blue', s=3, label='label=1')
    plt.legend()
    plt.show()
    plt.scatter(test_emb[pred==-1,0], test_emb[pred==-1,1], color='red', s=3, label='prediction=-1')
    plt.scatter(test_emb[pred==1,0], test_emb[pred==1,1], color='blue', s=3, label='prediction=1')
    plt.legend()
    plt.show()
    fpr, tpr, thresholds = metrics.roc_curve(data['train_labels'], pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
    # test set
    test_features2 = np.hstack([data['test_features'], np.ones(data['test_features'].shape[0]).reshape(-1,1)])
    pred = np.sign(test_features2.dot(w))
    print('test accuracy:', len(np.where(pred==data['test_labels'])[0])/len(data['test_labels']))
    test_emb = PCA().fit_transform(data['test_features'])
    plt.scatter(test_emb[data['test_labels']==-1,0], test_emb[data['test_labels']==-1,1], color='red', s=3, label='label=-1')
    plt.scatter(test_emb[data['test_labels']==1,0], test_emb[data['test_labels']==1,1], color='blue', s=3, label='label=1')
    plt.legend()
    plt.show()
    plt.scatter(test_emb[pred==-1,0], test_emb[pred==-1,1], color='red', s=3, label='prediction=-1')
    plt.scatter(test_emb[pred==1,0], test_emb[pred==1,1], color='blue', s=3, label='prediction=1')
    plt.legend()
    plt.show()
    fpr, tpr, thresholds = metrics.roc_curve(data['test_labels'], pred, pos_label=1)
    print(metrics.auc(fpr, tpr))
