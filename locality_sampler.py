# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 09:57:28 2019

@author: marjor
"""

#%%
import numpy as np
from sklearn.neighbors import KDTree

#%%
def gen_Qw(data,psu,ssu,M):
    """ Takes data as numpy matrix """
    T = KDTree(data)
    _, Q = T.query(data, k = M)
    N = Q.shape[0]; 
    w = psu * np.bincount(Q.flatten()) / N * ssu / M
    #w = np.zeros(N)
#    for i in range(N):
#        w[i] = psu * np.sum( Q == i ) / N * ssu / M

    return(Q,w)

#%%
def locality_sampler(psu,ssu,Q,w):
    ps = np.random.randint(len(w),size = psu)
    ss = []
    
    for i in ps:
        ss = np.append(ss,np.random.choice(Q[i,:],size = ssu))
    ss = np.unique(ss)
    return(ss)

#%%
 # minor error in above
def locality_sampler2(psu,ssu,Q,w):
    ps = np.random.randint(len(w),size = psu)
    ss = []
    
    for i in ps:
        ss = np.append(ss,np.random.choice(Q[i,:],size = ssu, replace = False))
    ss = np.unique(ss)
    return(ss)
#%% 
def get_pseupoch(w,p):
    w_min = np.min(w)
    w_min = np.maximum(w_min, 1e-2)
    n = 0; x = 0
    while x<p:
        n = n + 1
        x = 1 - (1-w_min)**n
    return(n)
        
    
#%%
def local_batchify(*arrays, **kwargs):
#    mean_psu = 1
#    mean_ssu = 100
#    mean_M = 60
#
#    var_psu = 3
#    var_ssu = 7
#    var_M = 10
    
    mean_psu = 1
    mean_ssu = 256
    mean_M = 300
    
    var_psu = 2
    var_ssu = 8
    var_M = 10
    
    if arrays[0].shape[1] > 50:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=0.50)
        temp = pca.fit_transform(arrays[0])
        mean_Q, mean_w = gen_Qw(temp, mean_psu, mean_ssu, mean_M)
        var_Q, var_w = gen_Qw(temp, var_psu, var_ssu, var_M)
    else:
        mean_Q, mean_w = gen_Qw(arrays[0], mean_psu, mean_ssu, mean_M)
        var_Q, var_w = gen_Qw(arrays[0], var_psu, var_ssu, var_M)
    arrays = (*arrays, mean_w, var_w)
    while True:
        batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w).astype(np.int32)
        yield [a[batch] for a in arrays]
        
#%%
if __name__ == '__main__':
    def generate_data(num_samples=20):
        def f(x):
            return x * np.sin(x) # x**3
        X = np.random.uniform(0, 10, size=num_samples)
        x = np.linspace(-4, 14, 250)
        y = f(X) + 0.3*np.random.randn(num_samples) + 0.3*X*np.random.randn(num_samples) # added input dependent noise
        return np.atleast_2d(X).T, y, np.atleast_2d(x).T, f
    
    X, y, x, f = generate_data(1000)
    
    name = 'psu_6'
    
    mean_psu = 1
    mean_ssu = 100
    mean_M = 105
    
    var_psu = 2
    var_ssu = 8
    var_M = 10
    
    def local_batchify2(*arrays, **kwargs):
        mean_Q, mean_w = gen_Qw(arrays[0], mean_psu, mean_ssu, mean_M)
        var_Q, var_w = gen_Qw(arrays[0], var_psu, var_ssu, var_M)
        arrays = (*arrays, mean_w, var_w)
        while True:
            batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w).astype(np.int32)
            yield [a[batch] for a in arrays]
    
    
    batches = local_batchify2(X, y)
    
    import matplotlib.pyplot as plt
    plt.close('all')
    for i in range(5):
        plt.figure()
        plt.plot(X, y, 'k.', markersize=15)
        xb, yb, _, _ = next(batches)
        plt.plot(xb, yb, 'ro', markersize=15)
        plt.savefig(name + '_' + str(i))
        plt.axis('off')
        
    
    
             