#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:02:03 2019

@author: nsde
"""

#%%
import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tqdm import tqdm
from torch import distributions as D
from locality_sampler import gen_Qw, locality_sampler, get_pseupoch, locality_sampler2
import seaborn as sns
from utils import t_likelihood
sns.set() # default seaborn plot settings

#%%
n_neurons = 50

#%%
def generate_data(num_samples=20):
    def f(x):
        return x * np.sin(x) # x**3
    X = np.random.uniform(0, 10, size=num_samples)
    x = np.linspace(-4, 14, 250)
    y = f(X) + 0.3*np.random.randn(num_samples) + 0.3*X*np.random.randn(num_samples) # added input dependent noise
    return torch.tensor(np.atleast_2d(X).T).to(torch.float32), \
           torch.tensor(y).to(torch.float32), \
           torch.tensor(np.atleast_2d(x).T).to(torch.float32), \
           f

#%%
def gp(X, y, x):
    import gpytorch
    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(X, y, likelihood)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    n_iter = 7000
    it = 0
    while it <= n_iter:
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, y)
        loss.backward()
        if it % 500 == 0: print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
        optimizer.step()
        it+=1
        
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        observed_pred = likelihood(model(x))
        mean = observed_pred.mean
        var = observed_pred.variance.sqrt()
    return mean.reshape(-1,1).numpy(), var.reshape(-1,1).numpy()

#%%
def neuralnet(X, y, x):
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons, 1))
            self.var = nn.Sequential(nn.Linear(1, n_neurons),
                                     nn.Sigmoid(),
                                     nn.Linear(n_neurons, 1),
                                     nn.Softplus())
            
        def forward(self, x):
            return self.mean(x), self.var(x)
    
    model = NNModel()    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    n_iter = 10000
    it = 0
    while it <= n_iter:
        optimizer.zero_grad()
        m, v = model(X)
        v = torch.tensor([0.002]) if it < n_iter/2 else v
        loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).mean()
        loss.backward() 
        if it % 500 == 0: print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
        optimizer.step()
        it+=1
    
    model.eval()
    with torch.no_grad():
        mean, var = model(x)
    return mean.numpy(), var.sqrt().numpy()

#%%
def dropout(X, y, x):
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.Dropout(p=0.05),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons, 1),
                                      nn.Dropout(p=0.05))
            
        def forward(self, x):
            return self.mean(x)
    
    model = NNModel()    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    n_iter = 10000
    it = 0
    while it <= n_iter:
        optimizer.zero_grad()
        m = model(X)
        loss = ((m.flatten()-y)**2).mean()
        loss.backward() 
        if it % 500 == 0: print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
        optimizer.step()
        it+=1
    
    with torch.no_grad():
        samples = torch.zeros(x.shape[0], 1000)
        for i in range(1000):
            samples[:,i] = model(x).flatten()
        mean, var = samples.mean(dim=1), samples.var(dim=1)
    return mean.numpy(), var.sqrt().numpy()

#%%
def ensemble(X, y, x):
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons, 1))
            self.var = nn.Sequential(nn.Linear(1, n_neurons),
                                     nn.Sigmoid(),
                                     nn.Linear(n_neurons, 1),
                                     nn.Softplus())
            
        def forward(self, x):
            return self.mean(x), self.var(x)
    
    ms, vs = [ ], [ ]
    for i in range(5):
        model = NNModel()    
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        n_iter = 10000
        it = 0
        while it <= n_iter:
            optimizer.zero_grad()
            m, v = model(X)
            v = torch.tensor([0.002]) if it < n_iter/2 else v
            loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).mean()
            loss.backward() 
            if it % 500 == 0: print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
            optimizer.step()
            it+=1
    
        model.eval()
        with torch.no_grad():
            mean, var = model(x)
            ms.append(mean)
            vs.append(var)
    
    with torch.no_grad():
        ms = torch.stack(ms)
        vs = torch.stack(vs)    
        mean = ms.mean(dim=0)
        var = (vs + ms**2).mean(dim=0) - mean**2
            
    return mean.numpy(), var.sqrt().numpy()

#%%
def bnn(X, y, x):
    from tensorflow_probability import edward2 as ed
    X = X.numpy()
    y = y.numpy().reshape(-1, 1)
    x = x.numpy()

    def Net(features):
        W0 = ed.Normal(loc = tf.zeros([1,n_neurons]), scale=10*tf.ones([1,n_neurons]), name='W0')
        b0 = ed.Normal(loc = tf.zeros(n_neurons), scale=10*tf.ones(n_neurons), name='b0')
        W1 = ed.Normal(loc = tf.zeros([n_neurons,1]), scale=10*tf.ones([n_neurons,1]), name='W1')
        b1 = ed.Normal(loc = tf.zeros(1), scale=10*tf.ones(1), name='b1')
        
        h = tf.sigmoid(tf.matmul(features, W0) + b0)
        mean = tf.matmul(h, W1) + b1
        
        noise_std = ed.HalfNormal(scale=tf.ones([1]), name="noise_std")    
        
        return ed.Normal(loc=mean, scale=noise_std, name='predictions')
    
    log_joint = ed.make_log_joint_fn(Net)
    
    def target_log_prob_fn(W0, b0, W1, b1, noise_std):
        return log_joint(features = X,
                         W0 = W0, b0 = b0, W1 = W1, b1 = b1,
                         noise_std = noise_std,
                         predictions = y)
    
    num_results = int(20e3) #number of hmc iterations
    n_burnin = int(5e3)     #number of burn-in steps
    step_size = 0.01
    num_leapfrog_steps = 10

    kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps)
    
    states, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=n_burnin,
            kernel=kernel,
            current_state=[
                    tf.zeros([1,n_neurons], name='init_W0'),
                    tf.zeros([n_neurons], name='init_b0'),
                    tf.zeros([n_neurons,1], name='init_W1'),
                    tf.zeros([1], name='init_b1'),
                    tf.ones([1], name='init_noise_std'),
                    ]
            )
    W0, b0, W1, b1, noise_std = states
    
    with tf.Session() as sess:
        [W0_, b0_, W1_, b1_, noise_std_, accepted_] = sess.run(
                [W0, b0, W1, b1, noise_std, kernel_results.is_accepted])

    W0_samples = W0_[n_burnin:]
    b0_samples = b0_[n_burnin:]
    W1_samples = W1_[n_burnin:]
    b1_samples = b1_[n_burnin:]
    noise_std_samples = noise_std_[n_burnin:]
    accepted_samples = accepted_[n_burnin:]

    print('Acceptance rate: %0.1f%%' % (100*np.mean(accepted_samples)))
    
    from scipy.special import expit as sigmoid
    
    def NpNet(features, W0, b0, W1, b1, noise):
        h = sigmoid(np.matmul(features, W0) + b0)
        return np.matmul(h, W1) + b1# + noise*np.random.randn()
    
    out = [NpNet(x, W0_samples[i], b0_samples[i], W1_samples[i],
                 b1_samples[i], noise_std_samples[i]) for i in range(len(W0_samples))]
    
    out = np.array(out)
    y_pred = out.mean(axis=0)
    sigma = out.std(axis=0)
    return y_pred, sigma

#%%
def john(X, y, x):
    from sklearn.cluster import KMeans
    from utils import dist
    from itertools import chain

    mean_psu = 1
    mean_ssu = 50
    mean_M = 60
    
    var_psu = 3
    var_ssu = 7
    var_M = 10
    
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(np.concatenate([X], axis=0))
    c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    class translatedSigmoid(nn.Module):
        def __init__(self):
            super(translatedSigmoid, self).__init__()
            self.beta = nn.Parameter(torch.tensor([1.5]))
            
        def forward(self, x):
            beta = torch.nn.functional.softplus(self.beta)
            alpha = -beta*(6.9077542789816375)
            return torch.sigmoid((x+alpha)/beta)
    
    class GPNNModel(nn.Module):
        def __init__(self):
            super(GPNNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons, 1))
            self.alph = nn.Sequential(nn.Linear(1,n_neurons),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons,1),
                                      nn.Softplus())
            self.bet = nn.Sequential(nn.Linear(1,n_neurons),
                                     nn.Sigmoid(),
                                     nn.Linear(n_neurons,1),
                                     nn.Softplus())
            self.trans = translatedSigmoid()
            
        def forward(self, x, switch):
            d = dist(x, c)
            d_min = d.min(dim=1, keepdim=True)[0]
            s = self.trans(d_min)
            mean = self.mean(x)
            if switch:
                a = self.alph(x)
                b = self.bet(x)
                gamma_dist = D.Gamma(a+1e-8, 1.0/(b+1e-8))
                if self.training:
                    samples_var = gamma_dist.rsample(torch.Size([50]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([2000]))
                    x_var = (1.0/(samples_var+1e-8))
                var = (1-s) * x_var + s*torch.tensor([3.5**2]) # HYPERPARAMETER
                
            else:
                var = torch.tensor([0.05])
            return mean, var
    
    model = GPNNModel()
    optimizer = torch.optim.Adam(model.mean.parameters(), lr=1e-2)
    optimizer2 = torch.optim.Adam(chain(model.alph.parameters(),
                                        model.bet.parameters(),
                                        model.trans.parameters()), lr=1e-3)
    
    n_iter = 6000
    it = 0
    mean_Q, mean_w = gen_Qw(X, mean_psu, mean_ssu, mean_M)
    var_Q, var_w = gen_Qw(X, var_psu, var_ssu, var_M)
    mean_pseupoch = get_pseupoch(mean_w,0.5)
    var_pseupoch = get_pseupoch(var_w,0.5)
    opt_switch = 1
    mean_w = torch.Tensor(mean_w)
    var_w = torch.Tensor(var_w)
    model.train()

    while it < n_iter:
        model.train()
        switch = 1.0 if it >5000 else 0.0
        
        if it % 11: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        if not switch:
            optimizer.zero_grad();
            m, v = model(X, switch)
            loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).sum()
            loss.backward()
            optimizer.step()
        else:
            if opt_switch % 2 == 0:    
                for b in range(mean_pseupoch):
                    optimizer.zero_grad()
                    batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w)
                    m, v = model(X[batch], switch)
                    loss = -(-v.log() - ((m.flatten()-y[batch])**2).reshape(1,-1,1) / (2 * v)) / mean_w[batch].reshape(1,-1,1)
                    loss = loss.sum() # why the f*** is it so slow
                    loss.backward()
                    optimizer.step()
            else:
                for b in range(var_pseupoch):
                    optimizer2.zero_grad()
                    batch = locality_sampler2(var_psu,var_ssu,var_Q,var_w)
                    m, v = model(X[batch], switch)
                    diff = ((m.flatten()-y[batch])**2).reshape(1,-1,1)
                    loss = -(-(diff.log() / 2 + diff/v + v.log() / 2)) / var_w[batch].reshape(1,-1,1)
                    loss = loss.sum() # why the f*** is it so slow
                    loss.backward()
                    optimizer2.step()
                    
        if it % 500 == 0:
            model.eval()
            m, v = model(X, switch)
            loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).mean()
            print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
        it+=1
        
    model.eval()
    with torch.no_grad():
        mean, var = model(x, switch)
    return mean.numpy(), var.mean(dim=0).sqrt().numpy()

#%%
def ens_john(X, y, x):
    from sklearn.cluster import KMeans
    from utils import dist
    from itertools import chain

    mean_psu = 1
    mean_ssu = 50
    mean_M = 60
    
    var_psu = 3
    var_ssu = 7
    var_M = 10
    
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(np.concatenate([X], axis=0))
    c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    class translatedSigmoid(nn.Module):
        def __init__(self):
            super(translatedSigmoid, self).__init__()
            self.beta = nn.Parameter(torch.tensor([1.5]))
            
        def forward(self, x):
            beta = torch.nn.functional.softplus(self.beta)
            alpha = -beta*(6.9077542789816375)
            return torch.sigmoid((x+alpha)/beta)
    
    class GPNNModel(nn.Module):
        def __init__(self):
            super(GPNNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons, 1))
            self.alph = nn.Sequential(nn.Linear(1,n_neurons),
                                      nn.Sigmoid(),
                                      nn.Linear(n_neurons,1),
                                      nn.Softplus())
            self.bet = nn.Sequential(nn.Linear(1,n_neurons),
                                     nn.Sigmoid(),
                                     nn.Linear(n_neurons,1),
                                     nn.Softplus())
            self.trans = translatedSigmoid()
            
        def forward(self, x, switch):
            d = dist(x, c)
            d_min = d.min(dim=1, keepdim=True)[0]
            s = self.trans(d_min)
            mean = self.mean(x)
            if switch:
                a = self.alph(x)
                b = self.bet(x)
                gamma_dist = D.Gamma(a+1e-8, 1.0/(b+1e-8))
                if self.training:
                    samples_var = gamma_dist.rsample(torch.Size([50]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([2000]))
                    x_var = (1.0/(samples_var+1e-8))
                var = (1-s) * x_var + s*torch.tensor([3.5**2]) # HYPERPARAMETER
                
            else:
                var = torch.tensor([0.05])
            return mean, var

    ens_mean, ens_var = [ ], [ ]
    for i in range(5):
        model = GPNNModel()
        optimizer = torch.optim.Adam(model.mean.parameters(), lr=1e-2)
        optimizer2 = torch.optim.Adam(chain(model.alph.parameters(),
                                            model.bet.parameters(),
                                            model.trans.parameters()), lr=1e-3)
        
        n_iter = 6000
        it = 0
        mean_Q, mean_w = gen_Qw(X, mean_psu, mean_ssu, mean_M)
        var_Q, var_w = gen_Qw(X, var_psu, var_ssu, var_M)
        mean_pseupoch = get_pseupoch(mean_w,0.5)
        var_pseupoch = get_pseupoch(var_w,0.5)
        opt_switch = 1
        mean_w = torch.Tensor(mean_w)
        var_w = torch.Tensor(var_w)
        model.train()
    
        while it < n_iter:
            model.train()
            switch = 1.0 if it >5000 else 0.0
            
            if it % 11: opt_switch = opt_switch + 1 # change between var and mean optimizer
            
            if not switch:
                optimizer.zero_grad();
                m, v = model(X, switch)
                loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).sum()
                loss.backward()
                optimizer.step()
            else:
                if opt_switch % 2 == 0:    
                    for b in range(mean_pseupoch):
                        optimizer.zero_grad()
                        batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w)
                        m, v = model(X[batch], switch)
                        loss = -t_likelihood(y[batch], m, v, mean_w[batch])#-(-v.log() - ((m.flatten()-y[batch])**2).reshape(1,-1,1) / (2 * v)) / mean_w[batch].reshape(1,-1,1)
                        loss = loss.sum() # why the f*** is it so slow
                        loss.backward()
                        optimizer.step()
                else:
                    for b in range(var_pseupoch):
                        optimizer2.zero_grad()
                        batch = locality_sampler2(var_psu,var_ssu,var_Q,var_w)
                        m, v = model(X[batch], switch)
                        loss = -t_likelihood(y[batch], m, v, var_w[batch])#-(-(diff.log() / 2 + diff/v + v.log() / 2)) / var_w[batch].reshape(1,-1,1)
                        loss = loss.sum() # why the f*** is it so slow
                        loss.backward()
                        optimizer2.step()
                        
            if it % 500 == 0:
                model.eval()
                m, v = model(X, switch)
                loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).mean()
                print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
            it+=1
            
        model.eval()
        with torch.no_grad():
            mean, var = model(x, switch)
        ens_mean.append(mean)
        ens_var.append(var.mean(dim=0))
    
    ens_mean = torch.stack(ens_mean)
    ens_var = torch.stack(ens_var)
    
    mean = ens_mean.mean(dim=0)
    var = (ens_var + ens_mean**2).mean(dim=0) - mean**2
    
    return mean.numpy(), var.sqrt().numpy()

#%%
def plot(X, y, x, y_pred, sigma, label, save=False):
    X = X.numpy(); y = y.numpy(); x = x.numpy()
#    true_std = (0.3+0.3*x)*((0<x) & (x<10)) + 100*((x<0) | (10<x))
#    true_std[true_std > 50] = np.nan
#    valid = np.isfinite(true_std)
#    upper = f(x) + 1.96 * true_std
#    lower = f(x) - 1.96 * true_std
    
    fig, ax = plt.subplots()
    ax.plot(x, f(x), ':', label=r'$f(x) = x\,\sin(x)$', lw=3)
    ax.plot(X.ravel(), y, '.', markersize=10, label=u'Observations', lw=3)
    ax.plot(x, y_pred, '-', label=label + ' prediction', lw=3)
    #plt.plot(x[valid], upper[valid], 'k', alpha=0.5)
    #plt.plot(x[valid], lower[valid], 'k', alpha=0.5)
    
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.96 * sigma,
                           (y_pred + 1.96 * sigma)[::-1]]),
            alpha=.3, ec='None', label=label+': 95% confidence interval')
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.axis([-4, 14, -15, 15])
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        
    if save:
        plt.savefig('figs/test_' + label + '_' + str(n_neurons) + '.pdf', format='pdf', bbox_inches = "tight")
        plt.savefig('figs/test_' + label + '_' + str(n_neurons) + '.png', format='png', bbox_inches = "tight")

#%%
def plot2(X, y, x, sigma, labels, save=False):
    X = X.numpy(); y = y.numpy(); x = x.numpy()
    
    true_std = (0.3+0.3*x)*((0<x) & (x<10)) + 100*((x<0) | (10<x))
    true_std[true_std > 50] = np.nan
    valid = np.isfinite(true_std)
    
    fig, ax = plt.subplots()
    ax.plot(x[valid], true_std[valid], 'k', label='True std', lw=3)
    for s, l in zip(sigma, labels):
        ax.plot(x, s, '-', label=l, lw=3)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('std(x)')
    ax.legend(loc='best', fontsize=15)
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        
    if save:
        plt.savefig('figs/variance_' + str(n_neurons) + '.pdf', format='pdf', bbox_inches = "tight")
        plt.savefig('figs/variance_' + str(n_neurons) + '.png', format='png', bbox_inches = "tight")
    
#%%
if __name__ == '__main__':
    # Generate data
    X, y, x, f = generate_data(num_samples=500)

    # Fit models
    y_pred, sigma, labels = [ ], [ ], [ ]
#    print("================ Fitting GP ================")
#    y_pred1, sigma1 = gp(X, y, x)
#    y_pred.append(y_pred1); sigma.append(sigma1); labels.append('GP')
#    print("================ Fitting NeuNet ================")
#    y_pred2, sigma2 = neuralnet(X, y, x)
#    y_pred.append(y_pred2); sigma.append(sigma2); labels.append('NN')
#    print("================ Fitting McDnn ================")
#    y_pred3, sigma3 = dropout(X, y, x)
#    y_pred.append(y_pred3); sigma.append(sigma3); labels.append('MC-Dropout')
#    print("================ Fitting Ensnn ================")
#    y_pred4, sigma4 = ensemble(X, y, x)
#    y_pred.append(y_pred4); sigma.append(sigma4); labels.append('Ens-NN')
#    print("================ Fitting BNN ================")
#    y_pred5, sigma5 = bnn(X, y, x)
#    y_pred.append(y_pred5); sigma.append(sigma5); labels.append('BNN')
    print("================ Fitting JOHN ================")
    y_pred6, sigma6 = john(X, y, x)
    y_pred.append(y_pred6); sigma.append(sigma6); labels.append('Combined')
#    print("================ Fitting Ens-JOHN ================")
#    y_pred7, sigma7 = ens_john(X, y, x)
#    y_pred.append(y_pred7); sigma.append(sigma7); labels.append('Combined')
    
    #%%
    # Plot
    plt.close('all')
    for yp, s, l in zip(y_pred, sigma, labels):
        plot(X, y, x, yp, s, l, save=False)
    plot(X, y, x, y_pred[-1], sigma[-1], labels[-1], save=True)
    plot2(X, y, x, sigma, labels, save=True)
    
