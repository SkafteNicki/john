#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 11:23:32 2019

@author: nsde
"""

#%%
import torch
from torch import nn
import numpy as np
import csv
from matplotlib import pyplot as plt
#import tensorflow as tf
#import tensorflow_probability as tfp
#from tensorflow_probability import distributions as tfd
from tqdm import tqdm
from torch import distributions as D
from locality_sampler import gen_Qw, locality_sampler, get_pseupoch, locality_sampler2
import seaborn as sns
sns.set() # default seaborn plot settings

#%%
n_neurons = 50

#%%
def safefloat(x):
    try:
        y = float(x)
    except:
        y = np.NaN
    return y

#%%
def get_data(plot=False):
    # Read data file and split into numpy arrays
    f = open('data/weather/SD_b2dates_trimmed.csv')
    f.readline() # skip first line
    data = csv.reader(f, delimiter=',')

    date = list()
    prcp = list()
    snow = list()
    snwd = list()
    tmax = list()
    tmin = list()
    for row in data:
        date.append(row[0])
        prcp.append(safefloat(row[1]))
        snow.append(safefloat(row[2]))
        snwd.append(safefloat(row[3]))
        tmax.append(safefloat(row[4]))
        tmin.append(safefloat(row[5]))
    f.close()

    maxtemp1d = np.array(tmax)

    # Reshape to be have a year-month-date axes
    first_year = int(date[0].split('-')[0])
    last_year  = int(date[-1].split('-')[0])
    num_years = last_year - first_year + 1
    maxtemp = np.NaN * np.ones((num_years, 12, 31))
    for mt,currdate in zip(maxtemp1d, date):
        year, month, day = currdate.split('-')
        maxtemp[int(year)-first_year, int(month)-1, int(day)-1] = mt

    # Extract a random training set
    train = np.array([maxtemp[np.random.randint(low=0, high=num_years), m, d] for m in range(12) for d in range(31)])  
    
    # Compute mean/std estimates for entire dataset (ground truth)
    mu = np.nanmean(maxtemp, axis=0).flatten()
    s  = np.nanstd(maxtemp, axis=0).flatten()

    # Plot training data and ground truth
    if plot:
        plt.plot(train, 'o')
        plt.plot(mu)
        plt.plot(mu+2.0*s)
        plt.plot(mu-2.0*s)
        plt.show()
    
    comp = np.array([np.arange(len(train)), train, mu, s]).T
    comp = comp[~np.isnan(comp).any(axis=1)]
    
    return comp[:,0], comp[:,1], comp[:,2], comp[:,3]

#%%
def gp(X, y, x):
    if X.shape[0] > 2000: # do not run gp for large datasets
        return np.nan, np.nan
    import GPy
    d = X.shape[1]
    kernel = GPy.kern.RBF(d, ARD=True)
    model = GPy.models.GPRegression(X.numpy(), y.numpy().reshape(-1, 1), kernel, normalizer=True)
    
    model.constrain_positive(' ') # ensure positive hyperparameters
    model.optimize()
    
    mean, var = model.predict(x.numpy())
    _, Xvar = model.predict(X.numpy())
    return mean, np.sqrt(var).flatten(), np.sqrt(Xvar).flatten()

#%%
def neuralnet(X, y, x):
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.LeakyReLU(),
                                      nn.Linear(n_neurons, 1))
            self.var = nn.Sequential(nn.Linear(1, n_neurons),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_neurons, 1),
                                     nn.Softplus())
            
        def forward(self, x):
            return self.mean(x), self.var(x)
    
    model = NNModel()    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    n_iter = 15000
    it = 0
    while it <= n_iter:
        optimizer.zero_grad()
        m, v = model(X)
        v = torch.tensor([10.0]) if it < n_iter/2 else (v+0.01)
        loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).mean()
        loss.backward() 
        if it % 500 == 0: print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
        optimizer.step()
        it+=1
    
    model.eval()
    with torch.no_grad():
        mean, var = model(x)
        _, Xvar = model(X)
    return mean.numpy(), var.sqrt().numpy(), Xvar.sqrt().numpy()

#%%
def dropout(X, y, x):
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.Dropout(p=0.01),
                                      nn.LeakyReLU(),
                                      nn.Linear(n_neurons, 1),
                                      nn.Dropout(p=0.01))
            
        def forward(self, x):
            return self.mean(x)
    
    model = NNModel()    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    n_iter = 15000
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
        samples = torch.zeros(X.shape[0], 1000)
        for i in range(1000):
            samples[:,i] = model(X).flatten()
        Xvar = samples.var(dim=1)
    return mean.numpy(), var.sqrt().numpy(), Xvar.sqrt().numpy()

#%%
def ensemble(X, y, x):
    class NNModel(nn.Module):
        def __init__(self):
            super(NNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.LeakyReLU(),
                                      nn.Linear(n_neurons, 1))
            self.var = nn.Sequential(nn.Linear(1, n_neurons),
                                     nn.LeakyReLU(),
                                     nn.Linear(n_neurons, 1),
                                     nn.Softplus())
            
        def forward(self, x):
            return self.mean(x), self.var(x)
    
    ms, vs, xms, xvs = [ ], [ ], [ ], [ ]
    for i in range(5):
        model = NNModel()    
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        n_iter = 15000
        it = 0
        while it <= n_iter:
            optimizer.zero_grad()
            m, v = model(X)
            v = torch.tensor([0.002]) if it < n_iter/2 else (v+0.01)
            loss = -(-v.log()/2 - (m.flatten()-y)**2 / (2 * v)).mean()
            loss.backward() 
            if it % 500 == 0: print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))
            optimizer.step()
            it+=1
    
        model.eval()
        with torch.no_grad():
            mean, var = model(x)
            ms.append(mean)
            vs.append(var)
            mean, var = model(X)
            xms.append(mean)
            xvs.append(var)
    
    with torch.no_grad():
        ms = torch.stack(ms)
        vs = torch.stack(vs)    
        mean = ms.mean(dim=0)
        var = (vs + ms**2).mean(dim=0) - mean**2
        
        xms = torch.stack(xms)
        xvs = torch.stack(xvs)
        Xvar = (xvs + xms**2).mean(dim=0) - xms.mean(dim=0)**2
            
    return mean.numpy(), var.sqrt().numpy(), Xvar.sqrt().numpy()

#%%
def bnn(X, y, x):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability import edward2 as ed
    X = X.numpy()
    y = y.numpy().reshape(-1, 1)
    x = x.numpy()

    def Net(features):
        W0 = ed.Normal(loc = tf.zeros([1,n_neurons]), scale=10*tf.ones([1,n_neurons]), name='W0')
        b0 = ed.Normal(loc = tf.zeros(n_neurons), scale=10*tf.ones(n_neurons), name='b0')
        W1 = ed.Normal(loc = tf.zeros([n_neurons,1]), scale=10*tf.ones([n_neurons,1]), name='W1')
        b1 = ed.Normal(loc = tf.zeros(1), scale=10*tf.ones(1), name='b1')
        
        h = tf.nn.leaky_relu(tf.matmul(features, W0) + b0, alpha=0.01)
        mean = tf.matmul(h, W1) + b1
        
        noise_std = ed.HalfNormal(scale=1*tf.ones([1]), name="noise_std")    
        
        return ed.Normal(loc=mean, scale=noise_std, name='predictions')
    
    log_joint = ed.make_log_joint_fn(Net)
    
    def target_log_prob_fn(W0, b0, W1, b1, noise_std):
        return log_joint(features = X,
                         W0 = W0, b0 = b0, W1 = W1, b1 = b1,
                         noise_std = noise_std,
                         predictions = y)
    
    num_results = int(10e3) #number of hmc iterations
    n_burnin = int(5e3)     #number of burn-in steps
    step_size = 0.0001
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
    
    #from scipy.special import expit as sigmoid
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha*x)
    
    def NpNet(features, W0, b0, W1, b1, noise):
        h = leaky_relu(np.matmul(features, W0) + b0)
        return np.matmul(h, W1) + b1 + noise*np.random.randn()
    
    out = [NpNet(x, W0_samples[i], b0_samples[i], W1_samples[i],
                 b1_samples[i], noise_std_samples[i]) for i in range(len(W0_samples))]
    
    out = np.array(out)
    y_pred = out.mean(axis=0)
    sigma = out.std(axis=0)
    
    out = [NpNet(X, W0_samples[i], b0_samples[i], W1_samples[i],
                 b1_samples[i], noise_std_samples[i]) for i in range(len(W0_samples))]
    out = np.array(out)
    
    return y_pred, sigma, out.std(axis=0)

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
    
    kmeans = KMeans(n_clusters=20)
    kmeans.fit(np.concatenate([X], axis=0))
    c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    class translatedSigmoid(nn.Module):
        def __init__(self):
            super(translatedSigmoid, self).__init__()
            self.beta = nn.Parameter(torch.tensor([110.0]))
            
        def forward(self, x):
            beta = torch.nn.functional.softplus(self.beta)
            alpha = -beta*(6.9077542789816375)
            return torch.sigmoid((x+alpha)/beta)
    
    class GPNNModel(nn.Module):
        def __init__(self):
            super(GPNNModel, self).__init__()
            self.mean = nn.Sequential(nn.Linear(1, n_neurons),
                                      nn.LeakyReLU(),
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
                a = self.alph(x) * 2
                b = self.bet(x) / 2
                gamma_dist = D.Gamma(a+1e-4, b+1e-4)
                if self.training:
                    samples_var = gamma_dist.rsample(torch.Size([50]))
                    x_var = (1.0/(samples_var+1e-4))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([2000]))
                    x_var = (1.0/(samples_var+1e-4))
                
                var = (1-s) * x_var + s*((20.0**2)*torch.ones_like(x_var)) # HYPERPARAMETER
            else:
                var = torch.tensor([15.0])
            return mean, var
    
    model = GPNNModel()
    optimizer = torch.optim.Adam(model.mean.parameters(), lr=1e-2)
    optimizer2 = torch.optim.Adam(chain(model.alph.parameters(),
                                        model.bet.parameters(),
                                        model.trans.parameters()), lr=1e-3)
    
    n_iter = 40000
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
        switch = 1.0 if it >10000 else 0.0
        
        #if it % 11: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        if not switch:
            optimizer.zero_grad();
            m, v = model(X, switch)
            loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).sum()
            loss.backward()
            optimizer.step()
        else:
            if opt_switch % 2 == 0:    
                #for b in range(mean_pseupoch):
                optimizer.zero_grad()
                batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w)
                m, v = model(X[batch], switch)
                loss = -(-v.log()/2 - ((m.flatten()-y[batch])**2).reshape(1,-1,1) / (2 * v)) / mean_w[batch].reshape(1,-1,1)
                loss = loss.sum() # why the f*** is it so slow
                loss.backward()
                optimizer.step()
            else:
                #for b in range(var_pseupoch):
                optimizer2.zero_grad()
                batch = locality_sampler2(var_psu,var_ssu,var_Q,var_w)
                m, v = model(X[batch], switch)
                loss = -(-v.log()/2 - ((m.flatten()-y[batch])**2).reshape(1,-1,1) / (2 * v)) / var_w[batch].reshape(1,-1,1)
                loss = loss.sum() # why the f*** is it so slow
                loss.backward()
                optimizer2.step()
                    
        if it % 500 == 0:
            model.eval()
            m, v = model(X, switch)
            loss = -(-v.log() - (m.flatten()-y)**2 / (2 * v)).mean()
            print('Iter {0}/{1}, Loss {2}'.format(it, n_iter, loss.item()))

        
#        if it % 1000 == 0:
#            plot(X, y, X, m.detach().numpy(), v.detach().mean(dim=0).sqrt().numpy(), str(it), save=False)

        it+=1
        
    model.eval()
    with torch.no_grad():
        mean, var = model(x, switch)
        _, Xvar = model(X, switch)
    print(list(model.trans.parameters()))
    return mean.numpy(), var.mean(dim=0).sqrt().numpy(), Xvar.mean(dim=0).sqrt().numpy()
   
#%%
def plot2(X, y, x, sigma, labels, save=False):
    X = X.numpy(); y = y.numpy(); x = x.numpy()
    
    plt.figure()
    
    for s, l in zip(sigma, labels):
        plt.plot(x, s, '-', label=l + ' std')
    
    plt.xlabel('$x$')
    plt.ylabel('std(x)')
    plt.legend(loc='best', fontsize=15)
    
    if save:
        plt.savefig('figs/weather_variance_' + str(n_neurons) + '.pdf', format='pdf', bbox_inches = "tight")
        plt.savefig('figs/weather_variance_' + str(n_neurons) + '.png', format='png', bbox_inches = "tight")
        
#%%
def plot3(X, y, x, sigma, sigma_true, labels, save=False):
    X = X.numpy(); y = y.numpy(); x = x.numpy()
    
    plt.figure()
    
    for s, l in zip(sigma, labels):
        diff = abs(sigma_true - s.flatten()).mean()
        plt.plot(X, abs(sigma_true - s.flatten()), '-', label=l + ':' + str(diff.round(2)))
        
    plt.xlabel('$x$')
    plt.ylabel('$| \sigma^* - \hat{\sigma}| $', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    
    if save:
        plt.savefig('figs/weather_variance_diff_' + str(n_neurons) + '.pdf', format='pdf', bbox_inches = "tight")
        plt.savefig('figs/weather_variance_diff_' + str(n_neurons) + '.png', format='png', bbox_inches = "tight")

#%%
def plot(X, y, x, y_pred, sigma, xvar, mu_true, sigma_true, label, save=False):
    X = X.numpy(); y = y.numpy(); x = x.numpy()
    
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(label)
    ax.plot(X, y, 'k.', markersize=5, label='Observations')
    ax.plot(x, y_pred, '-', label=label + ' prediction', lw=3)
    
    ax.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y_pred - 1.96 * sigma,
                           (y_pred + 1.96 * sigma)[::-1]]),
            alpha=.2, ec='None', label=label+': 95% confidence interval', c='#B8C4DC')
    
    ax.plot(X, mu - 1.96 * sigma_true, color='g', lw=2)
    ax.plot(X, mu + 1.96 * sigma_true, color='g', lw=2)
        
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    plt.title('Err={0}'.format(np.mean(np.abs(sigma_true-xvar.flatten())).round(2)), fontsize=40)
    ax.axis([0, 365, 0, 110])
    
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    
    if save:
        plt.savefig('figs/weather_test_' + label + '_' + str(n_neurons) + '.pdf', format='pdf', bbox_inches = "tight")
        plt.savefig('figs/weather_test_' + label + '_' + str(n_neurons) + '.png', format='png', bbox_inches = "tight")

        
#%%
if __name__ == '__main__':
    # Generate data
    X, y, mu, var = get_data()
    size = (X.max() - X.min()) / 50
    x = np.linspace(X.min()-size, X.max()+size, 500)
    
    X = torch.tensor(X.reshape(-1,1)).to(torch.float32)
    y = torch.tensor(y).to(torch.float32)
    x = torch.tensor(x.reshape(-1,1)).to(torch.float32)
    
    # Fit models
    y_pred, sigma, xvar, labels = [ ], [ ], [ ], [ ]
#    print("================ Fitting GP ================")
#    y_pred1, sigma1, xvar1 = gp(X, y, x)
#    y_pred.append(y_pred1); sigma.append(sigma1); xvar.append(xvar1); labels.append('GP')
#    print("================ Fitting NeuNet ================")
#    y_pred2, sigma2, xvar2 = neuralnet(X, y, x)
#    y_pred.append(y_pred2); sigma.append(sigma2); xvar.append(xvar2);  labels.append('NN')
#    print("================ Fitting McDnn ================")
#    y_pred3, sigma3, xvar3 = dropout(X, y, x)
#    y_pred.append(y_pred3); sigma.append(sigma3); xvar.append(xvar3); labels.append('MC-Dropout')
#    print("================ Fitting Ensnn ================")
#    y_pred4, sigma4, xvar4 = ensemble(X, y, x)
#    y_pred.append(y_pred4); sigma.append(sigma4); xvar.append(xvar4); labels.append('Ens-NN')
#    print("================ Fitting BNN ================")
    y_pred5, sigma5, xvar5 = bnn(X, y, x)
    y_pred.append(y_pred5); sigma.append(sigma5); xvar.append(xvar5); labels.append('BNN')
#    print("================ Fitting JOHN ================")
#    y_pred6, sigma6, xvar6 = john(X, y, x)
#    y_pred.append(y_pred6); sigma.append(sigma6); xvar.append(xvar6); labels.append('Combined')
    
    #%%
    # Plot
    plt.close('all')
    for yp, s, l, xv in zip(y_pred, sigma, labels, xvar):
        plot(X, y, x, yp, s, xv, mu, var, l, save=False)
#   
#    plot2(X, y, x, sigma, labels, save=True)
#    plot3(X, y, x, xvar, var, labels, save=True)    
