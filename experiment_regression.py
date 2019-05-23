#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 10:06:31 2019

@author: nsde
"""

#%%
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import argparse, math, os
from tqdm import tqdm
from itertools import chain
from utils import timer, batchify, normalize_y, normal_log_prob, RBF, \
    Norm2, OneMinusX, PosLinear, Reciprocal, normal_log_prob_w_prior, t_likelihood
    

#%%
def argparser():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    gs = parser.add_argument_group('General settings')
    gs.add_argument('--model', type=str, default='john', help='model to use',
                    choices=['gp', 'sgp', 'nn', 'mcdnn', 'ensnn', 'bnn', 'rbfnn', 'gpnn', 'john'])
    gs.add_argument('--dataset', type=str, default='boston', help='dataset to use')
    gs.add_argument('--seed', type=int, default=1, help='random state of data-split')
    gs.add_argument('--test_size', type=float, default=0.1, help='test set size, as a procentage')
    gs.add_argument('--repeats', type=int, default=20, help='number of repeatitions')
    gs.add_argument('--silent', type=bool, default=True, help='suppress warnings')
    gs.add_argument('--cuda', type=bool, default=True, help='use cuda')
    
    ms = parser.add_argument_group('Model specific settings')
    ms.add_argument('--batch_size', type=int, default=512, help='batch size')
    ms.add_argument('--shuffel', type=bool, default=True, help='shuffel data during training')
    ms.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ms.add_argument('--iters', type=int, default=10000, help='number of iterations')
    ms.add_argument('--mcmc', type=int, default=500, help='number of mcmc samples')
    ms.add_argument('--inducing', type=int, default=500, help='number of inducing points')
    ms.add_argument('--n_clusters', type=int, default=500, help='number of cluster centers')
    ms.add_argument('--n_models', type=int, default=5, help='number of ensemble')
    
    # Parse and return
    args = parser.parse_args()
    return args
    
#%%
def gp(args, X, y, Xval, yval):
    if X.shape[0] > 2000: # do not run gp for large datasets
        return np.nan, np.nan
    import GPy
    d = X.shape[1]
    kernel = GPy.kern.RBF(d, ARD=True)
    model = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel, normalizer=True)
    
    model.constrain_positive(' ') # ensure positive hyperparameters
    model.optimize()
    
    y_pred, cov = model.predict(Xval, full_cov=True)
    cov += 1e-4*np.diag(np.ones(cov.shape[0]))
    y_pred = y_pred.flatten()
    log_px = -1/2*(np.linalg.slogdet(cov)[1] \
             + (yval-y_pred).T.dot(np.linalg.inv(cov).dot(yval-y_pred)) \
             + d*math.log(2*math.pi)) / Xval.shape[0]
    
    rmse = math.sqrt(((yval-y_pred)**2).mean())
    return log_px, rmse

#%%
def sgp(args, X, y, Xval, yval):
    if X.shape[0] > 30000: # do not run spg for large datasets
        return np.nan, np.nan
    import GPy
    args.inducing = min(args.inducing, X.shape[0])
    d = X.shape[1]
    kernel = GPy.kern.RBF(d, ARD=True)
    model = GPy.models.SparseGPRegression(X, y.reshape(-1, 1), kernel, normalizer=True,
                                          num_inducing=args.inducing)
    model.constrain_positive(['rbf.variance', 'rbf.lengthscale', 'Gaussian_noise.variance'])
    model.optimize()
    
    y_pred, cov = model.predict(Xval, full_cov=True)
    y_pred = y_pred.flatten()

    log_px = -1/2*(np.linalg.slogdet(cov)[1] \
             + (yval-y_pred).T.dot(np.linalg.inv(cov).dot(yval-y_pred)) \
             + d*math.log(2*math.pi)) / Xval.shape[0]
    
    rmse = math.sqrt(((yval-y_pred)**2).mean())
    return log_px, rmse

#%%
def nn(args, X, y, Xval, yval):
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    
    y, y_mean, y_std = normalize_y(y)
    
    mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                               torch.nn.ReLU(),
                               torch.nn.Linear(n_neurons, 1))
    var = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                              torch.nn.ReLU(),
                              torch.nn.Linear(n_neurons, 1),
                              torch.nn.Softplus())
    if torch.cuda.is_available() and args.cuda: 
        mean.cuda(); var.cuda(); 
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    optimizer = torch.optim.Adam(chain(mean.parameters(),
                                       var.parameters()), lr=args.lr)
    it = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        optimizer.zero_grad()
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
        loss = normal_log_prob(label, m, v).sum()
        (-loss).backward()
        optimizer.step()
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = mean(data), var(data)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = normal_log_prob(label, m, v)    
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()

#%%
def mcdnn(args, X, y, Xval, yval):
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    
    mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                               torch.nn.Dropout(p=0.05),
                               torch.nn.ReLU(),
                               torch.nn.Linear(n_neurons, 1),
                               torch.nn.Dropout(p=0.05))
    
    if torch.cuda.is_available() and args.cuda: 
        mean.cuda()
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    optimizer = torch.optim.Adam(mean.parameters(), lr=args.lr)
    
    it = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        optimizer.zero_grad()
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m = mean(data)
        loss = (m - label).abs().pow(2.0).mean()
        loss.backward()
        optimizer.step()
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    samples = torch.zeros(Xval.shape[0], args.mcmc).to(device)
    for i in range(args.mcmc):
        samples[:,i] = mean(data).flatten()
    m, v = samples.mean(dim=1), samples.var(dim=1)
    
    log_probs = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_probs.mean().item(), rmse.item()

#%%
def ensnn(args, X, y, Xval, yval):
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    
    y, y_mean, y_std = normalize_y(y)
    
    ms, vs = [ ], [ ]
    for m in range(args.n_models): # initialize differently
        mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(n_neurons, 1))
        var = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(n_neurons, 1),
                                  torch.nn.Softplus())
        if torch.cuda.is_available() and args.cuda: 
            mean.cuda(); var.cuda(); 
            device=torch.device('cuda')
        else:
            device=torch.device('cpu')
        
        optimizer = torch.optim.Adam(chain(mean.parameters(),
                                           var.parameters()), lr=args.lr)
        it = 0
        progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
        batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
        while it < args.iters:
            switch = 0.0#1.0 if it > args.iters/2 else 0.0
            optimizer.zero_grad()
            data, label = next(batches)
            data = torch.tensor(data).to(torch.float32).to(device)
            label = torch.tensor(label).to(torch.float32).to(device)
            m, v = mean(data), var(data)
            v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
            loss = normal_log_prob(label, m, v).sum()
            (-loss).backward()
            optimizer.step()
            it+=1
            progressBar.update()
            progressBar.set_postfix({'loss': loss.item()})
        progressBar.close()
        
        data = torch.tensor(Xval).to(torch.float32).to(device)
        label = torch.tensor(yval).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        m = m * y_std + y_mean
        v = v * y_std**2
        
        ms.append(m)
        vs.append(v)
    
    ms = torch.stack(ms)
    vs = torch.stack(vs)
        
    m = ms.mean(dim=0)
    v = (vs + ms**2).mean(dim=0) - m**2
    
    log_px = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()
    
#%%
def bnn(args, X, y, Xval, yval):
    import tensorflow as tf
    import tensorflow_probability as tfp
    from tensorflow_probability import distributions as tfd
    tf.reset_default_graph()
    
    y, y_mean, y_std = normalize_y(y)
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
        
    def VariationalNormal(name, shape, constraint=None):
        means = tf.get_variable(name+'_mean',
                                initializer=tf.ones([1]),
                                constraint=constraint)
        stds = tf.get_variable(name+'_std',
                               initializer=-1.0*tf.ones([1]))
        return tfd.Normal(loc=means, scale=tf.nn.softplus(stds))
    
    x_p = tf.placeholder(tf.float32, shape=(None, X.shape[1]))
    y_p = tf.placeholder(tf.float32, shape=(None, 1))
    
    with tf.name_scope('model', values=[x_p]):
        layer1 = tfp.layers.DenseFlipout(
                units=n_neurons,
                activation='relu',
                kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(),
                bias_posterior_fn = tfp.layers.default_mean_field_normal_fn()
                )
        layer2 = tfp.layers.DenseFlipout(
                units=1,
                activation='linear',
                kernel_posterior_fn = tfp.layers.default_mean_field_normal_fn(),
                bias_posterior_fn = tfp.layers.default_mean_field_normal_fn()
                )
        predictions = layer2(layer1(x_p))
        noise = VariationalNormal('noise', [1], constraint=tf.keras.constraints.NonNeg())
        pred_distribution = tfd.Normal(loc=predictions,
                                       scale=noise.sample())
        
    neg_log_prob = -tf.reduce_mean(pred_distribution.log_prob(y_p))
    kl_div = sum(layer1.losses + layer2.losses) / X.shape[0]
    elbo_loss = neg_log_prob + kl_div
    
    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        train_op = optimizer.minimize(elbo_loss)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        it = 0
        progressBar = tqdm(desc='Training BNN', total=args.iters, unit='iter')
        batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
        while it < args.iters:
            data, label = next(batches)
            _, l = sess.run([train_op, elbo_loss], feed_dict={x_p: data, y_p: label.reshape(-1, 1)})
            progressBar.update()
            progressBar.set_postfix({'loss': l})
            it+=1
        progressBar.close()
    
        W0_samples = layer1.kernel_posterior.sample(1000)
        b0_samples = layer1.bias_posterior.sample(1000)
        W1_samples = layer2.kernel_posterior.sample(1000)
        b1_samples = layer2.bias_posterior.sample(1000)
        noise_samples = noise.sample(1000)
    
        W0, b0, W1, b1, n = sess.run([W0_samples,
                                      b0_samples,
                                      W1_samples,
                                      b1_samples,
                                      noise_samples])
    
    def sample_net(x, W0, b0, W1, b1, n):
        h = np.maximum(np.matmul(x[np.newaxis], W0) + b0[:, np.newaxis, :], 0.0)
        return np.matmul(h, W1) + b1[:, np.newaxis, :] + n[:, np.newaxis, :] * np.random.randn()
        
    samples = sample_net(Xval, W0, b0, W1, b1, n)

    m = samples.mean(axis=0)
    v = samples.var(axis=0)
    
    m = m * y_std + y_mean
    v = v * y_std**2
    
    log_probs = normal_log_prob(yval, m, v)
    rmse = math.sqrt(((m.flatten() - yval)**2).mean())
    
    return log_probs.mean(), rmse

#%%
def rbfnn(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    cluster_alg = KMeans(args.n_clusters)
    cluster_alg.fit(X)
    c = cluster_alg.cluster_centers_
    if torch.cuda.is_available() and args.cuda: 
        c = torch.tensor(c).to(torch.float32).to('cuda')
    else:
        c = torch.tensor(c).to(torch.float32)
    
    mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                               torch.nn.ReLU(),
                               torch.nn.Linear(n_neurons, 1))
    var = torch.nn.Sequential(RBF(None, None, c, 1.0),
                              PosLinear(args.n_clusters, 1, bias=False),
                              Reciprocal(0.1),
                              PosLinear(1, 1, bias=False))

    if torch.cuda.is_available() and args.cuda: 
        mean.cuda(); var.cuda(); 
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    optimizer = torch.optim.Adam(chain(mean.parameters(),
                                       var.parameters()), lr=args.lr)
    it = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        optimizer.zero_grad()
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
        loss = normal_log_prob(label, m, v).sum()
        (-loss).backward()
        optimizer.step()
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = mean(data), var(data)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()
                           
#%%
def gpnn(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    cluster_alg = KMeans(args.n_clusters)
    if args.dataset != 'year_prediction':
        cluster_alg.fit(np.concatenate([X], axis=0))
    else:
        cluster_alg.fit(X[np.random.randint(0, X.shape[0], size=(10000))])
    c = cluster_alg.cluster_centers_
    if torch.cuda.is_available() and args.cuda: 
        c = torch.tensor(c).to(torch.float32).to('cuda')
    else:
        c = torch.tensor(c).to(torch.float32)
        
    mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                               torch.nn.ReLU(),
                               torch.nn.Linear(n_neurons, 1))
    var = torch.nn.Sequential(RBF(None, None, c, 1.0),
                              torch.nn.Linear(args.n_clusters, args.n_clusters, bias=False),
                              Norm2(dim=1),
                              torch.nn.Sigmoid(),
                              OneMinusX(),
                              PosLinear(1, 1, bias=False))
    
    if torch.cuda.is_available() and args.cuda: 
        mean.cuda(); var.cuda(); 
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    optimizer = torch.optim.Adam(chain(mean.parameters(),
                                       var.parameters()), lr=args.lr)
    it = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        optimizer.zero_grad()
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        v = switch*(v+1e-4) + (1-switch)*torch.tensor([0.02**2], device=device)
        loss = normal_log_prob(label, m, v).sum()
        (-loss).backward()
        optimizer.step()
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = mean(data), var(data)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = normal_log_prob(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()    
    
#%%
def john(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    from utils import dist
    from itertools import chain
    from torch import distributions as D
    from locality_sampler import gen_Qw, locality_sampler2
    from sklearn.decomposition import PCA
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    mean_psu = 1
    mean_ssu = 40
    mean_M = 50

    var_psu = 2
    var_ssu = 10
    var_M = 10
    
    num_draws_train = 20
    kmeans = KMeans(n_clusters=args.n_clusters)
    if args.dataset != 'year_prediction':
        kmeans.fit(np.concatenate([X], axis=0))
    else:
        kmeans.fit(X[np.random.randint(0, X.shape[0], size=(10000))])
    c = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    if torch.cuda.is_available() and args.cuda: 
        c = torch.tensor(c).to(torch.float32).to('cuda')
    else:
        c = torch.tensor(c).to(torch.float32)
        
    class translatedSigmoid(torch.nn.Module):
        def __init__(self):
            super(translatedSigmoid, self).__init__()
            self.beta = torch.nn.Parameter(torch.tensor([1.5]))
            
        def forward(self, x):
            beta = torch.nn.functional.softplus(self.beta)
            alpha = -beta*(6.9077542789816375)
            return torch.sigmoid((x+alpha)/beta)
    
    class GPNNModel(torch.nn.Module):
        def __init__(self):
            super(GPNNModel, self).__init__()
            self.mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(n_neurons, 1))
            self.alph = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(n_neurons, 1),
                                      torch.nn.Softplus())
            self.bet = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(n_neurons, 1),
                                     torch.nn.Softplus())
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
                    samples_var = gamma_dist.rsample(torch.Size([num_draws_train]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([1000]))
                    x_var = (1.0/(samples_var+1e-8))
                var = (1-s) * x_var + s*torch.tensor([y_std**2], device=x.device) # HYPERPARAMETER
                
            else:
                var = 0.05*torch.ones_like(mean)
            return mean, var
    
    model = GPNNModel()
    if torch.cuda.is_available() and args.cuda: 
        model.cuda()
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')

    optimizer = torch.optim.Adam(model.mean.parameters(), lr=1e-2)
    optimizer2 = torch.optim.Adam(chain(model.alph.parameters(),
                                        model.bet.parameters(),
                                        model.trans.parameters()), lr=1e-4)
    mean_Q, mean_w = gen_Qw(X, mean_psu, mean_ssu, mean_M)
    
    if X.shape[0] > 100000 and X.shape[1] > 10:
        pca = PCA(n_components=0.5)
        temp = pca.fit_transform(X)
        var_Q, var_w = gen_Qw(temp, var_psu, var_ssu, var_M)
    else:    
        var_Q, var_w = gen_Qw(X, var_psu, var_ssu, var_M)
    
    #mean_pseupoch = get_pseupoch(mean_w,0.5)
    #var_pseupoch = get_pseupoch(var_w,0.5)
    opt_switch = 1
    mean_w = torch.tensor(mean_w).to(torch.float32).to(device)
    var_w = torch.tensor(var_w).to(torch.float32).to(device)
    model.train()
    
    X = torch.tensor(X).to(torch.float32).to(device)
    y = torch.tensor(y).to(torch.float32).to(device)
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    it = 0
    while it < args.iters:
        switch = 1.0 if it > args.iters/2.0 else 0.0
        
        if it % 11: opt_switch = opt_switch + 1 # change between var and mean optimizer
        with torch.autograd.detect_anomaly():
            data, label = next(batches)
            if not switch:
                optimizer.zero_grad();
                m, v = model(data, switch)
                loss = -t_likelihood(label.reshape(-1,1), m, v.reshape(1,-1,1)) / X.shape[0]
                loss.backward()
                optimizer.step()
            else:
                if opt_switch % 2 == 0:    
                    #for b in range(mean_pseupoch):
                    optimizer.zero_grad()
                    batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w)
                    m, v = model(X[batch], switch)
                    loss = -t_likelihood(y[batch].reshape(-1,1), m, v, mean_w[batch]) / X.shape[0]
                    loss.backward()
                    optimizer.step()
                else:
                    #for b in range(var_pseupoch):
                    optimizer2.zero_grad()
                    batch = locality_sampler2(var_psu,var_ssu,var_Q,var_w)
                    m, v = model(X[batch], switch)
                    loss = -t_likelihood(y[batch].reshape(-1,1), m, v, var_w[batch]) / X.shape[0]
                    loss.backward()
                    optimizer2.step()
            
        if it % 500 == 0: 
            m, v = model(data, switch)
            loss = -(-v.log()/2 - ((m.flatten()-label)**2).reshape(1,-1,1) / (2 * v)).mean()
            print('Iter {0}/{1}, Loss {2}'.format(it, args.iters, loss.item()))
        it+=1
        
    model.eval()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    with torch.no_grad():
        m, v = model(data, switch)
    m = m * y_std + y_mean
    v = v * y_std**2
    #log_px = normal_log_prob(label, m, v).mean(dim=0) # check for correctness
    log_px = t_likelihood(label.reshape(-1,1), m, v) / Xval.shape[0]# check
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()
    
#%%
if __name__ == '__main__':
    args = argparser() # get input arguments
    if args.silent:
        import warnings
        warnings.filterwarnings("ignore")
    print("==================== Training model {0} on dataset {1} ====================".format(
            args.model, args.dataset))
    
    # Load data
    dataset = np.load('data/regression_datasets/' + args.dataset + '.npz')
    X, y = dataset['data'], dataset['target']
    
    log_score, rmse_score = [ ], [ ]
    # Train multiple models
    T = timer()
    for i in range(args.repeats):
        print("==================== Model {0}/{1} ====================".format(i+1, args.repeats))
        # Make train/test split
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                        test_size=args.test_size,
                                                        random_state=(i+1)*args.seed)
        
        # Normalize data
        scaler = preprocessing.StandardScaler()
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
        
        # Fit and score model
        T.begin()
        try:
            logpx, rmse = eval(args.model)(args, Xtrain, ytrain, Xtest, ytest)
        except Exception as e:
            print('encountered error:', e)
            logpx, rmse = np.nan, np.nan
        T.end()
        log_score.append(logpx); rmse_score.append(rmse)
        
    log_score = np.array(log_score)
    rmse_score = np.array(rmse_score)
    
    # Save results
    if not 'results/regression_results' in os.listdir():
        os.makedirs('results/regression_results/', exist_ok=True)
    np.savez('results/regression_results/' + args.dataset + '_' + args.model, 
             log_score=log_score,
             rmse_score=rmse_score,
             timings=np.array(T.timings))
    
    # Print the results
    print('log(px): {0:.3f} +- {1:.3f}'.format(log_score.mean(), log_score.std()))
    print('rmse:    {0:.3f} +- {1:.3f}'.format(rmse_score.mean(), rmse_score.std()))
    T.res()