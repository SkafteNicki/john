#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 10:35:43 2019

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
    Norm2, OneMinusX, PosLinear, Reciprocal, normal_log_prob_w_prior , t_likelihood

#%%
def argparser():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    gs = parser.add_argument_group('General settings')
    gs.add_argument('--model', type=str, default='gp', help='model to use',
                    choices=['nn', 'nnls', 'nnmv', 'nnlsmv', 'jn', 'jnls', 'jnmv', 'jnlsmv'])
    gs.add_argument('--dataset', type=str, default='boston', help='dataset to use')
    gs.add_argument('--seed', type=int, default=1, help='random state of data-split')
    gs.add_argument('--test_size', type=float, default=0.1, help='test set size, as a procentage')
    gs.add_argument('--repeats', type=int, default=20, help='number of repeatitions')
    gs.add_argument('--silent', type=bool, default=True, help='suppress warnings')
    gs.add_argument('--cuda', type=bool, default=False, help='use cuda')
    
    ms = parser.add_argument_group('Model specific settings')
    ms.add_argument('--batch_size', type=int, default=512, help='batch size')
    ms.add_argument('--shuffel', type=bool, default=True, help='shuffel data during training')
    ms.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ms.add_argument('--iters', type=int, default=10000, help='number of iterations')
    ms.add_argument('--mcmc', type=int, default=100, help='number of mcmc samples')
    ms.add_argument('--inducing', type=int, default=500, help='number of inducing points')
    ms.add_argument('--n_clusters', type=int, default=500, help='number of cluster centers')
    ms.add_argument('--n_models', type=int, default=5, help='number of ensemble')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
def local_batchify(*arrays, **kwargs):
    from locality_sampler import gen_Qw, locality_sampler, locality_sampler2
    mean_psu = 1
    mean_ssu = 100
    mean_M = 150

    var_psu = 3
    var_ssu = 7
    var_M = 15
    
    mean_Q, mean_w = gen_Qw(arrays[0], mean_psu, mean_ssu, mean_M)
    var_Q, var_w = gen_Qw(arrays[0], var_psu, var_ssu, var_M)
    arrays = (*arrays, mean_w, var_w)
    while True:
        batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w).astype(np.int32)
        yield [a[batch] for a in arrays]

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
def nnls(args, X, y, Xval, yval):
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
    batches = local_batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        optimizer.zero_grad()
        data, label, _, _ = next(batches)
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
def nnmv(args, X, y, Xval, yval):
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
    
    optimizer = torch.optim.Adam(mean.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(var.parameters(), lr=args.lr)
                                       
    it = 0; opt_switch = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        if it % 11 == 0 and switch: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        
        if opt_switch % 2 == 0:    
            optimizer.zero_grad()
            m, v = mean(data), var(data)
            v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
            loss = -(-v.log() - (m.flatten()-label)**2 / (2 * v))
            loss = loss.sum() 
            loss.backward()
            optimizer.step()
        else:
            optimizer2.zero_grad()
            m, v = mean(data), var(data)
            loss = -(-v.log() - (m.flatten()-label)**2 / (2 * v))
            loss = loss.sum() 
            loss.backward()
            optimizer2.step()

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
def nnlsmv(args, X, y, Xval, yval):
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
    
    optimizer = torch.optim.Adam(mean.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(var.parameters(), lr=args.lr)
                                       
    it = 0; opt_switch = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = local_batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        if it % 11 == 0 and switch: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        data, label, mean_w, var_w = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        mean_w = torch.tensor(mean_w).to(torch.float32).to(device)
        var_w = torch.tensor(var_w).to(torch.float32).to(device)
        
        if opt_switch % 2 == 0:    
            optimizer.zero_grad()
            m, v = mean(data), var(data)
            v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
            loss = -(-v.log() - (m.flatten()-label)**2 / (2 * v)) / mean_w
            loss = loss.sum() 
            loss.backward()
            optimizer.step()
        else:
            optimizer2.zero_grad()
            m, v = mean(data), var(data)
            loss = -(-v.log() - (m.flatten()-label)**2 / (2 * v)) / var_w
            loss = loss.sum() 
            loss.backward()
            optimizer2.step()

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
def jn(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    from utils import dist
    from torch import distributions as D
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    kmeans = KMeans(n_clusters=args.n_clusters)
    kmeans.fit(np.concatenate([X], axis=0))
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
                                      torch.nn.Sigmoid(),
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
                    samples_var = gamma_dist.rsample(torch.Size([20]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([1000]))
                    x_var = (1.0/(samples_var+1e-8)).mean(dim=0)
                var = (1-s) * x_var + s*torch.tensor([3.5**2], device=x.device) # HYPERPARAMETER
                
            else:
                var = torch.tensor([0.05], device=x.device)
            return mean, var
    
    model = GPNNModel()
    if torch.cuda.is_available() and args.cuda: 
        model.cuda()
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    it = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        optimizer.zero_grad()
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m, v = model(data, switch)
        v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
        loss = (-(-v.log()/2 - ((m.flatten()-label)**2).reshape(1,-1,1) / (2 * v))).sum()
        loss.backward()
        optimizer.step()
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = model(data, switch)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = normal_log_prob_w_prior(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()

#%%
def jnls(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    from utils import dist
    from torch import distributions as D
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    kmeans = KMeans(n_clusters=args.n_clusters)
    kmeans.fit(np.concatenate([X], axis=0))
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
                                      torch.nn.Sigmoid(),
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
                    samples_var = gamma_dist.rsample(torch.Size([20]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([1000]))
                    x_var = (1.0/(samples_var+1e-8)).mean(dim=0)
                var = (1-s) * x_var + s*torch.tensor([3.5**2], device=x.device) # HYPERPARAMETER
                
            else:
                var = torch.tensor([0.05], device=x.device)
            return mean, var
    
    model = GPNNModel()
    if torch.cuda.is_available() and args.cuda: 
        model.cuda()
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    it = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = local_batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        optimizer.zero_grad()
        data, label, _, _ = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m, v = model(data, switch)
        m = m.reshape(1,-1,1)
        v = switch*v + (1-switch)*torch.tensor([0.02**2], device=device)
        loss = (-(-v.log()/2 - ((m.flatten()-label)**2).reshape(1,-1,1) / (2 * v))).sum()
        loss.backward()
        optimizer.step()
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = model(data, switch)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = normal_log_prob_w_prior(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()

#%%
def jnmv(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    from utils import dist
    from torch import distributions as D
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    kmeans = KMeans(n_clusters=args.n_clusters)
    kmeans.fit(np.concatenate([X], axis=0))
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
                                      torch.nn.Sigmoid(),
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
                    samples_var = gamma_dist.rsample(torch.Size([20]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([1000]))
                    x_var = (1.0/(samples_var+1e-8)).mean(dim=0)
                var = (1-s) * x_var + s*torch.tensor([3.5**2], device=x.device) # HYPERPARAMETER
                
            else:
                var = torch.tensor([0.05], device=x.device)
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
                                       
    it = 0; opt_switch = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        if it % 11 == 0 and switch: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        data, label = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        
        if opt_switch % 2 == 0:    
            optimizer.zero_grad()
            m, v = model(data, switch)
            loss = -(-v.log()/2 - ((m.flatten()-label)**2).reshape(1,-1,1) / (2 * v))
            loss = loss.sum() 
            loss.backward()
            optimizer.step()
        else:
            optimizer2.zero_grad()
            m, v = model(data, switch)
            loss = -(-v.log() - ((m.flatten()-label)**2).reshape(1,-1,1) / (2 * v))
            loss = loss.sum() 
            loss.backward()
            optimizer2.step()

        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = model(data, switch)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = normal_log_prob_w_prior(label, m, v)
    rmse = ((label - m.flatten())**2).mean().sqrt()
    return log_px.mean().item(), rmse.item()

#%%
def jnlsmv(args, X, y, Xval, yval):
    from sklearn.cluster import KMeans
    from utils import dist
    from torch import distributions as D
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    kmeans = KMeans(n_clusters=args.n_clusters)
    kmeans.fit(np.concatenate([X], axis=0))
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
                                      torch.nn.Sigmoid(),
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
                    samples_var = gamma_dist.rsample(torch.Size([20]))
                    x_var = (1.0/(samples_var+1e-8))
                else:
                    samples_var = gamma_dist.rsample(torch.Size([1000]))
                    x_var = (1.0/(samples_var+1e-8)).mean(dim=0)
                var = (1-s) * x_var + s*torch.tensor([3.5**2], device=x.device) # HYPERPARAMETER
                
            else:
                var = torch.tensor([0.05], device=x.device)
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
                                       
    it = 0; opt_switch = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    batches = local_batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        if it % 11 == 0 and switch: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        data, label, mean_w, var_w = next(batches)
        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        mean_w = torch.tensor(mean_w).to(torch.float32).to(device)
        var_w = torch.tensor(var_w).to(torch.float32).to(device)
        
        if opt_switch % 2 == 0:    
            #for b in range(mean_pseupoch):
            optimizer.zero_grad()
            #batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w)
            m, v = model(data, switch)
            loss = -t_likelihood(label.reshape(-1,1), m, v, mean_w) / X.shape[0]
            loss.backward()
            optimizer.step()
        else:
            #for b in range(var_pseupoch):
            optimizer2.zero_grad()
            #batch = locality_sampler2(var_psu,var_ssu,var_Q,var_w)
            m, v = model(data, switch)
            loss = -t_likelihood(label.reshape(-1,1), m, v, var_w) / X.shape[0]
            loss.backward()
            optimizer2.step()

        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
    progressBar.close()
    
    data = torch.tensor(Xval).to(torch.float32).to(device)
    label = torch.tensor(yval).to(torch.float32).to(device)
    m, v = model(data, switch)
    m = m * y_std + y_mean
    v = v * y_std**2
    log_px = t_likelihood(label.reshape(-1,1), m, v)
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
        logpx, rmse = eval(args.model)(args, Xtrain, ytrain, Xtest, ytest)
        T.end()
        log_score.append(logpx); rmse_score.append(rmse)
        
    log_score = np.array(log_score)
    rmse_score = np.array(rmse_score)
    
    # Save results
    if not 'results/contrib_results' in os.listdir():
        os.makedirs('results/contrib_results/', exist_ok=True)
    np.savez('results/contrib_results/' + args.dataset + '_' + args.model, 
             log_score=log_score,
             rmse_score=rmse_score,
             timings=np.array(T.timings))
    
    # Print the results
    print('log(px): {0:.3f} +- {1:.3f}'.format(log_score.mean(), log_score.std()))
    print('rmse:    {0:.3f} +- {1:.3f}'.format(rmse_score.mean(), rmse_score.std()))
    T.res()
