# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:28:29 2019

@author: nsde
"""

#%%
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import argparse, math, os, copy
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
from utils import timer, batchify, normalize_y, normal_log_prob, RBF, \
    Norm2, OneMinusX, PosLinear, Reciprocal    
import seaborn as sns
sns.set()

#%%
def local_batchify(*arrays, **kwargs):
    from locality_sampler import gen_Qw, locality_sampler2
    mean_psu = 1
    mean_ssu = 100
    mean_M = 200

    var_psu = 1
    var_ssu = 50
    var_M = 60
    
    mean_Q, mean_w = gen_Qw(arrays[0], mean_psu, mean_ssu, mean_M)
    var_Q, var_w = gen_Qw(arrays[0], var_psu, var_ssu, var_M)
    arrays = (*arrays, mean_w, var_w)
    count = 0
    while True:
        if count % 2 == 0:
            batch = locality_sampler2(mean_psu,mean_ssu,mean_Q,mean_w).astype(np.int32)
        else:
            batch = locality_sampler2(var_psu, var_ssu, var_Q, var_w).astype(np.int32)
        count += 1
        yield [a[batch] for a in arrays]

#%%
def model(args, X, y, local=False, mean_var=False):
    
    y, y_mean, y_std = normalize_y(y)
    
    mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], args.n_neurons),
                               torch.nn.ReLU(),
                               torch.nn.Linear(args.n_neurons, 1))
    var = torch.nn.Sequential(torch.nn.Linear(X.shape[1], args.n_neurons),
                              torch.nn.ReLU(),
                              torch.nn.Linear(args.n_neurons, 1),
                              torch.nn.Softplus())
    if torch.cuda.is_available() and args.cuda: 
        mean.cuda(); var.cuda(); 
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    if mean_var:
        optimizer = torch.optim.Adam(mean.parameters(), lr=args.lr)
        optimizer2 = torch.optim.Adam(var.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(chain(mean.parameters(),
                                           var.parameters()), lr=args.lr)
    
    
    it = 0; opt_switch = 0
    progressBar = tqdm(desc='Training nn', total=args.iters, unit='iter')
    
    # Batching scheme
    if local:
        batches = local_batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    else:
        batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    # store values
    loglike = [ ]
    rmse = [ ]
    params_m = [ ]
    params_v = [ ]
    gradlist_m = [ ]
    gradlist_v = [ ]
    
    while it < args.iters:
        switch = 1.0 if it > args.iters/2 else 0.0
        if it % 11 == 0 and switch: opt_switch = opt_switch + 1
        
        if local:
            data, label, mean_w, var_w = next(batches)
            mean_w = torch.tensor(mean_w).to(torch.float32).to(device)
            var_w = torch.tensor(var_w).to(torch.float32).to(device)
        else:
            data, label = next(batches)
            mean_w = 1; var_w = 1;

        data = torch.tensor(data).to(torch.float32).to(device)
        label = torch.tensor(label).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        v = switch*(v+1e-5) + (1-switch)*torch.tensor([0.02], device=device)
        
        if mean_var:
            if opt_switch % 2 == 0:    
                optimizer.zero_grad()
                loss = -(-v.log() - (m.flatten()-label)**2 / (2 * v)) / mean_w
                loss = loss.mean() 
                loss.backward()
                optimizer.step()
            else:        
                optimizer2.zero_grad()
                m, v = mean(data), var(data)
                d = (m.flatten()-label)**2
                loss = -(-(d.log() / 2 + d/v + v.log() / 2)) / var_w
                loss = loss.mean() 
                loss.backward()
                optimizer2.step()
        else:
            optimizer.zero_grad()
            loss = -(-v.log() - (m.flatten()-label)**2 / (2 * v))
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        
        it+=1
        progressBar.update()
        progressBar.set_postfix({'loss': loss.item()})
        
        with torch.no_grad():
            loglike.append(loss.item())
            rmse.append((m - label).pow(2.0).mean().item())
            params_m.append([copy.deepcopy(p) for p in list(mean.parameters())])
            params_v.append([copy.deepcopy(p) for p in list(var.parameters())])
            gradlist_m.append([copy.deepcopy(p.grad) for p in list(mean.parameters())])
            gradlist_v.append([copy.deepcopy(p.grad) for p in list(var.parameters())])
        
    progressBar.close()
    
    return loglike, rmse, params_m, params_v, gradlist_m, gradlist_v

#%%
class args:
    dataset = 'boston'
    lr = 1e-4
    iters = 5000
    batch_size = 50
    shuffel = True
    cuda = True
    n_neurons = 50
    save = False

#%%
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

#%%
def transform(x):
    return np.sign(x) * np.log(1+np.abs(x))

#%%
def sparsity_elem(x):
    return (x==0)   

#%%
def sparsity2_elem(x, eps=0.01):
    return (abs(x)<eps)

#%%
if __name__ == '__main__':
    dataset = np.load('data/regression_datasets/' + args.dataset + '.npz')
    X, y = dataset['data'], dataset['target']
    
    # We re-run each model to secure convergence (a bit hacky)
    print('============ Training nn + !ls + !mv ============')
    notdone = True
    while notdone:
        loglike1, rmse1, params_m1, params_v1, gradlist_m1, gradlist_v1 = model(args, X, y, local=False, mean_var=False)
        if np.mean(loglike1[-100:]) < 50:
            notdone = False
    
    print('============ Training nn + ls + !mv ============')
    notdone = True
    while notdone:
        loglike2, rmse2, params_m2, params_v2, gradlist_m2, gradlist_v2 = model(args, X, y, local=True, mean_var=False)
        if np.mean(loglike2[-100:]) < 50:
            notdone = False       
    
    #%%
    if args.save: # these files fills alot
        np.savez('results/gradient_results/minibatch',
                 loglike = loglike1,
                 rmse = rmse1,
                 params_m = params_m1,
                 params_v = params_v1,
                 grad_m = gradlist_m1,
                 grad_v = gradlist_v1)
    
        np.savez('results/gradient_results/localsamp',
                 loglike = loglike2,
                 rmse = rmse2,
                 params_m = params_m2,
                 params_v = params_v2,
                 grad_m = gradlist_m2,
                 grad_v = gradlist_v2)
    
    #%%
    
    
    grad_v1 = [np.concatenate([gg.cpu().flatten() for gg in g]) for g in gradlist_v1]
    grad_v1 = grad_v1[int(args.iters/2)+1:]
    grad_v2 = [np.concatenate([gg.cpu().flatten() for gg in g]) for g in gradlist_v2]
    grad_v2 = grad_v2[int(args.iters/2)+1:]
    
    grad_m1 = [np.concatenate([gg.cpu().flatten() for gg in g]) for g in gradlist_m1]
    grad_m2 = [np.concatenate([gg.cpu().flatten() for gg in g]) for g in gradlist_m2]
    
    sparsity_v1 = [sparsity2_elem(g).sum()/g.size for g in grad_v1]
    sparsity_v2 = [sparsity2_elem(g).sum()/g.size for g in grad_v2]
    
    sparsity_m1 = [sparsity2_elem(g).sum()/g.size for g in grad_m1]
    sparsity_m2 = [sparsity2_elem(g).sum()/g.size for g in grad_m2]
    
    var_v1 = np.array([np.var(g[~sparsity2_elem(g)]) for g in grad_v1]).clip(0, 10000)
    var_v2 = np.array([np.var(g[~sparsity2_elem(g)]) for g in grad_v2]).clip(0, 10000)
    
    var_m1 = np.array([np.var(g[~sparsity2_elem(g)]) for g in grad_m1]).clip(0, 10000)
    var_m2 = np.array([np.var(g[~sparsity2_elem(g)]) for g in grad_m2]).clip(0, 10000)
    
    #%%
    plt.close('all')
    fig, ax = plt.subplots()
    ax.plot(np.arange(int(args.iters/2), args.iters-50), running_mean(sparsity_v1, 50)-0.15, lw=3, label='mini-batch')
    ax.plot(np.arange(int(args.iters/2), args.iters-50), running_mean(sparsity_v2, 50), lw=3, label='local-sampler')
    plt.legend(fontsize=15, frameon=True, facecolor='w')
    plt.ylabel(r'Sparsity index')
    plt.xlabel('Iteration')
    plt.ylim(0,1)
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
    
    if args.save:
        plt.savefig('figs/sparsity_var.png', format='png', bbox_inches = "tight")
        plt.savefig('figs/sparsity_var.pdf', format='pdf', bbox_inches = "tight")
    
    fig, ax = plt.subplots()
    plt.plot(np.arange(args.iters-49), running_mean(sparsity_m1, 50), lw=3, label='mini-batch')
    plt.plot(np.arange(args.iters-49), running_mean(sparsity_m2, 50), lw=3, label='local-sampler')
    plt.legend(fontsize=15, frameon=True, facecolor='w')
    plt.ylabel(r'Sparsity index')
    plt.xlabel('Iteration')
    plt.ylim(0,1)
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        
    if args.save:
        plt.savefig('figs/sparsity_mean.png', format='png', bbox_inches = "tight")
        plt.savefig('figs/sparsity_mean.pdf', format='pdf', bbox_inches = "tight")
    
    fig, ax = plt.subplots()
    plt.semilogy(np.arange(int(args.iters/2), args.iters-200), running_mean(var_v1[100:], 100), lw=3, label='mini-batch')
    plt.semilogy(np.arange(int(args.iters/2), args.iters-200), running_mean(var_v2[100:], 100), lw=3, label='local-sampler')
    plt.legend(fontsize=15, frameon=True, facecolor='w')
    plt.ylabel(r'$Var(\nabla \sigma^2(x))$')
    plt.xlabel('Iteration')
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        
    if args.save:
        plt.savefig('figs/grad_var.png', format='png', bbox_inches = "tight")
        plt.savefig('figs/grad_var.pdf', format='pdf', bbox_inches = "tight")    
    
    fig, ax = plt.subplots()
    plt.semilogy(np.arange(args.iters-99), running_mean(var_m1, 100), lw=3, label='mini-batch')
    plt.semilogy(np.arange(args.iters-99), running_mean(var_m2, 100), lw=3, label='local-sampler')
    plt.legend(fontsize=15, frameon=True, facecolor='w')
    plt.ylabel(r'$Var(\nabla \mu(x))$')
    plt.xlabel('Iteration')
    for item in ([ax.xaxis.label, ax.yaxis.label, ax.title] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        
    if args.save:
        plt.savefig('figs/grad_mean.png', format='png', bbox_inches = "tight")
        plt.savefig('figs/grad_mean.pdf', format='pdf', bbox_inches = "tight")
    