#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 08:11:10 2019

@author: nsde
"""
#%%
import torch
from torch import nn
from torch import distributions as D
import numpy as np
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from utils import batchify, dist, translatedSigmoid, RBF, PosLinear, Reciprocal
from itertools import chain
from locality_sampler import gen_Qw, locality_sampler, get_pseupoch, local_batchify
from sklearn.cluster import KMeans
sns.set()

#%%
n_neurons = 50

#%%
class basemodel(nn.Module):
    def __init__(self, latent_size=2, cuda=True):
        super(basemodel, self).__init__()
        self.switch = 0.0
        self.latent_size = latent_size
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        
    def encoder(self, x):
        return self.enc_mu(x), self.enc_var(x)

    def decoder(self, z):
        x_mu, x_var = self.dec_mu(z), self.dec_var(z)
        x_var = self.switch * x_var + (1-self.switch)*torch.tensor([0.02**2], device=z.device)
        return x_mu, x_var

    def sample(self, N):
        z = torch.randn(N, self.latent_size, device=self.device)
        x_mu, x_var = self.decoder(z)
        return x_mu, x_var
    
    def forward(self, x, beta=1.0, epsilon=1e-5):
        z_mu, z_var = self.encoder(x)
        q_dist = D.Independent(D.Normal(z_mu, z_var.sqrt()+epsilon), 1)
        z = q_dist.rsample()
        x_mu, x_var = self.decoder(z)
        p_dist = D.Independent(D.Normal(x_mu, x_var.sqrt()+epsilon), 1)
        
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        log_px = p_dist.log_prob(x)
        kl = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = log_px - beta*kl
        return elbo.mean(), log_px.mean(), kl.mean(), x_mu, x_var, z, z_mu, z_var
    
    def evaluate(self, X, L=1000):
        with torch.no_grad():
            x_mu, x_var = self.sample(L)
            parzen_dist = D.Independent(D.Normal(x_mu, x_var.sqrt()), 1)
            elbolist, logpxlist, parzen_score = [ ], [ ], [ ]
            for x in tqdm(X, desc='evaluating', unit='samples'):
                x = torch.tensor(x.reshape(1, -1), device=self.device)
                elbo, logpx, kl, _, _, _, _, _ = self.forward(x)
                elbolist.append(elbo.item())
                logpxlist.append(logpx.item())
                parzen_score.append(parzen_dist.log_prob(x).mean().item())
            
            return np.array(elbolist), np.array(logpxlist), np.array(parzen_score)
        
#%%
class vae(basemodel):
    def __init__(self, latent_size=2, cuda=True):
        super(vae, self).__init__(latent_size, cuda)
        
        self.enc_mu = nn.Sequential(nn.Linear(4, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, self.latent_size))
        self.enc_var = nn.Sequential(nn.Linear(4, n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, self.latent_size),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, 4))
        self.dec_var = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, 4),
                                     nn.Softplus())
    
    def fit(self, Xtrain, n_iters=100, lr=1e-3, batch_size=256, beta=1.0):
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        it = 0
        batches = batchify(Xtrain, batch_size = batch_size, shuffel=True)
        progressBar = tqdm(desc='training', total=n_iters, unit='iter')
        loss, var = [ ], [ ]
        while it < n_iters:
            optimizer.zero_grad()
            
            self.switch = 1.0 if it > n_iters/2 else 0.0
            anneling = np.minimum(1, it/n_iters)*beta
            
            x = torch.tensor(next(batches)[0], device=self.device)
            elbo, log_px, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
            
            (-elbo).backward()
            optimizer.step()
            
            progressBar.update()
            progressBar.set_postfix({'elbo': (-elbo).item()})
            loss.append((-elbo).item())
            var.append(x_var.mean().item())
            it+=1
        progressBar.close()
        return loss, var

#%%
class rbf(basemodel):
    def __init__(self, latent_size=2, cuda=True):
        super(rbf, self).__init__(latent_size, cuda)
        
        self.enc_mu = nn.Sequential(nn.Linear(4, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, self.latent_size))
        self.enc_var = nn.Sequential(nn.Linear(4, n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, self.latent_size),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, 4))
        self.dec_var = nn.Sequential(RBF(self.latent_size, 50, None, 5),
                                     PosLinear(50, 1, bias=False),
                                     Reciprocal(0.1),
                                     PosLinear(1, 1, bias=False))
        
    def fit(self, Xtrain, n_iters=100, lr=1e-3, batch_size=256, n_clusters=50, beta=1.0):
        self.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        it = 0
        batches = batchify(Xtrain, batch_size = batch_size, shuffel=True)
        progressBar = tqdm(desc='training', total=n_iters, unit='iter')
        loss, var = [ ], [ ]
        while it < n_iters:
            optimizer.zero_grad()
            
            self.switch = 1.0 if it > n_iters/2 else 0.0
            anneling = np.minimum(1, it/n_iters)*beta
            if self.switch and (it % 100 == 0 or not hasattr(self, "C")):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(self.encoder(torch.tensor(Xtrain).to(self.device))[0].detach().numpy())
                self.C = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                rbf = list(self.dec_var.children())[0]
                rbf.points = nn.Parameter(self.C)
                
            x = torch.tensor(next(batches)[0], device=self.device)
            elbo, log_px, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
            
            (-elbo).backward()
            optimizer.step()
            
            progressBar.update()
            progressBar.set_postfix({'elbo': (-elbo).item()})
            loss.append((-elbo).item())
            var.append(x_var.mean().item())
            it+=1
        progressBar.close()
        return loss, var

#%%
class john(basemodel):
    def __init__(self, latent_size=2, cuda=True):
        super(john, self).__init__(latent_size, cuda)
        self.opt_switch = 0
        
        self.enc_mu = nn.Sequential(nn.Linear(4, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, self.latent_size))
        self.enc_var = nn.Sequential(nn.Linear(4, n_neurons),
                                     nn.ReLU(),
                                     nn.Linear(n_neurons, self.latent_size),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                    nn.ReLU(),
                                    nn.Linear(n_neurons, 4))
        self.alpha = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                   nn.ReLU(),
                                   nn.Linear(n_neurons, 4),
                                   nn.Softplus())
        self.beta = nn.Sequential(nn.Linear(self.latent_size, n_neurons),
                                  nn.Sigmoid(),
                                  nn.Linear(n_neurons, 4),
                                  nn.Softplus())
        
    def decoder(self, z):
        x_mu = self.dec_mu(z)
        if self.switch:
            d = dist(z, self.C)
            d_min = d.min(dim=1, keepdim=True)[0]
            s = translatedSigmoid(d_min, -6.907*0.03, 0.03)
            alpha = self.alpha(z)
            beta = self.beta(z)
            gamma_dist = D.Gamma(alpha+1e-6, beta+1e-6)
            if self.training:
                samples_var = gamma_dist.rsample([100])
                x_var = (1.0/(samples_var+1e-6))
            else:
                samples_var = gamma_dist.rsample([1000])
                x_var = (1.0/(samples_var+1e-6)).mean(dim=0)
            x_var = (1-s) * x_var + s*(10*torch.ones_like(x_var))
        else:
            x_var = (0.02**2)*torch.ones_like(x_mu)
            
        return x_mu, x_var        
    
    def fit(self, Xtrain, n_iters=100, lr=1e-3, batch_size=250, n_clusters=50, beta=1.0):
        self.train()
        
        optimizer1 = torch.optim.Adam(chain(self.enc_mu.parameters(),
                                            self.enc_var.parameters(),
                                            self.dec_mu.parameters()),
                                      lr=lr)
        optimizer2 = torch.optim.Adam(chain(self.enc_mu.parameters(),
                                            self.enc_var.parameters(),
                                            self.dec_mu.parameters()),
                                      lr=lr)
        optimizer3 = torch.optim.Adam(chain(self.alpha.parameters(),
                                            self.beta.parameters()),
                                      lr=lr)
            
        it = 0
        batches = local_batchify(Xtrain)
        progressBar = tqdm(desc='training', total=n_iters, unit='iter')
        loss, var = [ ], [ ]
        while it < n_iters:
            self.switch = 1.0 if it > n_iters/2 else 0.0
            anneling = np.minimum(1, it/n_iters)*beta
            self.opt_switch = (self.opt_switch+1) if (it % 100 == 0 and self.switch) else self.opt_switch
            if self.switch and (it % 100 == 0 or not hasattr(self, "C")):
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans.fit(self.encoder(torch.tensor(Xtrain).to(self.device))[0].detach().numpy())
                self.C = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            
            x, mean_w, var_w = next(batches)
            x = torch.tensor(x).to(torch.float32).to(self.device)
            mean_w = torch.tensor(mean_w).to(torch.float32).to(self.device)
            var_w = torch.tensor(var_w).to(torch.float32).to(self.device)
            
            if not self.switch:
                optimizer1.zero_grad()
                elbo, _, _, _, x_var, _, _, _ = self.forward(x, anneling)
                (-elbo).backward()
                optimizer1.step()
            else:
                _, logpx, kl, x_mu, x_var, z, z_mu, z_var = self.forward(x, anneling)
                if self.opt_switch % 2 == 0:
                    optimizer2.zero_grad()
                    elbo = logpx - kl
                    (-elbo).backward()
                    optimizer2.step()
                else:
                    optimizer3.zero_grad()
                    elbo = logpx - kl
                    (-elbo).backward()
                    optimizer3.step()
                
                elbo, _, _, _, _, _, _, _, = self.forward(x, anneling)
                
            progressBar.update()
            progressBar.set_postfix({'elbo': (-elbo).item()})
            loss.append((-elbo).item())
            var.append(x_var.mean().item())
            it+=1
          
        progressBar.close()
        return loss, var  

#%%
if __name__ == '__main__':
    modelname = 'john'
    
    Xtrain = np.zeros((500, 4))
    with open('data/data_v2.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for i, row in enumerate(reader):
            if i!=0:
                Xtrain[i-1] = [float(r) for r in row[1:]]
    Xval = np.zeros((500, 4))
    with open('data/data_v2_eval.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for i, row in enumerate(reader):
            if i!=0:
                Xval[i-1] = [float(r) for r in row[1:]]
    Xtrain = Xtrain.astype('float32')
    Xval = Xval.astype('float32')
 
    use_cuda = False
    if modelname == 'vae':
        model = vae(cuda=use_cuda)
    elif modelname == 'rbf':
        model = rbf(cuda=use_cuda)
    elif modelname == 'john':
        model = john(cuda=use_cuda)
        
    if use_cuda:
        model.cuda();
    
    loss, var = model.fit(Xtrain, n_iters=10000, lr=1e-3, beta=1.0)
    model.eval()
      
    _, _, _, x_mu, x_var, z, z_mu, z_var = model.forward(torch.tensor(Xtrain).to(model.device))
    x_samp = x_mu + x_var * torch.randn_like(x_var)
    
    #%%
    def savefig(c):
        plt.savefig('figs/2m4d_' + modelname + '_' + str(c) + '.pdf', format='pdf')
        plt.savefig('figs/2m4d_' + modelname + '_' + str(c) + '.png', format='png')
    
    def set_axes(fig):
        for i in range(4):
            for j in range(4):
                if j>i:
                    fig.axes[i,j].axis([
                        Xtrain[:,j].min()-0.5, Xtrain[:,j].max()+0.5,
                        Xtrain[:,i].min()-0.5, Xtrain[:,i].max()+0.5
                        ])
                if i>j:
                    fig.axes[i,j].axis([
                        Xtrain[:,j].min()-0.5, Xtrain[:,j].max()+0.5,
                        Xtrain[:,i].min()-0.5, Xtrain[:,i].max()+0.5
                        ])
    
    # plot
    plt.close('all')
  
    fig1 = sns.PairGrid(pd.DataFrame(Xtrain))
    fig1 = fig1.map_upper(plt.scatter, edgecolor="w")
    fig1 = fig1.map_lower(sns.kdeplot, cmap="Blues_d")
    fig1 = fig1.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig1)
    #savefig(1)
      
    fig2 = sns.PairGrid(pd.DataFrame(x_samp.detach().numpy()))
    fig2 = fig2.map_upper(plt.scatter, edgecolor="w")
    fig2 = fig2.map_lower(sns.kdeplot, cmap="Blues_d")
    fig2 = fig2.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig2)
    #savefig(2)
    
    fig3 = plt.figure()
    plt.plot(loss, lw=2)
    plt.xlabel('iter')
    plt.ylabel('elbo')
    #savefig(3)
    
    fig4 = plt.figure()
    sns.scatterplot(z[:,0].detach(), z[:,1].detach())
    if hasattr(model, "C"):
        sns.scatterplot(model.C.detach()[:,0], model.C.detach()[:,1])
    plt.axis([-4,4,-4,4])
    #savefig(4)    
    
    fig5 = plt.figure()
    plt.plot(var, lw=2)
    plt.xlabel('iter')
    plt.ylabel('mean var')
    #savefig(5)
    
    fig6 = plt.figure()
    grid = np.stack([m.flatten() for m in np.meshgrid(np.linspace(-4,4,200), np.linspace(4,-4,200))]).T.astype('float32')
    _, x_var = model.decoder(torch.tensor(grid).to(model.device))
    plt.imshow(x_var.detach().sum(dim=1).reshape(200,200), extent=[-4, 4, -4, 4])
    plt.colorbar()
    sns.scatterplot(z[:,0].detach(), z[:,1].detach())
    plt.axis('off')
    savefig(8)
    
    x_mu, x_var = model.sample(1000)
    x_samp = x_mu + x_var * torch.randn_like(x_var)
    x_samp = x_samp[x_var.sum(dim=1) < 10]
    fig7 = sns.PairGrid(pd.DataFrame(x_samp.detach().numpy()))
    fig7 = fig7.map_upper(plt.scatter, edgecolor="w")
    fig7 = fig7.map_lower(sns.kdeplot, cmap="Blues_d")
    fig7 = fig7.map_diag(sns.kdeplot, lw=3, legend=False)
    set_axes(fig7)
    #savefig(7)
    
    