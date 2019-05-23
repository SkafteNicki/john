#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:29:39 2019

@author: nsde
"""

#%%
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import argparse, math, os, time
from tqdm import tqdm
from itertools import chain
from utils import timer, batchify, normalize_y, normal_log_prob, RBF, \
    Norm2, OneMinusX, PosLinear, Reciprocal, normal_log_prob_w_prior, t_likelihood

#%%
def argparser():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    gs = parser.add_argument_group('General settings')
    gs.add_argument('--model', type=str, default='john', help='model to use')
    gs.add_argument('--dataset', type=str, default='boston', help='dataset to use')
    gs.add_argument('--seed', type=int, default=1, help='random state of data-split')
    gs.add_argument('--repeats', type=int, default=10, help='number of repeatitions')
    gs.add_argument('--silent', type=bool, default=True, help='suppress warnings')
    gs.add_argument('--cuda', type=bool, default=True, help='use cuda')
    gs.add_argument('--sample_size', type=float, default=0.01, help='fraction of pool to add after each iteration')
    gs.add_argument('--al_iters', type=int, default=10, help='number of AL iterations')
    
    ms = parser.add_argument_group('Model specific settings')
    ms.add_argument('--batch_size', type=int, default=512, help='batch size')
    ms.add_argument('--shuffel', type=bool, default=True, help='shuffel data during training')
    ms.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    ms.add_argument('--iters', type=int, default=1000, help='number of iterations')
    ms.add_argument('--mcmc', type=int, default=100, help='number of mcmc samples')
    ms.add_argument('--inducing', type=int, default=500, help='number of inducing points')
    ms.add_argument('--n_clusters', type=int, default=500, help='number of cluster centers')
    ms.add_argument('--n_models', type=int, default=5, help='number of ensemble')
    
    # Parse and return
    args = parser.parse_args()
    return args

#%%
def gp(args, X, y, Xpool, ypool, Xtest, ytest):
    if X.shape[0] > 2000: # do not run gp for large datasets
        return np.nan, np.nan, np.nan, np.nan
    import GPy
    d = X.shape[1]
    kernel = GPy.kern.RBF(d, ARD=True)
    model = GPy.models.GPRegression(X, y.reshape(-1, 1), kernel, normalizer=True)
    
    model.constrain_positive(' ') # ensure positive hyperparameters
    model.optimize()
    
    m, v = model.predict(Xpool)
    
    y_pred, cov = model.predict(Xtest, full_cov=True)
    cov += 1e-4*np.diag(np.ones(cov.shape[0]))
    y_pred = y_pred.flatten()
    log_px = -1/2*(np.linalg.slogdet(cov)[1] \
             + (ytest-y_pred).T.dot(np.linalg.inv(cov).dot(ytest-y_pred)) \
             + d*math.log(2*math.pi)) / Xtest.shape[0]
    
    rmse = math.sqrt(((ytest-y_pred)**2).mean())
    return log_px, rmse, m.flatten(), v.flatten()

#%%
def nn(args, X, y, Xpool, ypool, Xtest, ytest):
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
    
    with torch.no_grad():
        data = torch.tensor(Xpool).to(torch.float32).to(device)
        label = torch.tensor(ypool).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        pool_m = m * y_std + y_mean
        pool_v = v * y_std**2
        
        data = torch.tensor(Xtest).to(torch.float32).to(device)
        label = torch.tensor(ytest).to(torch.float32).to(device)
        m, v = mean(data), var(data)
        m = m * y_std + y_mean
        v = v * y_std**2
        test_log_px = normal_log_prob(label, m, v)
        test_rmse = ((label - m.flatten())**2).mean().sqrt()
    
    return test_log_px.mean().item(), \
            test_rmse.item(), \
            pool_m.cpu().flatten().numpy(), \
            pool_v.cpu().flatten().numpy()

#%%
def mcdnn(args, X, y, Xpool, ypool, Xtest, ytest):
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    
#    y, y_mean, y_std = normalize_y(y)
    
    mean = torch.nn.Sequential(torch.nn.Linear(X.shape[1], n_neurons),
                               torch.nn.Dropout(p=0.1),
                               torch.nn.ReLU(),
                               torch.nn.Linear(n_neurons, 1),
                               torch.nn.Dropout(p=0.1))
    
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
    
    with torch.no_grad():
        data = torch.tensor(Xpool).to(torch.float32).to(device)
        label = torch.tensor(ypool).to(torch.float32).to(device)
        samples = torch.zeros(Xpool.shape[0], args.mcmc).to(device)
        for i in range(args.mcmc):
            samples[:,i] = mean(data).flatten()
        pool_m, pool_v = samples.mean(dim=1), samples.var(dim=1)
#        pool_m = m * y_std + y_mean
#        pool_v = v * y_std**2
    
        data = torch.tensor(Xpool).to(torch.float32).to(device)
        label = torch.tensor(ypool).to(torch.float32).to(device)
        samples = torch.zeros(Xpool.shape[0], args.mcmc).to(device)
        for i in range(args.mcmc):
            samples[:,i] = mean(data).flatten()
        m, v = samples.mean(dim=1), samples.var(dim=1)
        #m = m * y_std + y_mean
        #v = v * y_std**2
        
        test_log_px = normal_log_prob(label, m, v)
        test_rmse = ((label - m.flatten())**2).mean().sqrt()
    return test_log_px.mean().item(), \
            test_rmse.item(), \
            pool_m.cpu().flatten().numpy(), \
            pool_v.cpu().flatten().numpy()

#%%
def ensnn(args, X, y, Xpool, ypool, Xtest, ytest):
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    
    y, y_mean, y_std = normalize_y(y)
    
    ms_pool, vs_pool, ms, vs = [ ], [ ], [ ], [ ]
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
        
        with torch.no_grad():
            data = torch.tensor(Xpool).to(torch.float32).to(device)
            label = torch.tensor(ypool).to(torch.float32).to(device)
            m, v = mean(data), var(data)
            pool_m = m * y_std + y_mean
            pool_v = v * y_std**2
        
            data = torch.tensor(Xtest).to(torch.float32).to(device)
            label = torch.tensor(ytest).to(torch.float32).to(device)
            m, v = mean(data), var(data)
            m = m * y_std + y_mean
            v = v * y_std**2
        
        ms_pool.append(pool_m)
        vs_pool.append(pool_v)
        ms.append(m)
        vs.append(v)
    
    ms_pool = torch.stack(ms_pool)
    vs_pool = torch.stack(vs_pool)
    ms = torch.stack(ms)
    vs = torch.stack(vs)
    
    pool_m = ms_pool.mean(dim=0)
    pool_v = (vs_pool + ms_pool**2).mean(dim=0) - pool_m**2
    
    m = ms.mean(dim=0)
    v = (vs + ms**2).mean(dim=0) - m**2
    test_log_px = normal_log_prob(label, m, v)
    test_rmse = ((label - m.flatten())**2).mean().sqrt()
    return test_log_px.mean().item(), \
            test_rmse.item(), \
            pool_m.cpu().flatten().numpy(), \
            pool_v.cpu().flatten().numpy()

#%%
def john(args, X, y, Xpool, ypool, Xtest, ytest):
    from sklearn.cluster import KMeans
    from utils import dist
    from itertools import chain
    from torch import distributions as D
    from locality_sampler import gen_Qw, locality_sampler2
    
    if args.dataset == 'protein' or args.dataset == 'year_prediction':
        n_neurons = 100
    else:
        n_neurons = 50
    args.n_clusters = min(args.n_clusters, X.shape[0])
    
    y, y_mean, y_std = normalize_y(y)
    
    mean_psu = 1
    mean_ssu = 50
    mean_M = 60

    var_psu = 2
    var_ssu = 10
    var_M = 15
    
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
    
    mean_Q, mean_w = gen_Qw(X, mean_psu, mean_ssu, mean_M)
    var_Q, var_w = gen_Qw(X, var_psu, var_ssu, var_M)
    #mean_pseupoch = get_pseupoch(mean_w,0.5)
    #var_pseupoch = get_pseupoch(var_w,0.5)
    opt_switch = 1
    mean_w = torch.Tensor(mean_w).to(device)
    var_w = torch.Tensor(var_w).to(device)
    model.train()
    
    X = torch.tensor(X).to(torch.float32).to(device)
    y = torch.tensor(y).to(torch.float32).to(device)
    batches = batchify(X, y, batch_size = args.batch_size, shuffel=args.shuffel)
    
    it = 0    
    while it < args.iters:
        switch = 1.0 if it > args.iters/2.0 else 0.0
        
        if it % 11: opt_switch = opt_switch + 1 # change between var and mean optimizer
        
        data, label = next(batches)
        if not switch:
            optimizer.zero_grad();
            m, v = model(data, switch)
            loss = -(-v.flatten().log() - (m.flatten()-label)**2 / (2 * v.flatten())).sum()
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
    
    with torch.no_grad():
        data = torch.tensor(Xpool).to(torch.float32).to(device)
        label = torch.tensor(ypool).to(torch.float32).to(device)
        m, v = model(data, switch)
        pool_m = m * y_std + y_mean
        pool_v = (v * y_std**2).mean(dim=0)
        
        data = torch.tensor(Xtest).to(torch.float32).to(device)
        label = torch.tensor(ytest).to(torch.float32).to(device)
        m, v = model(data, switch)
        m = m * y_std + y_mean
        v = v * y_std**2
        test_log_px = t_likelihood(label.reshape(-1,1), m, v)
        test_rmse = ((label - m.flatten())**2).mean().sqrt()
    
    return test_log_px.mean().item(), \
            test_rmse.item(), \
            pool_m.cpu().flatten().numpy(), \
            pool_v.cpu().flatten().numpy()

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
        time.sleep(0.5)
        # Make train/pool/test split
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                        test_size=0.2,
                                                        random_state=(i+1)*args.seed)
        
        Xtrain, Xpool, ytrain, ypool = train_test_split(Xtrain, ytrain,
                                                        test_size=0.75,
                                                        random_state=(i+1)*args.seed)
        
        T.begin()
        log_score.append([ ]); rmse_score.append([ ])
        args.sample_size = int(Xpool.shape[0] * args.sample_size)
        al_iters = int(np.minimum(np.ceil(Xpool.shape[0]/args.sample_size), args.al_iters))
        # Active learning time
        for j in range(al_iters):
            print("======== AL iter {0}/{1} ========".format(j+1, al_iters))
            time.sleep(0.5)
            # Normalize data
            scaler = preprocessing.StandardScaler()
            scaler.fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xpool = scaler.transform(Xpool)
            Xtest = scaler.transform(Xtest)
            
            # Fit and score model
            logpx, rmse, m, v = eval(args.model)(args, Xtrain, ytrain, Xpool, ypool, Xtest, ytest)
            
            # Select point 10 points with highest variance = most uncertain about
            idx = np.argsort(v)[::-1][:args.sample_size]
   
            # Add to train, remove from test
            Xtrain = np.concatenate((Xtrain, Xpool[idx]), axis=0)
            ytrain = np.concatenate((ytrain, ypool[idx]), axis=0)
            Xpool = np.delete(Xpool, (idx), axis=0)
            ypool = np.delete(ypool, (idx), axis=0)
            
            log_score[-1].append(logpx)
            rmse_score[-1].append(rmse)
        T.end()
        
    log_score = np.array(log_score)
    rmse_score = np.array(rmse_score)
            
    # Save results
    if not 'results/active_learning_results' in os.listdir():
        os.makedirs('results/active_learning_results/', exist_ok=True)
    np.savez('results/active_learning_results/' + args.dataset + '_' + args.model, 
             log_score=log_score,
             rmse_score=rmse_score,
             timings=np.array(T.timings))
    
    # Print the results
    T.res()