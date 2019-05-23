# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 16:09:47 2019

@author: nsde
"""

#%%
import torch
from torch import nn
from torch import distributions as D
from torch.nn import functional as F
from torch.autograd import Variable
import time, math
import numpy as np
import os
import gzip
import scipy.io as sio
import pickle

#%%
class Norm2(nn.Module):
    def __init__(self, dim=1):
        super(Norm2, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.sum(x**2, dim=self.dim, keepdim=True)

#%%
class PosLinear(nn.Linear):
    def forward(self, x):
        if self.bias is None:
            return F.linear(x, F.softplus(self.weight))
        else:
            return F.linear(x, F.softplus(self.weight), F.softplus(self.bias))

#%%
class Reciprocal(nn.Module):
    def __init__(self, b=0.0):
        super(Reciprocal, self).__init__()
        self.b = b

    def forward(self, x):
        return torch.reciprocal(x + self.b)
    
#%%
class OneMinusX(nn.Module):
    def forward(self, x):
        return 1 - x
    
#%%
class RBF(nn.Module):
    def __init__(self, dim=None, num_points=None, points=None, beta=1.0):
        super(RBF, self).__init__()
        assert (points is None and dim is not None and num_points is not None) \
            or (points is not None and dim is None and num_points is None), \
            'Either dim or num_points has to be defined else points'
        if points is None:
            self.points = nn.Parameter(torch.randn(num_points, dim))
        else:
            self.points = points
        if isinstance(beta, torch.Tensor):
            self.beta = beta.view(1, -1)
        else:
            self.beta = beta

    def __dist2__(self, x):
        x_norm = (x**2).sum(1).view(-1, 1)
        points_norm = (self.points**2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)

    def forward(self, x):
        D2 = self.__dist2__(x) # |x|-by-|points|
        return torch.exp(-self.beta * D2)

#%%
class RBF2(nn.Module):
    def __init__(self, dim=None, num_points=None, points=None, beta=1.0):
        super(RBF2, self).__init__()
        self.points = torch.randn(num_points, dim).cuda()
        self.beta = beta

    def __dist2__(self, x):
        x_norm = (x**2).sum(1).view(-1, 1)
        points_norm = (self.points**2).sum(1).view(1, -1)
        d2 = x_norm + points_norm - 2.0 * torch.mm(x, self.points.transpose(0, 1))
        return d2.clamp(min=0.0)

    def forward(self, x):
        D2 = self.__dist2__(x) # |x|-by-|points|
        return torch.exp(-self.beta * D2)



#%%
class gmm(D.Distribution):
    def __init__(self, MixtureDist, ComponentDist):
        self.MixtureDist = MixtureDist # D.Categorical with K shape
        self.ComponentDist = ComponentDist # D.Distribution with K x d shape
        
    def log_prob(self, x):
        log_prob = self.ComponentDist.log_prob(x[:,None,:]) # N x k
        log_mix = torch.log_softmax(self.MixtureDist.logits, dim=-1)
        return torch.logsumexp(log_prob + log_mix, dim=1)

#%%
class positivelinear(nn.Module):
    def __init__(self, size_out):
        super(positivelinear, self).__init__()
        self.a = nn.Parameter(torch.randn(size_out))
        self.b = nn.Parameter(torch.randn(size_out))

    def forward(self, x):
        return nn.functional.softplus(self.a) * x + nn.functional.softplus(self.b)

#%%
def translatedSigmoid(x, trans, scale):
    x = (x + trans) / scale
    return torch.sigmoid(x)

#%%
def dist(X, Y): # X:  N x d , Y: M x d
    dist =  X.norm(p=2, dim=1, keepdim=True)**2 + \
            Y.norm(p=2, dim=1, keepdim=False)**2 - \
            2*torch.matmul(X, Y.t())
    return dist.clamp(0.0) # N x M

#%%
def plotpairwise(X, fig):
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(4,4,1+i*4+j)
            if i == j:
                ax.hist(X[:,i])
            else:
                ax.scatter(X[:,i], X[:,j], s=1)

#%%
class timer(object):
    ''' Small class for logging time consumption of models '''
    def __init__(self):
        self.timings = [ ]
        self.start = 0 
        self.stop = 0 
        
    def begin(self):
        self.start = time.time()
        
    def end(self):
        self.stop = time.time()
        self.timings.append(self.stop - self.start)
        
    def res(self):
        print('Total train time: {0:.3f}'.format(np.array(self.timings).sum()))
        print('Train time per model: {0:.3f}'.format(np.array(self.timings).mean()))

#%%
def batchify(*arrays, batch_size = 10, shuffel=True):
    """ Function that defines a generator that keeps outputting new batches
        for the input arrays infinit times.
    Arguments:
        *arrays: a number of arrays all assume to have same length along the
            first dimension
        batch_size: int, size of each batch 
        shuffel: bool, if the arrays should be shuffeled when we have reached
            the end
    """
    N = arrays[0].shape[0]
    c = -1
    while True:
        c += 1
        if c*batch_size >= N: # reset if we reach end of array
            c = 0
            if shuffel:
                perm_idx = np.random.permutation(N)
                arrays = [a[perm_idx] for a in arrays]
        lower = c*batch_size
        upper = (c+1)*batch_size
        yield [a[lower:upper] for a in arrays]

#%%
def normal_log_prob(x, mean, var):
    c = - 0.5 * math.log(2*math.pi)
    if isinstance(x, np.ndarray): # numpy implementation
        return c - np.log(var.flatten())/2 - (x - mean.flatten())**2 / (2*var.flatten())
    else: # torch implementation
        return c - var.flatten().log()/2 - (x - mean.flatten())**2 / (2*var.flatten())

#%%
def normal_log_prob_w_prior(x, mean, var):
    c = - 0.5 * math.log(2*math.pi)
    if isinstance(x, np.ndarray): # numpy implementation
        return c - np.log(var.mean(dim=0).flatten())/2 - (x - mean.flatten())**2 / (2*var).mean(dim=0).flatten()
    else: # torch implementation
        return c - var.mean(dim=0).flatten().log()/2 - (x - mean.flatten())**2 / (2*var).mean(dim=0).flatten()

#%%
def t_likelihood(x, mean, var, w = None):
    w = torch.ones(x.shape[0], device=x.device) if w is None else w
        
    c = -0.5*math.log(2*math.pi)
    A = logmeanexp(c - var.log()/2 - (x - mean)**2 / (2*var), dim=0)
        # mean shape : [batch,dim], var shape [num,draws,batch,dim]
        # shape [batch, dim]
    A = A / w.reshape(-1,1)
        # mean over batch size with inclusion prob if given
        # mean over dim
    A = A.sum() 
    
    return A
#%%
def normalize_y(y):
    y_mean = y.mean()
    y_std = y.std()
    y = (y - y_mean) / y_std
    return y, y_mean, y_std

#%%
def logmeanexp(inputs, dim=0):
    input_max = inputs.max(dim=dim)[0]
    return (inputs - input_max).exp().mean(dim=dim).log() + input_max

#%%
class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])
        
    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)
            if x.is_cuda:
                epsilon = epsilon.cuda()

            return x * epsilon
        else:
            return x
        
def Gaus_Dropout(p):
    return GaussianDropout(p/(1-p))

#%%
def _loader(path, kind='train'):
    ''' Specialized loaded meant to be used to load the mnist dataset '''
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte.gz' % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    images = images.reshape(-1, 1, 28, 28)
    return images, labels

#%%
def _unpickle(file):
    ''' Small function to unpickle files, used to extract cifar10 dataset '''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#%%
def to_float(*arrays):
    return [a.astype('float32') for a in arrays]

#%%
def get_mnist():
    ''' Returns the mnist dataset '''
    Xtrain, ytrain = _loader('data/image_datasets/mnist/', kind='train')
    Xtest, ytest = _loader('data/image_datasets/mnist/', kind='t10k')
    return to_float(Xtrain, ytrain, Xtest, ytest)

#%%
def get_fashionmnist():
    ''' Returns the fashion mnist dataset '''
    Xtrain, ytrain = _loader('data/image_datasets/fashionmnist/', kind='train')
    Xtest, ytest = _loader('data/image_datasets/fashionmnist/', kind='t10k')
    return to_float(Xtrain, ytrain, Xtest, ytest)

#%% 
def get_cifar10():
    ''' Returns the cifar10 dataset '''
    Xtrain = np.zeros((50000, 3072))
    ytrain = np.zeros((50000, ))
    for i in range(1, 6):
        data = _unpickle('data/image_datasets/cifar10/data_batch_' + str(i))
        Xtrain[(i-1)*10000:i*10000] = data[b'data']
        ytrain[(i-1)*10000:i*10000] = data[b'labels']
    Xtrain = Xtrain.reshape(-1, 3, 32, 32)
    data = _unpickle('data/image_datasets/cifar10/test_batch')
    Xtest = data[b'data'].reshape(-1, 3, 32, 32)
    ytest = np.array(data[b'labels'])
    return to_float(Xtrain, ytrain, Xtest, ytest)

#%% 
def get_svhn():
    ''' Returns the svhn dataset '''
    train = sio.loadmat('data/image_datasets/svhn/train_32x32.mat')
    test = sio.loadmat('data/image_datasets/svhn/test_32x32.mat')
    Xtrain = train['X'].transpose(3, 2, 0, 1)
    Xtest = test['X'].transpose(3, 2, 0, 1)
    ytrain = train['y'].flatten()
    ytest = test['y'].flatten()
    return to_float(Xtrain, ytrain, Xtest, ytest)

#%%
def get_image_dataset(name):
    ''' Get a dataset based on its name '''
    d = {'mnist': get_mnist, 'fashionmnist': get_fashionmnist,
         'cifar10': get_cifar10, 'svhn': get_svhn}
    return d[name]()