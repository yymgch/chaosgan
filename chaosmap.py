# -*- coding: utf-8 -*-
# chapsmap.py
#%%

import numpy as np

#from randomtest import X0

def f_logistic(X, param=4.0):
  ''' logistic map
      (batch-compatible):
  '''
  return param*X*(1.0-X)

def f_tent(X, param=2.0):
  ''' logistic map
      (batch-compatible):
  '''
  return (X<1.0/2) * param*X + (X>=1.0/2) * param * (1.0-X)


def f_henon(X, param=[1.2, 0.4]):
  ''' 
     Henon map
      a 2-dimensional chaotic map
      (batch-compatible)
      X: state variable (2,batch_size) or (2)
      param: parameter: (2) or (2,batch_size)
      param[0]: a
      param[1]: b
  '''

  x1 = param[0] - X[0]**2 + param[1] * X[1]
  y1 = X[0]
  return np.array([x1, y1])


def f_ikeda(X, param=[1.0, 0.4, 0.9, 6.0]):
  ''' 
     Ikeda map
      2-dimensional chaotic map
      (batch-compatible)
      X' shape  should be [2]  or [2, batch_size]
      param's shape should be [4] or [4, batch_size]
  '''
  tau = param[1] - param[3]/(1.0 + X[0]**2 + X[1]**2)
  x1 = param[0] + param[2] * (X[0] * np.cos(tau) - X[1] * np.sin(tau))
  y1 = param[2] * (X[0] * np.sin(tau) + X[1] * np.cos(tau))
  return np.array([x1, y1])


def iterate_f_batch(X0, f, tmax=10, transient=0, **args):
  '''
   iterates map f from initial state X0 and return trajectory.
   For using with mini-batch.
   args:
     X0: initial state with shape (dimension of x, batch_size)
     f: function f(x, args)
     tmax: max_step to iterate
     transient: step of transient not included in returns
   returns:
    Xs : array with shape (batch_size, tmax+1, dim_x)
  '''
  d = X0.shape[0]  #  (dimension)
  if X0.ndim >= 2:
    nb = X0.shape[1]  # batch_size
  else:
    nb = 1

  #X = np.zeros((tmax+1, nb, d), dtype=float)
  Xs = []  # list for trajectories
  #print(X.shape)
  X = X0  # set initial conditions

  # transient
  for t in range(transient):
    X = f(X, **args)

  Xs.append(X)
  for t in range(tmax):
    X = f(X, **args)
    Xs.append(X)
  Xs = np.array(Xs)
  # set order  (batch_size, timestep, dimension)

  return Xs.transpose([2, 0, 1]).copy()

def iterate_f_batch_with_noise(X0, f, s=0.0, tmax=10, transient=0, **args):
  '''
   iterates map f from initial state X0 and return trajectory.
   For using with mini-batch. use small noise for avoiding convergence due to rounding error
   args:
     X0: initial state with shape (dimension of x, batch_size)
     f: function f(x, args)
     s: noise standard deviation
     tmax: max_step to iterate
     transient: step of transient not included in returns
   returns:
    Xs : array with shape (batch_size, tmax+1, dim_x)
  '''
  d = X0.shape[0]  #  (dimension)
  if X0.ndim >= 2:
    nb = X0.shape[1]  # 
  else:
    nb = 1

  #X = np.zeros((tmax+1, nb, d), dtype=float)
  Xs = []  # 
  #print(X.shape)
  X = X0  #

  # transient
  for t in range(transient):
    X = f(X, **args) + s * np.random.randn(*X.shape)

  Xs.append(X)
  for t in range(tmax):
    X = f(X, **args) + s * np.random.randn(*X.shape) 
    Xs.append(X)
  Xs = np.array(Xs)
  # set order as (batch_size, timestep, dimension)
  return Xs.transpose([2, 0, 1]).copy()
  

def random_x0(shape, x0min, x0max):
  # in shape, dimension must be last (like  (bs, dim))
  return (x0max-x0min)*np.random.rand(*shape)+x0min



# %% just for test
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  bs = 10
  tmax =1000

  # small noise for tent map
  s = 1e-14
  X0 = np.random.rand(1,bs)
  X_tent = iterate_f_batch_with_noise(X0, f_tent, s=s, tmax=tmax, transient=0, param=2.0)
  # X_tent = X_tent.transpose([2, 1, 0])

  plt.plot(X_tent[0,0:-1], X_tent[0,1:], '.')
  plt.show()
  plt.plot(X_tent[:,:,0].T)

# %% Henon map
  x0min = np.array([-0.1 , -0.1]).reshape((2,1))
  x0max = np.array([0.1, 0.1]).reshape((2,1))
  param = np.array([1.4, 0.3])
  X0 = (x0max-x0min)*np.random.rand(2, bs) + x0min #type:ignore

  X_henon = iterate_f_batch(X0, f_henon, tmax=tmax, transient=0, param=param)

  plt.plot(X_henon[0,:,0], X_henon[0,:,1], '.')
  plt.show()

# %%
