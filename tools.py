
import matplotlib.pyplot as plt
import numpy as np
from chaosmap import iterate_f_batch, iterate_f_batch_with_noise


def plot_hist2d(x, y, bins=50, cmap="Blues", xlim=None, ylim=None):
  '''
  plotting histogram
  x: xvalues
  y: y values
  bins: num of bins
  cmap: colormap
  '''
  if xlim is None or ylim is None:
    hist_range = None
  else:
    hist_range = [[xlim[0], xlim[1]], [ylim[0], ylim[1]]]

  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  H = ax.hist2d(x, y, bins=bins, range=hist_range, cmap=cmap)

  #ax.set_aspect('equal')
  if not (xlim is None):
    ax.set_xlim(xlim[0], xlim[1])
  if not (ylim is None):
    ax.set_ylim(ylim[0], ylim[1])

  fig.colorbar(H[3], ax=ax)
  return fig


def random_x0(shape, x0min, x0max) -> np.ndarray:
  # in shape, dimension must be last (like  (bs, dim))
  return (x0max-x0min)*np.random.rand(*shape)+x0min


class TrajGenerator:
  '''
      Trajectory Generator
      is a class that generates batch of trajectories of chaotic map f.
      It is a generator (in the sense of Python).

  '''

  def __init__(self, dim, batch_size, f, tmax, gen_param, transient=0,
               x0min=None, x0max=None,
               use_noise=False, s_noise=0.0):
    '''
      dim: dimension of phase space
      batch_size: batch_size
      f: function f(X,args) (batch_size,dim)->(batch_size,dim), args contains parameters
      tmax: maximum iteration
      gen_param: gen_param(batch_size) function that generate parameters of size (batch_size, batchsize)
      transient: transient
      use_noise: True if treating stochastic development
    '''
    self.batch_size = batch_size
    self.dim = dim
    self.f = f
    self.tmax = tmax
    self.transient = transient
    self.gen_param = gen_param
    self.use_noise = use_noise  # for Bernoulli shift to avoid convergence due to rounding
    self.s_noise = s_noise

    if x0min is None:
      x0min = np.zeros((self.dim, 1))
    if x0max is None:
      x0max = np.ones((self.dim, 1))
    self.x0min = x0min.reshape(self.dim, 1)
    self.x0max = x0max.reshape(self.dim, 1)

  def __call__(self, x0=None, params=None):
    ''' 
      returns generator-iterator
      x0: initial condition (dim, batch)
      params: parameters
    '''
    if x0 is None:
      x0 = (self.x0max - self.x0min) * \
          (np.random.rand(self.dim, self.batch_size)) + self.x0min
    if params is None:
      params = self.gen_param(self.batch_size)
    while True:
      if not self.use_noise:
        X_batch = iterate_f_batch(
            x0, self.f, tmax=self.tmax, transient=self.transient, param=params)  # generate one batch
      else:
        X_batch = iterate_f_batch_with_noise(
            x0, self.f, tmax=self.tmax, transient=self.transient, s=self.s_noise, param=params)  # generate one batch

      x0 = (self.x0max - self.x0min) * \
          (np.random.rand(self.dim, self.batch_size)) + self.x0min
      yield X_batch, params
    #def iterate_f_batch_with_noise(X0, f, s=0, tmax=10, transient=0, **args):


class TransitionErrors:
  '''
  Transition Errors calculator
  calculates errors between X and f(X)
  '''

  def __init__(self, f):
    self.f = f

  def __call__(self, X, p=[]):
    '''
    X: (bs,timestep,dim)
    p: parameters having (pdim, bs) or (pdim) shape (pdim is a number of parameters)
    '''
    self.X = X
    bs = int(X.shape[0])  # batch_size
    l = int(X.shape[1])  # length
    ndim = int(X.shape[2])  # dimension
    pdim = int(p.shape[0])  # number of parameters type:ignore
    # print(f'bs:{bs} l:{l} ndim:{ndim} pdim:{pdim}')
    X_r = X.reshape((bs * l, ndim)).transpose()  # (ndim, bs*l) shape
    if p is None:
      fx_r = self.f(X_r)  # type:ignore
    else:
      #p = p.transpose() # (bs,dim)
      if p.ndim == 1:
        p = np.stack([p] * bs, axis=1)  # make (pdim, bs) array
      p_r = p.reshape([pdim, 1, bs]).repeat(
          l, axis=1).reshape([pdim, bs * l])  # (pdim,bs*l)
      fx_r = self.f(X_r, p_r)  # p:(bs,dim), fx_r:(ndim,bs*l) #type:ignore

    fx = fx_r.transpose().reshape((bs, l, ndim))

    self.er = fx[:, 0:l - 1, :] - X[:, 1:l, :]
    self.sqer = (self.er)**2
    # over batch. mean error for each timestep and variable
    self.mse_td = np.mean(self.sqer, axis=0)
    # over dim. mean error for each timestep
    self.mse_t = np.mean(self.mse_td, axis=1)

    self.mse = np.mean(self.sqer)  # mean square error
    self.rmse = np.sqrt(self.mse)  # root mean square error

    self.nrmse = self.rmse / fx.std()  # normalized root mean square error

    self.summary = {'er': self.er, 'sqer': self.sqer, 'mse_td': self.mse_td, 'mse_t': self.mse,
                    'rmse': self.rmse, 'nrmse': self.nrmse}
    return self.summary
