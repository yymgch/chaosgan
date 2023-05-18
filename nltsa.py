# nltsa.py
# library for nonlinear time-series analysis
#
# - Delay_embedding
# - Wayland algorithm measuring determinism
# - surrogate data generation
# - estimation of maximal Lyapunov exponent(Kantz1994)
#

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression

from chaosmap import f_logistic, iterate_f_batch


def delay_embedding(x, m=2, tau=1) -> np.ndarray:
  """
  performing delay embedding

  args:
  x: 2dim or 3dim array
  m: embedding dimension become m*(original_dimension)
  tau: delay

  return:
    y:   y(n)= (x(n), x(n+tau), ..., x(n+tau*(m-1))), (n = 1,..., N-tau*(m-1))
    has (N-(tau*(m-1))), m) dimension
  """
  array_dim = x.ndim
  if array_dim == 2:
    N = x.shape[0]  # number of data
    d = x.shape[1]
  elif array_dim == 3:
    N = x.shape[1]
    d = x.shape[2]
  else:
    print('Error: array dim must be 2 or 3.')
    return np.array([])

  X = []
  # making delay-embedding vectors by shifting and appending
  if array_dim == 2:
    for l in range(m):
      X.append(x[l * tau:(N - tau * (m - 1 - l)), :].copy())
  elif array_dim == 3:
    for l in range(m):
      X.append(x[:, l * tau:(N - tau * (m - 1 - l)), :].copy())

  X = np.concatenate(X, axis=-1)
  return X


def wayland(x, k_nearest, n_res=0, step=1):
  """
  Walyand test
    args:
    x: time series (already embedded ) to m dim (t_len, m)
    k_nearest: number of nearest points for a reference point
    n_res: number of reference points to compute e_trans: default is 0 which mean use all points
  """
  len_t = x.shape[0]
  dim = x.shape[1]

  # select reference point
  if n_res == 0:
    ind_ref = np.arange(0, len_t - 1, dtype=int)
  else:
    if len_t - 1 < n_res:
      print('Error n_res is too large')
      return (0, 0)
    ind_ref = np.random.choice(len_t - 1, n_res)

  e_transs = []
  x_refs = x[ind_ref, :]  # 参照点
  # kdtree for nearest neighbor search # do not use last points
  tree = KDTree(x[:-step], leaf_size=2)
  # distance and index of nearest neighbors
  dist, ind_nns = tree.query(x_refs, k=k_nearest + 1, dualtree=True)

  for t_ref, ind_nearest in zip(ind_ref, ind_nns):

    # moving of reference points and nearby points
    v_j = x[ind_nearest + step, :] - x[ind_nearest, :]

    v_center = np.mean(v_j, axis=0)  # mean vector of moving

    e_trans_i = np.mean(np.linalg.norm(
        v_j - v_center, axis=1) / np.linalg.norm(v_center))

    e_transs.append(e_trans_i)

  e_transs = np.array(e_transs)

  med_e_trans = np.median(e_transs)

  return med_e_trans, e_transs


def e_wayland(x, m, k_nearest=10, n_res=0):
  '''
    embedding and apply wayland analysis
  '''
  xd = delay_embedding(x, m, tau=1)  # delay embedding
  m_e_tr, e_tr = wayland(xd, k_nearest=k_nearest, n_res=n_res)
  return m_e_tr


# estimation of largest Lyapunov exponent


def count_neighbors(x, r):
  """
  count the number of neighborhood points within a ball with given radius r
  """
  tree = KDTree(x, leaf_size=2)

  neighbor_ind = tree.query_radius(x, r=r, )
  # taking away own points
  for i, ind_i in enumerate(neighbor_ind):
    neighbor_ind[i] = ind_i[ind_i != i]  # type:ignore

  # counting
  n_neibs = np.array([np.size(ind_i) for ind_i in neighbor_ind])
  frac_zero_neighbor = np.count_nonzero(n_neibs == 0) / n_neibs.size

  print("number of neighborhood points:")
  print(f"radius: {r:.3g}")
  print(f'minimum number: {np.min(n_neibs)}')
  print(f'maximum number: {np.max(n_neibs)}')
  print(f'mean number: {np.mean(n_neibs):.5g}')
  print(f'fraction of zero-neighbor point: {frac_zero_neighbor}')
  return neighbor_ind


def average_local_divergence(x, r, steps=np.arange(0, 5)):
  """
  measure the development of difference between nearby points as a function of time step
  args:
    x: multidimensional timeseries, 2-dim array (len_t, dim)
    r: radius for searching neighborhood points
    steps: array for time-steps
  """

  max_step = steps[-1]
  n_data = x.shape[0] - max_step
  tree = KDTree(x[:-max_step], leaf_size=2)

  neighbor_ind = tree.query_radius(x[:-max_step], r=r,)
  # print(f'len:{len(neighbor_ind)}')
  # taking away own points
  for i, ind_i in enumerate(neighbor_ind):
    neighbor_ind[i] = ind_i[ind_i != i]  # type:ignore

  # counting
  n_neibs = np.array([np.size(ind_i) for ind_i in neighbor_ind])

  log_m_d = []  # = np.zeros((n_data, max_step + 1))

  for i, inds in enumerate(neighbor_ind):
    if np.size(inds) != 0:
      # development of difference between nearby points
      d_is = np.zeros(max_step + 1,)
      # print(d0.shape) #
      for j in steps:
        d_j = np.mean(np.linalg.norm(x[inds + j] - x[i + j], axis=1))
        d_is[j] = d_j

      log_m_d.append(np.log(d_is))
  log_m_d = np.array(log_m_d)
  s_n = np.mean(log_m_d, axis=0)

  return s_n, steps, log_m_d


def estimate_largest_lyapunov_exponent(x: np.ndarray, m: int, steps, ind_step_fit, radius):
  '''
  This algorithm computes the maximum Lyapunov exponent using Kantz's method.

  Args:
    x: A 2D time series data of shape (len_t, dim).
    m: The embedding dimension.
    steps: A list of time step numbers, e.g., [0, 1, 2, 3, 4].
    ind_step_fit: The index range of steps to be used for slope estimation.
    radius: The radius for searching nearest neighbors.

  Returns:
    lle: The maximum Lyapunov exponent.
    log_md: A list of average local divergence rates for each time step.
    step_b: A list of time step numbers.
    s_fit: The fitting result of linear regression of log_md against step_b.  
  '''

  xd = delay_embedding(x, m, tau=1)  # delay embedding
  ind_neib = count_neighbors(xd, radius)  # check neighbor densities
  log_md, steps, data = average_local_divergence(xd, r=radius, steps=steps)

  # fit
  step_b = steps[ind_step_fit].reshape(-1, 1)
  reg = LinearRegression().fit(step_b, log_md[ind_step_fit])
  s_fit = reg.predict(step_b)
  lle = reg.coef_[0]  # estimated largest lyapunov exponent
  return lle, log_md, step_b, s_fit


# surrogate data generation

def rs_surrogate(x):
  """
  Random Shuffle surrogate
  make shuffled data
  args:
    x: (len_t, dim) array
  """
  x_sh = x.copy()
  np.random.shuffle(x_sh)
  return x_sh


def fourier_transform_surrogate(x):
  """
  making power-spectrum-preserving randomized data.
  generated random data has same power-spectrum of x while phases of every
  frequency were shuffled.
  args:
    x: one-dim or 2-dim array
  returns:
    x_surrogate: randomized data (2-dim array)
  """
  if x.ndim == 1:
    x = np.reshape(x, [-1, 1])

  y = fft(x, axis=0)  # Fourier transform
  N = y.shape[0]  # type:ignore
  # random phase shift. using mirroring for real-valued time series.
  if N % 2 == 0:
    l = N // 2 - 1
    r = np.exp(2j * np.pi * np.random.rand(l, 1))  # random rotation to phase.
    v = np.concatenate(
        [[[1, ], ], r, [[1.]], np.conj(r[::-1, :])])  # type:ignore

  else:
    l = (N - 1) // 2
    r = np.exp(2j * np.pi * np.random.rand(l, 1))
    v = np.concatenate([[[1, ], ], r, np.conj(r[::-1])])  # type:ignore
  # print(y.shape)
  # print(v.shape)

  y_fss = y * v  # randomize phase by multiplying random phase shift.
  # inverse FFT (and remove imaginary part for remove errors)  #type:ignore
  x_fss = np.real(ifft(y_fss, axis=0))  # type:ignore
  return x_fss


def aaft_surrogate(x):
  """
  making power-spectrum- and distribution-preserving randomized data.
  generated random data has same power-spectrum of x while phases of every
  frequency were shuffled. distribution of x is also preserved.
  if x is 2-dim array, each dimension are independently shuffled.

  args:
    x: one-dim or 2-dim array
  returns:
    x_surrogate: randomized data
  """
  if x.ndim == 1:
    x = np.reshape(x, [-1, 1])
  N = x.shape[0]
  d = x.shape[1]
  ys = []
  for k in range(d):
    # generate gaussian random sample
    g = np.random.randn(N,)
    r = np.sort(g)  # sorted Gaussian random sample
    ind_x = np.argsort(x[:, k])
    x_sort = np.sort(x[:, k])
    # Two argsort will result in an array containing the ranks.
    rank_x = np.argsort(ind_x)
    g = r[rank_x]  # g is like inverse transform of x

    h = fourier_transform_surrogate(g)
    ind_h = np.argsort(h, axis=0)
    rank_h = np.argsort(ind_h, axis=0)
    y = x_sort[rank_h].copy()
    ys.append(y)
  ys = np.concatenate(ys, axis=1)
  return ys
