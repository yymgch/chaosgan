# nltsa.py
# nonlinear time-series analysis
# 実装済み:
# 時系列の遅延埋め込み delay_embedding
# Waylandの手法: wayland
# 未実装
# サロゲートデータ作成
# 最大リアプノフ指数の推定(Kantz1994)
#

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression

from chaosmap import f_logistic, iterate_f_batch


def delay_embedding(x, m=2, tau=1) -> np.ndarray :
  """
  遅延埋め込み
  args:
  x: 2dim or 3dim array
  m: embedding dimension become m*(original_dimension)
  tau: delay
  it returns
  return:
    y:   y(n)= (x(n), x(n+tau), ..., x(n+tau*(m-1))), (n = 1,..., N-tau*(m-1))
    has (N-(tau*(m-1))), m) dimension
  """
  array_dim = x.ndim
  if array_dim == 2:
    N = x.shape[0]  # データ点数
    d = x.shape[1]
  elif array_dim == 3:
    N = x.shape[1]
    d = x.shape[2]
  else:
    print('Error: array dim must be 2 or 3.')
    return np.array([])

  X = []
  # m個重ねて遅延座標系を作る
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

  # 参照点として使う点を選ぶ
  if n_res == 0:
    ind_ref = np.arange(0, len_t - 1, dtype=int)
  else:
    if len_t - 1 < n_res:
      print('Error n_res is too large')
      return (0,0)
    ind_ref = np.random.choice(len_t - 1, n_res)

  e_transs = []
  x_refs = x[ind_ref, :]  # 参照点
  # kdtree for nearest neighbor search # 遷移がわからないので最後は使わない
  tree = KDTree(x[:-step], leaf_size=2)
  # distance and index of nearest neighbors
  dist, ind_nns = tree.query(x_refs, k=k_nearest + 1, dualtree=True)

  for t_ref, ind_nearest in zip(ind_ref, ind_nns):
    # v0 = x[t_ref+step,:] - x[t_ref,:] # 参照点の移動
    v_j = x[ind_nearest + step, :] - x[ind_nearest, :]  # 近傍点の移動 (v0も含むことに注意)
    # print(v_j.shape)
    v_center = np.mean(v_j, axis=0)  # 平均移動ベクトル (m-dim)

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
  count the number of neighborhood points within a ball with given radius
  """
  tree = KDTree(x, leaf_size=2)

  neighbor_ind = tree.query_radius(x, r=r, )
  # taking away own points
  for i, ind_i in enumerate(neighbor_ind):
    neighbor_ind[i] = ind_i[ind_i != i]

  # counting
  n_neibs = np.array([np.size(ind_i) for ind_i in neighbor_ind])
  frac_zero_neighbor = np.count_nonzero(n_neibs == 0) / n_neibs.size

  print("number of neighborhood points:")
  print(f'embedding dimension: {x.shape[1]}')
  print(f"radius: {r:.3g}")
  print(f'minimum number: {np.min(n_neibs)}')
  print(f'maximum number: {np.max(n_neibs)}')
  print(f'mean number: {np.mean(n_neibs):.5g}')
  print(f'fraction of zero-neighbor point: {frac_zero_neighbor}')
  return neighbor_ind


def average_local_divergence(x, r, steps=np.arange(0, 5), d=1):
  """
  measure the development of difference between nearby points as time step
  x: multidimensional timeseries, 2-dim array (len_t, dim)
  r: radius for searching neighborhood points
  steps: array for time-steps
  d: dimension of the original signal

  """

  max_step = steps[-1]
  n_data = x.shape[0] - max_step
  tree = KDTree(x[:-max_step], leaf_size=2)

  neighbor_ind = tree.query_radius(x[:-max_step], r=r,)
  # print(f'len:{len(neighbor_ind)}')
  # taking away own points
  for i, ind_i in enumerate(neighbor_ind):
    neighbor_ind[i] = ind_i[ind_i != i]

  # counting
  n_neibs = np.array([np.size(ind_i) for ind_i in neighbor_ind])

  log_m_d = []  # = np.zeros((n_data, max_step + 1))

  for i, inds in enumerate(neighbor_ind):
    if np.size(inds) != 0: # 0-neighbor pointは計算から除く
      d_is = np.zeros(max_step + 1,)  # i との差の発展
      # print(d0.shape) #
      for j in steps:
        # d_j = np.mean(np.linalg.norm(x[inds + j] - x[i + j], axis=1))
        # -1:最も未来の点で比較する．delay embedding が，時刻の昇順になっているベクトルを使っているため．
        # 多次元の信号を埋め込んだ場合は，最後の時間ステップの部分を比較する．
        # d_is[j] = np.mean( np.sqrt(np.sum((x[inds + j,-2:-1] - x[i + j,-2:-1])**2, axis=1)))        
        # d_is[j] = np.mean(np.abs(x[inds + j,-1] - x[i + j,-1])) # -1:最も未来の点で比較する．これは意味がある

        d_is[j] = np.mean( np.sqrt(np.sum((x[inds + j,(-1-d):] - x[i + j,(-1-d):])**2, axis=1)))
                          

      log_m_d.append(np.log(d_is))
  log_m_d = np.array(log_m_d)
  s_n = np.mean(log_m_d, axis=0)

  return s_n, steps, log_m_d


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

  y = fft(x, axis=0)  # フーリエ変換
  N = y.shape[0] #type:ignore
  # ランダム位相．　ミラーリングのための設定
  if N % 2 == 0:
    l = N // 2 - 1
    r = np.exp(2j * np.pi * np.random.rand(l, 1))  # 位相は[0,2pi]で一様
    v = np.concatenate([[[1, ], ], r, [[1.]], np.conj(r[::-1, :])]) #type:ignore

  else:
    l = (N - 1) // 2
    r = np.exp(2j * np.pi * np.random.rand(l, 1))
    v = np.concatenate([[[1, ], ], r, np.conj(r[::-1])]) #type:ignore
  # print(y.shape)
  # print(v.shape)

  y_fss = y * v  # 振幅を変えずに位相をランダム化する．
  x_fss = np.real(ifft(y_fss, axis=0))  # フーリエ逆変換する． 誤差のためにのこる虚部を除去する #type:ignore
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
    r = np.sort(g)  # 順番にならんだガウス乱数
    ind_x = np.argsort(x[:, k])
    x_sort = np.sort(x[:, k])
    rank_x = np.argsort(ind_x)  # argsortを２回すると順位が入った配列になる．
    g = r[rank_x]  # x を逆変換したようなもの

    h = fourier_transform_surrogate(g)
    ind_h = np.argsort(h, axis=0)
    rank_h = np.argsort(ind_h, axis=0)
    y = x_sort[rank_h].copy()
    ys.append(y)
  ys = np.concatenate(ys, axis=1)
  return ys


# %%
if __name__ == "__main__":

  bs = 5  # batch_size
  transient = 1000
  tmax = 100000
  t_total = transient + tmax
  s_noise = 0.01

  x0 = np.random.rand(bs, 1)
  params = 4.0  # value of a
  xs = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)
  xs = xs.transpose([2, 1, 0])
  d0, d1, d2 = xs.shape
  x_noise = xs + s_noise * np.random.randn(d0, d1, d2)
# %%
  plt.plot(xs[0, 0:-1, 0], xs[0, 1:, 0], '.')
  plt.plot(x_noise[0, 0:-1, 0], x_noise[0, 1:, 0], '.', markersize=1)
# %% delay embedding
  xd = delay_embedding(xs[0], m=2, tau=1)
  plt.plot(xd[:, 0], xd, '.')
  # print(xs[0].shape)
  # print(xd.shape)  # N-tau*(m-1)+1
  xd_noise = delay_embedding(x_noise[0], m=2, tau=1)

# % delay embedding of batch data
  xd_batch = delay_embedding(xs, m=3, tau=2)
  # print(xd_batch.shape)
  plt.plot(xd_batch[0, :, 0], xd_batch[0, :, 2], '.')

# %% Wayland test
  # without noise
  med_e, e_transs = wayland(xd, 50)
  print(f'median of e_trans: {med_e}')
  # noisy data
  med_e_noise, e_trans_noise = wayland(xd_noise, 50)
  print(f'median of e_trans(noisy data): {med_e_noise}')


# %% 異なる近傍数，ノイズの強度と埋め込み次元の組み合わせで試す．
# %% サロゲートデータと比較して試す．
  x0 = np.random.rand(1, 1)
  xs = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)
  xs = xs.transpose([2, 1, 0]).reshape([-1, 1])
  x_noise = xs + s_noise * np.random.randn(xs.shape[0], xs.shape[1])

  xd = delay_embedding(xs, m=2, tau=1)
  xd_noise = delay_embedding(x_noise, m=2, tau=1)

  # RS surrogate
  x_rss = rs_surrogate(xs)
  plt.plot(x_rss[0:100, 0])
  xd_rss = delay_embedding(x_rss, m=2, tau=1)
  med_e_rss, e_trans_noise = wayland(xd_rss, 100)
  print(f'med(e) (RS): {med_e_rss}')
# %%
  # FT surrogate

  xd_fss = fourier_transform_surrogate(xd)
  plt.plot(xd_fss[0:100, 0])
  plt.show()
# %%
  # 以下はパワースペクトルが変わらないことを確かめる
  y = fft(xd, axis=0)
  psp_y = np.real(y * np.conj(y)) #type:ignore
  plt.plot(psp_y[1:100, 0])

  y_fss = fft(xd_fss, axis=0)
  psp_y_fss = np.real(y_fss * np.conj(y_fss)) #type:ignore
  plt.plot(psp_y_fss[1:100, 0])
# %%  AAFT surrogate
  x_aafts = aaft_surrogate(xd)

  plt.plot(x_aafts[0:100, 0])
  plt.show()

  y = fft(xd, axis=0)
  psp_y = np.real(y * np.conj(y)) #type:ignore
  plt.plot(psp_y[0:100, 0])

  y_aafts = fft(x_aafts, axis=0)
  psp_y_aafts = np.real(y_aafts * np.conj(y_aafts))# type:ignore
  plt.plot(psp_y_aafts[0:100, 0])

  # ys = []
  # for k in range(d):
  #   #generate gaussian random sample
  #   g = np.random.randn(N,)
  #   r = np.sort(g) # 順番にならんだガウス乱数
  #   ind_x = np.argsort(x[:,k])
  #   x_sort = np.sort(x[:,k])
  #   rank_x = np.argsort(ind_x) # argsortを２回すると順位が入った配列になる．
  #   g = r[rank_x] # x を逆変換したようなもの

  #   h = fourier_transform_surrogate(g)
  #   ind_h = np.argsort(h, axis=0)
  #   rank_h = np.argsort(ind_h, axis=0)
  #   y = x_sort[rank_h].copy()
  #   ys.append(y)
  # ys = np.concatenate(ys,axis=1)

# %% 埋め込み次元ごとに，サロゲートデータと比較

  # パラメータ
  ms = np.array(range(1, 6))
  N_surrogate = 100
  transient = 1000
  tmax = 100000
  k_nearest = 100
  n_res = 1000  # used in wayland method

  # データ配列
  len_ms = ms.shape[0]
  med_e_orig = []
  med_e_rss = []
  med_e_fts = []
  med_e_aafts = []
  # 元時系列の作成
  x0 = np.random.rand(1, 1)
  params = 4.0  # value of a
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])

  for m in ms:
    print(f'm={m}')
    xd = delay_embedding(x, m, tau=1)  # delay embedding
    m_e_tr, e_tr = wayland(xd, k_nearest=100, n_res=n_res)
    med_e_orig.append(m_e_tr)

    x_rss = rs_surrogate(x)
    xd_rss = delay_embedding(x_rss, m, tau=1)
    m_e_tr, e_tr = wayland(xd_rss, k_nearest=100, n_res=n_res)
    med_e_rss.append(m_e_tr)

    x_fts = fourier_transform_surrogate(x)
    xd_fts = delay_embedding(x_fts, m, tau=1)
    m_e_tr, e_tr = wayland(xd_fts, k_nearest=100, n_res=n_res)
    med_e_fts.append(m_e_tr)

    x_aafts = aaft_surrogate(x)
    xd_aafts = delay_embedding(x_aafts, m, tau=1)
    m_e_tr, e_tr = wayland(xd_aafts, k_nearest=100, n_res=n_res)
    med_e_aafts.append(m_e_tr)

  med_e_orig = np.array(med_e_orig)
  med_e_rss = np.array(med_e_rss)
  med_e_fts = np.array(med_e_fts)
  med_e_aafts = np.array(med_e_aafts)

# %%
  plt.plot(ms, med_e_orig, 'o-', label='original')
  plt.plot(ms, med_e_rss, 'x-', label='RS')
  plt.plot(ms, med_e_fts, 'v-', label='FT')
  plt.plot(ms, med_e_aafts, '^-', label='AAFT')
  plt.xlabel('m: embedding dimension')
  plt.ylabel('med(e_trans)')
  plt.legend()

# %% Noiseデータで同じことをする．
  s_noise = 0.01
  # データ配列
  len_ms = ms.shape[0]
  med_e_orig = []
  med_e_rss = []
  med_e_fts = []
  med_e_aafts = []
  # 元時系列の作成
  x0 = np.random.rand(1, 1)
  params = 4.0  # value of a
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])
  # adding Noise
  x = x + s_noise * np.random.randn(x.shape[0], x.shape[1])

  for m in ms:
    print(f'm={m}')
    xd = delay_embedding(x, m, tau=1)  # delay embedding
    m_e_tr, e_tr = wayland(xd, k_nearest=100, n_res=n_res)
    med_e_orig.append(m_e_tr)

    x_rss = rs_surrogate(x)
    xd_rss = delay_embedding(x_rss, m, tau=1)
    m_e_tr, e_tr = wayland(xd_rss, k_nearest=100, n_res=n_res)
    med_e_rss.append(m_e_tr)

    x_fts = fourier_transform_surrogate(x)
    xd_fts = delay_embedding(x_fts, m, tau=1)
    m_e_tr, e_tr = wayland(xd_fts, k_nearest=100, n_res=n_res)
    med_e_fts.append(m_e_tr)

    x_aafts = aaft_surrogate(x)
    xd_aafts = delay_embedding(x_aafts, m, tau=1)
    m_e_tr, e_tr = wayland(xd_aafts, k_nearest=100, n_res=n_res)
    med_e_aafts.append(m_e_tr)

  med_e_orig = np.array(med_e_orig)
  med_e_rss = np.array(med_e_rss)
  med_e_fts = np.array(med_e_fts)
  med_e_aafts = np.array(med_e_aafts)
# %% ノイズつきの結果
  plt.plot(ms, med_e_orig, 'o-', label='original')
  plt.plot(ms, med_e_rss, 'x-', label='RS')
  plt.plot(ms, med_e_fts, 'v-', label='FT')
  plt.plot(ms, med_e_aafts, '^-', label='AAFT')
  plt.xlabel('m: embedding dimension')
  plt.ylabel('med(e_trans)')
  plt.legend()
# %% ノイズ強度に対するWayland 指標の変化

  m = 1  # 次元
  sn_min_log10 = -5
  sn_max_log10 = 1
  sn_step_log10 = 0.25
  n_step = int(1 + (sn_max_log10 - sn_min_log10) / sn_step_log10)
  s_noises = np.logspace(sn_min_log10, sn_max_log10, n_step, base=10)
  # データ配列
  len_s = s_noises.shape[0]
  med_e_orig = []
  med_e_rss = []
  med_e_fts = []
  med_e_aafts = []

  for s_noise in s_noises:
    # 元時系列の作成
    x0 = np.random.rand(1, 1)
    params = 4.0  # value of a
    x = iterate_f_batch(
        x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
    x = x.transpose([2, 1, 0]).reshape([-1, 1])
    # adding Noise
    x = x + s_noise * np.random.randn(x.shape[0], x.shape[1])

    print(f's={s_noise}')
    xd = delay_embedding(x, m, tau=1)  # delay embedding
    m_e_tr, e_tr = wayland(xd, k_nearest=100, n_res=n_res)
    med_e_orig.append(m_e_tr)

    x_rss = rs_surrogate(x)
    xd_rss = delay_embedding(x_rss, m, tau=1)
    m_e_tr, e_tr = wayland(xd_rss, k_nearest=100, n_res=n_res)
    med_e_rss.append(m_e_tr)

    x_fts = fourier_transform_surrogate(x)
    xd_fts = delay_embedding(x_fts, m, tau=1)
    m_e_tr, e_tr = wayland(xd_fts, k_nearest=100, n_res=n_res)
    med_e_fts.append(m_e_tr)

    x_aafts = aaft_surrogate(x)
    xd_aafts = delay_embedding(x_aafts, m, tau=1)
    m_e_tr, e_tr = wayland(xd_aafts, k_nearest=100, n_res=n_res)
    med_e_aafts.append(m_e_tr)

  med_e_orig = np.array(med_e_orig)
  med_e_rss = np.array(med_e_rss)
  med_e_fts = np.array(med_e_fts)
  med_e_aafts = np.array(med_e_aafts)

# %% ノイズつきの結果
  plt.plot(s_noises, med_e_orig, 'o-', label='original')
  plt.plot(s_noises, med_e_rss, 'x-', label='RS')
  plt.plot(s_noises, med_e_fts, 'v-', label='FT')
  plt.plot(s_noises, med_e_aafts, '^-', label='AAFT')
  plt.xlabel('s: noise amplitude')
  plt.ylabel('med(e_trans)')
  plt.legend()  # %%
  plt.xscale('log')

  #############################################################
# %% ここから Kantz & Schreiber(1994)の最大リアプノフ推定手法
  steps = np.arange(0, 15)
  ind_step_fit = np.arange(0, 6)  # slopeの推定に用いる範囲 (データ見ながら決めるべき)
  x0 = np.random.rand(1, 1)
  params = 4.0  # value of a
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])

  ind_neib = count_neighbors(x, 1e-3)
  log_md, steps, data = average_local_divergence(x, r=0.001, steps=np.arange(0, 15))
  # %% linear fittingをする．

  step_b = steps[ind_step_fit].reshape(-1, 1)
  reg = LinearRegression().fit(step_b, log_md[ind_step_fit])
  s_fit = reg.predict(step_b)
  print(f'fitted slope (Lyapunov Exp. Est.): {reg.coef_}')
  # print(f'bias:{reg.intercept_}')
  print(f'true Lyapunov exp.: {np.log(2.0)}')
#%%

  plt.plot(steps, np.exp(log_md), 'o-', label='S(k)')
  plt.plot(step_b[:, 0], np.exp(s_fit), label='linear fit')
  plt.plot(steps[0:9], np.exp(np.log(2.0) * steps[0:9] - 7), label='log(2) slope')
  plt.yscale('log')
  plt.xlabel('k (step)')
  plt.ylabel('exp(s(k)) ')
  plt.legend()

#%%  次元を変えてやってみる．

  # パラメータ
  ms = np.array(range(1, 6))  # 1-5次元
  steps = np.arange(0, 15)
  ind_step_fit = np.arange(0, 6)  # slopeの推定に用いる範囲 (データ見ながら決めるべき)
  step_b = steps[ind_step_fit].reshape(-1, 1)

  transient = 1000
  tmax = 100000
  radius = 0.01  # 半径
  params = 4.0  # value of a

  x0 = np.random.rand(1, 1)
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])

  # データ配列
  len_ms = ms.shape[0]

  log_mds = []

  for m in ms:
    print(f'm={m}')
    xd = delay_embedding(x, m, tau=1)  # delay embedding
    ind_neib = count_neighbors(xd, radius)
    log_md_m, steps, data = average_local_divergence(xd, r=radius, steps=np.arange(0, 15))
    log_mds.append(log_md_m)
    # fit
    reg = LinearRegression().fit(step_b, log_md_m[ind_step_fit])
    s_fit = reg.predict(step_b)
    print(f'fitted slope (Lyapunov Exp. Est.): {reg.coef_[0]}')
    # print(f'bias:{reg.intercept_}')
    print(f'true Lyapunov exp.: {np.log(2.0)}')

  log_mds = np.array(log_mds).T  # (steps, dim)

#%%
  plt.plot(steps, np.exp(log_mds), 'o-', label='s(k)')
  plt.plot(steps[0:6], np.exp(np.log(2.0) * steps[0:6] - 4.5), label='log(2) slope')
  plt.yscale('log')
  plt.xlabel(r'$k$')
  plt.ylabel(r'$\exp(S(k))$')

#%% ノイズがある場合

  # パラメータ
  ms = np.array(range(1, 5))  # 1-4次元
  steps = np.arange(0, 15)
  ind_step_fit = np.arange(1, 5)  # slopeの推定に用いる範囲 (データ見ながら決めるべき)
  step_b = steps[ind_step_fit].reshape(-1, 1)

  transient = 1000
  tmax = 100000
  radius = 0.02  # 半径
  params = 4.0  # value of a
  s_noise = 0.02

  x0 = np.random.rand(1, 1)
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])
  x = x + s_noise * np.random.randn(x.shape[0], x.shape[1])

  # データ配列
  len_ms = ms.shape[0]

  log_mds = []
  lyap_est_m = []

  for m in ms:
    print(f'm={m}')
    xd = delay_embedding(x, m, tau=1)  # delay embedding
    ind_neib = count_neighbors(xd, radius)
    log_md_m, steps, data = average_local_divergence(xd, r=radius, steps=np.arange(0, 15))
    log_mds.append(log_md_m)
    # fit
    reg = LinearRegression().fit(step_b, log_md_m[ind_step_fit])
    s_fit = reg.predict(step_b)
    print(f'fitted slope (an estimation of largest Lyapunov exponent): {reg.coef_[0]}')
    lyap_est_m.append(reg.coef_[0])
    # print(f'bias:{reg.intercept_}')
    print(f'true Lyapunov exp.: {np.log(2.0)}')

  log_mds = np.array(log_mds).T  # (steps, dim)
  lyap_est_m = np.array(lyap_est_m)

#%%
  m_label = list(map(str, ms))
  plt.plot(xd[:, 0], xd[:, 1], '.', markersize=0.2)
  plt.show()
  plt.plot(steps, np.exp(log_mds), 'o-', label=m_label)
  plt.plot(steps[1:5], np.exp(np.log(2.0) * steps[1:5] - 3.8), '-', label='log(2) slope')
  plt.yscale('log')
  plt.xlabel(r'$k$')
  plt.ylabel(r'$\exp(S(k))$')
  plt.legend()

  plt.show()
  plt.plot(ms, lyap_est_m)

# %% TODO: サロゲートデータとの比較

  # パラメータ
  ms = np.array(range(1, 5))  # 1-4次元
  steps = np.arange(0, 15)
  ind_step_fit = np.arange(1, 5)  # slopeの推定に用いる範囲 (データ見ながら決めるべき)
  step_b = steps[ind_step_fit].reshape(-1, 1)

  transient = 1000
  tmax = 100000
  radius = 0.02  # 半径
  params = 4.0  # value of a
  s_noise = 1e-2

  x0 = np.random.rand(1, 1)
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])
  x = x + s_noise * np.random.randn(x.shape[0], x.shape[1])
  # making surrogate data
  x_rss = rs_surrogate(x)
  x_fts = fourier_transform_surrogate(x)
  x_aafts = aaft_surrogate(x)

  m = 2
  log_mds = []
  lyap_est = []

  print(f'm={m}')
  xd = delay_embedding(x, m, tau=1)  # delay embedding

  ind_neib = count_neighbors(xd, radius)
  log_md, steps, data = average_local_divergence(xd, r=radius, steps=steps)
  log_mds.append(log_md)

  # fit
  reg = LinearRegression().fit(step_b, log_md_m[ind_step_fit]) #type:ignore
  s_fit = reg.predict(step_b)
  print(f'fitted slope (an estimation of largest Lyapunov exponent): {reg.coef_[0]}')
  lyap_est.append(reg.coef_[0])
  # print(f'bias:{reg.intercept_}')
  print(f'true Lyapunov exp.: {np.log(2.0)}')

  print('--Random Shuffle surrogate--')
  xd_rss = delay_embedding(x_rss, m, tau=1)  # delay embedding
  ind_neib = count_neighbors(xd_rss, radius)
  log_md_rss, steps, data = average_local_divergence(xd_rss, r=radius, steps=steps)
  log_mds.append(log_md_rss)
  # fit
  reg = LinearRegression().fit(step_b, log_md_rss[ind_step_fit])
  s_fit = reg.predict(step_b)
  print(f'fitted slope (an estimation of largest Lyapunov exponent): {reg.coef_[0]}')
  lyap_est.append(reg.coef_[0])

  print('--Fourier Transform surrogate--')
  xd_fts = delay_embedding(x_fts, m, tau=1)  # delay embedding
  ind_neib = count_neighbors(xd_fts, radius)
  log_md_fts, steps, data = average_local_divergence(xd_fts, r=radius, steps=steps)
  log_mds.append(log_md_fts)
  # fit
  reg = LinearRegression().fit(step_b, log_md_fts[ind_step_fit])
  s_fit = reg.predict(step_b)
  print(f'fitted slope (an estimation of largest Lyapunov exponent): {reg.coef_[0]}')
  lyap_est.append(reg.coef_[0])

  print('--Amplitude-Adjusted Fourier Transform surrogate--')
  xd_aafts = delay_embedding(x_aafts, m, tau=1)  # delay embedding
  ind_neib = count_neighbors(xd_aafts, radius)
  log_md_aafts, steps, data = average_local_divergence(xd_aafts, r=radius, steps=steps)
  log_mds.append(log_md_aafts)
  # fit
  reg = LinearRegression().fit(step_b, log_md_aafts[ind_step_fit])
  s_fit = reg.predict(step_b)
  print(f'fitted slope (an estimation of largest Lyapunov exponent): {reg.coef_[0]}')
  lyap_est.append(reg.coef_[0])

  log_mds = np.array(log_mds).T
#%%

  labels = ['original', 'RS', 'FT', 'AAFT']
  plt.plot(xd[:, 0], xd[:, 1], '.', markersize=0.2)
  plt.show()
  plt.plot(steps, np.exp(log_mds), 'o-', label=labels)
  plt.plot(steps[1:5], np.exp(np.log(2.0) * steps[1:5] - 3.8), '-', label='log(2) slope')
  plt.yscale('log')
  plt.xlabel(r'$k$')
  plt.ylabel(r'$\exp(S(k))$')
  plt.legend()

  plt.show()
  plt.plot(lyap_est)

  #####################################
# %% 最近傍検索の使い方．
  ##################################
  x = xs.reshape([-1, 1])
  print(x.shape)
  from sklearn.neighbors import KDTree

  tree = KDTree(x, leaf_size=2)
  dist, ind = tree.query(x[:2], k=3, )
  print(ind)

  dist, ind = tree.query(x, k=5, )


# %% 近傍点を図示してみる．
  #ms = np.array(range(1, 5)) # 1-4次元
  m = 2

  steps = np.arange(0, 15)
  ind_step_fit = np.arange(1, 5)  # slopeの推定に用いる範囲 (データ見ながら決めるべき)
  step_b = steps[ind_step_fit].reshape(-1, 1)

  transient = 1000
  tmax = 10000
  radius = 0.04  # 半径
  params = 4.0  # value of a
  s_noise = 5e-2

  x0 = np.random.rand(1, 1)
  x = iterate_f_batch(
      x0, f_logistic, tmax=tmax, transient=transient, param=params)  # (dim, t_len, batch)
  x = x.transpose([2, 1, 0]).reshape([-1, 1])
  x = x + s_noise * np.random.randn(x.shape[0], x.shape[1])

  xd = delay_embedding(x, m, tau=1)  # delay embedding
  ind_neib = count_neighbors(xd, radius)
  n_neib = list(map(len, ind_neib))

# %%
  ref_p = 3
  plt.plot(xd[:, 0], xd[:, 1], '.', markersize=0.5)
  plt.plot(xd[ind_neib[ref_p], 0], xd[ind_neib[ref_p], 1], 'x')
  plt.plot(xd[ind_neib[ref_p] + 2, 0], xd[ind_neib[ref_p] + 2, 1], 'x')

  plt.plot(xd[ref_p, 0], xd[ref_p, 1], 'o')
  # plt.xlim([xd[ref_p,0]-0.2, xd[ref_p,0]+0.2])
  # plt.ylim([xd[ref_p,1]-0.2, xd[ref_p,1]+0.2])
  plt.gca().set_aspect('equal')


# %%
