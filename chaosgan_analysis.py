# -*- coding: utf-8 -*-
# chaosgan_analysis.py
# chaosgan 結果の解析．

# TODO: 埋め込み次元 m を変えてLyapunov指数の推定を行う
# TODO: それぞれのm でのスロープを書く

#%%
from nltsa import count_neighbors, average_local_divergence
from nltsa import rs_surrogate, fourier_transform_surrogate, aaft_surrogate
from nltsa import delay_embedding, wayland, e_wayland

from tools import   plot_hist2d, random_x0, TrajGenerator, TransitionErrors
from parzen import ParzenWindow, jsd_parzen, kld_parzen
from models import make_generator_model, make_discriminator_model, ChaosGANTraining
from chaosmap import f_henon, f_logistic, f_ikeda, f_tent, iterate_f_batch, iterate_f_batch_with_noise
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import time
from IPython import display
import os

# import other files
import importlib
import chaosmap
import models
import parzen

importlib.reload(chaosmap)
importlib.reload(models)
importlib.reload(parzen)

# fixing random seed
np.random.seed(100)
tf.random.set_seed(101)

# %%
GEN_IMAGES_DIR = 'gen_images'
output_length_about = 1000
from models import BATCH_SIZE,  NOISE_DIM, SIGMA_PARZEN, FULLY_CONV, INIT_CH

S_NOISE =  1e-14 # small noise for tent or Bernouii map

# BATCH_SIZE = 100
# ITERATION = 1000_000
# NOISE_DIM = 100
# SIGMA_PARZEN = 0.02

# FULLY_CONV = True
# INIT_CH = 64
# S_NOISE = 1e-14


# set map name here
mapname = 'logistic'
use_noise = True if mapname=='tent' else False # Tent mapのときはsmall noiseを入れて値が収束しないようにする


checkpoint_dir = os.path.join('training_checkpoints', mapname)
save_image_dir = os.path.join(GEN_IMAGES_DIR, mapname)
saved_data_dir = os.path.join('saved_data', mapname)
result_logfile = os.path.join('saved_data', mapname, 'analysis_result.log')

os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(saved_data_dir, exist_ok=True)


plt.rcParams["font.size"] = 18


def add_log(message, path, display=True):
  # log fileに書き込む

  if not os.path.exists(path):
    opt = 'w'
  else:
    opt = 'a'

  with open(path, opt) as f:
    print(message, file=f)
  if display:
    print(message)

def min_ave_max(y, axis=-1):
  '''returns min, average, and max of y along the specified axis'''
  y_min = np.min(y, axis=axis)
  y_max = np.max(y, axis=axis)
  y_mean = np.mean(y, axis=axis)
  return y_min, y_mean, y_max

#%%
if __name__ == "__main__":
  # GPU memory を使いすぎない設定 (一部のマシンではこれをいれないでKeras使うとエラーになることもある)
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(
            physical_devices[k]))
  else:
    print("Not enough GPU hardware devices available")

  os.makedirs(GEN_IMAGES_DIR, exist_ok=True)

  import matplotlib as mpl
  mpl.rc('pdf', fonttype=42)


  def is_env_notebook():
    """Determine whether is the environment Jupyter Notebook"""
    if 'get_ipython' not in globals():
        # Python shell
        return False
    from IPython import get_ipython
    env_name = get_ipython().__class__.__name__
    if env_name == 'TerminalInteractiveShell':
        # IPython shell
        return False
    # Jupyter Notebook
    return True

  if not is_env_notebook():
    print('use AGG')
    mpl.use('Agg')  # off-screen use of matplotlib
#%%

# setting map-specific variables
dim_map = 0
f_map = ''
f_param = ''
x0min = 0
x0max = 0

if mapname == 'ikeda':
  f_map = f_ikeda
  dim_map = 2
  x0min = np.array([-0.1, -0.1])
  x0max = np.array([0, 0])
elif mapname == 'logistic':
  f_map = f_logistic
  dim_map = 1
  x0min = np.array([0.0])
  x0max = np.array([1.0])
elif mapname == 'henon':
  f_map = f_henon
  dim_map = 2
  x0min = np.array([-0.1, -0.1])
  x0max = np.array([0.1, 0.1])
elif mapname == 'tent':
  f_map = f_tent
  dim_map = 1
  x0min = np.array([0.0])
  x0max = np.array([1.0])

# %% make model training object and restore weights
cgantr = ChaosGANTraining(batch_size=BATCH_SIZE, noise_dim=NOISE_DIM,
                          output_length_about=output_length_about, dim_map=dim_map)
# making models
generator, discriminator = cgantr.make_models()
generator_optimizer, discriminator_optimizer = cgantr.set_optimizer()
# set checkpoint
cgantr.set_checkpoint(checkpoint_dir)
# set image dir
cgantr.set_image_dir(save_image_dir)
# tmax = generator.output_shape[1]-1

# %%　入力を入れてみてアウトプットの形をみる．
cgantr.build_model_by_input()
# #%%
#   noise = tf.random.normal(cgantr.input_shape)  # 乱数発生器
#   generated_timeseries = cgantr.generator(noise, training=False)
#   print("generated_timeseries.shape={}".format(generated_timeseries.shape))
#   d = cgantr.discriminator(generated_timeseries)
#   print("discriminator output shape: {}".format(d.shape))
add_log(cgantr.generator.summary(), result_logfile)
add_log(cgantr.discriminator.summary(), result_logfile)

#%% draw a model architecture and save
tf.keras.utils.plot_model(cgantr.generator, to_file=os.path.join(GEN_IMAGES_DIR, 'generator.png'), show_shapes=True)

#%%
tf.keras.utils.plot_model(cgantr.discriminator, to_file=os.path.join(GEN_IMAGES_DIR, 'discriminator.png'), show_shapes=True)
# %% dataset をつくる

def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # 池田写像のパラメータ(固定)

def param_henon(bs): return (1.2, 0.4)  # 池田写像のパラメータ(固定)

def param_tent(bs): return (2.0,)  # Tent写像のパラメータ(固定)


def gen_as(bs, low=4.0, high=4.0):
    '''return parameter values for logistic maps'''
    return np.random.uniform(low=low, high=high, size=(1, bs))

use_noise = False
s_n = 0

if mapname == 'ikeda':
  f_param = param_ikeda
elif mapname == 'logistic':
  f_param = gen_as
elif mapname == 'henon':
  f_param = param_henon
elif mapname == 'tent':
  f_param = param_tent
  use_noise = True
  s_n = S_NOISE    

trjgen = TrajGenerator(batch_size=BATCH_SIZE, dim=dim_map,
                        f=f_map, tmax=cgantr.tmax, gen_param=f_param, transient=100,
                        use_noise=use_noise, s_noise=s_n, #type:ignore
                        x0min=x0min, x0max=x0max)
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds = tf.data.Dataset.from_generator(
    trjgen, output_types=(tf.float64, tf.float64)).prefetch(buffer_size=AUTOTUNE)

# fixed seed for sampling
seed = tf.random.normal(cgantr.input_shape)


#%% restore model
cgantr.checkpoint.restore(tf.train.latest_checkpoint(cgantr.checkpoint_dir))

#%%tr_result の読み込み
import pickle
with open("saved_data/tr_result.pkl", "rb") as f:
  tr_result = pickle.load(f)

#%% making graphs
fig = plt.figure()
ax = fig.add_subplot(111) #type:ignore
ax.plot(tr_result['itrs'], tr_result['rmses'],
        'C0', label='transition error(RMSE)', linewidth=1)
ax2 = ax.twinx()
ax2.plot(tr_result['itrs'], tr_result['mloglikes'],
          'C1', label='-(log likelihood)', linewidth=1, alpha=0.7)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right', fontsize=14)
ax.set_xlabel('iteration')
ax.set_ylabel('RMSE')
ax.set_yscale('log')
ax2.set_ylabel('-(log likelihood)')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "errors_and_likelihood.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "errors_and_likelihood.pdf"))
plt.show()
plt.close(fig)

#%% checking ranges of input position to output position

pulse_input = np.zeros_like(seed)
pos_pulse = int(seed.shape[1]/2) # 中心に入れる．
pulse_input[:,pos_pulse,:] = 1


output_pulse = generator(pulse_input)
dif_from_base = np.abs(output_pulse[0,:,0] - output_pulse[0,0,0])
depend_range = (dif_from_base > 1e-5)
dep_ind = np.where(depend_range)[0]
print(dep_ind)
len_dep = np.max(dep_ind) - np.min(dep_ind) +1

print(f'length of dependency is {len_dep}')

fig, axs = plt.subplots(3,1, figsize=(10,7))
axs[0].plot(pulse_input[0,:,0], label='input')
axs[0].legend()
axs[1].plot(output_pulse[0,:,0], label='output')
axs[1].legend()
axs[2].plot(1.0*np.array(depend_range[:]), label='dependence')
axs[2].legend()

fig.savefig(os.path.join(save_image_dir, "dependency_range.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "dependency_range.pdf"))

# %% sample
X_nc, p_nc = next(iter(ds)) # get numerical data of trajectory
X_nc = X_nc.numpy()
p_nc = p_nc.numpy() # p is array of parameters

X_sample = generator(seed, training=False).numpy()


#%% Estimation of error
batch_size = BATCH_SIZE
errors = TransitionErrors(f=f_map)
#error_summary = errors_map(X_sample,p=np.array(param_ikeda(bs=batch_size))) #Transition Error

error_summary = errors(X_sample, p=np.array(f_param(bs=batch_size))) # type:ignore
# error_nc_summary = errors(X_nc, p=np.array(param_ikeda(bs=batch_size)))
add_log(f"RMSE: {error_summary['rmse']}", result_logfile) #calculating RMSE

#%% timeseries sample
#  plt.plot(X_nc[0,0:100,0])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #type:ignore
ax.plot(np.linspace(0, 10, 10) + X_sample[0:10, 200:400, 0].T)
ax.tick_params(left=False, labelleft=False)
fig.tight_layout()

fig.savefig(os.path.join(save_image_dir, "timeseries_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "timeseries_final.pdf"))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #type:ignore
ax.plot(np.linspace(0, 10, 10) + X_nc[0:10, 0:200, 0].T)
ax.tick_params(left=False, labelleft=False)
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "timeseries_nc.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "timeseries_nc.pdf"))
plt.show()



# %% return map
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(1, 1, 1) #type:ignore
if dim_map == 1:
  ax.plot(X_sample[:, 0:-1, 0].flatten(),
          X_sample[:, 1:, 0].flatten(), '.', markersize=1, alpha=0.2)
  ax.set_xlabel(r'$x_t$')
  ax.set_ylabel(r'$x_{t+1}$')
else:
  ax.plot(X_sample[:, :, 0].flatten(),
          X_sample[:, :, 1].flatten(), '.', markersize=1, alpha=0.1)
#ax.set_aspect('equal')
if mapname == 'ikeda':
  ax.set_xlim((-0.5, 2.5))
  ax.set_ylim((-2.5, 1.5)) #type:ignore
if mapname == 'henon':
  ax.set_xlim((-2, 2.1))
  ax.set_ylim((-2, 2)) #type:ignore
if mapname == 'logistic':
  ax.set_xlim((-0.2, 1.2))
  ax.set_ylim((-0.2, 1.2)) #type:ignore
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xy_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xy_final.pdf"))

fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(1, 1, 1) #type:ignore
if dim_map == 1:
  ax.plot(X_nc[:, 0:-2, 0].flatten(),
          X_nc[:, 1:-1, 0].flatten(), '.', markersize=0.2)
  ax.set_xlabel(r'$x_t$')
  ax.set_ylabel(r'$x_{t+1}$')

else:
  ax.plot(X_nc[:, :, 0].flatten(),
          X_nc[:, :, 1].flatten(), '.', markersize=0.2)
#ax.set_aspect('equal')
if mapname == 'ikeda':
  ax.set_xlim((-0.5, 2.5))
  ax.set_ylim((-2.5, 1.5)) #type:ignore
elif mapname == 'henon':
  ax.set_xlim((-2, 2.1))
  ax.set_ylim((-2, 2)) #type:ignore
if mapname == 'logistic':
  ax.set_xlim((-0.2, 1.2))
  ax.set_ylim((-0.2, 1.2)) #type:ignore

fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xy_nc.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xy_nc.pdf"))


#%% density (1-dim, x-only)
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(1, 1, 1) #type:ignore

p_sample, bins_sample, _ = ax.hist(X_sample[:, :, 0].flatten(), bins=200,
                                    density=1, color='red', alpha=0.5, label='GAN') #type:ignore
p_nc, bins_nc, _ = ax.hist(X_nc[:, :, 0].flatten(), bins=200,
                            density=1, color='blue', alpha=0.5, label='Training') #type:ignore

ax.legend(loc='upper center')
ax.set_xlabel('x')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xhist1d_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xhist1d_final.pdf"))
fig.show()

# cumulative density
fig = plt.figure(figsize=(4,4))
ax = fig.add_subplot(1, 1, 1) #type:ignore
d_x_sample = bins_sample[1] - bins_sample[0]  # bin幅
d_x_nc = bins_nc[1] - bins_nc[0]
ax.plot(bins_sample[1:], np.cumsum(p_sample)
        * d_x_sample, color='red', label='GAN')
ax.plot(bins_nc[1:], np.cumsum(p_nc) * d_x_nc, color='blue', label='Training')
ax.set_xlabel('x')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "x_cdf.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "x_cdf.pdf"))
plt.show()

#%% Parzen window analysis (own code)
SIGMA_PARZEN = 0.02
p_sample_pw = ParzenWindow(
    X_sample.shape[-1], sigma=SIGMA_PARZEN, X_sample=X_sample, max_parallel=5e8)
p_nc_pw = ParzenWindow(
    X_nc.shape[-1], sigma=SIGMA_PARZEN, X_sample=X_nc, max_parallel=5e8)

loglikes_xnc = p_sample_pw.log_like(X_nc)
m_loglike = tf.reduce_mean(p_sample_pw.log_like(X_nc)).numpy()
#%%
if dim_map == 1:
  x_val = np.arange(-0.1, 1.1, 0.0005)
  plt.plot(x_val, p_sample_pw.pdf(x_val),  color='red', label='GAN')
  plt.plot(x_val, p_nc_pw.pdf(x_val), color='blue', label='Training')
  plt.legend()
  plt.tight_layout()
  plt.savefig(os.path.join(save_image_dir, "x_pdf_parzen.png"), dpi=300)
  plt.savefig(os.path.join(save_image_dir, "x_pdf_parzen.pdf"))

elif dim_map == 2:
  if mapname == 'ikeda':
    x_val = np.arange(-0.5, 2.5, 0.002)
    y_val = np.arange(-2.5, 1.5, 0.005)     
  elif mapname == 'henon':
    x_val = np.arange(-2, 2, 0.005)
    y_val = np.arange(-2, 2, 0.005)
  else:
    x_val = np.arange(-2, 2, 0.005)
    y_val = np.arange(-2, 2, 0.005)

  XX, YY = np.meshgrid(x_val, y_val)
  xy = np.stack((XX.flatten(), YY.flatten()), axis=1)

  p_sample = tf.reshape(p_sample_pw.pdf(xy), (y_val.size, x_val.size))
  p_nc = tf.reshape(p_nc_pw.pdf(xy), (y_val.size, x_val.size))

  plt.imshow(p_sample, origin='lower', cmap='Blues',
              extent=(x_val[0], x_val[-1], y_val[0], y_val[-1]))
  plt.xlabel('x')
  plt.ylabel('y')
  plt.tight_layout()
  plt.savefig(os.path.join(save_image_dir, "xy_pdf_parzen.png"), dpi=300)
  plt.savefig(os.path.join(save_image_dir, "xy_pdf_parzen.pdf"))
  plt.show()

  plt.imshow(p_nc, origin='lower', cmap='Blues', extent=(
      x_val[0], x_val[-1], y_val[0], y_val[-1]))
  plt.xlabel('x')
  plt.ylabel('y')
  plt.tight_layout()
  plt.savefig(os.path.join(save_image_dir, "xy_nc_pdf_parzen.png"), dpi=300)
  plt.savefig(os.path.join(save_image_dir, "xy_nc_pdf_parzen.pdf"))

add_log(f"mean log likelihood = {m_loglike}", result_logfile)
#%%
#mloglikes.append(-mloglike)
# Kullback-Libler Divergence
kld = kld_parzen(p_nc_pw, p_sample_pw, one_leave_out=True)
jsd = jsd_parzen(
    X_nc, X_sample, dim=X_sample.shape[-1], sigma=SIGMA_PARZEN, one_leave_out=True)
add_log(f'KLD: {kld}', result_logfile)
add_log(f'jsd: {jsd}', result_logfile)

#%%  Parzen window (KDE) using scipy
#   from scipy.stats import gaussian_kde
#   # reshape
#   x_sample_all = X_sample.reshape([-1, X_sample.shape[-1]])
#   x_nc_all = X_nc.reshape([-1, X_sample.shape[-1]])
#   kde_sample = gaussian_kde(x_sample_all)

#%% Kernel Density Estimation using scikit-learn
# from sklearn.neighbors import KernelDensity
# import numpy as np
# # X_nc に 数値計算の教師信号 [batch_size, t_length, dim]
# # X_sample に generatorの時系列(ほぼ同サイズ)　が入ってる．
# # SIGMA_PARZEN はウインドウの幅を決めるパラメータ

# kde_gen = KernelDensity(kernel='gaussian', bandwidth=SIGMA_PARZEN, atol=1e-6, #type:ignore
#                         rtol=1e-6, leaf_size=20).fit(X_sample.reshape([-1, X_sample.shape[-1]])) #type:ignore
# kde_nc = KernelDensity(kernel='gaussian', bandwidth=SIGMA_PARZEN, atol=1e-6, #type:ignore
#                         rtol=1e-6, leaf_size=20).fit(X_nc.reshape([-1, X_nc.shape[-1]])) #type:ignore

# loglikes = []
# for x in X_nc:
#   log_like_x = kde_gen.score_samples(x)  # log-likelihood
#   print('.', end='')
#   loglikes.append(log_like_x)
# print()
# m_loglike = np.mean(np.concatenate(loglikes))  # 平均対数尤度

# # 教師データの分布からの尤度
# loglikes_p_nc = []
# for x in X_nc:
#   log_like_x = kde_nc.score_samples(x, )  # log-likelihood
#   print('.', end='')
#   loglikes_p_nc.append(log_like_x)
# print()

# m_loglike_p_nc = np.mean(np.concatenate(loglikes_p_nc))  # 平均対数尤度

# kld = m_loglike_p_nc - m_loglike

# print(f"mean log likelihood = {m_loglike}")
# print(f"KL Divergence = {kld}")
#%%
# 密度関数のグラフ
# if dim_map == 1:
#   x_val = np.arange(-0.1, 1.1, 0.0005)
#   plt.plot(x_val, np.exp(kde_gen.score_samples(
#       x_val.reshape([-1, 1]))), color='red', label='GAN')
#   plt.plot(x_val, np.exp(kde_nc.score_samples(
#       x_val.reshape([-1, 1]))), '--', color='blue', label='Training')
#   plt.legend()

  #plt.plot(X_sample[0], p_x)
#%%
  # import tensorflow_probability as tfp
  # tfd = tfp.distributions
  # mix = 0.3
  # bimix_gauss = tfd.Mixture(
  #   cat=tfd.Categorical(probs=[mix, 1.-mix]),
  #   components=[
  #     tfd.Normal(loc=-1., scale=0.1),
  #     tfd.Normal(loc=+1., scale=0.5),
  # ])
  # x = tf.linspace(-2., 3., int(1e4))
  # plt.plot(x, bimix_gauss.prob(x))
  # #tfp.distributions.MultivariateNormalDiag(loc=x_sample_all, scale=SIGMA_PARZEN)
# %% density map (generated data)

if dim_map == 1:
  x2 = np.stack([X_sample[:, 0:-2, 0].flatten(),
                  X_sample[:, 1:-1, 0].flatten()], axis=1)
  x2_nc = np.stack([X_nc[:, 0:-2, 0].flatten(),
                    X_nc[:, 1:-1, 0].flatten()], axis=1)

else:
  x2 = X_sample[:, :, 0:2].reshape(
      (X_sample.shape[0] * X_sample.shape[1], X_sample.shape[2]))
  x2_nc = X_nc[:, :, 0:2].reshape(
      (X_nc.shape[0] * X_nc.shape[1], X_nc.shape[2]))
if mapname == 'ikeda':
  ax.set_xlim((-0.5, 2.5))
  ax.set_ylim((-2.5, 1.5)) #type:ignore
elif mapname == 'henon':
  ax.set_xlim((-2.0, 2.1))
  ax.set_ylim((-2.0, 2.1)) #type:ignore

fig = plot_hist2d(x2[:, 0], x2[:, 1], bins=100,
                  cmap="Blues",)
if mapname == 'ikeda':
  plt.xlim((-0.5, 2.0))
  plt.ylim((-2.5, 1.0))
  plt.xlabel(r'$x_{t}$')
  plt.ylabel(r'$y_{t}$')

elif mapname == 'logistic' or mapname == 'tent':
  plt.xlim((-0.2, 1.2))
  plt.ylim((-0.2, 1.2))
  plt.xlabel(r'$x_{t}$')
  plt.ylabel(r'$x_{t+1}$')
elif mapname == 'henon':
  plt.xlim((-2.0, 2.1))
  plt.ylim((-2.0, 2.1))
  plt.xlabel(r'$x_{t}$')
  plt.ylabel(r'$y_{t}$')

fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xhist2d_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xhist2d_final.pdf"))
plt.show()
#%% density map (nc data)
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# H = ax.hist2d(x2_true[:, 0], x2_true[:, 1], bins=50, cmap="Blues")

# ax.set_aspect('equal')
# fig.colorbar(H[3],ax=ax)
fig = plot_hist2d(x2_nc[:, 0], x2_nc[:, 1], bins=100,
                  cmap="Blues",)
if mapname == 'ikeda':
  plt.xlim((-0.5, 2.0))
  plt.ylim((-2.5, 1.0))
elif mapname == 'logistic':
  plt.xlim((-0.2, 1.2))
  plt.ylim((-0.2, 1.2))
  plt.xlabel(r'$x_{t}$')
  plt.ylabel(r'$x_{t+1}$')
elif mapname == 'henon':
  plt.xlim((-2.0, 2.1))
  plt.ylim((-2.0, 2.0))
elif mapname == 'tent':
  plt.xlim((-0.2, 1.2))
  plt.ylim((-0.2, 1.2))
  plt.xlabel(r'$x_{t}$')
  plt.ylabel(r'$x_{t+1}$')

fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xhist2d_nc.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xhist2d_nc.pdf"))
plt.show()


# %% example of error time-series
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #type:ignore
ax.plot(np.sum(errors.sqer, 2)[0, 0:1000])  # 二乗誤差の時系列
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|^2$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "error_timeseries_sqer_long.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_sqer_long.pdf"))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #type:ignore
ax.plot(np.sum(errors.sqer, 2)[0, 0:100])  # 二乗誤差の時系列
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|^2$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "error_timeseries_sqer_short.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_sqer_short.pdf"))
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #type:ignore
ax.plot(np.sum(np.abs(errors.er[0, :,:]),1))  # 誤差の時系列
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "error_timeseries_er_long.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_er_long.pdf"))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1) #type:ignore
ax.plot(np.sum(np.abs(errors.er[0, 0:100,:]),1))  # 誤差の時系列
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "error_timeseries_er_short.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_er_short.pdf"))
plt.show()


# # %%
#   bin_log = np.logspace(-3, 0.4, 41)
#   #hi,x_hi = np.histogram(np.sqrt( np.sum(errors.sqer,2)  ).flatten(),bins=1000, density=True)
#   hi, x_hi = np.histogram(np.sqrt(np.sum(errors.sqer, 2)).flatten(), bins=bin_log, density=True)

#   fig = plt.figure()
#   ax = fig.add_subplot(1, 1, 1)
#   #ax.hist(np.sqrt( np.sum(errors.sqer,2)  ).flatten(),bins=bin_log, density=True)
#   ax.loglog(x_hi[1:], hi, '.-')
#   #ax.semilogy(x_hi[1:],hi, '.-')
#   ax.set_xscale("log")
#   ax.set_yscale("log")
#   ax.set_xlabel(r'$|e_{n}|$')

#   fig.savefig(os.path.join(save_image_dir, "error_histogram_loglog.png"), dpi=300)
# # %%
#   fig = plt.figure()
#   ax = fig.add_subplot(1, 1, 1)

#   ax.set_xlabel(r'$|e_{n}|$')

#   ax.semilogy(x_hi[1:], hi, '.-')
#   fig.savefig(os.path.join(save_image_dir, "error_histogram_semilog.png"))

# # %%

#   fig = plt.figure()
#   ax = fig.add_subplot(1, 1, 1)

#   ax.plot(x_hi[1:], hi, '.-')
#   ax.set_xlabel(r'$|e_{n}|$')

#   fig.savefig(os.path.join(save_image_dir, "error_histogram.png"))

# # %% １乗誤差
#   e_h, e_h_x = np.histogram(errors.er.flatten(), bins=10000)
#   plt.plot(e_h_x[1:], e_h)
#   plt.xlim([-0.2, 0.2])
# # %%
#   abs_e_h, abs_e_h_x = np.histogram(np.abs(errors.er.flatten()), bins=10000)
#   plt.plot(abs_e_h_x[1:], abs_e_h)
#   plt.yscale('log')
# # %% log normal?
#   import scipy.stats

#   log_er = np.log(np.abs(errors.er))
#   loge_h, loge_h_x = np.histogram(log_er.flatten(), bins=100, density=True)
#   m_log = np.mean(log_er.flatten())
#   std_log = np.std(log_er.flatten())

#   p_log = scipy.stats.norm.pdf(loge_h_x, m_log, std_log)

#   plt.plot(loge_h_x[1:], loge_h)
#   plt.plot(loge_h_x, p_log)
# # %%
#   plt.plot(np.var((errors.er), axis=1).flatten())
# # %%
#   x_cauchy = np.random.standard_cauchy((100, 1000))
#   plt.plot(np.var(x_cauchy, axis=1))


# # %%
#   hi, x_hi = np.histogram(np.abs(errors.er - np.mean(errors.er)).flatten(), bins=1000, density=True)
#   plt.loglog(x_hi[1:], hi)
# # %%
#   A = scipy.stats.lognorm.fit(np.abs(errors.er).flatten())
#   px = scipy.stats.lognorm.pdf(x_hi, A[0], A[1], A[2])
#   plt.plot(x_hi[1:], hi, x_hi, px)




###############
# %% 
# Deterministic property
###############



# 長いsequenceを生成する．
bs = 1
transient = 1000
tmax = 100000
t_total = transient + tmax
# s_noise_nc = 0.01 
# params = 4.0  # value of a

# x0 = np.random.rand(dim_map, bs)
x0 = random_x0((bs, dim_map), x0min, x0max).transpose( [1, 0])  # (dim_map, bs) type:ignore

#%% generate long sequence
params=None
if mapname == 'logistic':
  params = 4.0  # value of a
elif mapname == 'ikeda':
  def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # 池田写像のパラメータ(固定)
  params = param_ikeda(bs)
elif mapname == 'henon':
  def param_henon(bs): return (1.2, 0.4)  #
  params = param_henon(bs)
elif mapname=='tent':
  def param_tent(bs): return (2.0,)  # Tent写像のパラメータ(固定)
  params = param_tent(bs)

if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
print(x.shape)
x = x.reshape([-1, dim_map])

init_shape = [bs, tmax // 8 + 13, 64]
seed = tf.random.normal(init_shape)
X_sample = generator(seed, training=False).numpy()
# print(f'X_sample.shape: {X_sample.shape}')
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array

# 長さの調整
if x_sample.shape[0] != x.shape[0]:
  lt = min(x_sample.shape[0] != x.shape[0])
  x_sample = x_sample[0:lt,:]
  x = x[0:lt,:]
add_log(f'length of time series={x_sample.shape[0]}', result_logfile)

# %% 誤差の推定 (埋め込んだ空間における真のアトラクタとの距離によって定義する)
# 比較の結果，シンプルなTransitionErrorsの方が良い．

# transition error
errors = TransitionErrors(f=f_map)
error_summary = errors(x_sample.reshape([1,-1,dim_map]), p=np.array(f_param(bs=bs))) # type:ignore
#%%
#from sklearn.neighbors import KDTree

xd = delay_embedding(x, 2, 1) # embedding to d*2 dimension
xd_sample = delay_embedding(x_sample, 2, tau=1)

# ここの部分の処理は，chaosgan_errors.pyで詳しくやる．
# tree = KDTree(xd, leaf_size=2)
# d_atr, ind = tree.query(xd_sample, k=1, dualtree=True, )
# med_d_atr = np.median(d_atr)
# m_d_atr = np.mean(d_atr)
# add_log(f'distance-to-attractor as error index', result_logfile)
# add_log(f'med(d_atr)={med_d_atr}', result_logfile)
# add_log(f'mean(d_atr)={m_d_atr}', result_logfile)


#%% 
# ノイズ時系列を作る( 元の時系列にRMSEを標準偏差に持つ正規分布のランダムノイズを加える)
x_noise = x + errors.rmse * np.random.randn(x.shape[0], x.shape[1])
# サロゲートデータを作る
x_rss = rs_surrogate(x)
x_fts = fourier_transform_surrogate(x)
x_aafts = aaft_surrogate(x)

#%%  Wayland test (1回だけやってみる例)
med_, e_ = wayland(xd_sample, 50) #type:ignore

# %% Wayland

ms = np.array(range(1, 6))# m: 埋め込みの次元

N_surrogate = 100 # サロゲートデータの個数
k_nearest = 100 # 近傍の個数
n_res = 1000 # 基準点の個数
# n_res = 1000  # used in wayland method

# データ配列
len_ms = ms.shape[0]
med_e_orig = []
med_e_noise = []
med_e_gan = []
med_e_rss = []
med_e_fts = []
med_e_aafts = []

for m in ms:
  print(f'm={m}')
  med_e_orig.append(e_wayland(x, m, k_nearest, n_res))
  med_e_noise.append(e_wayland(x_noise, m, k_nearest, n_res))
  med_e_gan.append(e_wayland(x_sample, m, k_nearest, n_res))
  med_e_rss.append(e_wayland(x_rss, m, k_nearest, n_res))
  med_e_fts.append(e_wayland(x_fts, m, k_nearest, n_res))
  med_e_aafts.append(e_wayland(x_aafts, m, k_nearest, n_res))

med_e_orig = np.array(med_e_orig)
med_e_gan = np.array(med_e_gan)
med_e_noise = np.array(med_e_noise)
med_e_rss = np.array(med_e_rss)
med_e_fts = np.array(med_e_fts)
med_e_aafts = np.array(med_e_aafts)
# %%
plt.plot(ms, med_e_orig, 'o-', label='Training', color='blue')
plt.plot(ms, med_e_noise, '*-', label='noisy')
plt.plot(ms, med_e_gan, 's-', label='GAN', color='red')
plt.plot(ms, med_e_rss, 'x-', label='RS')
plt.plot(ms, med_e_fts, 'v-', label='FT')
plt.plot(ms, med_e_aafts, '^-', label='AAFT')
plt.xlabel('m')
plt.ylabel(r'med($E_{trans}$)')
plt.legend(bbox_to_anchor=(1.0, 1.1), fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "wayland_m.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "wayland_m.pdf"))

# %%
plt.plot(ms, med_e_orig, 'o-', label='Training', color='blue')
plt.plot(ms, med_e_noise, '*-', label='noisy')
plt.plot(ms, med_e_gan, 's-', label='GAN', color='red')
plt.xlabel('m')
plt.ylabel(r'med($E_{trans}$)')
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "wayland_m_enlarge.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "wayland_m_enlarge.pdf"))


# %% statistical test 
N_surrogate = 100
med_e_orig = []  # (len(ms), N_surrogate)の2次元配列になる
med_e_noise = []
med_e_gan = []
med_e_rss = []
med_e_fts = []
med_e_aafts = []
for m in ms:
  print(f'm={m}')
  med_e_orig.append(e_wayland(x, m, k_nearest, n_res))
  # med_e_noise.append(e_wayland(x_noise, m, k_nearest, n_res))
  med_e_gan.append(e_wayland(x_sample, m, k_nearest, n_res))
  med_e_noise_m = []  
  med_e_rss_m = []
  med_e_fts_m = []
  med_e_aafts_m = []
  for j in range(N_surrogate):
    # print(j)
    x_noise = x + errors.rmse * np.random.randn(x.shape[0], x.shape[1])
    x_rss = rs_surrogate(x)
    x_fts = fourier_transform_surrogate(x)
    x_aafts = aaft_surrogate(x)

    med_e_noise_m.append(e_wayland(x_noise, m, k_nearest, n_res))
    med_e_rss_m.append(e_wayland(x_rss, m, k_nearest, n_res))
    med_e_fts_m.append(e_wayland(x_fts, m, k_nearest, n_res))
    med_e_aafts_m.append(e_wayland(x_aafts, m, k_nearest, n_res))
  med_e_noise.append(np.array(med_e_noise_m))
  med_e_rss.append(np.array(med_e_rss_m))
  med_e_fts.append(np.array(med_e_fts_m))
  med_e_aafts.append(np.array(med_e_aafts_m))
med_e_orig = np.array(med_e_orig)
med_e_gan = np.array(med_e_gan)
med_e_noise = np.array(med_e_noise)
med_e_rss = np.array(med_e_rss)
med_e_fts = np.array(med_e_fts)
med_e_aafts = np.array(med_e_aafts)
#%%



e_min, m_e, e_max = min_ave_max(med_e_rss)
# %%
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111) #type:ignore
ax.plot(ms, med_e_orig, 'o-', label='Training', color='blue')
ax.plot(ms, med_e_gan, 's-', label='GAN', color='red')
# ax.plot(ms, med_e_noise, '*-', label='noisy')

e_min, m_e, e_max = min_ave_max(med_e_noise)
ax.errorbar(ms, m_e, yerr=np.stack(
    [(m_e - e_min), (e_max - m_e)]), fmt='x-', capsize=3, label='Noisy')
e_min, m_e, e_max = min_ave_max(med_e_rss)
ax.errorbar(ms, m_e, yerr=np.stack(
    [(m_e - e_min), (e_max - m_e)]), fmt='d-', capsize=3, label='RS')
e_min, m_e, e_max = min_ave_max(med_e_fts)
ax.errorbar(ms, m_e, yerr=np.stack(
    [(m_e - e_min), (e_max - m_e)]), fmt='v-', capsize=3, label='FTS')
e_min, m_e, e_max = min_ave_max(med_e_aafts)
ax.errorbar(ms, m_e, yerr=np.stack(
    [(m_e - e_min), (e_max - m_e)]), fmt='^-', capsize=3, label='AAFT')
# plt.plot(ms, med_e_fts, 'v-', label='FT')
# plt.plot(ms, med_e_aafts, '^-', label='AAFT')
ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$\bar{E}_{trans}$')
# plt.legend(bbox_to_anchor=(1.0, 1.1), fontsize=14)
plt.legend( fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join(save_image_dir, "wayland_m.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "wayland_m.pdf"))
#%%
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111) #type:ignore
ax.plot(ms, med_e_orig, 'o-', label='Training', color='blue')
ax.plot(ms, med_e_gan, 's-', label='GAN', color='red')
# ax.plot(ms, med_e_noise, '*-', label='noisy')
e_min, m_e, e_max = min_ave_max(med_e_noise)
ax.errorbar(ms, m_e, yerr=np.stack(
    [(m_e - e_min), (e_max - m_e)]), fmt='x-', capsize=3, label='Noisy')


ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$\bar{E}_{trans}$')
ax.legend(fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join(save_image_dir, "wayland_m_enlarge.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "wayland_m_enlarge.pdf"))


#%% test results
#
m = 2
add_log(f' m={2}', result_logfile)
add_log(f'e_trans of original sequence: {med_e_orig[m-1]}', result_logfile)
# add_log(f'e_trans of noisy sequence: {med_e_noise[m-1]}', result_logfile)
add_log(f'e_trans of GAN sequence: {med_e_gan[m-1]}', result_logfile)

add_log('----------', result_logfile)
add_log('test against noise surrogate', result_logfile)
e_min, m_e, e_max = min_ave_max(med_e_noise)
add_log(
    f'minimum e_trans of {N_surrogate} sets of surrogate data: {e_min[m-1]}', result_logfile)
add_log(
    f'Significance with p=0.01:  {med_e_gan[m-1] <e_min[m-1]}', result_logfile)

add_log('----------', result_logfile)
add_log('test against RS surrogate', result_logfile)
e_min, m_e, e_max = min_ave_max(med_e_rss)
add_log(
    f'minimum e_trans of {N_surrogate} sets of surrogate data: {e_min[m-1]}', result_logfile)
add_log(
    f'Significance with p=0.01:  {med_e_gan[m-1] <e_min[m-1]}', result_logfile)

add_log('----------', result_logfile)
add_log('test against FT surrogate', result_logfile)
e_min, m_e, e_max = min_ave_max(med_e_fts)
add_log(
    f'minimum e_trans of {N_surrogate} sets of surrogate data: {e_min[m-1]}', result_logfile)
add_log(
    f'Significance with p=0.01:  {med_e_gan[m-1] <e_min[m-1]}', result_logfile)

add_log('----------', result_logfile)
add_log('test against AAFTS surrogate', result_logfile)
e_min, m_e, e_max = min_ave_max(med_e_aafts)
add_log(
    f'minimum e_trans of {N_surrogate} sets of surrogate data: {e_min[m-1]}', result_logfile)
add_log(
    f'Significance with p=0.01:  {med_e_gan[m-1] <e_min[m-1]}', result_logfile)


####################
# %%  
# Estimation of Maximal Lyapunov Exponent
####################

#  Generating long sequence
bs = 1
transient = 1000
tmax = 100000
t_total = transient + tmax
s_noise = 0.01
x0 = random_x0((bs, dim_map), x0min, x0max).transpose([1, 0]) # set initial values

if mapname == 'logistic':
  params = 4.0  # value of a
elif mapname == 'ikeda':
  def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # Parameter of Ikeda map
  params = param_ikeda(bs)
elif mapname == 'henon':
  def param_henon(bs): return (1.2, 0.4)  #
  params = param_henon(bs)
elif mapname=='tent':
  def param_tent(bs): return (2.0,)  # parameter of tent map
  params = param_tent(bs)

# generate original sequence
if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
x = x.reshape([-1, dim_map]) # original sequence (2-dim array)

init_shape = [bs, tmax // 8 + 13, 64]
seed = tf.random.normal(init_shape)


# generate GAN-sequence
X_sample = generator(seed, training=False).numpy()
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array　（tmax, dim_x)

# %% 誤差の推定 (埋め込んだ空間における真のアトラクタとの距離によって定義する)
errors = TransitionErrors(f=f_map)
error_summary = errors(X_sample, p=np.array(f_param(bs=bs))) # type:ignore
# xd = delay_embedding(x, 2, 1)
# xd_sample = delay_embedding(x_sample, 2, tau=1)
# tree = KDTree(xd, leaf_size=2)
# d_atr, ind = tree.query(xd_sample, k=1, dualtree=True, )
# med_d_atr = np.median(d_atr)
# m_d_atr = np.mean(d_atr)
# add_log(f'distance-to-attractor as error index', result_logfile)
# add_log(f'med(d_atr)={med_d_atr}', result_logfile)
# add_log(f'mean(d_atr)={m_d_atr}', result_logfile)

x_noise = x + errors.rmse * np.random.randn(x.shape[0], x.shape[1])
x_rss = rs_surrogate(x)
x_fts = fourier_transform_surrogate(x)
x_aafts = aaft_surrogate(x)


# %% parameters


m = 2
steps = np.arange(0, 15)  # step to forward
ind_step_fit = np.arange(1, 5)  # slopeの推定に用いる範囲 (データ見ながら決めるべき)
# 最近傍を探す半径
if (mapname == 'logistic') or (mapname=='tent'):
  radius = 0.01
elif (mapname == 'ikeda') or (mapname == 'henon'):
  radius = 0.08
else:
  radius = 0.01
# %%

def estimate_largest_lyapunov_exponent(x:np.ndarray, m:int, steps, ind_step_fit, radius):
  '''
  Kantzの方法により最大リアプノフ指数を求めるアルゴリズムを実行する
  args:
    x: (len_t, dim)の2次元時系列データ
    m: 埋め込み次元
    steps: 時間発展のステップ数のリスト e.g., [0,1,2,3,4]
    ind_step_fit: stepsの中で，slopeの推定に用いる範囲のindex
    radius: 最近傍を探す半径
  return:
    lle: 最大リアプノフ指数
    log_md: 時間発展のステップ数に対する平均局所発散率のリスト
    step_b: 時間発展のステップ数のリスト
    s_fit: step_bに対するlog_mdの線形回帰によるフィッティング結果
  '''
  dim_orig = x.shape[1]
  xd = delay_embedding(x, m, tau=1)  # delay embedding
  ind_neib = count_neighbors(xd, radius)  # check neighbor densities
  log_md, steps, data = average_local_divergence(xd, r=radius, steps=steps, d=dim_orig)

  # fit
  step_b = steps[ind_step_fit].reshape(-1, 1)
  reg = LinearRegression().fit(step_b, log_md[ind_step_fit])
  s_fit = reg.predict(step_b)
  lle = reg.coef_[0]  # estimated largest lyapunov exponent
  add_log(
      f'fitted slope (an estimation of largest Lyapunov exponent): {reg.coef_[0]}', result_logfile)
  return lle, log_md, step_b, s_fit
# %%
log_mds = []
lyap_est = []
add_log(f'm={m}', result_logfile)
add_log('----- Training -------', result_logfile)
lle_nc, log_md, step_b, s_fit = estimate_largest_lyapunov_exponent(
    x, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
log_mds.append(log_md)
add_log('----- Noise -------', result_logfile)
lle_noise, log_md_noise, step_b, s_fit_noise = estimate_largest_lyapunov_exponent(
    x_noise, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
log_mds.append(log_md_noise)

add_log('----- GAN -------', result_logfile)
lle_gan, log_md_gan, step_b_gan, s_fit_gan = estimate_largest_lyapunov_exponent(
    x_sample, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
log_mds.append(log_md_gan)
#%%
# lyap_est.append(reg.coef_[0])
# print(f'bias:{reg.intercept_}')
print(f'true Lyapunov exp.: {np.log(2.0)}')

plt.plot(steps, np.exp(log_md), 'o-b', label='Training')
plt.plot(steps, np.exp(log_md_gan), 's-r', label='GAN')
plt.plot(steps, np.exp(log_md_noise), 'x-', label='Noisy')

plt.yscale('log')
plt.xlabel('k: steps')
plt.ylabel('exp(S(k))')

plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(save_image_dir, "lle_estimation_1.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "lle_estimation_1.pdf"))

# %% noiseを変えたものをいくつか作って比較
sigmas = np.logspace(-4, 0, num=5, base=10)

x0 = random_x0((bs, dim_map), x0min, x0max).transpose([1, 0])

if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
x = x.reshape([-1, dim_map])

init_shape = [1, tmax // 8 + 13, 64]
seed = tf.random.normal(init_shape)
X_sample = generator(seed, training=False).numpy()
# print(f'X_sample.shape: {X_sample.shape}')
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array

xd = delay_embedding(x, 2, 1)
xd_sample = delay_embedding(x_sample, 2, tau=1)
x_rss = rs_surrogate(x)
x_fts = fourier_transform_surrogate(x)
x_aafts = aaft_surrogate(x)
# %%
log_mds = []
add_log(f'm={m}', result_logfile)
add_log('----- Training -------', result_logfile)
lle_nc, log_md, step_b, s_fit = estimate_largest_lyapunov_exponent(
    x, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
log_mds.append(log_md)

add_log('----- GAN -------', result_logfile)
lle_gan, log_md_gan, step_b_gan, s_fit_gan = estimate_largest_lyapunov_exponent(
    x_sample, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
log_mds.append(log_md_gan)

add_log('----- Noise -------', result_logfile)
lles_noises = []
log_mds_noises = []
for s_noise in sigmas:
  add_log(f's_noise={s_noise}', result_logfile)
  x_noise = x + s_noise * np.random.randn(x.shape[0], x.shape[1])
  lle_noise, log_md_noise, step_b, s_fit_noise = estimate_largest_lyapunov_exponent(
      x_noise, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
  log_mds_noises.append(log_md_noise)
  lles_noises.append(lle_noise)

log_mds_noises = np.array(log_mds_noises)
lles_noises = np.array(lles_noises)
# %%
#%% plotting LLE with various noise
plt.plot(steps, np.exp(log_md), 'o-b', label='n.c', markersize=6)
plt.plot(steps, np.exp(
    log_mds_noises[1]), '^--g', label=r'$\sigma =0.001$', markersize=6)
plt.plot(steps, np.exp(log_mds_noises[2]),
          'v--g', label=r'$\sigma =0.01$', markersize=6)
plt.plot(steps, np.exp(log_mds_noises[3]),
          'x--g', label=r'$\sigma =0.1$', markersize=6)
plt.plot(steps, np.exp(log_md_gan), 's-r', label='GAN', markersize=6)

plt.yscale('log')
plt.xlabel('n (steps)')
plt.ylabel('exp(S(n))')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "lle_estimations.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "lle_estimations.pdf"))

# %%
#  差分（傾き）の観察
d_log_md = np.diff(log_md)
d_log_md_gan = np.diff(log_md_gan)
d_log_md_noises = np.diff(log_mds_noises, axis=1)

plt.plot(steps[1:], d_log_md, 'o-b', label='original', markersize=6)
plt.plot(steps[1:], d_log_md_gan, 's-r', label='GAN', markersize=6)
plt.plot(steps[1:], d_log_md_noises[1],
          '^--g', label='s=0.001', markersize=6)
plt.plot(steps[1:], d_log_md_noises[2], 'v--g', label='s=0.01', markersize=6)

plt.plot(steps[1:], d_log_md_noises[3], 'x-g', label='s=0.1', markersize=6)
plt.xlabel('n (steps)')
plt.ylabel('S(n)-S(n-1)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "lle_slopes.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "lle_slopes.pdf"))
#%%
#  結果の値の表示
plt.plot(sigmas, lles_noises, '^-', label='noisy')
plt.plot([5e-5, 1], [lle_nc, lle_nc], '-', color='blue', label='Training')
plt.plot([5e-5, 1], [lle_gan, lle_gan], ':', color='red', label='GAN')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$\lambda$')
plt.legend()
plt.grid()
plt.xscale('log')
plt.tight_layout()

plt.savefig(os.path.join(save_image_dir, "lle_results.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "lle_results.pdf"))

# %%  サロゲートデータとの比較

N_surrogate = 100 # 本番では100
# lle_rss, log_md_rss, step_b, s_fit_rss = estimate_largest_lyapunov_exponent(
#     x_rss, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
# lle_fts, log_md_fts, step_b, s_fit_fts = estimate_largest_lyapunov_exponent(
#     x_fts, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
# lle_aafts, log_md_aafts, step_b, s_fit_aafts = estimate_largest_lyapunov_exponent(
#     x_aafts, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)


lles_orig = []  # lles: largest lyapunov exponents
lles_gan = [] # (N_surrogate)の配列
lles_noise = [] # (N_surrogate)のリスト
lles_rss = []  # 
lles_fts = []
lles_aafts = []

log_mds_orig = [] # (n_steps)の配列
log_mds_gan = [] #  (n_steps)の配列
log_mds_noise = [] # (N_surrogate, n_steps)の２次元配列になる予定
log_mds_rss = []
log_mds_fts = []
log_mds_aafts = []
#%%
# GAN data
for j in range(N_surrogate):
  add_log(f'------- {j+1}th -------', result_logfile)
  # original data
  x0 = random_x0((bs, dim_map), x0min, x0max).transpose([1, 0])
  if use_noise:
    x = iterate_f_batch_with_noise(
        x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
  else:
    x = iterate_f_batch(
        x0, f_map, tmax=tmax, transient=transient, param=params)
  x = x.reshape([-1, dim_map]) # reshape to 2dim(timax,dim_x)
  #gan-generated data

  init_shape = [1, tmax // 8 + 13, 64]
  seed = tf.random.normal(init_shape)
  X_sample = generator(seed, training=False).numpy()
  x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array　（tmax, dim_x)

  # ランダムデータ生成
  x_noise = x + errors.rmse * np.random.randn(x.shape[0], x.shape[1])
  x_rss = rs_surrogate(x)
  x_fts = fourier_transform_surrogate(x)
  x_aafts = aaft_surrogate(x)

  #training data (original)
  lle_orig, log_md_orig, step_b, s_fit = estimate_largest_lyapunov_exponent(
    x, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
  lles_orig.append(lle_orig)
  log_mds_orig.append(log_md_orig)

  lle_gan, log_md_gan, step_b, s_fit = estimate_largest_lyapunov_exponent(
      x_sample, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)
  lles_gan.append(lle_gan)
  log_mds_gan.append(log_md_gan)

  lle_noise, log_md_noise, step_b, s_fit = estimate_largest_lyapunov_exponent(
      x_noise, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)  

  lle_rss, log_md_rss, step_b, s_fit = estimate_largest_lyapunov_exponent(
      x_rss, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)  

  lle_fts, log_md_fts, step_b, s_fit = estimate_largest_lyapunov_exponent(
      x_fts, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)  

  lle_aafts, log_md_aafts, step_b, s_fit = estimate_largest_lyapunov_exponent(
      x_aafts, m=2, steps=steps, ind_step_fit=ind_step_fit, radius=radius)

  lles_noise.append(lle_noise)
  lles_rss.append(lle_rss)
  lles_fts.append(lle_fts)
  lles_aafts.append(lle_aafts)
  
  # 同じように, log_md_xx を log_mds_xxにappendする
  log_mds_noise.append(log_md_noise) 
  log_mds_rss.append(log_md_rss) 
  log_mds_fts.append(log_md_fts)
  log_mds_aafts.append(log_md_aafts)

#%%
# numpy配列に直す
lles_orig = np.array(lles_orig) # (n_steps, N_surrogate)の配列
lles_gan = np.array(lles_gan)
lles_noise = np.array(lles_noise) # (n_steps, N_surrogate)の配列
lles_rss = np.array(lles_rss)
lles_fts = np.array(lles_fts)
lles_aafts = np.array(lles_aafts)

log_mds_orig = np.array(log_mds_orig)
log_mds_gan = np.array(log_mds_gan)
log_mds_noise = np.array(log_mds_noise)
log_mds_rss = np.array(log_mds_rss)
log_mds_fts = np.array(log_mds_fts)
log_mds_aafts = np.array(log_mds_aafts)

# %%
fig, ax = plt.subplots(figsize=(6, 4))

s_min, m_s, s_max = min_ave_max(log_mds_orig, axis=0)
s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='o-b', capsize=3, label='Training', markersize=6)

s_min, m_s, s_max = min_ave_max(log_mds_gan, axis=0)
s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='s-r', capsize=3, label='GAN', markersize=6)

s_min, m_s, s_max = min_ave_max(log_mds_noise, axis=0)
s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='x-', capsize=3, label='Noisy')
s_min, m_s, s_max = min_ave_max(log_mds_rss, axis=0)
s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='d-', capsize=3, label='RS')
s_min, m_s, s_max = min_ave_max(log_mds_fts, axis=0)
s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='v-', capsize=3, label='FT')
s_min, m_s, s_max = min_ave_max(log_mds_aafts, axis=0)
s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='^-', capsize=3, label='AAFT')


ax.set_yscale('log')
ax.set_xlabel('k (steps)')
ax.set_ylabel('exp(S(k))')
ax.set_xlim([-0.5,11.0]) #type:ignore
ax.legend(fontsize=14)
# x軸の値を１ずつ表示
ax.set_xticks(np.arange(0,11))
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "lle_estimations_surrogate.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "lle_estimations_surrogate.pdf"))

#%% bar-graphs of lle, comparison between original, gan, noise, rss, fts, aafts
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(0, np.mean(lles_orig), yerr=np.std(lles_orig), color='b',  capsize=3, align='center',  ecolor='black', label='Training')
ax.bar(1, np.mean(lles_gan), yerr=np.std(lles_gan), color='r', capsize=3, align='center',  ecolor='black', label='GAN')
ax.bar(2, np.mean(lles_noise), yerr=np.std(lles_noise), capsize=3, align='center',  ecolor='black', label='Noisy')
ax.bar(3, np.mean(lles_rss), yerr=np.std(lles_rss), capsize=3, align='center',  ecolor='black', label='RS')
ax.bar(4, np.mean(lles_fts), yerr=np.std(lles_fts), capsize=3, align='center',  ecolor='black', label='FT')
ax.bar(5, np.mean(lles_aafts), yerr=np.std(lles_aafts), capsize=3, align='center',  ecolor='black', label='AAFT')
ax.plot([-0.5,6], [np.log(2), np.log(2)], '--k', label='Theoretical')


ax.set_ylabel('LLE')
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['Training', 'GAN', 'Noisy', 'RS', 'FT', 'AAFT'], fontsize=12)
ax.legend(fontsize=12, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "lle_estimations_bar.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "lle_estimations_bar.pdf"))

#%% log 出力
add_log('-------LLE estimation--------', result_logfile)
add_log(f'Lyapunov Exponent Estimation m={2}', result_logfile)
add_log(f'Theoretical value of LLE: {np.log(2):.6}', result_logfile)

add_log(f'LLE estimation from training data (mean+-std): {np.mean(lles_orig):.6} +- {np.std(lles_orig):.6}', result_logfile)
add_log(f'LLE estimation from GAN data (mean+-std): {np.mean(lles_gan):.6} +- {np.std(lles_gan):.6}', result_logfile)
add_log(f'LLE estimation from noisy data (mean+-std): {np.mean(lles_noise):.6} +- {np.std(lles_noise):.6}', result_logfile)
add_log(f'LLE estimation from RS data (mean+-std): {np.mean(lles_rss):.6} +- {np.std(lles_rss):.6}', result_logfile)
add_log(f'LLE estimation from FT data (mean+-std): {np.mean(lles_fts):.6} +- {np.std(lles_fts):.6}', result_logfile)
add_log(f'LLE estimation from AAFT data (mean+-std): {np.mean(lles_aafts):.6} +- {np.std(lles_aafts):.6}', result_logfile)

#%%
# m を変えて上記のリアプノフ指数の推定を多数回行う．それぞれのm で，傾きを出力しておく．

N_surrogate=100 #
ms = np.arange(1,6, dtype=int)
radius_m = 0.0001**(1/ms) # scaling of radius with m
radius_m = [0.001, 0.01, 0.05, 0.1, 0.1]
#%%
lles_m = np.zeros((6,len(ms),  N_surrogate)) #6 の内訳は training, gan, noisy, rs, fts, aaft
log_mds_m = []
for i in range(6):
  li = []
  for j in range(len(ms)):
    li.append([])
  log_mds_m.append(li)


 #%%
for m in ms:
  print(f'm={m}')


  for j in range(N_surrogate):
    add_log(f'-------m={m}, {j+1} trial -------', result_logfile)
    # original data
    x0 = random_x0((bs, dim_map), x0min, x0max).transpose([1, 0]) # initial condition
    # generate sequence (original)
    if use_noise:
      x = iterate_f_batch_with_noise(
          x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
    else:
      x = iterate_f_batch(
          x0, f_map, tmax=tmax, transient=transient, param=params)
    x = x.reshape([-1, dim_map]) # reshape to 2dim(timax,dim_x)

    #gan-generated data
    init_shape = [1, tmax // 8 + 13, 64]
    seed = tf.random.normal(init_shape)

    X_sample = generator(seed, training=False).numpy() # generate GAN sequence
    x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array　（tmax, dim_x)

    # ランダムデータ生成 using original data
    x_noise = x + errors.rmse * np.random.randn(x.shape[0], x.shape[1]) # noise data
    x_rss = rs_surrogate(x)
    x_fts = fourier_transform_surrogate(x)
    x_aafts = aaft_surrogate(x)

    x_timeseries = [x, x_sample, x_noise, x_rss, x_fts, x_aafts]
    x_names = ['original', 'GAN', 'noisy', 'RS', 'FT', 'AAFT']
    for i, x_ in enumerate(x_timeseries):
      add_log(f'----- {x_names[i]} -------', result_logfile)
      lle, log_md, step_b, s_fit = estimate_largest_lyapunov_exponent(
          x_, m=m, steps=steps, ind_step_fit=ind_step_fit, radius=radius_m[m-1])
      add_log(f'an estimation of largest Lyapunov exponent: {lle}', result_logfile)

      # i-th condition, m embedding dim, j-th trial の結果を保存
      lles_m[i,m-1,j] = lle        
      log_mds_m[i][m-1].append(log_md)

# log_mds_m is a list of shape (6, max(m), N_surrogate ). each element have ( n_steps) array.

#%% save figures
# %%
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111) #type:ignore
ax.plot([0.5,5.3], [np.log(2), np.log(2)], '--k', label='Theoretical')

l_min, m_l, l_max = min_ave_max(lles_m[0])
ax.errorbar(ms, m_l, yerr=np.stack(
    [(m_l - l_min), (l_max - m_l)]), fmt='o-', color='blue',capsize=3, label='Training')
l_min, m_l, l_max = min_ave_max(lles_m[1])
ax.errorbar(ms, m_l, yerr=np.stack(
    [(m_l - l_min), (l_max - m_l)]), fmt='s-', color='red',capsize=3, label='GAN')

# ax.plot(ms, m_e, 'o-', label='Training', color='blue')
# ax.plot(ms, med_e_gan, 's-', label='GAN', color='red')
# ax.plot(ms, med_e_noise, '*-', label='noisy')

l_min, m_l, l_max = min_ave_max(lles_m[2])
ax.errorbar(ms, m_l, yerr=np.stack(
    [(m_l - l_min), (l_max - m_l)]), fmt='x-', capsize=3, label='Noisy')
l_min, m_l, l_max = min_ave_max(lles_m[3])
ax.errorbar(ms, m_l, yerr=np.stack(
    [(m_l - l_min), (l_max - m_l)]), fmt='d-', capsize=3, label='RS')
l_min, m_l, l_max = min_ave_max(lles_m[4])
ax.errorbar(ms, m_l, yerr=np.stack(
    [(m_l - l_min), (l_max - m_l)]), fmt='v-', capsize=3, label='FTS')
l_min, m_l, l_max = min_ave_max(lles_m[5])
ax.errorbar(ms, m_l, yerr=np.stack(
    [(m_l - l_min), (l_max - m_l)]), fmt='^-', capsize=3, label='AAFT')
# plt.plot(ms, med_e_fts, 'v-', label='FT')
# plt.plot(ms, med_e_aafts, '^-', label='AAFT')
ax.set_xlabel(r'$m$')
ax.set_ylabel(r'LLE $\lambda$')
# plt.legend(bbox_to_anchor=(1.0, 1.1), fontsize=14)
# x軸のticks 1-5を表示
ax.set_xticks(np.arange(1,6))
ax.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))

fig.tight_layout()
plt.savefig(os.path.join(save_image_dir, "lle_ms.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "lle_ms.pdf"))

# %%
#TODO: m=1 -- 5 のそれぞれで，傾きのプロットを行う．

log_mds_m = np.array(log_mds_m) # (6, max(m), N_surrogate, n_steps)の配列
for m in ms:

  fig, ax = plt.subplots(figsize=(6, 4))

  
  s_min, m_s, s_max = min_ave_max(log_mds_m[0,m-1,:,:], axis=0)
  s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
  ax.errorbar(steps, m_s, yerr=np.stack(
      [(m_s - s_min), (s_max - m_s)]), fmt='o-b', capsize=3, label='Training', markersize=6)

  s_min, m_s, s_max = min_ave_max(log_mds_m[1,m-1,:,:], axis=0)
  s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
  ax.errorbar(steps, m_s, yerr=np.stack(
      [(m_s - s_min), (s_max - m_s)]), fmt='s-r', capsize=3, label='GAN', markersize=6)

  s_min, m_s, s_max = min_ave_max(log_mds_m[2,m-1,:,:], axis=0)
  s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
  ax.errorbar(steps, m_s, yerr=np.stack(
      [(m_s - s_min), (s_max - m_s)]), fmt='x-', capsize=3, label='Noisy')
  
  s_min, m_s, s_max = min_ave_max(log_mds_m[3,m-1,:,:], axis=0)
  s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
  ax.errorbar(steps, m_s, yerr=np.stack(
      [(m_s - s_min), (s_max - m_s)]), fmt='d-', capsize=3, label='RS')
  
  s_min, m_s, s_max = min_ave_max(log_mds_m[4,m-1,:,:], axis=0)
  s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
  ax.errorbar(steps, m_s, yerr=np.stack(
      [(m_s - s_min), (s_max - m_s)]), fmt='v-', capsize=3, label='FT')
  
  s_min, m_s, s_max = min_ave_max(log_mds_m[5,m-1,:,:], axis=0)
  s_min = np.exp(s_min); m_s = np.exp(m_s); s_max = np.exp(s_max)
  ax.errorbar(steps, m_s, yerr=np.stack(
      [(m_s - s_min), (s_max - m_s)]), fmt='^-', capsize=3, label='AAFT')


  ax.set_yscale('log')
  ax.set_xlabel('k (steps)')
  ax.set_ylabel('exp(S(k))')
  ax.set_xlim([-0.5,11.0]) #type:ignore
  ax.legend(fontsize=14)
  # x軸の値を１ずつ表示
  ax.set_xticks(np.arange(0,11))
  ax.set_title(f"m={m}")
  fig.tight_layout()
  fig.savefig(os.path.join(save_image_dir, f"lle_ms_slope_surrogate_m{m}.png"), dpi=300)
  fig.savefig(os.path.join(save_image_dir, f"lle_ms_slope_surrogate_m{m}.pdf"))


  plt.show()



# %% リカレンスプロット (pending中)
if False:
  pass
  def recurrence_plot(x, theta):
    """
    x: (t_len, dim) timeseries data
    theta: threshold
    """
    x1 = np.reshape(x, [1, -1, x.shape[-1]])
    x2 = np.reshape(x, [-1, 1, x.shape[-1]])
    Dij = np.sqrt(np.sum((x2 - x1)**2, axis=2))
    R = 1.0 * (Dij < theta)
    return R, Dij


# %%
  m = 2
  N_rp = 10000

  if (mapname == 'logistic') or (mapname == 'tent'):
    thr_rp = 0.1
    s_noise = 0.01
  elif (mapname == 'ikeda') or (mapname == 'henon'):
    thr_rp = 0.6
    s_noise = 0.1
  x_noise = x + 0.1 * np.random.randn(x.shape[0], x.shape[1])

  xd = delay_embedding(x, m=m, tau=1)
  xd_gan = delay_embedding(x_sample, m=m, tau=1)
  xd_noise = delay_embedding(x_noise, m=m, tau=1)

  R_orig, Dij_orig = recurrence_plot(xd[0:N_rp, :], thr_rp)
  fig, ax = plt.subplots(figsize=(20, 20))
  ax = plt.imshow(R_orig, interpolation='none', cmap='gray')

# %%
  R_gan, Dij_gan = recurrence_plot(xd_gan[0:N_rp, :], thr_rp)
  fig, ax = plt.subplots(figsize=(20, 20))
  ax = plt.imshow(R_gan, interpolation='none', cmap='gray')


# %%
  R_noise, Dij_noise = recurrence_plot(xd_noise[0:N_rp, :], thr_rp)
  fig, ax = plt.subplots(figsize=(20, 20))
  ax = plt.imshow(R_gan, interpolation='none', cmap='gray')


# %%
  cont_diag_orig = R_orig[0:-1, 0:-1] * R_orig[1:, 1:]
  cont_diag_gan = R_gan[0:-1, 0:-1] * R_gan[1:, 1:]
  cont_diag_noise = R_noise[0:-1, 0:-1] * R_noise[1:, 1:]

# %%
  ncd_orig = np.sum(cont_diag_orig, axis=1)
  ncd_gan = np.sum(cont_diag_gan, axis=1)
  ncd_noise = np.sum(cont_diag_noise, axis=1)
# %%
  plt.plot(ncd_gan)
# %%
  plt.hist(ncd_gan, bins=100)
  plt.show()
  plt.hist(ncd_orig, bins=100)
  plt.show()
  plt.hist(ncd_noise, bins=100)
  plt.show()

# %% TODO: powerlaw パッケージを使ってフィッティングを試す．
# %% 異常点検出を使って，分析する．(いまいちなので廃棄)

  xd_sample = delay_embedding(x_sample, m=2, tau=1)

  from sklearn.neighbors import LocalOutlierFactor

  clf = LocalOutlierFactor(n_neighbors=200)
  label = clf.fit_predict(xd_sample)
# %%
  plt.plot(xd_sample[:, 0], xd_sample[:, 1], '.')
  plt.plot(xd_sample[label == -1, 0], xd_sample[label == -1, 1], 'xg')

# %%
  plt.plot(clf.negative_outlier_factor_)

# %%
  sigma_d_atr = np.sqrt(np.mean(d_atr**2))
  add_log(sigma_d_atr, result_logfile)
  z_d_atr = np.squeeze(d_atr / sigma_d_atr)

  plt.hist(z_d_atr, bins=1000)
# %%
  e_thr = 5
  plt.plot(xd_sample[:, 0], xd_sample[:, 1], '.')
  plt.plot(xd_sample[z_d_atr > e_thr, 0], xd_sample[z_d_atr > e_thr, 1], 'xg')

# %%
  bins = np.arange(0, 100, 1)
  hi = plt.hist(z_d_atr, bins=bins, density=True)

# %%
