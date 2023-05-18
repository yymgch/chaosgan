# -*- coding: utf-8 -*-
# chaosgan_analysis.py
# Nonlinear time series analysis of time series of chaosgan.


#%%
from models import BATCH_SIZE,  NOISE_DIM, SIGMA_PARZEN
import pickle
from nltsa import count_neighbors, average_local_divergence
from nltsa import rs_surrogate, fourier_transform_surrogate, aaft_surrogate
from nltsa import delay_embedding, wayland, e_wayland

from tools import plot_hist2d, random_x0, TrajGenerator, TransitionErrors
from parzen import ParzenWindow, jsd_parzen, kld_parzen
from models import ChaosGANTraining
from chaosmap import f_henon, f_logistic, f_ikeda, f_tent, iterate_f_batch, iterate_f_batch_with_noise
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# import matplotlib.cm as cm  # type:ignore
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

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

S_NOISE = 1e-14  # small noise for tent or Bernouii map

# set map name here
mapname = 'logistic'
# adding small noise for tent map
use_noise = True if mapname == 'tent' else False


checkpoint_dir = os.path.join('training_checkpoints', mapname)
save_image_dir = os.path.join(GEN_IMAGES_DIR, mapname)
saved_data_dir = os.path.join('saved_data', mapname)
result_logfile = os.path.join('saved_data', mapname, 'analysis_result.log')

os.makedirs(save_image_dir, exist_ok=True)
os.makedirs(saved_data_dir, exist_ok=True)


plt.rcParams["font.size"] = 18


def add_log(message, path, display=True):
  # recording to log file.

  if not os.path.exists(path):
    opt = 'w'
  else:
    opt = 'a'

  with open(path, opt) as f:
    print(message, file=f)
  if display:
    print(message)


if __name__ == "__main__":
  # GPU memory setting
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

# %%　入力を入れてみてアウトプットの形をみる．
cgantr.build_model_by_input()
# #%%

add_log(cgantr.generator.summary(), result_logfile)
add_log(cgantr.discriminator.summary(), result_logfile)
# %% dataset をつくる


def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # Parameters of Ikeda map.


def param_henon(bs): return (1.2, 0.4)  # Parameters of Henon map.


def param_tent(bs): return (2.0,)  # Parameters of Henon map


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
                       use_noise=use_noise, s_noise=s_n,  # type:ignore
                       x0min=x0min, x0max=x0max)
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds = tf.data.Dataset.from_generator(
    trjgen, output_types=(tf.float64, tf.float64)).prefetch(buffer_size=AUTOTUNE)

# fixed seed for sampling
seed = tf.random.normal(cgantr.input_shape)


#%% restore model
cgantr.checkpoint.restore(tf.train.latest_checkpoint(cgantr.checkpoint_dir))

#%%loading tr_result  for making figures
with open("saved_data/tr_result.pkl", "rb") as f:
  tr_result = pickle.load(f)

#%% making graphs
fig = plt.figure()
ax = fig.add_subplot(111)  # type:ignore
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

#%% checking pulse response of generator to visualize dependence range on input

pulse_input = np.zeros_like(seed)
pos_pulse = int(seed.shape[1]/2)  # pulse to the center of input
pulse_input[:, pos_pulse, :] = 1


output_pulse = generator(pulse_input)
dif_from_base = np.abs(output_pulse[0, :, 0] - output_pulse[0, 0, 0])
depend_range = (dif_from_base > 1e-5)
dep_ind = np.where(depend_range)[0]
print(dep_ind)
len_dep = np.max(dep_ind) - np.min(dep_ind) + 1

print(f'length of dependency is {len_dep}')

fig, axs = plt.subplots(3, 1, figsize=(10, 7))
axs[0].plot(pulse_input[0, :, 0], label='input')
axs[0].legend()
axs[1].plot(output_pulse[0, :, 0], label='output')
axs[1].legend()
axs[2].plot(1.0*np.array(depend_range[:]), label='dependence')
axs[2].legend()

fig.savefig(os.path.join(save_image_dir, "dependency_range.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "dependency_range.pdf"))

# %% sample
X_nc, p_nc = next(iter(ds))  # get numerical data of trajectory
X_nc = X_nc.numpy()
p_nc = p_nc.numpy()  # p is array of parameters

X_sample = generator(seed, training=False).numpy()


#%% Estimation of error
batch_size = BATCH_SIZE
errors = TransitionErrors(f=f_map)
#error_summary = errors_map(X_sample,p=np.array(param_ikeda(bs=batch_size))) #Transition Error

error_summary = errors(X_sample, p=np.array(
    f_param(bs=batch_size)))  # type:ignore
add_log(f"RMSE: {error_summary['rmse']}", result_logfile)  # calculating RMSE

#%% timeseries sample
#  plt.plot(X_nc[0,0:100,0])
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # type:ignore
ax.plot(np.linspace(0, 10, 10) + X_sample[0:10, 200:400, 0].T)
ax.tick_params(left=False, labelleft=False)
fig.tight_layout()

fig.savefig(os.path.join(save_image_dir, "timeseries_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "timeseries_final.pdf"))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # type:ignore
ax.plot(np.linspace(0, 10, 10) + X_nc[0:10, 0:200, 0].T)
ax.tick_params(left=False, labelleft=False)
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "timeseries_nc.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "timeseries_nc.pdf"))
plt.show()


# %% return map
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)  # type:ignore
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
  ax.set_ylim((-2.5, 1.5))  # type:ignore
if mapname == 'henon':
  ax.set_xlim((-2, 2.1))
  ax.set_ylim((-2, 2))  # type:ignore
if mapname == 'logistic':
  ax.set_xlim((-0.2, 1.2))
  ax.set_ylim((-0.2, 1.2))  # type:ignore
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xy_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xy_final.pdf"))

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1, 1, 1)  # type:ignore
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
  ax.set_ylim((-2.5, 1.5))  # type:ignore
elif mapname == 'henon':
  ax.set_xlim((-2, 2.1))
  ax.set_ylim((-2, 2))  # type:ignore
if mapname == 'logistic':
  ax.set_xlim((-0.2, 1.2))
  ax.set_ylim((-0.2, 1.2))  # type:ignore

fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xy_nc.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xy_nc.pdf"))


#%% density (1-dim, x-only)
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)  # type:ignore

p_sample, bins_sample, _ = ax.hist(X_sample[:, :, 0].flatten(), bins=200,
                                   density=1, color='red', alpha=0.5, label='GAN')  # type:ignore
p_nc, bins_nc, _ = ax.hist(X_nc[:, :, 0].flatten(), bins=200,
                           density=1, color='blue', alpha=0.5, label='Training')  # type:ignore

ax.legend(loc='upper center')
ax.set_xlabel('x')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "xhist1d_final.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "xhist1d_final.pdf"))
fig.show()

# cumulative density
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)  # type:ignore
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
  ax.set_ylim((-2.5, 1.5))  # type:ignore
elif mapname == 'henon':
  ax.set_xlim((-2.0, 2.1))
  ax.set_ylim((-2.0, 2.1))  # type:ignore

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
ax = fig.add_subplot(1, 1, 1)  # type:ignore
ax.plot(np.sum(errors.sqer, 2)[0, 0:1000])  # time series of squeared error
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|^2$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir,
            "error_timeseries_sqer_long.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_sqer_long.pdf"))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # type:ignore
ax.plot(np.sum(errors.sqer, 2)[0, 0:100])
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|^2$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir,
            "error_timeseries_sqer_short.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_sqer_short.pdf"))
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # type:ignore
ax.plot(np.sum(np.abs(errors.er[0, :, :]), 1))  # error time series
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir,
            "error_timeseries_er_long.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_er_long.pdf"))
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # type:ignore
ax.plot(np.sum(np.abs(errors.er[0, 0:100, :]), 1))
ax.set_xlabel('n')
ax.set_ylabel(r'$|e_{n}|$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir,
            "error_timeseries_er_short.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "error_timeseries_er_short.pdf"))
plt.show()


###############
# %% Deterministic property
###############


# set parameters to generate one long time series
bs = 1
transient = 1000
tmax = 100000
t_total = transient + tmax


x0 = random_x0((bs, dim_map), x0min, x0max).transpose(
    [1, 0])  # (dim_map, bs) type:ignore

#%% generate long sequence
params = None
if mapname == 'logistic':
  params = 4.0  # value of a
elif mapname == 'ikeda':
  def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # Ikeda
  params = param_ikeda(bs)
elif mapname == 'henon':
  def param_henon(bs): return (1.2, 0.4)  # Henon
  params = param_henon(bs)
elif mapname == 'tent':
  def param_tent(bs): return (2.0,)  # Tent
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
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array

# set length of time series
if x_sample.shape[0] != x.shape[0]:
  lt = min(x_sample.shape[0] != x.shape[0])
  x_sample = x_sample[0:lt, :]
  x = x[0:lt, :]
add_log(f'length of time series={x_sample.shape[0]}', result_logfile)

# %% errors

# transition error
errors = TransitionErrors(f=f_map)
error_summary = errors(x_sample.reshape(
    [1, -1, dim_map]), p=np.array(f_param(bs=bs)))  # type:ignore
#%%

xd = delay_embedding(x, 2, 1)  # embedding to d*2 dimension
xd_sample = delay_embedding(x_sample, 2, tau=1)


#%% generate noisy time series
x_noise = x + errors.rmse * np.random.randn(x.shape[0], x.shape[1])

# # surrogate
# x_rss = rs_surrogate(x)
# x_fts = fourier_transform_surrogate(x)
# x_aafts = aaft_surrogate(x)

# #%%  Wayland test (1回だけやってみる例)
# med_, e_ = wayland(xd_sample, 50)  # type:ignore

# %% Wayland

ms = np.array(range(1, 6))  # m: 埋め込みの次元

N_surrogate = 100  # number of surrogate data
k_nearest = 100  # number of nearest neighbors
n_res = 1000  # number of reference point


# data arrays
# len_ms = ms.shape[0]
# med_e_orig = []
# med_e_noise = []
# med_e_gan = []
# med_e_rss = []
# med_e_fts = []
# med_e_aafts = []

# for m in ms:
#   print(f'm={m}')
#   med_e_orig.append(e_wayland(x, m, k_nearest, n_res))
#   med_e_noise.append(e_wayland(x_noise, m, k_nearest, n_res))
#   med_e_gan.append(e_wayland(x_sample, m, k_nearest, n_res))
#   med_e_rss.append(e_wayland(x_rss, m, k_nearest, n_res))
#   med_e_fts.append(e_wayland(x_fts, m, k_nearest, n_res))
#   med_e_aafts.append(e_wayland(x_aafts, m, k_nearest, n_res))

# med_e_orig = np.array(med_e_orig)
# med_e_gan = np.array(med_e_gan)
# med_e_noise = np.array(med_e_noise)
# med_e_rss = np.array(med_e_rss)
# med_e_fts = np.array(med_e_fts)
# med_e_aafts = np.array(med_e_aafts)
# # %%
# plt.plot(ms, med_e_orig, 'o-', label='Training', color='blue')
# plt.plot(ms, med_e_noise, '*-', label='noisy')
# plt.plot(ms, med_e_gan, 's-', label='GAN', color='red')
# plt.plot(ms, med_e_rss, 'x-', label='RS')
# plt.plot(ms, med_e_fts, 'v-', label='FT')
# plt.plot(ms, med_e_aafts, '^-', label='AAFT')
# plt.xlabel('m')
# plt.ylabel(r'med($E_{trans}$)')
# plt.legend(bbox_to_anchor=(1.0, 1.1), fontsize=14)
# plt.tight_layout()
# plt.savefig(os.path.join(save_image_dir, "wayland_m.png"), dpi=300)
# plt.savefig(os.path.join(save_image_dir, "wayland_m.pdf"))

# %%
# plt.plot(ms, med_e_orig, 'o-', label='Training', color='blue')
# plt.plot(ms, med_e_noise, '*-', label='noisy')
# plt.plot(ms, med_e_gan, 's-', label='GAN', color='red')
# plt.xlabel('m')
# plt.ylabel(r'med($E_{trans}$)')
# plt.legend(fontsize=14)
# plt.tight_layout()
# plt.savefig(os.path.join(save_image_dir, "wayland_m_enlarge.png"), dpi=300)
# plt.savefig(os.path.join(save_image_dir, "wayland_m_enlarge.pdf"))


# %% statistical test

# TODO: use dict to avoid repetition
med_e_orig = []  # shape will be (len(ms), N_surrogate)
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


def min_ave_max(y, axis=-1):
  '''returns min, average, and max of y along the specified axis'''
  y_min = np.min(y, axis=axis)
  y_max = np.max(y, axis=axis)
  y_mean = np.mean(y, axis=axis)
  return y_min, y_mean, y_max


e_min, m_e, e_max = min_ave_max(med_e_rss)
# %%
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)  # type:ignore
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
plt.legend(fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join(save_image_dir, "wayland_m.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "wayland_m.pdf"))
#%%
fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)  # type:ignore
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
# %%  Estimation of Maximal Lyapunov Exponent
####################

#  Generating long sequence
bs = 1
transient = 1000
tmax = 100000
t_total = transient + tmax
s_noise = 0.01
x0 = random_x0((bs, dim_map), x0min, x0max).transpose(
    [1, 0])  # set initial values

if mapname == 'logistic':
  params = 4.0  # value of a
elif mapname == 'ikeda':
  def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # Parameter of Ikeda map
  params = param_ikeda(bs)
elif mapname == 'henon':
  def param_henon(bs): return (1.2, 0.4)  #
  params = param_henon(bs)
elif mapname == 'tent':
  def param_tent(bs): return (2.0,)  # parameter of tent map
  params = param_tent(bs)

# generate original sequence
if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
x = x.reshape([-1, dim_map])  # original sequence (2-dim array)

init_shape = [bs, tmax // 8 + 13, 64]
seed = tf.random.normal(init_shape)


# generate GAN-sequence
X_sample = generator(seed, training=False).numpy()
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array　（tmax, dim_x)

# %% 誤差の推定 (埋め込んだ空間における真のアトラクタとの距離によって定義する)
errors = TransitionErrors(f=f_map)
error_summary = errors(X_sample, p=np.array(f_param(bs=bs)))  # type:ignore
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
if (mapname == 'logistic') or (mapname == 'tent'):
  radius = 0.01
elif (mapname == 'ikeda') or (mapname == 'henon'):
  radius = 0.08
else:
  radius = 0.01
# %%


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

# %%  comparison to surrogate data

N_surrogate = 100

lles_orig = []  # lles: largest lyapunov exponents
lles_gan = []  # (N_surrogate)
lles_noise = []  # (N_surrogate)
lles_rss = []  #
lles_fts = []
lles_aafts = []

log_mds_orig = []  # (n_steps)
log_mds_gan = []  # (n_steps)
log_mds_noise = []  # (N_surrogate, n_steps)
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
  x = x.reshape([-1, dim_map])  # reshape to 2dim(timax,dim_x)
  #gan-generated data
  seed = tf.random.normal(cgantr.input_shape)

  X_sample = generator(seed, training=False).numpy()
  x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array　（tmax, dim_x)

  # random data generation
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

  log_mds_noise.append(log_md_noise)
  log_mds_rss.append(log_md_rss)
  log_mds_fts.append(log_md_fts)
  log_mds_aafts.append(log_md_aafts)

#%%
# to numpy array
lles_orig = np.array(lles_orig)  # (n_steps, N_surrogate)
lles_gan = np.array(lles_gan)
lles_noise = np.array(lles_noise)  # (n_steps, N_surrogate)
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
s_min = np.exp(s_min)
m_s = np.exp(m_s)
s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='o-b', capsize=3, label='Training', markersize=6)

s_min, m_s, s_max = min_ave_max(log_mds_gan, axis=0)
s_min = np.exp(s_min)
m_s = np.exp(m_s)
s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='s-r', capsize=3, label='GAN', markersize=6)

s_min, m_s, s_max = min_ave_max(log_mds_noise, axis=0)
s_min = np.exp(s_min)
m_s = np.exp(m_s)
s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='x-', capsize=3, label='Noisy')
s_min, m_s, s_max = min_ave_max(log_mds_rss, axis=0)
s_min = np.exp(s_min)
m_s = np.exp(m_s)
s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='d-', capsize=3, label='RS')
s_min, m_s, s_max = min_ave_max(log_mds_fts, axis=0)
s_min = np.exp(s_min)
m_s = np.exp(m_s)
s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='v-', capsize=3, label='FT')
s_min, m_s, s_max = min_ave_max(log_mds_aafts, axis=0)
s_min = np.exp(s_min)
m_s = np.exp(m_s)
s_max = np.exp(s_max)
ax.errorbar(steps, m_s, yerr=np.stack(
    [(m_s - s_min), (s_max - m_s)]), fmt='^-', capsize=3, label='AAFT')


ax.set_yscale('log')
ax.set_xlabel('k (steps)')
ax.set_ylabel('exp(S(k))')
ax.set_xlim([-0.5, 11.0])  # type:ignore
ax.legend(fontsize=14)

ax.set_xticks(np.arange(0, 11))
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir,
            "lle_estimations_surrogate.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "lle_estimations_surrogate.pdf"))

#%% bar-graphs of lle, comparison between original, gan, noise, rss, fts, aafts
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(0, np.mean(lles_orig), yerr=np.std(lles_orig), color='b',
       capsize=3, align='center',  ecolor='black', label='Training')
ax.bar(1, np.mean(lles_gan), yerr=np.std(lles_gan), color='r',
       capsize=3, align='center',  ecolor='black', label='GAN')
ax.bar(2, np.mean(lles_noise), yerr=np.std(lles_noise),
       capsize=3, align='center',  ecolor='black', label='Noisy')
ax.bar(3, np.mean(lles_rss), yerr=np.std(lles_rss),
       capsize=3, align='center',  ecolor='black', label='RS')
ax.bar(4, np.mean(lles_fts), yerr=np.std(lles_fts),
       capsize=3, align='center',  ecolor='black', label='FT')
ax.bar(5, np.mean(lles_aafts), yerr=np.std(lles_aafts),
       capsize=3, align='center',  ecolor='black', label='AAFT')
ax.plot([-0.5, 6], [np.log(2), np.log(2)], '--k', label='Theoretical')


ax.set_ylabel('LLE')
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(
    ['Training', 'GAN', 'Noisy', 'RS', 'FT', 'AAFT'], fontsize=12)
ax.legend(fontsize=12, loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "lle_estimations_bar.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "lle_estimations_bar.pdf"))

#%% log
add_log('-------LLE estimation--------', result_logfile)
add_log(f'Lyapunov Exponent Estimation m={2}', result_logfile)
add_log(f'Theoretical value of LLE: {np.log(2):.6}', result_logfile)

add_log(
    f'LLE estimation from training data (mean+-std): {np.mean(lles_orig):.6} +- {np.std(lles_orig):.6}', result_logfile)
add_log(
    f'LLE estimation from GAN data (mean+-std): {np.mean(lles_gan):.6} +- {np.std(lles_gan):.6}', result_logfile)
add_log(
    f'LLE estimation from noisy data (mean+-std): {np.mean(lles_noise):.6} +- {np.std(lles_noise):.6}', result_logfile)
add_log(
    f'LLE estimation from RS data (mean+-std): {np.mean(lles_rss):.6} +- {np.std(lles_rss):.6}', result_logfile)
add_log(
    f'LLE estimation from FT data (mean+-std): {np.mean(lles_fts):.6} +- {np.std(lles_fts):.6}', result_logfile)
add_log(
    f'LLE estimation from AAFT data (mean+-std): {np.mean(lles_aafts):.6} +- {np.std(lles_aafts):.6}', result_logfile)
