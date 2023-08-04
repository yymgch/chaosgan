# -*- coding: utf-8 -*-
# chaosgan_errors.py
# # chaosgan により生成したデータの誤差に関する分析を行う
#%%
from models import BATCH_SIZE, NOISE_DIM, FULLY_CONV
import pandas as pd
import powerlaw
# from nltsa import count_neighbors, average_local_divergence
# from nltsa import rs_surrogate, fourier_transform_surrogate, aaft_surrogate
from nltsa import delay_embedding # , wayland, e_wayland
from tools import plot_hist2d, random_x0, TrajGenerator, TransitionErrors
# from parzen import ParzenWindow, jsd_parzen, kld_parzen
from models import make_generator_model, make_discriminator_model, ChaosGANTraining
from chaosmap import f_henon, f_logistic, f_ikeda, f_tent, iterate_f_batch, iterate_f_batch_with_noise
# from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KDTree
import datetime
import scipy
import scipy.stats
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

plt.rcParams["savefig.dpi"] = 300

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

# %%


#誤差はなにをもとにするか
er_sig = 'transition'  # d_er をd として使う
#er_sig = 'discrepancy' # d_atr をdとして使う

GEN_IMAGES_DIR = 'gen_images'
output_length_about = 1000
S_NOISE = 1e-14


# set map name here
# TODO: use argparse
mapname = 'logistic' # logistic, tent, henon or ikeda
checkpoint_dir = os.path.join('training_checkpoints', mapname)
save_image_dir = os.path.join(GEN_IMAGES_DIR, mapname)
saved_data_dir = os.path.join('saved_data', mapname)
result_logfile = os.path.join('saved_data', mapname, 'error_result.log')

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

  os.makedirs(save_image_dir, exist_ok=True)
  os.makedirs(saved_data_dir, exist_ok=True)

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
dim_map = 0
f_map = ''
f_param = ''
x0min = 0
x0max = 0

add_log(
    f'Adding log message of chaosgan_errors.py\n{datetime.datetime.today()}', result_logfile)

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

cgantr = ChaosGANTraining(batch_size=BATCH_SIZE, noise_dim=NOISE_DIM,
                          output_length_about=output_length_about, dim_map=dim_map)

generator, discriminator = cgantr.make_models()
generator_optimizer, discriminator_optimizer = cgantr.set_optimizer()
cgantr.set_checkpoint(checkpoint_dir)
# set image dir
cgantr.set_image_dir(GEN_IMAGES_DIR)
# tmax = generator.output_shape[1]-1

# %%　入力を入れてみてアウトプットの形をみる．
cgantr.build_model_by_input()
# #%%
#   noise = tf.random.normal(cgantr.input_shape)  # 乱数発生器
#   generated_timeseries = cgantr.generator(noise, training=False)
#   print("generated_timeseries.shape={}".format(generated_timeseries.shape))
#   d = cgantr.discriminator(generated_timeseries)
#   print("discriminator output shape: {}".format(d.shape))

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
                        use_noise=use_noise, s_noise=s_n,
                        x0min=x0min, x0max=x0max)
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds = tf.data.Dataset.from_generator(
    trjgen, output_types=(tf.float64, tf.float64)).prefetch(buffer_size=AUTOTUNE)

# fixed seed for sampling
seed = tf.random.normal(cgantr.input_shape)


#%% restore model
cgantr.checkpoint.restore(tf.train.latest_checkpoint(cgantr.checkpoint_dir))

# %%

# 長いsequenceを生成する．
bs = 1
transient = 1000
tmax = 100000
t_total = transient + tmax
# s_noise = 0.01 #使われていないのでコメントアウトした


if mapname == 'logistic':
  params = 4.0  # value of a
elif mapname == 'ikeda':
  def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # 池田写像のパラメータ(固定)
  params = param_ikeda(bs)
elif mapname == 'henon':
  def param_henon(bs): return (1.2, 0.4)  #
  params = param_henon(bs)
elif mapname == 'tent':
  def param_tent(bs): return (2.0,)  # Tent写像のパラメータ(固定)
  params = param_tent(bs)
else:
  params = None

x0 = random_x0((bs, dim_map), x0min, x0max).transpose(
    [1, 0])  # (dim_map, bs) #type:ignore

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
print(f'X_sample.shape: {X_sample.shape}')
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array

xd = delay_embedding(x, 2, 1)  # 埋め込み
xd_sample = delay_embedding(x_sample, 2, tau=1)  # 埋め込み

print(xd.shape)
print(xd_sample.shape)

# %%　データのチェック
plt.rcParams["font.size"] = 18

if dim_map == 1:
  plt.plot(x_sample[0:-1, 0], x_sample[1:, 0], '.', alpha=1, ms=1)
  plt.plot(x[0:-1, 0], x[1:, 0], '.', ms=2)
elif dim_map == 2:
  plt.plot(x_sample[:, 0], x_sample[:, 1], '.', alpha=1, ms=1)
  plt.plot(x[:, 0], x[:, 1], '.', ms=0.5)
plt.xlabel('x(n)')
plt.ylabel('x(n+1)')
plt.tight_layout()
# plt.savefig(os.path.join(save_image_dir, "x_sample_and_nc.png"))
# plt.savefig(os.path.join(save_image_dir, "x_sample_and_nc.pdf"))

#%%
# Transition Error
errors = TransitionErrors(f=f_map)
error_summary = errors(X_sample, p=np.array(f_param(bs=bs)))  # type:ignore

er = errors.er.reshape((-1, dim_map))
er_abs = np.abs(errors.er).reshape((-1, dim_map))
d_er = np.sqrt(np.sum(er_abs**2, axis=1))
rmse = np.sqrt(np.mean(np.sum(er_abs**2, axis=1)))
plt.plot(d_er)


# %% discrepancy from true-attractor
tree = KDTree(xd, leaf_size=2)
d_atr, ind = tree.query(xd_sample, k=1, dualtree=True)  # 真のアトラクタからの距離の時系列
d_atr = d_atr[:, 0]
plt.rcParams["font.size"] = 16

###################
#%%選択
##############
d = np.array(0.0)
if er_sig == 'transition':
  d = d_er
elif er_sig == 'discrepancy':
  d = d_atr

#%%

plt.plot(d)
plt.xlabel(r'$t$')
plt.ylabel(r'$d_t$')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_timeseries_all.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_timeseries_all.pdf"))

plt.show()

plt.plot(d[0:2000])
plt.xlabel(r'$t$')
plt.ylabel(r'$d_t$')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_timeseries_short.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_timeseries_short.pdf"))
plt.show()

plt.plot(d[0:500])
plt.xlabel(r'$t$')
plt.ylabel(r'$d_t$')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_timeseries_very_short.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_timeseries_very_short.pdf"))

# %% dが平均0の正規分布に従う乱数の絶対値を観測したと仮定したときの分散，標準偏差
sigma_d = np.sqrt(np.mean(d**2))  # （平均0なので自由度は減らないので，Nで割って良い）
add_log(f'sigma_d= {sigma_d}', result_logfile)

# %% 安定性をみる checking stability
var_cum = np.cumsum(d**2) / (1 + np.arange(d.shape[0]))
plt.plot(var_cum)  # checking convergence
# %% to z-value
z_d = np.squeeze(d / sigma_d)
plt.plot(z_d)
# %% histogram
d_density, d_bin, batches = plt.hist(d, bins=1000, density=True)

############
#   drawing complementary cumulative distribution function
############
#%%
sd = np.sort(d)  # ascending order 分布のグラフのx軸としても使う
d_rank = (np.array(range(sd.size)) + 1)
P_d = 1 - (d_rank - 1) / d_rank.size  # 累積補分布

norm_dist = scipy.stats.distributions.norm(scale=sigma_d)
# norm_dist = scipy.stats.distributions.norm(scale=0.013)
P_norm_x = 1 - 2 * (norm_dist.cdf(sd) - 0.5)  # ガウス分布と仮定したときの累積補分布

# %% 指数分布を仮定したときのフィッティング
loc, scale = scipy.stats.distributions.expon.fit(d, floc=0)
exp_dist = scipy.stats.distributions.expon(scale=scale)
P_exp = exp_dist.sf(sd)  # 累積補分布, 指数

# %% 対数正規分布を仮定したときのフィッティング
s, loc, scale = scipy.stats.distributions.lognorm.fit(d, floc=0)
lognorm_dist = scipy.stats.distributions.lognorm(s, loc=loc, scale=scale)
P_lognorm = lognorm_dist.sf(sd)  # 累積補分布

# %% ガンマ分布を仮定したときのフィッティング
a, loc, scale = scipy.stats.distributions.gamma.fit(d, floc=0)
gamma_dist = scipy.stats.distributions.gamma(a, loc=loc, scale=scale)
P_gamma = gamma_dist.sf(sd)  # 累積補分布


# %% z-value への変換, 外れ値を排除してからfit
z_d = (d - np.mean(d))/np.std(d)

# z>3 の値を排除
d_tranc = d[z_d < 3.0]
print(d_tranc.shape)
sigma_d_tranc = np.sqrt(np.mean(d_tranc**2))  # （平均0なので自由度は減らないので，Nで割って良い）
print(f'sigma_d= {sigma_d_tranc}')
norm_dist_tranc = scipy.stats.distributions.norm(scale=sigma_d_tranc)
# norm_dist = scipy.stats.distributions.norm(scale=0.013)
P_norm_tranc = 1 - 2 * (norm_dist_tranc.cdf(sd) - 0.5)  # ガウス分布と仮定したときの累積補分布


# %%　累積補分布のフィッティング


plt.plot(sd, P_d, '.-', label='data')
plt.plot(sd, P_norm_x, label='normal')
# plt.plot(sd, P_norm_tranc, label='normal_tranc')
plt.plot(sd, P_exp, label='exponential')
plt.plot(sd, P_lognorm, ':', label='log-normal')
plt.plot(sd, P_gamma, ':', label='gamma')
plt.xlabel(r'$d$')
plt.ylabel(r'$\bar{P}(d)$')
plt.legend()
#plt.yscale('log')
#plt.xscale('log')
# plt.ylim([1e-5, 10])
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_ccdf_normal.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_ccdf_normal.pdf"))

plt.show()

plt.plot(sd, P_d, '.-', label='data')
plt.plot(sd, P_norm_x, '--', label='normal')
# plt.plot(sd, P_norm_tranc, ':', label='normal_tranc')
plt.plot(sd, P_exp, '-.', label='exponential')
plt.plot(sd, P_lognorm, ':', label='log-normal')
plt.plot(sd, P_gamma, ':', label='gamma')
plt.xlabel(r'$d$')
plt.ylabel(r'$\bar{P}(d)$')
plt.legend()
plt.yscale('log')
#plt.xscale('log')
plt.ylim([1e-5, 10])
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_ccdf_semilog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_ccdf_semilog.pdf"))
plt.show()

plt.plot(sd, P_d, '.-', label='data')
plt.plot(sd, P_norm_x, '--', label='normal')
# plt.plot(sd, P_norm_tranc, ':', label='normal_tranc')
plt.plot(sd, P_exp, '-.', label='exponential')
plt.plot(sd, P_lognorm, ':', label='log-normal')
plt.plot(sd, P_gamma, ':', label='gamma')
plt.xlabel(r'$d$')
plt.ylabel(r'$\bar{P}(d)$')
plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-5, 10])
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_ccdf_loglog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_ccdf_loglog.pdf"))

plt.show()


# %% density functions
log_bin = np.logspace(np.log10(0.001), np.log10(1.0), 200)
d_density, d_bin, batches = plt.hist(
    d, bins=log_bin, density=True)  # type:ignore

#%% truncated
d_density_tranc, d_bin_tranc, batches_tranc = plt.hist(
    d_tranc, bins=log_bin, density=True)  # type:ignore
# %%
plt.plot(d_bin[1:], d_density, '.', label='data')
plt.plot(d_bin, 2 * norm_dist.pdf(d_bin), label='normal')
# plt.plot(d_bin, 2 * norm_dist_tranc.pdf(d_bin), label='normal_tranc')
plt.plot(d_bin, exp_dist.pdf(d_bin), label='exp')
plt.plot(d_bin, lognorm_dist.pdf(d_bin), label='log-normal')
plt.plot(d_bin, gamma_dist.pdf(d_bin), label='gamma')
plt.legend()
plt.xlabel(r'$d$')
plt.ylabel('density')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_pdf_1.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_pdf_1.pdf"))

plt.show()

plt.plot(d_bin[1:], d_density, '.', label='data')
plt.plot(d_bin, 2 * norm_dist.pdf(d_bin), label='normal')
# plt.plot(d_bin, 2 * norm_dist_tranc.pdf(d_bin), label='normal_tranc')
plt.plot(d_bin, exp_dist.pdf(d_bin), label='exp')
plt.plot(d_bin, lognorm_dist.pdf(d_bin), label='log-normal')
plt.plot(d_bin, gamma_dist.pdf(d_bin), label='gamma')
plt.legend()
plt.xlim([-0.01, 0.1])
plt.xlabel(r'$d$')
plt.ylabel('density')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_pdf_2.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_pdf_2.pdf"))
plt.show()

plt.plot(d_bin[1:], d_density, '.', label='data')
plt.plot(d_bin, 2 * norm_dist.pdf(d_bin), label='normal')
# plt.plot(d_bin, 2 * norm_dist_tranc.pdf(d_bin), label='normal_tranc')
plt.plot(d_bin, exp_dist.pdf(d_bin), label='exp')
plt.plot(d_bin, lognorm_dist.pdf(d_bin), label='log-normal')
plt.plot(d_bin, gamma_dist.pdf(d_bin), label='gamma')
plt.legend()
# plt.xlim([-0.01, 0.1])
plt.xlabel(r'$d$')
plt.ylabel('density')
plt.yscale('log')
plt.ylim([1e-5, 100])
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_pdf_semilog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_pdf_semilog.pdf"))
plt.show()

plt.plot(d_bin[0:-1], d_density, '.', label='data')
plt.plot(d_bin, 2 * norm_dist.pdf(d_bin), label='normal')
# plt.plot(d_bin, 2 * norm_dist_tranc.pdf(d_bin), label='normal_tranc')
plt.plot(d_bin, exp_dist.pdf(d_bin), label='exp')
plt.plot(d_bin, lognorm_dist.pdf(d_bin), label='log-normal')
# plt.plot(d_bin, gamma_dist.pdf(d_bin), label='gamma')
plt.legend()
# plt.xlim([-0.01, 0.1])
plt.xlabel('d')
plt.ylabel('density')
plt.yscale('log')
plt.xscale('log')
plt.ylim([1e-5, 100])
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_pdf_loglog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "d_pdf_loglog.pdf"))

############
#% autocorrelation function of d
# purpose: check if there is no periodicity in d (check if there is no periodicity due to filter size or upsampling)
############

d_nom = (d - d.mean())/d.std()  # normalization
pow_d = np.abs(np.fft.fft(d_nom))**2  # power spectrum
# calculating autocorrelation function by ifft. normalize by the number of elements.
ac_d = np.abs(np.fft.ifft(pow_d))/d_nom.shape[0]

fig, ax = plt.subplots()
ax.plot(ac_d[0:100], '.-')
ax.set_xlabel('s')
ax.set_ylabel('auto-correlation')
ax.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_autocorrelation.png"), dpi=300 )
plt.savefig(os.path.join(save_image_dir, "d_autocorrelation.pdf"))

#################################
#%% 
# N-step error
#################################

# 軌道
if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
x = x.reshape([-1, dim_map])
# 生成
init_shape = [1, tmax // 8 + 13, 64]
seed = tf.random.normal(init_shape)
X_sample = generator(seed, training=False).numpy()
print(f'X_sample.shape: {X_sample.shape}')
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array

# x_sample の値を初期値として，10ステップ先までの軌道を生成
n_init = 99990 # シフトする分，減らす
x0_sample = x_sample[0:n_init,:] # (n_init, dim_map)

pr = np.array(f_param(bs=x0_sample.shape[0])).T
#%% n-step trajectory
x_n_traj = [x0_sample] #(about_tmax, dim_map)
x_cur =  x0_sample
for n in range(10):
  x_next = f_map(x_cur, param=pr)
  x_n_traj.append(x_next)
  x_cur = x_next

x_n_traj = np.stack(x_n_traj, axis=0) # (10,n_init, dim_map)

plt.plot(x_n_traj[0,:,0], x_n_traj[1,:,0], '.') # return map


#%%


#%%


#%% make n-step GAN trajectory by shifting
x_n_traj_gan = []
#x0_gan = x_sample[0:n_init,:] # (n_init, dim_map)

for n in range(11):
  x_n_traj_gan.append(x_sample[n:n+n_init,:])
x_n_traj_gan = np.stack(x_n_traj_gan, axis=0) # (11,n_init, dim_map)

print(x_n_traj_gan.shape)
plt.plot(x_n_traj_gan[0,:,0], x_n_traj_gan[1,:,0], '.')

# remove trajectory that start outside [0,1] (for logistic map)
if mapname == 'logistic':
  in_01 = (x0_sample[:,0] < 1.0) & (x0_sample[:,0]>=0.0)

  x_n_traj_01 = x_n_traj[:, in_01, :]

  x0_sample_01 = x0_sample[in_01, :]
  plt.plot(x_n_traj_01[0,:,0], x_n_traj_01[1,:,0], '.')

  x_n_traj_gan_01 = x_n_traj_gan[:, in_01, :]

  sqer_nstep = np.sum((x_n_traj_gan_01 - x_n_traj_01)**2, axis=2) # (11, n_init) type:ignore
else:
  sqer_nstep = np.sum((x_n_traj_gan - x_n_traj)**2, axis=2) # (11, n_init) type:ignore


rmse_nstep = np.sqrt(np.mean(sqer_nstep, axis=1))
mae_nstep = np.mean(np.sqrt(sqer_nstep), axis=1)



fig, ax = plt.subplots()
ax.plot(np.arange(1,11),rmse_nstep[1:], '.-')
ax.plot(np.arange(1,6), 0.03*np.exp(np.log(2.0)*np.arange(1,6)), 'k--' )
ax.set_xlabel('n')
ax.set_ylabel('E(n)')
ax.set_yscale('log')
ax.set_ylim([1e-2,1])
ax.set_xticks(range(1,11))
ax.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_nstep_sqer.png"), dpi=300 )
plt.savefig(os.path.join(save_image_dir, "d_nstep_sqer.pdf"))

fig, ax = plt.subplots()
ax.plot(np.arange(1,11),mae_nstep[1:], '.-')
ax.plot(np.arange(1,6), 0.02*np.exp(np.log(2.0)*np.arange(1,6)), 'k--' )
ax.set_xlabel('n')
ax.set_ylabel('E(n)')
ax.set_yscale('log')
ax.set_ylim([1e-2,1])
ax.set_xticks(range(1,11))
ax.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "d_nstep_mae.png"), dpi=300 )
plt.savefig(os.path.join(save_image_dir, "d_nstep_mae.pdf"))

##############################
# %%
#   Statistics of fragment length 
###############################

if (mapname == 'logistic') or (mapname == 'tent'):
  threshold = 0.1
elif (mapname == 'ikeda') or (mapname == 'henon'):
  threshold = 0.4
else:
  threshold = 0.0
n_sample_seq = 100


def gen_fragment_length(threshold, n_sample_seq=100, tmax=100000):
  '''
    doing sampling multiple times and collecting sampling of fragment length 
    args:
      threshold: threshold value for fragment length
      n_sample_seq: number of sampling sequences
  '''

  
  len_fr = []
  x0 = random_x0((bs, dim_map), x0min, x0max).transpose(
      [1, 0])  # (dim_map, bs)
  if use_noise:
    x = iterate_f_batch_with_noise(
        x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
  else:
    x = iterate_f_batch(
        x0, f_map, tmax=tmax, transient=transient, param=params)
  x = x.reshape([-1, dim_map])
  xd = delay_embedding(x, 2, 1)  # teaching data
  tree = KDTree(xd, leaf_size=2)  # tree

  for n in range(n_sample_seq):
    seed = tf.random.normal(init_shape)  # random seed
    X_sample = generator(seed, training=False).numpy()  # generate sequence
    x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array (t_len, dim)
    xd_sample = delay_embedding(x_sample, 2, tau=1)  # delay-embedding
    d = np.array(0.0)
    if er_sig == 'transition':
      errors = TransitionErrors(f=f_map)
      error_summary = errors(
          X_sample, p=np.array(f_param(bs=bs)))  # type:ignore

      er = errors.er.reshape((-1, dim_map))
      er_abs = np.abs(errors.er).reshape((-1, dim_map))
      d_er = np.sqrt(np.sum(er_abs**2, axis=1))
      d = d_er
    elif er_sig == 'discrepancy':
      d_atr, ind = tree.query(
          xd_sample, k=1, dualtree=True)  # 真のアトラクタからの距離の時系列
      d_atr = d_atr[:, 0]
      d = d_atr

    fr_ind = np.where(d > threshold)  # threshold より高い値の位置
    l_fr = np.diff(fr_ind[0])  # 隣あうindex間の差を取ることにより長さを計算
    len_fr.append(l_fr)
    # print(n)

  len_fr = np.concatenate(len_fr, axis=0)
  return len_fr


len_fr = gen_fragment_length(threshold, n_sample_seq, tmax=tmax)
#plt.plot(l_fr)

#%% 大変位と判定された点のプロット

if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
x = x.reshape([-1, dim_map])
xd = delay_embedding(x, 2, 1)  # teaching data
tree = KDTree(xd, leaf_size=2)  # tree

seed = tf.random.normal(init_shape)  # random seed
X_sample = generator(seed, training=False).numpy()  # generate sequence
x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array (t_len, dim)
xd_sample = delay_embedding(x_sample, 2, tau=1)  # delay-embedding

# d_atr, ind = tree.query(xd_sample, k=1, dualtree=True)  # 真のアトラクタからの距離の時系列
# d_atr = d_atr[:, 0]

if er_sig == 'transition':
  errors = TransitionErrors(f=f_map)
  error_summary = errors(X_sample, p=np.array(f_param(bs=bs)))  # type:ignore

  er = errors.er.reshape((-1, dim_map))
  er_abs = np.abs(errors.er).reshape((-1, dim_map))
  d_er = np.sqrt(np.sum(er_abs**2, axis=1))
  d = d_er
elif er_sig == 'discrepancy':
  d_atr, ind = tree.query(xd_sample, k=1, dualtree=True)  # 真のアトラクタからの距離の時系列
  d_atr = d_atr[:, 0]
  d = d_atr

fr_ind = np.where(d > threshold)  # threshold より高い値の位置

# l_fr = np.diff(fr_ind[0])# 隣あうindex間の差を取ることにより長さを計算
#len_fr.append(l_fr)


fig, ax = plt.subplots()
if dim_map ==1:
  ax.plot(xd_sample[:, 0], xd_sample[:, 1], '.', ms=2)
  ax.plot(xd_sample[fr_ind[0], 0], xd_sample[fr_ind[0], 1], 'x', ms=4)
else:
  ax.plot(xd_sample[:, dim_map], xd_sample[:, dim_map+1], '.', ms=2)
  ax.plot(xd_sample[fr_ind[0], dim_map], xd_sample[fr_ind[0], dim_map+1], 'x', ms=4)

ax.text(0.8, 0.8, fr'$\lambda={threshold:.2}$')
fig.tight_layout()
fig.savefig(os.path.join(save_image_dir, "frl_large_d.png"), dpi=300)
fig.savefig(os.path.join(save_image_dir, "frl_large_d.pdf"))

# %% ヒストグラム
count, bin, batches = plt.hist(len_fr, bins=100, density=False)
plt.xlabel('length')
plt.ylabel('count')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "frl_hist.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "frl_hist.pdf"))

# %% semilog ヒストグラム
plt.plot(bin[1:], count, '.-')
# plt.xscale('log')
plt.xlabel('length', fontsize=18)
plt.ylabel('count', fontsize=18)
plt.yscale('log')
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "frl_hist_semilog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "frl_hist_semilog.pdf"))

# %% removing length 1
len_fr_mod = len_fr[len_fr > 4]
count, bin, batches = plt.hist(len_fr_mod, bins=100, density=False)
#%%
plt.plot(bin[1:], count, '.-')
plt.xlabel('length')
plt.ylabel('count')
# plt.xscale('log')
plt.yscale('log')
m_fr_mod = np.mean(len_fr_mod)
print(m_fr_mod)
plt.tight_layout()
plt.savefig(os.path.join(save_image_dir, "frl_hist_semilog_remove4.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "frl_hist_semilog_remove4.pdf"))


# %% trying analysis in various threshold
if (mapname == 'logistic') or (mapname == 'tent'):

  thresholds = [0.02, 0.05, 0.08, 0.1, 0.13, 0.15, 0.2]
else:
  thresholds = [0.05, 0.1, 0.2, 0.4, 0.5, 0.8]

n_sample_seq = 100  # サンプル時系列の数

len_frs = []  # 閾値ごとにリストをつくる
for _ in thresholds:
  len_frs.append([])
#%%
x0 = random_x0((bs, dim_map), x0min, x0max).transpose([1, 0])  # (dim_map, bs)
if use_noise:
  x = iterate_f_batch_with_noise(
      x0, f_map, tmax=tmax, transient=transient, param=params, s=S_NOISE)
else:
  x = iterate_f_batch(
      x0, f_map, tmax=tmax, transient=transient, param=params)
x = x.reshape([-1, dim_map])
xd = delay_embedding(x, 2, 1)
tree = KDTree(xd, leaf_size=2)

for n in range(n_sample_seq):
  seed = tf.random.normal(init_shape)
  X_sample = generator(seed, training=False).numpy()
  x_sample = X_sample[0, :tmax + 1, :]  # change to 2dim-array
  xd_sample = delay_embedding(x_sample, 2, tau=1)
  if er_sig == 'transition':
    errors = TransitionErrors(f=f_map)
    error_summary = errors(X_sample, p=np.array(f_param(bs=bs)))  # type:ignore

    er = errors.er.reshape((-1, dim_map))
    er_abs = np.abs(errors.er).reshape((-1, dim_map))
    d_er = np.sqrt(np.sum(er_abs**2, axis=1))
    d = d_er
  elif er_sig == 'discrepancy':
    d_atr, ind = tree.query(xd_sample, k=1, dualtree=True)  # 真のアトラクタからの距離の時系列
    d_atr = d_atr[:, 0]
    d = d_atr

  # threshold を変えて断片長をみてる．
  for list_fr, thr in zip(len_frs, thresholds):
    fr_ind = np.where(d > thr)
    l_fr = np.diff(fr_ind[0])
    list_fr.append(l_fr)
  # print(n)

for i in range(len(len_frs)):
  len_frs[i] = np.concatenate(len_frs[i], axis=0)

n_frs = np.array([len(ls) for ls in len_frs])  # フラグメントの数

#%%
probs = []
bins = []

x_max_bin = 4000
if mapname == 'logistic' or mapname == 'tent':
  x_max_bin = 4000
else:
  x_max_bin = 2000

for i in range(len(thresholds)):
  bin_size = min(len_frs[i].max(), 100)
  bin = np.linspace(1, x_max_bin, 100)
  prob, bin, batches = plt.hist(len_frs[i], bins=bin, density=True) #type:ignore
  probs.append(prob)
  bins.append(bin)


fig, ax = plt.subplots()
for i in range(len(thresholds)):
  lab = fr'$\lambda = {thresholds[i]:.3}$'
  ax.plot(bins[i][1:], probs[i], '.-', label=lab, ms=2)
ax.set_xlim((0, x_max_bin))
ax.legend()
ax.set_xlabel('fragment length', fontsize=18)
ax.set_ylabel('density', fontsize=18)
ax.set_yscale('log')
fig.tight_layout()


plt.savefig(os.path.join(save_image_dir, "frl_hist_thr_semilog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "frl_hist_thr_semilog.pdf"))

fig, ax = plt.subplots(figsize=(7, 4))
stl = ['-', '-', '--', ':', '-.']

if mapname == 'logistic':
  th_subset = [1,3,5,6]
else:
  th_subset = [0,1,2,3]


for j, i in enumerate(th_subset):
  lab = fr'$\lambda = {thresholds[i]:.3}$'
  ax.plot(bins[i][1:], probs[i], stl[j], label=lab, ms=2)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
ax.set_xlabel('fragment length')
ax.set_ylabel('density')
ax.set_yscale('log')
ax.set_xlim((0, x_max_bin/4))
fig.tight_layout()

plt.savefig(os.path.join(save_image_dir, "frl_hist_thr_semilog2.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "frl_hist_thr_semilog2.pdf"))

#%%

fig, ax = plt.subplots()
i = 3 # lambda = 0.1 for logistic map

lab = fr'$\lambda = {thresholds[i]:.3}$'
ax.plot(bins[i][1:], probs[i], '.-', label=lab, ms=2)
ax.set_xlim((0, 1200))
ax.legend()
ax.set_xlabel('fragment length', fontsize=18)
ax.set_ylabel('density', fontsize=18)
ax.set_yscale('log')
fig.tight_layout()

plt.savefig(os.path.join(save_image_dir, "frl_hist_l01_semilog.png"), dpi=300)
plt.savefig(os.path.join(save_image_dir, "frl_density_l01_semilog.pdf"))


## ここまで 3/1
## 論文で使用したのはここまで．ここから先は実験的なもの．

#%% k-ステップ誤差をつくる

errors = TransitionErrors(f=f_map)
error_summary = errors(X_sample, p=np.array(f_param(bs=bs)))  # type:ignore

er = errors.er.reshape((-1, dim_map))
er_abs = np.abs(errors.er).reshape((-1, dim_map))
d_er = np.sqrt(np.sum(er_abs**2, axis=1))

def f_logi_safe(X, epsi=1e-10,a=4.0):
  X_new = a*X*(1.0-X)
  X_new = np.minimum(np.maximum(epsi,X_new), 1-epsi)
  return X_new


def k_step_error(X, f, k_max):
  '''
    args:
      X: (len_t,dim_map)
      f: map
      k_max: maximum progress
  '''
  len_t = X.shape[0]
  dim_map = X.shape[1]

  D = -0.2*np.ones((len_t, len_t))
  X_cur = X.transpose()  # (dim_map, len_t)
  print(X_cur.shape)
  Xk = []
  for k in range(k_max):
    Xk.append(X_cur)
    X_cur = f(X_cur)
  Xk.append(X_cur)
  Xk = np.array(Xk).transpose((2,0,1))  # k_max, dim_map, len_t -> (len_t, k_max+1, dim_map) 
  print(Xk.shape)

  for n in range(len_t):
    m_max = min(n+k_max,len_t-1)
    k_m = m_max-n
    # print(f'{n},{m_max}, {k_m}' )
    # print( Xk[n,:k_m+1,:].shape)
    d_nk =  np.sum((X[n:(m_max+1),:]- Xk[n,:(k_m+1),:])**2,1)  # shape (k_m)
    D[n,n:m_max+1] = d_nk


  return D


D = k_step_error(X_sample[0, 0:1000, :], f_logi_safe, 100)
#%%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
ax.imshow(np.log(D[300:400,300:400]))

#%%
# %% TODO: powerlawパッケージを使ってみる

# %% dから分布を推定する
d_atr.shape  # (100000,)

# %% 分布へのフィッティング
fit = powerlaw.Fit(d_atr, xmin=np.min(d_atr))  # xmin指定して，低い部分をカットしないようにしておく．
fit2 = powerlaw.Fit(d_atr)

# %%
# print(fit.power_law.alpha) # べき分布の パラメータ alpha の値 (pdfのべきの指数)
# print(fit.power_law.sigma) # べき分布の sigma の値
# 尤度比検定．値が正なら前者によりフィットしている． (R, p) pはsignificance
(R, p) = fit.distribution_compare('power_law', 'exponential')
add_log(f'powerlaw vs exponential: {(R,p)}', result_logfile)
# 尤度比検定．値が正なら前者によりフィットしている． (R, p) pはsignificance
(R, p) = fit.distribution_compare('truncated_power_law', 'exponential')
add_log(f'truncated powerlaw vs exponential: {(R,p)}', result_logfile)
# 尤度比検定．値が正なら前者によりフィットしている． (R, p) pはsignificance
(R, p) = fit.distribution_compare('stretched_exponential', 'exponential')
add_log(f'Stretched exponential vs exponential: {(R,p)}', result_logfile)
(R, p) = fit.distribution_compare('stretched_exponential',
                                  'truncated_power_law')  # 尤度比検定．値が正なら前者によりフィットしている． (R, p) pはsignificance
add_log(
    f'Stretched exponential vs truncated power law: {(R,p)}', result_logfile)
# まとめ stretched exponential (わいぶる)に最もフィットするらしいが？　しかし二つの分布のミクスチャーとみれそうな分布なので，あまり意味ないのでは？
# %% 表を作る

candidate_distributions = ('power_law', 'lognormal', 'exponential',
                           'truncated_power_law', 'stretched_exponential')
ratios_df = pd.DataFrame(index=candidate_distributions,
                         columns=candidate_distributions)
p_df = pd.DataFrame(index=candidate_distributions,
                    columns=candidate_distributions)

#%% 総当たり
for dist_name1 in candidate_distributions:
  for dist_name2 in candidate_distributions:
    if dist_name1 == dist_name2:
      continue
    (R, p) = fit.distribution_compare(dist_name1, dist_name2)
    ratios_df.loc[dist_name1, dist_name2] = R
    p_df.loc[dist_name1, dist_name2] = p
signif_sign = ratios_df.applymap(np.sign) * (p_df < 0.1)
# %%
print(signif_sign)
#%%
print(p_df)
# %% 分布にフィットしたPDFグラフをみてみる．

fig_pdf, ax = plt.subplots(1, 1, figsize=(8, 6))

fit.plot_pdf(label='data', linewidth=2, ax=ax)
fit.stretched_exponential.plot_pdf(label='stretched_exp', ax=ax)
fit.exponential.plot_pdf(label='exp', ax=ax)
fit.truncated_power_law.plot_pdf(label='truncated power law', ax=ax)
fit.lognormal.plot_pdf(label='log-normal', ax=ax)
# fit.power_law.plot_pdf(label= 'power_law')
ax.set_ylim([10e-5, 1000.0]) #type:ignore
#plt.xscale('linear')
ax.set_xlabel('d')
ax.set_ylabel('density')
# plt.legend()
ax.legend(bbox_to_anchor=(1.5, 1.0), fontsize=14)
plt.savefig(os.path.join(save_image_dir, "plaw_d_loglog.png"))
plt.savefig(os.path.join(save_image_dir, "plaw_d_loglog.pdf"))

#%% ccdf

fig_ccdf, ax = plt.subplots(1, 1, figsize=(8, 5))
fit.plot_ccdf(label='data', linewidth=2, ax=ax)
fit.stretched_exponential.plot_ccdf(label='stretched_exp', ax=ax)

fit.exponential.plot_ccdf(label='exp', ax=ax)
fit.truncated_power_law.plot_ccdf(label='truncated power law', ax=ax)
fit.lognormal.plot_ccdf(label='log-normal', ax=ax)
# fit.power_law.plot_pdf(label= 'power_law')

if (mapname == 'logistic'):
  ax.set_ylim([10e-6, 1.0]) #type:ignore
elif (mapname == 'ikeda') or (mapname == 'henon'):
  ax.set_ylim([10e-6, 0.6]) #type:ignore
elif (mapname == 'tent'):
  ax.set_ylim([10e-6, 1.1]) #type:ignore

#plt.xscale('linear')
ax.set_xlabel('d')
ax.set_ylabel('1-P(d)')
ax.legend(bbox_to_anchor=(1.0, 1.1), fontsize=14)
fig_ccdf.tight_layout()

plt.savefig(os.path.join(save_image_dir, "plaw_d_ccdf_loglog.png"))
plt.savefig(os.path.join(save_image_dir, "plaw_d_ccdf_loglog.pdf"))


# %% 長さl の分布
threshold = 0.05
len_fr = gen_fragment_length(threshold=threshold, n_sample_seq=100)
#%%
fit_frl = powerlaw.Fit(len_fr, discrete=True, xmin=5)
# %%
fit_frl.distribution_compare('exponential', 'power_law')

# %%
fit_frl.plot_pdf(label='data', linewidth=3)
fit_frl.exponential.plot_pdf(label='exp')
fit_frl.stretched_exponential.plot_pdf(label='Weibull')
plt.legend()
plt.savefig(os.path.join(save_image_dir, "plaw_frl_loglog.png"))
plt.savefig(os.path.join(save_image_dir, "plaw_frl_loglog.pdf"))
# plt.xscale('linear')

# %%
fit_frl.distribution_compare('exponential', 'stretched_exponential')

