# -*- coding: utf-8 -*-
#Training ChaosGAN
#%% import


import pickle
from models import make_generator_model, make_discriminator_model, ChaosGANTraining
from tools import plot_hist2d, TrajGenerator, TransitionErrors, random_x0
from chaosmap import f_henon, f_logistic, f_ikeda, f_tent, iterate_f_batch, iterate_f_batch_with_noise
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
# from tensorflow.keras import layers
from models import BATCH_SIZE, NOISE_DIM, SIGMA_PARZEN, FULLY_CONV

from IPython import display
import os

# import other files
import importlib
import chaosmap
import models

importlib.reload(chaosmap)
importlib.reload(models)


GEN_IMAGES_DIR = 'gen_images'
output_length_about = 1000
ITERATION = 250_000
# ITERATION = 1000


GIF_OUTPUT = True
S_NOISE = 1e-14

# fixing random seed
np.random.seed(100)
tf.random.set_seed(101)


# set map name here
mapname = 'logistic'  # logistic ikeda tent henon
checkpoint_dir = os.path.join('training_checkpoints', mapname)
save_image_dir = os.path.join(GEN_IMAGES_DIR, mapname)
saved_data_dir = os.path.join('saved_data', mapname)
result_logfile = os.path.join('saved_data', mapname, 'train_result.log')
gif_image_dir = os.path.join(GEN_IMAGES_DIR, mapname, 'train_gif')

#%%


def disp_1d(X_sample):
  plt.plot(X_sample[:, 0:-2, 0].T, X_sample[:, 1:-1, 0].T, '.', markersize=0.5)
  plt.show()
  plt.plot(np.linspace(0, 4, 5) + X_sample[0:5, 0:100, 0].T)
  plt.show()


def disp_2d(X_sample):
  plt.plot(X_sample[:, :, 0].T, X_sample[:, :, 1].T, '.', markersize=0.5)
  plt.show()
  plt.plot(np.linspace(0, 4, 5) + X_sample[0:5, 0:100, 0].T)
  plt.show()


#%%
if __name__ == "__main__":
  # setting GPU memory growth mode
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(
            physical_devices[k]))
  else:
    print("Not enough GPU hardware devices available")

  os.makedirs(save_image_dir, exist_ok=True)
  #matplotlib off-screen switching
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


# %% making discriminator and generator model

# setting map and parameters

# dummy for avoiding warning
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
elif mapname == 'henon':
  f_map = f_henon
  dim_map = 2
  x0min = np.array([-0.1, -0.1])
  x0max = np.array([0.1, 0.1])
elif mapname == 'logistic':
  f_map = f_logistic
  dim_map = 1
  x0min = np.array([0.0])
  x0max = np.array([1.0])
elif mapname == 'tent':
  f_map = f_tent
  dim_map = 1
  x0min = np.array([0.0])
  x0max = np.array([1.0])

cgantr = ChaosGANTraining(batch_size=BATCH_SIZE, noise_dim=NOISE_DIM,
                          output_length_about=output_length_about, dim_map=dim_map,
                          fully_conv=FULLY_CONV,
                          sigma_parzen=SIGMA_PARZEN)

# models and optimizers
generator, discriminator = cgantr.make_models()
generator_optimizer, discriminator_optimizer = cgantr.set_optimizer()
# set checkpoints
cgantr.set_checkpoint(checkpoint_dir)
# set image dir
cgantr.set_image_dir(save_image_dir)
# tmax = generator.output_shape[1]-1

# %%　checking output shape

cgantr.build_model_by_input()

# %% making dataset


def param_ikeda(bs): return (1.0, 0.4, 0.9, 6.0)  # parameter of Ikeda map


def param_henon(bs): return (1.2, 0.4)  # Henon


def param_tent(bs): return (2.0,)  # Tent


def gen_as(bs, low=4.0, high=4.0):
    '''return parameter values for logistic maps'''
    return np.random.uniform(low=low, high=high, size=(1, bs))  # logistic


use_noise = False
s_noise = 0

if mapname == 'ikeda':
  f_param = param_ikeda
elif mapname == 'logistic':
  f_param = gen_as
elif mapname == 'henon':
  f_param = param_henon
elif mapname == 'tent':
  f_param = param_tent
  use_noise = True
  s_noise = S_NOISE

# trajectory generator for chaos map
trjgen = TrajGenerator(batch_size=BATCH_SIZE, dim=dim_map,
                       f=f_map, tmax=cgantr.tmax, gen_param=f_param, transient=100,
                       use_noise=use_noise, s_noise=s_noise,  # type:ignore
                       x0min=x0min, x0max=x0max)  # type:ignore
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds = tf.data.Dataset.from_generator(
    trjgen, output_types=(tf.float64, tf.float64)).prefetch(buffer_size=AUTOTUNE)

# fixed seed for sampling
seed = tf.random.normal(cgantr.input_shape)

# %% checking time series
x, params_batch = next(iter(ds))
print(x.shape)
if dim_map == 2:
  plt.plot(x[0, :, 0], x[0, :, 1], '.')
if dim_map == 1:
  plt.plot(x[0, 0:100, 0], x[0, 1:101, 0], '.')
  plt.show()
  plt.plot(x[0, :, 0])

# %% run training loop
tr_result = cgantr.train(ds, ITERATION, f_map, f_param, sample_seed=seed)
#%% saving tr_result
os.makedirs('saved_data', exist_ok=True)

with open("saved_data/tr_result.pkl", "wb") as f:
  pickle.dump(tr_result, f)


with open("saved_data/tr_result.pkl", "rb") as f:
  tr_result = pickle.load(f)

#%% restore model
# cgantr.checkpoint.restore(tf.train.latest_checkpoint(cgantr.checkpoint_dir))

# %% sample
X_nc, p_nc = next(iter(ds))
X_nc = X_nc.numpy()
p_nc = p_nc.numpy()

X_sample = generator(seed, training=False).numpy()

#%% making graphs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(tr_result['itrs'], tr_result['rmses'],
        'C0', label='transition error(RMSE)')
ax2 = ax.twinx()
ax2.plot(tr_result['itrs'], tr_result['mloglikes'],
         'C1', label='-(log likelihood)')
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper right')
ax.set_xlabel('iteration')
ax.set_ylabel('RMSE')
ax2.set_ylabel('-(log likelihood)')
fig.savefig(os.path.join(GEN_IMAGES_DIR, "errors_and_likelihood.png"))
plt.show()
plt.close(fig)
