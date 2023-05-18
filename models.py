# -*- coding: utf-8 -*-
# models.py

import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tools import plot_hist2d, TrajGenerator, TransitionErrors, random_x0
import time
from IPython import display
from parzen import ParzenWindow, jsd_parzen, kld_parzen

BATCH_SIZE = 100
NOISE_DIM = 100
SIGMA_PARZEN = 0.02
FULLY_CONV = True

INIT_CH = 64


def make_generator_model(input_dim=100, initial_length=120, dim=1, crop_size=[0, 0], padding='valid', fully_conv=True):
  ''' 
    returns a model of generator using fully convolutional 1D-CNN
    args:
      input_dim: dimension of input noise
      initial_length: length of initial timeseries
      dim: dimension of output timeseries
      crop_size: cropping size of output timeseries (unused)
      padding: padding mode of convolutional layers (default is valid)
      fully_conv: if True, the first layer is convolutional, otherwise dense
  '''
  initial_channel = 64
  #initial_length = output_length//4
  model = tf.keras.Sequential()

  if fully_conv:
    model.add(layers.Input(shape=[None, initial_channel]))
  else:
    model.add(layers.Dense(initial_length*initial_channel, use_bias=True,
                           input_shape=(input_dim,)))  # dense

    model.add(layers.ELU())
    model.add(layers.Reshape((initial_length, 64)))

  print(model.output_shape)
  #assert model.output_shape == (None, initial_length, 64)

  #model.add(layers.UpSampling1D(size=2))
  model.add(layers.UpSampling1D(size=2))
  model.add(layers.Conv1D(64, 7, strides=1,
            padding=padding, use_bias=True))  # conv1
  print(model.output_shape)
  #assert model.output_shape == (None, initial_length*2 -10, 64)
  model.add(layers.ELU())

  model.add(layers.Conv1D(64, 7, strides=1,
            padding=padding, use_bias=True))  # conv2
  # print(model.output_shape)
  model.add(layers.ELU())

  model.add(layers.UpSampling1D(size=2))
  model.add(layers.Conv1D(64, 7, strides=1,
            padding=padding, use_bias=True))  # conv3
  print(model.output_shape)
  #assert model.output_shape == (None, initial_length*2 -10, 64)
  model.add(layers.ELU())

  model.add(layers.Conv1D(64, 7, strides=1,
            padding=padding, use_bias=True))  # conv4
  # print(model.output_shape)
  model.add(layers.ELU())

  # model.add(layers.UpSampling1D(size=2))
  model.add(layers.UpSampling1D(size=2))
  model.add(layers.Conv1D(64, 7, strides=1,
            padding=padding, use_bias=True))  # conv5
  model.add(layers.ELU())
  print(model.output_shape)

  model.add(layers.Conv1D(64, 7, strides=1,
            padding=padding, use_bias=True))  # conv6
  model.add(layers.ELU())

  model.add(layers.Conv1D(8, 7, strides=1,
            padding=padding, use_bias=True))  # conv7
  model.add(layers.ELU())

  model.add(layers.Conv1D(8, 7, strides=1,
            padding=padding, use_bias=True))  # conv8
  model.add(layers.ELU())

  model.add(layers.Conv1D(dim, 7, strides=1,
            padding=padding, use_bias=True))  # conv9
  print(model.output_shape)
  #cropping

  # model.add(layers.Cropping1D(cropping=crop_size))
  print(model.output_shape)

  #assert model.output_shape == (None, output_length, dim)

  return model

# discriminator


def make_discriminator_model(dim=1, crop_size=(10, 10), padding='valid'):
  ''' 
    return a model of discriminator using fully convolutional 1D-CNN
    args:
      dim: dimension of input timeseries
      crop_size: cropping size of input timeseries (unused)
      padding: padding mode of convolutional layers (default is valid)

  '''

  model = tf.keras.Sequential()
  model.add(layers.Conv1D(64, 5, strides=1, padding=padding,
                          input_shape=[None, dim], use_bias=True))
  model.add(layers.ELU())
  #model.add(layers.BatchNormalization())

  model.add(layers.Conv1D(64, 5, strides=1,
            padding=padding, use_bias=True))  # conv1
  model.add(layers.ELU())
  #model.add(layers.BatchNormalization())

  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 5, strides=1,
            padding=padding, use_bias=True))  # conv2
  model.add(layers.ELU())

  model.add(layers.Conv1D(64, 5, strides=1,
            padding=padding, use_bias=True))  # conv3
  model.add(layers.ELU())

  model.add(layers.Conv1D(64, 5, strides=1,
            padding=padding, use_bias=True))  # conv4
  model.add(layers.ELU())

  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 5, strides=2,
            padding=padding, use_bias=True))  # conv5
  model.add(layers.ELU())
  #model.add(layers.BatchNormalization())

  model.add(layers.Conv1D(64, 5, strides=2,
            padding=padding, use_bias=True))  # conv6
  model.add(layers.ELU())

  model.add(layers.Conv1D(64, 5, strides=2,
            padding=padding, use_bias=True))  # conv7
  model.add(layers.ELU())

  model.add(layers.GlobalAveragePooling1D())
  model.add(layers.Dense(1))

  return model


class ChaosGANTraining:
  # class for GAN training

  def __init__(self, batch_size=100, noise_dim=100,
               output_length_about=1000,
               dim_map=1, sigma_parzen=0.01, fully_conv=True, initial_ch=INIT_CH):
    self.batch_size = batch_size
    self.noise_dim = noise_dim
    self.output_length_about = output_length_about
    self.initial_length = 2 + self.output_length_about // 8
    self.dim_map = dim_map
    self.crop_size = [25, 25]  # unused
    self.fully_conv = fully_conv
    self.initial_ch = initial_ch  # dim of initial channel of generator

    self.sigma_parzen = sigma_parzen

    if self.fully_conv:
      self.input_shape = [self.batch_size,
                          self.initial_length, self.initial_ch]
    else:
      self.input_shape = [self.batch_size, self.noise_dim]

  def make_models(self,):
    # making models here
    self.generator = make_generator_model(input_dim=self.noise_dim,
                                          initial_length=self.initial_length, dim=self.dim_map,
                                          crop_size=self.crop_size,
                                          padding='valid', fully_conv=self.fully_conv)
    self.discriminator = make_discriminator_model(
        dim=self.dim_map, padding='valid')
    # self.tmax = self.generator.output_shape[1]-1
    return self.generator, self.discriminator

  def build_model_by_input(self,):
    # initialize the model by input some tensor and check the output shape.

    noise = tf.random.normal(self.input_shape)  # random generator

    generated_timeseries = self.generator(noise, training=False)
    print("generated_timeseries.shape={}".format(generated_timeseries.shape))

    d = self.discriminator(generated_timeseries)
    print("discriminator output shape: {}".format(d.shape))
    # get output shape
    self.tmax = generated_timeseries.shape[1]

  def set_optimizer(self, learning_rate=1e-4):
    self.generator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    self.discriminator_optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    return self.generator_optimizer, self.discriminator_optimizer

  def set_checkpoint(self, checkpoint_dir='./training_checkpoints'):
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                          discriminator_optimizer=self.discriminator_optimizer,
                                          generator=self.generator,
                                          discriminator=self.discriminator)
    return self.checkpoint

  def set_image_dir(self, image_dir='./gen_images'):
    '''
      set directory for saving generated images
    '''
    self.image_dir = image_dir
    if not os.path.exists(self.image_dir):
      os.mkdir(self.image_dir)
    return self.image_dir

  @tf.function
  def train_step(self, x_batch):
    '''
      train one step of D and G
      args:
        x_batch: batch of input timeseries
    '''
    noise = tf.random.normal(self.input_shape)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated = self.generator(noise, training=True)

      real_output = self.discriminator(x_batch, training=True)
      fake_output = self.discriminator(generated, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, self.generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(
        zip(gradients_of_generator, self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, self.discriminator.trainable_variables))

  
  def train(self, dataset, n_iteration, f_map, f_param, sample_seed=None, change_learning_rate=False, steps_lr_change=0):
    '''
      train the model (main loop)
    '''
    errors_map = TransitionErrors(f=f_map)  # calculating errors
    itrs = []  # preserve iteration steps
    rmses = []  # preserve transition errors
    mloglikes = []  # preserve log likelihoods

    itr_ds = iter(dataset)
    startt = time.time()
    for ii in range(n_iteration):
      if change_learning_rate and ii == steps_lr_change:
        pass
        # not implemented yet

      x_batch, _a_batch = next(itr_ds)
      self.train_step(x_batch)
      # print(x_batch.shape)

      # Save the model every 10 epochs
      if (ii + 1) % 10 == 0:
        print(f'\r iteration {ii+1}', end='')
      # save the model parameters every 500 epoch
      if (ii + 1) % 500 == 0:
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)

      if (ii + 1) % 1000 == 0 or ii == 0:

        X_data = x_batch
        X_sample = self.generator(sample_seed, training=False).numpy()
        # print(X_sample.shape)
        error_summary = errors_map(X_sample, p=np.array(
            f_param(bs=self.batch_size)))  # Transition Error
        itrs.append(ii + 1)
        rmses.append(error_summary['rmse'])

        # kernel density estimation (Parzen Window)
        p_sample_pw = ParzenWindow(
            X_sample.shape[-1], sigma=self.sigma_parzen, X_sample=X_sample, max_parallel=5e8)
        mloglike = tf.reduce_mean(p_sample_pw.log_like(X_data)).numpy()
        mloglikes.append(-mloglike)

        display.clear_output(wait=True)
        print(f'--Epoch: {ii+1}--')
        print('elapsed time for 100 iteration: {} sec'.format(time.time() - startt))
        print(f'rmse={error_summary["rmse"]:.4f}')
        print(f'mean log likelihood ={mloglike}')

        # maximum likelihood
        if self.dim_map == 2:
          fig1 = plt.figure()
          ax = fig1.add_subplot(111)
          ax.hist2d(X_sample[:, :, 0].flatten(), X_sample[:,
                                                          :, 1].flatten(), bins=50, cmap="Blues")
          plt.show()
          plt.close(fig1)

        elif self.dim_map == 1:
          print('histogram')

          fig1 = plt.figure()
          ax = fig1.add_subplot(111)
          ax.hist(X_sample[:, :, 0].flatten(), bins=50)
          plt.show()
          plt.close(fig1)
          fig_f = plt.figure()
          ax = fig_f.add_subplot(111)
          ax.plot(X_sample[:, 0:-2, 0].flatten(),
                  X_sample[:, 1:-1, 0].flatten(), '.', markersize=0.5)
          plt.show()
          plt.close(fig_f)
          print('return map')
          fig_h = plt.figure()
          ax = fig_h.add_subplot(111)
          ax.hist2d(X_sample[:, 0:-2, 0].flatten(), X_sample[:,
                    1:-1, 0].flatten(), bins=50, cmap="Blues")
          print('2dim histogram')
          plt.show()
          plt.close(fig_h)

          fig_xts = plt.figure()
          ax = fig_xts.add_subplot(111)
          ax.plot(X_sample[:, :, 0].transpose())
          print('timeseries')
          plt.show()
          plt.close(fig_xts)

        # error
        fig2 = plt.figure()
        ax1 = fig2.add_subplot(111)
        ax1.plot(itrs, rmses, 'C0', label='transition error(RMSE)')
        ax2 = ax1.twinx()
        ax2.plot(itrs, mloglikes, 'C1', label='-(log likelihood)')
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc='upper right')
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('RMSE')
        ax2.set_ylabel('-(log likelihood)')

        plt.show()
        plt.close(fig2)

        startt = time.time()

    # after-loop processing
    # save models
    self.checkpoint.save(file_prefix=os.path.join(
        self.checkpoint_dir, "ckpt-final"))
    # output statistics

    # return summary of learning curve
    tr = dict()
    tr['itrs'] = np.array(itrs)
    tr['rmses'] = np.array(rmses)
    tr['mloglikes'] = np.array(mloglikes)

    return tr


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# functions for loss functions of a discriminator a generator


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
