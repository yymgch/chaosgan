# -*- coding: utf-8 -*-
# parzen.py
# implementing Parzen window method for estimating probability density function using tensorflow

#%%

import matplotlib.pyplot as plt
import numpy as np

import time
import tensorflow as tf
# from tensorflow.keras import layers


# from chaosmap import f_logistic, f_henon, iterate_f_batch

class ParzenWindow:
  ''' 
     estimation of probability distribution function by Parzen window method.
     using tensorflow 
  '''

  def __init__(self, dim, sigma, X_sample, max_parallel=5e7):
    '''
    dim: dimension of data
    sigma: window size (std of gaussian)
    X_sample: sample data set [---,dim]
    max_parallel: max num of scalar values computed in parallell at once. longer data are divided and computed sequentially

    '''
    self.dim = dim
    self.sigma = tf.constant(sigma, dtype=tf.float32)
    self.X_sample = tf.reshape(tf.constant(
        X_sample, dtype=tf.float32), shape=[-1, dim])
    self.N_sample = int(self.X_sample.shape[0])
    self.max_parallel = max_parallel

  def pdf(self, X, disp_process=True):
    '''
     returns probability density function p(X)  constructed by Parzen estimator with self.X_sample
     X should be 2-dim array with shape (N_data, dim)
    '''
    #maxN_para = 1e7 #
    # return to shape(n,1,dim)
    X = tf.cast(tf.reshape(X, shape=[-1, 1, self.dim]), tf.float32)
    n = int(X.shape[0])  # type:ignore number of data

    # splitting data
    n_split = int(np.ceil(n * self.N_sample * self.dim / self.max_parallel))
    # print(f'n={n} n_split={n_split}')

    #
    if (n / n_split).is_integer():
      X_split = tf.split(X, n_split, 0)
    else:
      n_full = n // n_split
      # print(f'n_full={n_full}')
      n_split = int(np.ceil(n / n_full))
      # print(f'n_split={n_split}')
      n_split_list = [n_full] * (n_split - 1)
      n_split_list.append(int(n % n_full))
      # print(f'n_split_list={n_split_list}')
      X_split = tf.split(X, n_split_list, 0)

    # reshape to (1, N_sample, dim)
    X_sample_tr = tf.reshape(self.X_sample, (1, self.N_sample, self.dim))

    ps = []
    for j, xs in enumerate(X_split):

      pdf_seg = self._pdf_seg(xs, X_sample_tr)
      ps.append(pdf_seg)
      if disp_process and j % 10 == 0:
        print(f'\r {j+1}/{n_split} segments are processed', end='')

    if disp_process:
      print('')

    px = tf.concat(ps, axis=0) / (self.N_sample *
                                  (tf.sqrt(2.0 * np.pi) * self.sigma)**self.dim)  # normalization
    return px

  @tf.function
  def _pdf_seg(self, xs, X_sample_tr):
    # return matrix has (n_k, N_sample, dim) shape
    d = tf.square(xs - X_sample_tr)
    # where k is index of splitted sub-dataset
    # (i,j,:) component is X_k_i-X_data_j
    md = tf.reduce_sum(d, axis=2) / (self.sigma**2)
    pdf_seg = tf.reduce_sum(tf.math.exp(-0.5 * md), axis=1)
    return pdf_seg

  def log_like(self, X, disp_process=True):
    return tf.math.log(self.pdf(X, disp_process=disp_process))

  def pdf_one_leave_out(self, X, disp_process=True):
    '''
      returns probability density function p_i(x_i) 
      p_i was constructed N-1 samples from X_sample other than x_i
      
    '''

    # reshaping to (n,1,dim)
    X = tf.cast(tf.reshape(X, shape=[-1, 1, self.dim]), tf.float32)

    n = int(X.shape[0])  # type:ignore

    n_split = int(np.ceil(n * self.N_sample * self.dim / self.max_parallel))
    if (n / n_split).is_integer():
      X_split = tf.split(X, n_split, 0)
    else:
      n_full = int(n / (n_split - 1))
      n_split = int(np.ceil(n / n_full))
      n_split_list = [n_full] * (n_split - 1)
      n_split_list.append(int(n % n_full))
      X_split = tf.split(X, n_split_list, 0)

    # print(f'n_split={n_split}')

    X_sample_tr = tf.reshape(self.X_sample, (1, self.N_sample, self.dim))

    ps = []
    for j, xs in enumerate(X_split):
      # return matrix has (n_k, N_sample, dim) shape
      d = tf.square(xs - X_sample_tr)
      # where k is index of splitted sub dataset
      # (i,j,:) component is X_k_i-X_data_j
      md = tf.reduce_sum(d, axis=2) / (self.sigma**2)
      assert md.shape == (xs.shape[0], self.N_sample)
      # ．# subtract self contribution. Because distance is 0, exp(-0.5*0)=1
      pdf = tf.reduce_sum(tf.math.exp(-0.5 * md), axis=1) - 1
      ps.append(pdf)
      if disp_process:
        print(f'\r {j+1}/{np.sum(n_split)} segments are processed', end='')

    if disp_process:
      print('')

    px = tf.concat(ps, axis=0) / ((self.N_sample - 1) *
                                  (tf.sqrt(2.0 * np.pi) * self.sigma)**self.dim)  # 自分を含まないのでN-1 個
    return px

  def log_like_one_leave_out(self, X, disp_process=True):
    return tf.math.log(self.pdf_one_leave_out(X, disp_process=disp_process))

  def determine_sigma(self, pow_start, pow_end, steps, ):
    s_loggrid = np.logspace(pow_start, pow_end, steps)

    loglike_s = []
    for s in s_loggrid:
      self.sigma = s
      log_like = tf.reduce_mean(tf.math.log(
          self.pdf_one_leave_out(disp_process=False))).numpy()  # type:ignore
      loglike_s.append(log_like)
      print(f'sigma={s}, mean_loglikelihood={log_like}')
    loglike_s = np.array(loglike_s)
    max_index = np.argmax(loglike_s)
    s_max = s_loggrid[max_index]

    print(f's_max: {s_max}')

    s_loggrid2 = np.logspace(np.log10(s_max) - 1, np.log10(s_max) + 1, 11)
    loglike_s2 = []
    for s in s_loggrid2:
      self.sigma = s
      log_like = tf.reduce_mean(tf.math.log(
          self.pdf_one_leave_out(disp_process=False))).numpy()  # type:ignore
      loglike_s2.append(log_like)
      print(f'sigma={s:.5}, mean_loglikelihood={log_like}')
    loglike_s2 = np.array(loglike_s2)
    max_index = np.argmax(loglike_s2)
    s_max = s_loggrid2[max_index]
    print(f's_max: {s_max}')
    self.sigma = s_max
    # store results in a dict
    self.opt_sigma = {"s_loggrid": loglike_s, "s_loggrid2": loglike_s2}

    return s_max

  def __call__(self, X):
    return self.pdf(X)


def kld_parzen(p_pw, q_pw, one_leave_out=False):
  '''KL divergence estimation by Parzen window method
    p_pw:  parzen windon object for pdf p f
    q_pw: pdf q for parzen window
  '''
  if not one_leave_out:
    return tf.reduce_mean(p_pw.log_like(p_pw.X_sample)) - tf.reduce_mean(q_pw.log_like(p_pw.X_sample))
  else:
    return tf.reduce_mean(p_pw.log_like(p_pw.X_sample)) - tf.reduce_mean(q_pw.log_like(p_pw.X_sample))


def jsd_parzen(X_p, X_q, dim, sigma, one_leave_out=False):
  '''
    Jensen-Shannon Divergence estimation by Parzen window method
    X_p: data for a distribution
    X_q: data for another distribution
    dim: dimension
    sigma: smoothing parameter (std of Gaussian)

  '''
  p_pw = ParzenWindow(dim=dim, sigma=sigma, X_sample=X_p)
  q_pw = ParzenWindow(dim=dim, sigma=sigma, X_sample=X_q)
  X_p = X_p.reshape([-1, dim])
  X_q = X_q.reshape([-1, dim])
  X_pq = np.concatenate([X_p, X_q], axis=0)
  m_pw = ParzenWindow(dim=dim, sigma=sigma, X_sample=X_pq)

  jsd = 0.5 * (kld_parzen(p_pw, m_pw, one_leave_out=True) +
               kld_parzen(q_pw, m_pw, one_leave_out=True))
  # print(f'KL(p,m):{kld_parzen(p_pw, m_pw)}')
  # print(f'KL(q,m):{kld_parzen(q_pw, m_pw)}')
  return jsd


def pdf_logistic_true(x):
  # true probability density function for invariant measure of logistic map with a=4.0.
  # use np.maximum for avoiding diverge in the interval outside [0,1].
  return ((0.0 <= x) * (x <= 1.0)) / (np.pi * np.sqrt(np.maximum(1e-8, x * (1 - x))))
