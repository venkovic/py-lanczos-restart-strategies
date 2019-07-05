import sys; sys.path += ["../"]
from lanczos import lanczos
import numpy as np 

import pylab as pl 
pl.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}

figures_path = '../figures/'

def plot(m, eigvals, approx_eigvals, approx_eigvals_full, approx_eigvals_selective, 
         iterated_error_bound, iterated_error_bound_full, iterated_error_bound_selective):

  fig, ax = pl.subplots(1, 3, figsize=(11,3.4), sharey="row")
  lw = .8
  ax[0].set_title("No reorthogonalization")
  for i in range(m):
    ax[0].semilogy(range(1,m), approx_eigvals[i,:], "+")
  for i in range(m):
    ax[0].semilogy(range(1,m), approx_eigvals[i,:], "-", lw=lw)
  ax[0].semilogy(eigvals.shape[0]*[m+3], eigvals, "k_", lw=0)  

  ax[1].set_title("Full reorthogonalization")
  for i in range(m):
    ax[1].semilogy(range(1,m), approx_eigvals_full[i,:], "+")
  for i in range(m):
    ax[1].semilogy(range(1,m), approx_eigvals_full[i,:], "-", lw=lw)
  ax[1].semilogy(eigvals.shape[0]*[m+3], eigvals, "k_", lw=0)  

  ax[2].set_title("Selective reorthogonalization")
  for i in range(m):
    ax[2].semilogy(range(1,m), approx_eigvals_selective[i,:], "+")
  for i in range(m):
    ax[2].semilogy(range(1,m), approx_eigvals_selective[i,:], "-",  lw=lw)
  for j in range(3):
    ax[j].grid()
  ax[2].semilogy(eigvals.shape[0]*[m+3], eigvals, "k_", lw=0)
  ax[1].set_ylim(ax[0].get_ylim()); ax[2].set_ylim(ax[0].get_ylim())
  ax[0].set_ylabel("Approximate eigenvalues")
  fig.suptitle("Approximate eigenvalues, "+r"$\{\lambda_i(T_m)\}_{i=1}^m$")
  ax[0].set_xlabel("m"); ax[1].set_xlabel("m"); ax[2].set_xlabel("m")
  pl.savefig(figures_path+"example01_lanczos_a.png", bbox_inches='tight')
  #pl.show()  

  fig, ax = pl.subplots(1, 3, figsize=(11,3.4), sharey="row")
  lw = .8
  ax[0].set_title("No reorthogonalization")
  for i in range(m):
    ind = (iterated_error_bound[i,:]==None).sum()
    ax[0].semilogy(range(1,m), np.concatenate((iterated_error_bound[i,:ind],iterated_error_bound[i,ind:]/approx_eigvals[i,ind:])), "-", lw=lw)
  ax[1].set_title("Full reorthogonalization")
  for i in range(m):
    ind = (iterated_error_bound_full[i,:]==None).sum()
    ax[1].semilogy(range(1,m), np.concatenate((iterated_error_bound_full[i,:ind],iterated_error_bound_full[i,ind:]/approx_eigvals_full[i,ind:])), "-", lw=lw)
  ax[2].set_title("Selective reorthogonalization")
  for i in range(m):
    ind = (iterated_error_bound_selective[i,:]==None).sum()
    ax[2].semilogy(range(1,m), np.concatenate((iterated_error_bound_selective[i,:ind],iterated_error_bound_selective[i,ind:]/approx_eigvals_selective[i,ind:])), "-", lw=lw)
  ax[1].set_ylim(ax[0].get_ylim()); ax[2].set_ylim(ax[0].get_ylim())
  ax[0].set_ylim(1e-16,1e2)
  ax[0].set_ylabel("Iterated relative error bound")
  fig.suptitle("Iterated relative error bound, "+r"$\{|\beta_mS_{mi}/\lambda_i(T_m)|\}_{i=1}^m$")
  for j in range(3):
    ax[j].set_xlabel("m")
    ax[j].grid()
  pl.savefig(figures_path+"example01_lanczos_b.png", bbox_inches='tight')
  #pl.show()  

  fig, ax = pl.subplots(1, 3, figsize=(11,3.4), sharey="row")
  lw = .8
  ax[0].set_title("No reorthogonalization")
  for i in range(m):
    ind = (approx_eigvals[i,:]==None).sum()
    ax[0].semilogy(range(1,m), np.concatenate((approx_eigvals[i,:ind],np.abs((approx_eigvals[i,ind:]-eigvals[i])/eigvals[i]))), "-", lw=lw)
  ax[1].set_title("Full reorthogonalization")
  for i in range(m):
    ind = (approx_eigvals_full[i,:]==None).sum()
    ax[1].semilogy(range(1,m), np.concatenate((approx_eigvals_full[i,:ind],np.abs((approx_eigvals_full[i,ind:]-eigvals[i])/eigvals[i]))), "-", lw=lw)
  ax[2].set_title("Selective reorthogonalization")
  for i in range(m):
    ind = (approx_eigvals_selective[i,:]==None).sum()
    ax[2].semilogy(range(1,m), np.concatenate((approx_eigvals_selective[i,:ind],np.abs((approx_eigvals_selective[i,ind:]-eigvals[i])/eigvals[i]))), "-", lw=lw)
  ax[0].set_ylabel("Global error")
  fig.suptitle("Global error, "+r"$\{|\lambda_i(T_m)-\lambda_i(A)|/|\lambda_i(A)|\}_{i=1}^m$")
  for j in range(3):
    ax[j].set_xlabel("m")
    ax[j].grid()
  pl.savefig(figures_path+"example01_lanczos_c.png", bbox_inches='tight')
  #pl.show()
