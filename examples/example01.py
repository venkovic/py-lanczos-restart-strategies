import sys; sys.path += ["../"]
from lanczos import lanczos
import numpy as np 
import scipy.sparse as sparse
import scipy.sparse.linalg
import time

import pylab as pl 
pl.rcParams['text.usetex'] = True
params={'text.latex.preamble':[r'\usepackage{amssymb}',r'\usepackage{amsmath}']}

figures_path = '../figures/'

n = 1000
A = np.random.rand(n, n)
A = A.dot(A.T)

v = np.random.rand(n)
v /= np.linalg.norm(v)

m = 100
npairs = 10

eigvals = np.linalg.eigvalsh(A)[-1::-1]
t0 = time.time()
data = lanczos(A, npairs, m=m, eigvals=eigvals)
print("No reorthogonalization : %g" %(time.time()-t0)); t0 = time.time()
data_full = lanczos(A, npairs, m=m, reortho="full", eigvals=eigvals)
print("Full reorthogonalization : %g" %(time.time()-t0)); t0 = time.time()
data_selective = lanczos(A, npairs, m=m, reortho="selective", eigvals=eigvals)
print("Selective reorthogonalization : %g" %(time.time()-t0)); t0 = time.time()

approx_eigvals, approx_eigvals_full, approx_eigvals_selective = [], [], []
iterated_error_bound, iterated_error_bound_full, iterated_error_bound_selective = [], [], []
mm = len(data["approx_eigvals"])
mj = len(data["approx_eigvals"][-1])
for i in range(mj):
  approx_eigvals += [i*[None]+[d[i] for d in data["approx_eigvals"][i:]]]
  approx_eigvals_full += [i*[None]+[d[i] for d in data_full["approx_eigvals"][i:]]]
  approx_eigvals_selective += [i*[None]+[d[i] for d in data_selective["approx_eigvals"][i:]]]

  iterated_error_bound += [i*[None]+[d[i] for d in data["iterated_error_bound"][i:]]]
  iterated_error_bound_full += [i*[None]+[d[i] for d in data_full["iterated_error_bound"][i:]]]
  iterated_error_bound_selective += [i*[None]+[d[i] for d in data_selective["iterated_error_bound"][i:]]]

approx_eigvals = np.array(approx_eigvals)
approx_eigvals_full = np.array(approx_eigvals_full)
approx_eigvals_selective = np.array(approx_eigvals_selective)

iterated_error_bound = np.array(iterated_error_bound)
iterated_error_bound_full = np.array(iterated_error_bound_full)
iterated_error_bound_selective = np.array(iterated_error_bound_selective)

fig, ax = pl.subplots(1, 3, figsize=(11,3.4), sharey="row")
lw = .5
ax[0].set_title("No reorthogonalization")
for i in range(mj):
  ax[0].semilogy(approx_eigvals[i,:], "+")
for i in range(mj):
  ax[0].semilogy(approx_eigvals[i,:], "-", lw=lw)
ax[0].semilogy(eigvals.shape[0]*[mj+3], eigvals, "k_", lw=0)

ax[1].set_title("Full reorthogonalization")
for i in range(mj):
  ax[1].semilogy(approx_eigvals_full[i,:], "+")
for i in range(mj):
  ax[1].semilogy(approx_eigvals_full[i,:], "-", lw=lw)
ax[1].semilogy(eigvals.shape[0]*[mj+3], eigvals, "k_", lw=0)

ax[2].set_title("Selective reorthogonalization")
for i in range(mj):
  ax[2].semilogy(approx_eigvals_selective[i,:], "+")
for i in range(mj):
  ax[2].semilogy(approx_eigvals_selective[i,:], "-",  lw=lw)
ax[2].semilogy(eigvals.shape[0]*[mj+3], eigvals, "k_", lw=0)
ax[1].set_ylim(ax[0].get_ylim()); ax[2].set_ylim(ax[0].get_ylim())
ax[0].set_ylabel("Approximate eigenvalues")
ax[0].set_xlabel("m"); ax[1].set_xlabel("m"); ax[2].set_xlabel("m")
pl.savefig(figures_path+"example01_lanczos_a.png", bbox_inches='tight')
#pl.show()

fig, ax = pl.subplots(1, 3, figsize=(11,3.4), sharey="row")
lw = .8
ax[0].set_title("No reorthogonalization")
for i in range(mj):
  ind = (iterated_error_bound[i,:]==None).sum()
  ax[0].semilogy(np.concatenate((iterated_error_bound[i,:ind],iterated_error_bound[i,ind:]/approx_eigvals[i,ind:])), "-", lw=lw)
ax[1].set_title("Full reorthogonalization")
for i in range(mj):
  ind = (iterated_error_bound_full[i,:]==None).sum()
  ax[1].semilogy(np.concatenate((iterated_error_bound_full[i,:ind],iterated_error_bound_full[i,ind:]/approx_eigvals_full[i,ind:])), "-", lw=lw)
ax[2].set_title("Selective reorthogonalization")
for i in range(mj):
  ind = (iterated_error_bound_selective[i,:]==None).sum()
  ax[2].semilogy(np.concatenate((iterated_error_bound_selective[i,:ind],iterated_error_bound_selective[i,ind:]/approx_eigvals_selective[i,ind:])), "-", lw=lw)
ax[1].set_ylim(ax[0].get_ylim()); ax[2].set_ylim(ax[0].get_ylim())
ax[0].set_ylim(1e-16,1e2)
ax[0].set_ylabel("Iterated relative error bound")
for j in range(3):
  ax[j].set_xlabel("m")
  ax[j].grid()
pl.savefig(figures_path+"example01_lanczos_b.png", bbox_inches='tight')
#pl.show()


"""
V, alpha, beta = Lanczos(A, v, m)

n_eigvecs = 7
eigvals = get_eigvals(A, n_eigvecs)
err_bnd, iterated_err_bnd, approx_eigvals, global_err, local_err = [], [], [], [], []
for m_j in range(10, m+1):
  T = get_T(alpha[:m_j], beta[:m_j-1])
  Y, _approx_eigvals, _err_bnd, _iterated_err_bnd, _global_err, _local_err = get_approx_eigvecs(T, V[:,:m_j], A, beta[:m_j-1], n_eigvecs, eigvals=eigvals, output_format=1)
  approx_eigvals += [_approx_eigvals]
  err_bnd += [_err_bnd]
  iterated_err_bnd += [_iterated_err_bnd]
  global_err += [_global_err]
  local_err += [_local_err]
err_bnd, iterated_err_bnd = np.array(err_bnd), np.array(iterated_err_bnd)
global_err, local_err = np.array(global_err), np.array(local_err)

fig = pl.figure()
for i in range(n_eigvecs):
  pl.semilogy([_eigvals[i] for _eigvals in approx_eigvals])
  pl.xlabel("m")
  pl.ylabel(r"$\theta_k^{(m)}$")
pl.show()
"""


"""
err_, iterated_err_bnd, approx_eigvals, global_err, local_err = [], [], [], [], []
for m_j in range(10, m+1):
  T = get_T(alpha[:m_j], beta[:m_j-1])
  Y, _approx_eigvals, _err_bnd, _iterated_err_bnd, _global_err, _local_err = get_approx_eigvecs(T, V[:,:m_j], A, beta[:m_j-1], n_eigvecs, eigvals=eigvals, output_format=1)
  approx_eigvals += [_approx_eigvals]
  err_bnd += [_err_bnd]
  iterated_err_bnd += [_iterated_err_bnd]
  global_err += [_global_err]
  local_err += [_local_err]
err_bnd, iterated_err_bnd = np.array(err_bnd), np.array(iterated_err_bnd)
global_err, local_err = np.array(global_err), np.array(local_err)

fig = pl.figure()
for i in range(n_eigvecs):
  pl.semilogy([_eigvals[i] for _eigvals in approx_eigvals])
  pl.xlabel("m")
  pl.ylabel(r"$\theta_k^{(m)}$")
pl.show()

for i in range(n_eigvecs):
  fig = pl.figure()
  pl.semilogy(err_bnd[:,i], label=r"$\|Ay_{%d}^{(m)}-\theta_{%d}^{(m)}y_{%d}^{(m)}\|_2/|\lambda_{%d}(A)|$" %(i+1, i+1, i+1, i+1))
  pl.semilogy(iterated_err_bnd[:,i], label=r"$|\beta_m||(S_m)_{m,%d}|/|\lambda_{%d}(A)|$" %(i+1, i+1))
  pl.semilogy(global_err[:,i], label=r"$|\theta_{%d}^{(m)}-\lambda_{%d}(A)|/|\lambda_{%d}(A)|$" %(i+1, i+1, i+1))
  pl.semilogy(local_err[:,i], label=r"$\min_{j}|\theta_{%d}^{(m)}-\lambda_{j}(A)|/|\lambda_{%d}(A)|$" %(i+1, i+1))
  pl.xlabel("m")
  pl.ylabel("Error bound and iterated error bound")
  pl.legend()
  pl.show()
"""

# Q1:    HOW DO YOU ASSESS THE ACCURACY OF THOSE APPROXIMATION.
#        i.e. HOW DO YOU PICK m for a given n_eigvecs
#        You can not know this, see p. 239-240 of Dongara et al.'s book.
# R1/T1: See oscillations of error bounds (iterated or not).
#        Plot local/global errors as in p. 370-371 of Demmel's book.
#        Try
# T2:    Implement restart strategies.
#        Explicit vs implicit, see p. 239-240 of Dongara et al.'s book
# T3:    Measure and plot LOO?