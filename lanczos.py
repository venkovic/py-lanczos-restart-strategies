import numpy as np 
import scipy.sparse as sparse
import scipy.sparse.linalg

# See https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Matlab/
#     https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Matlab/LANCZOS_README.html



def lanczos(A, npairs, m=50, q1=None, reortho=None, eigvals=None):
  # Implementation #1 : Algo. 7.1 (Demmel, 1997);
  # (  we use this )    Algo. 10.1.1 (Golub, 2012)
  # (implementation)    Sec. 10.3.1 (Golub, 2012)---verison w. lighter storage
  #
  # Implementation #2 : Sec. 13.1 (Parlett, 1980); 
  # (greater stability) Algo. 6.5 (Saad, 2011);
  # 
  # Links between variants  : A({1,2},{6,7}) (Paige, 1972)
  #                           Algo. 36.1 (Trefethen and Bau, 1997)
  #
  # Preferred references : Chap. 13.{1,2,7} (Parlett, 1980)
  #                        10.3.1 (Golub, 2012)
  #
  alpha = np.zeros(m)  # (al_1, ..., al_m)
  beta = np.zeros(m+1) # (be_0, ..., be_m), be_0 := 0
  
  #   T_j    = sparse.diags(([be_1, ..., be_{j-1}],
  # (j-by-j)                 [al_1, ..., al_{j}]
  #                          [be_1, ..., be_{j-1}]), (-1,0,1))
  
  n = A.shape[0]
  Q = np.zeros((n, m+1))
  
  # Initialize first Lanczos vector
  if not isinstance(q1, np.ndarray):
    q1 = np.random.rand(n)
  q1 /= np.linalg.norm(q1)
  Q[:,0] = q1
  
  if isinstance(eigvals, np.ndarray):
    #data = {"approx_eigvecs":[], "approx_eigvals":[], "rel_iterated_error_bound":[], \
    #    "rel_error_bound":[], "global_error":[], "local_error":[]}
    data = {"approx_eigvecs":[], "approx_eigvals":[], "iterated_error_bound":[]}

  for j in range(m):
    u = A.dot(Q[:,j])
    alpha[j] = Q[:,j].dot(u)

    u -= alpha[j]*Q[:,j]
    if (j > 0):
      u -= beta[j]*Q[:,j-1]

    if (reortho != None) & (j > 1):
      if (reortho == "full"):
        for _ in range(2):
          for k in range(j):
            u -= u.dot(Q[:,k])*Q[:,k]

      elif (reortho == "selective"):
        for k in range(j):
          if (beta[j]*abs(S[j-1,k]) < tol):
            u -= u.dot(Y[:,k])*Y[:,k]

    beta[j+1] = np.linalg.norm(u)
    Q[:,j+1] = u/beta[j+1]

    # Get eigenpairs from tridiagonal
    if (j > 0): # For T_2, ..., T_m
      theta, S = scipy.linalg.eigh_tridiagonal(alpha[:j+1], beta[1:j+1], select="a")
      # theta[0] <= ... <= theta[j]
   
      if (reortho == "selective"):
        Tnorm = theta[j]/theta[0]
        tol = np.finfo(float).eps**.5*Tnorm

      Y = Q[:,:j+1].dot(S)
      data["approx_eigvecs"] += [Y[:,-1::-1]]
      data["approx_eigvals"] += [theta[-1::-1]]
      data["iterated_error_bound"] += [np.abs(beta[j+1]*S[j,-1::-1])]

      # data["rel_error_bound"], data["global_error"] and data["local_error"]
      # can be computed after


      # Get iterated error bounds, ...

      """
      if not isinstance(eigvals, np.ndarray):
        approx_eigvecs, approx_eigvals, iterated_error_bound = \
          get_approx_eigvecs(V[:,:j+1], A, alpha[:j+1], beta[1:j+1], beta[j+1], j+1)
      else:
        approx_eigvecs, approx_eigvals, rel_iterated_error_bound, \
        rel_error_bound, global_error, local_error = \
          get_approx_eigvecs(V[:,:j+1], alpha[:j+1], beta[1:j+1], beta[j+1], j+1, A=A, eigvals=eigvals)
        data["approx_eigvecs"] += [approx_eigvecs]
        data["approx_eigvals"] += [approx_eigvals]
        data["rel_iterated_error_bound"] += [rel_iterated_error_bound]
        data["rel_error_bound"] += [rel_error_bound]
        data["global_error"] += [global_error]
        data["local_error"] += [local_error]
      """

  if not isinstance(eigvals, np.ndarray):
    return approx_eigvecs, approx_eigvals, iterated_error_bound
  else:
    return data

def get_approx_eigvecs(V, alpha, beta, beta_j, n_eigvecs, A=None, eigvals=None):
  reduced_eigvals, reduced_eigvecs = scipy.linalg.eigh_tridiagonal(alpha, beta, select="a")
  reduced_eigvals = reduced_eigvals[-1::-1]
  reduced_eigvecs = reduced_eigvecs[:,-1::-1]
  # reduced eigpairs are sorted from most to less dominant

  # Form Ritz approximate eigenvectors of A
  Y = V.dot(reduced_eigvecs)

  # Iterated error bound:
  iterated_error_bound = [abs(beta_j)*abs(reduced_eigvecs[-1,i]) for i in range(n_eigvecs)]
  
  if not isinstance(eigvals, np.ndarray):
    #return Y, reduced_eigvals, iterated_error_bound
    return Y, reduced_eigvals, iterated_error_bound

  else:
    rel_iterated_error_bound = np.array(iterated_error_bound)/np.abs(eigvals[:n_eigvecs])

    # Exact error bound:
    error_bound = [np.linalg.norm(A.dot(Y[:,i])-reduced_eigvals[i]*Y[:,i]) for i in range(n_eigvecs)]
    rel_error_bound = np.array(error_bound)/np.abs(eigvals[:n_eigvecs])

    # GLobal and local errors, see p. 370 in Demmel's textbook:
    global_error = np.abs(reduced_eigvals-eigvals[:n_eigvecs])/np.abs(eigvals[:n_eigvecs])
    local_error = np.array([np.min(reduced_eigvals-eigvals[i]) for i in range(n_eigvecs)])/np.abs(eigvals[:n_eigvecs])
    return Y, reduced_eigvals, rel_iterated_error_bound, rel_error_bound, global_error, local_error