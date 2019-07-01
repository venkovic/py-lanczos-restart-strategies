import numpy as np 
import scipy.sparse as sparse
import scipy.sparse.linalg

# See https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Matlab/
#     https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Matlab/LANCZOS_README.html

def lanczos(A, m, v=None, reortho=None, eigvals=None):
  n = A.shape[0]
  n_eigvecs =20

  alpha = np.zeros(m)
  beta = np.zeros(m+1) # beta[0] := 0
  V = np.zeros((n, m+1))
  # (j+1)-th (j+1)-by-(j+1) tridiagonal, j \in [0, m):
  #   T(j+1) = Tridiag((alpha[:j+1], beta[1:j+1], alpha[:j+1]), (-1,0,1))

  if not isinstance(v, np.ndarray):
    v = np.random.rand(n)
  v /= np.linalg.norm(v)
  V[:,0] = v
  
  if isinstance(eigvals, np.ndarray):
    data = {"approx_eigvecs":[], "approx_eigvals":[], "rel_iterated_error_bound":[], \
        "rel_error_bound":[], "global_error":[], "local_error":[]}

  for j in range(m):
    
    w = A.dot(V[:,j])
    alpha[j] = w.dot(V[:,j])
    

    if (j == 0):
      w -= alpha[j]*V[:,j]
    else:
      w -= alpha[j]*V[:,j]+beta[j]*V[:,j-1]

    if (reortho == "full"):
      for reortho_id in range(2):
        for k in range(j):
          w -= w.dot(V[:,k])*V[:,k]
    elif (reortho == "selective"):
      pass
    
    beta[j+1] = np.linalg.norm(w)
    V[:,j+1] = w/beta[j+1]

    # Assemble tridiagonal T(j+1)
    T = get_T(alpha[:j+1], beta[1:j+1])

    if (j > 0): # For T(2), ..., T(m)
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

  if not isinstance(eigvals, np.ndarray):
    return approx_eigvecs, approx_eigvals, iterated_error_bound
  else:
    return data

def get_T(alpha, beta):
# Assemble tridiagonalization of A
  return sparse.diags([beta, alpha, beta], [-1, 0, 1])

def get_eigvals(A, n_eigvecs):
  eigvals = sparse.linalg.eigsh(A, k=n_eigvecs, return_eigenvectors=False)[-1::-1]
  # eigvals are here sorted from most to less dominant
  return eigvals

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
