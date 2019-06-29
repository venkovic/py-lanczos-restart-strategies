import numpy as np 
import scipy.sparse as sparse
import scipy.sparse.linalg

# See https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Matlab/
#     https://people.eecs.berkeley.edu/~demmel/ma221_Fall09/Matlab/LANCZOS_README.html

def Lanczos(A, m, v=None, reortho="selective", eigvals=None):
  n = A.shape[0]
  n_eigvecs =7

  if (type(v) == type(None)):
    v = np.random.rand(n)
  v /= np.linalg.norm(v)
  V = v.reshape((n, 1))
  
  alpha = []
  beta = [0.]
  
  if (type(eigvals) != type(None)):
    data = {"approx_eigvecs":[], "approx_eigvals":[], "rel_iterated_error_bound":[], \
        "rel_error_bound":[], "global_error":[], "local_error":[]}

  for j in range(m):
    w = A.dot(V[:,j])
    if (j > 0):
      w -= beta[j]*V[:,j-1]
    alpha += [w.dot(V[:,j])]
    w -= alpha[j]*V[:,j]
    beta += [np.linalg.norm(w)]
    V = np.concatenate([V, (w/beta[j+1]).reshape((n,1))], axis=1)
    T = get_T(alpha, beta[1:-1])

    if (j > 2):
      if (reortho == "full"):
        pass
      elif (reortho == "selective"):
        pass

    if (j > 1):
      if (type(eigvals) == type(None)):
        approx_eigvecs, approx_eigvals, iterated_error_bound = \
          get_approx_eigvecs(T, V[:,:-1], A, beta[1:j], j, eigvals=None)
      else:
        approx_eigvecs, approx_eigvals, rel_iterated_error_bound, \
        rel_error_bound, global_error, local_error = \
          get_approx_eigvecs(T, V[:,:-1], A, beta[1:j], j, eigvals=eigvals)
        data["approx_eigvecs"] += [approx_eigvecs]
        data["approx_eigvals"] += [approx_eigvals]
        data["rel_iterated_error_bound"] += [rel_iterated_error_bound]
        data["rel_error_bound"] += [rel_error_bound]
        data["global_error"] += [global_error]
        data["local_error"] += [local_error]

      T2 = np.max(np.abs(approx_eigvals))
    #print("j = %d" %(m)) 
  #return V[:,:-1], np.array(alpha), np.array(beta[1:-1])
  if (type(eigvals) == type(None)):
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


def get_approx_eigvecs(T, V, A, beta, n_eigvecs, eigvals=None):
  # Compute eigpairs of the tridiagonal
  reduced_eigvals, reduced_eigvecs = sparse.linalg.eigsh(T, k=n_eigvecs)
  reduced_eigvals = reduced_eigvals[-1::-1]
  reduced_eigvecs = reduced_eigvecs[:,-1::-1]
  # reduced eigpairs are now sorted from most to less dominant

  # Form Ritz approximate eigenvectors of A
  Y = V.dot(reduced_eigvecs)

  # Iterated error bound:
  iterated_error_bound = [abs(beta[-1])*abs(reduced_eigvecs[-1,i]) for i in range(n_eigvecs)]
  
  if (type(eigvals) == type(None)):
    #return Y, reduced_eigvals, iterated_error_bound
    return Y, reduced_eigvals, iterated_error_bound

  elif (type(eigvals) != type(None)):
    rel_iterated_error_bound = np.array(iterated_error_bound)/np.abs(eigvals[:n_eigvecs])

    # Exact error bound:
    error_bound = [np.linalg.norm(A.dot(Y[:,i])-reduced_eigvals[i]*Y[:,i]) for i in range(n_eigvecs)]
    rel_error_bound = np.array(error_bound)/np.abs(eigvals[:n_eigvecs])

    # GLobal and local errors, see p. 370 in Demmel's textbook:
    global_error = np.abs(reduced_eigvals-eigvals[:n_eigvecs])/np.abs(eigvals[:n_eigvecs])
    local_error = np.array([np.min(reduced_eigvals-eigvals[i]) for i in range(n_eigvecs)])/np.abs(eigvals[:n_eigvecs])
    return Y, reduced_eigvals, rel_iterated_error_bound, rel_error_bound, global_error, local_error 
