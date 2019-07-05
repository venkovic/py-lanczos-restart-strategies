import numpy as np 
import scipy.sparse as sparse
import scipy.sparse.linalg

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

  if not isinstance(eigvals, np.ndarray):
    return approx_eigvecs, approx_eigvals, iterated_error_bound
  else:
    return data