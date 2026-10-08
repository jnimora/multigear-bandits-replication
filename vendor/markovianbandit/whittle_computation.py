import numpy as np
import scipy.linalg
from numba import jit

NON_INDEXABLE = False
INDEXABLE_BUT_NOT_STRONGLY = 1
STRONGLY_INDEXABLE = 2
MULTICHAIN = -1

# make numpy raise division by zero and 0/0 error
np.seterr(divide='raise', invalid='raise')

def multichain_message(whittle_idx):
    """
    Message to be printed when the arm is multichain. 
    """
    print("The arm is multichain! -> try with discount factor 0.99999999 to get an approximate answer.")
    return MULTICHAIN, whittle_idx

def initialize_X_from_update(beta_P0, beta_P1, beta, X, pi, atol):
    """
    Compute Delta*A_inv as defined in Algorithm 2 of the paper
    and store the product in matrix X
    """
    dim = beta_P0.shape[0]
    i0 = 0 # state having null bias in non-discounted case
    mat_pol = np.copy(beta_P1)
    for i, a in enumerate(pi):
        if a: continue
        else: mat_pol[i, :] = beta_P0[i, :]
    Delta = beta_P1 - beta_P0
    if abs(1.0 - beta) < atol:
        mat_pol[:, i0] = -1.0
        mat_pol[i0, i0] = 0.0
        Delta[:, i0] = 0.0
    A = np.eye(dim, dtype=np.double) - mat_pol
    X[:,:] = scipy.linalg.solve(A.transpose(), Delta.transpose(), overwrite_a=True, overwrite_b=True, check_finite=False).transpose()


def find_mu_min(y, z, current_mu, atol):
    """
    Find the smallest mu_i^k
    """
    nb_elems = z.shape[0]
    mu_i_k = np.empty(nb_elems)
    for i in range(nb_elems):
        if abs(z[i]) < atol: # if z[i] == 0
            mu_i_k[i] = current_mu
        elif z[i] > atol and 1.0 - y[i] > atol: # if z[i] and 1 - y[i] are > 0
            mu_i_k[i] = current_mu + z[i]/(1.0-y[i])
        else:
            mu_i_k[i] = np.inf

    argmin = mu_i_k.argmin()
    return argmin, mu_i_k[argmin]


@jit
def update_W(W, sigma, X, k, atol, check_indexability=True, k0=0):
    n = X.shape[0]
    V = np.copy(X[:, sigma])
    if check_indexability:
        for l in range(k0+1, k):
            c = V[n-l]
            for i in range(n):
                V[i] = V[i] - c * W[l-1, i]
        c = 1.0 + V[n-k]
        if abs(c) < atol:
            raise ZeroDivisionError
        for i in range(n):
            W[k-1, i] = V[i] / c
    else:
        for l in range(k0+1, k):
            c = V[n-l]
            for i in range(n-l+1):
                V[i] = V[i] - c * W[l-1, i]
        c = 1.0 + V[n-k]
        if abs(c) < atol:
            raise ZeroDivisionError
        for i in range(n-k):
            W[k-1, i] = V[i] / c


def compute_whittle_indices(P0, P1, R0, R1, beta=1, check_indexability=True, verbose=False, atol=1e-12, number_of_updates='2n**0.1'):
    """
    Implementation of Algorithm 2 of the paper
    Test whether the problem is indexable
    and compute Whittle indices when the problem is indexable
    The indices are computed in increasing order

    Args:
    - P0, P1: transition matrix for rest and activate actions respectively
    - R0, R1: reward vector for rest and activate actions respectively
    - beta: discount factor
    - check_indexability: if True check whether the problem is indexable or not
    - number_of_updates: (default = '2n**0.1'): number of time that X^{k} is recomputed from scratch.
    """
    dim = P0.shape[0]
    assert P0.shape == P1.shape
    assert R0.shape == R1.shape
    assert R0.shape[0] == dim

    is_indexable = STRONGLY_INDEXABLE
    pi = np.ones(dim, dtype=np.double)
    sorted_sigmas = np.arange(dim)
    idx_in_sorted = np.arange(dim)
    whittle_idx = np.empty(dim, dtype=np.double)
    whittle_idx.fill(np.nan)
    X = np.empty((dim, dim), dtype=np.double, order='C')
    sorted_X = np.empty((dim, dim), dtype=np.double, order='C')
    beta_P0 = beta*P0
    beta_P1 = beta*P1
    W = np.empty((dim-1,dim), dtype=np.double, order='C')
    k0 = 0
    if number_of_updates == '2n**0.1':
        number_of_updates = int(2*dim**0.1)
    frequency_of_update = int(dim / max(1, number_of_updates))

    try:
        initialize_X_from_update(beta_P0, beta_P1, beta, X, pi, atol)
    except np.linalg.LinAlgError as error:
        if 'Matrix is singular' in str(error):
            return multichain_message(whittle_idx)
        else:
            raise error
    y = np.zeros(dim)
    z = R1 - R0 + X.dot(R1)
    argmin = np.argmin(z)
    sigma = sorted_sigmas[argmin]
    whittle_idx[sigma] = z[sigma]
    z -= whittle_idx[sigma]

    if verbose: print('       ', end='')
    for k in range(1, dim):
        if verbose: print('\b\b\b\b\b\b\b{:7}'.format(k), end='', flush=True)
        """
        1. We sort the states so that the 'non visited' states are the first "dim-k"
           To do so, we exchange only one column of all matrices.
        """
        tmp_s, idx_sigma = sorted_sigmas[dim-k], idx_in_sorted[sigma]
        idx_in_sorted[tmp_s], idx_in_sorted[sigma] = idx_in_sorted[sigma], dim-k
        sorted_sigmas[dim-k], sorted_sigmas[idx_sigma] = sigma, sorted_sigmas[dim-k]

        X[dim-k, :], X[idx_sigma, :] = X[idx_sigma, :], np.copy(X[dim-k, :])
        W[:k-1, dim-k], W[:k-1, idx_sigma] = W[:k-1, idx_sigma], np.copy(W[:k-1, dim-k])

        y[dim-k], y[idx_sigma] = y[idx_sigma], y[dim-k]
        z[dim-k], z[idx_sigma] = z[idx_sigma], z[dim-k]

        """
        2. If needed, we re-compute the matrix "beta times X". This should not be done too often.
        """
        if k > k0 + frequency_of_update:
            try:
                initialize_X_from_update(beta_P0, beta_P1, beta, X, pi, atol)
            except np.linalg.LinAlgError as error:
                if k < dim-1 and 'Matrix is singular' in str(error):
                    return multichain_message(whittle_idx)
                else:
                    raise error
            for i in range(dim):
                sorted_X[i] = np.copy(X[sorted_sigmas[i]])
            X = np.copy(sorted_X)
            k0 = k-1
        pi[sigma] = 0

        """
        3. We perform the recursive operations to compute beta*X, beta*y and beta*z.
        """
        try:
            update_W(W, sigma, X, k, atol, check_indexability, k0)
        except ZeroDivisionError:
            return multichain_message(whittle_idx)
        y += (1.0 - y[dim-k])*W[k-1]
        argmin, mu_min_k = find_mu_min(y[0:dim-k], z[0:dim-k], whittle_idx[sigma], atol)
        if np.isinf(mu_min_k):
            one_minus_y = 1.0 - y[dim-k:]
            if (one_minus_y < -atol).any() or (np.nonzero(abs(one_minus_y)<atol)[0] == np.nonzero(abs(z[dim-k:])<atol)[0]).any():
                is_indexable = NON_INDEXABLE
                print("Not indexable!")
                return is_indexable, whittle_idx
            else:
                for active_state in sorted_sigmas[0:dim-k]:
                    whittle_idx[active_state] = np.inf
                return is_indexable, whittle_idx
        next_sigma = sorted_sigmas[argmin]
        whittle_idx[next_sigma] = mu_min_k
        z -= (mu_min_k - whittle_idx[sigma])*(1.0-y)

        """
        4. If needed, we test if we violate the indexability condition.
        """
        if check_indexability and is_indexable:
            if (whittle_idx[sigma] + atol < mu_min_k) and ( z[dim-k:] > -atol ).any():
                is_indexable = NON_INDEXABLE
                print("Not indexable!")
                return is_indexable, whittle_idx
            elif ( y > 1.0 ).any():
                is_indexable = INDEXABLE_BUT_NOT_STRONGLY
        sigma = next_sigma
    if verbose: print('\b\b\b\b\b\b\b', end='')
    return is_indexable, whittle_idx
