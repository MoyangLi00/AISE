from sklearn.gaussian_process.kernels import RBF
import scipy
import numpy as np


def rbf_kernel(xa, xb, sigma_f=1, l=1):
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    kernel = sigma_f**2 * np.exp(sq_norm)
    return kernel

def gaussian_process_generation(grid_resolution=50, n_samples=100, sigma_f=1, l=1):
    X = np.expand_dims(np.linspace(-1,1,grid_resolution+1),1)
    Sigma = rbf_kernel(X, X, sigma_f,l)
    ys = np.random.multivariate_normal(mean=np.zeros(grid_resolution+1), cov=Sigma, size=n_samples)
    return ys

def piecewise_linear_func(break_pts, eval_pts):
    """
    break_pts: (K+1) x 2, which stores x, y coords of break points

    Return:
    coeffs: K x 2 coeff matrix stores alpha and beta
    """
    X0 = break_pts[:-1]
    X1 = break_pts[1:]
    alpha = (X1[:,1] - X0[:,1])/(X1[:,0] - X0[:,0])
    beta = (X1[:,0]*X0[:,1] - X1[:,1]*X0[:,0])/(X1[:,0]-X0[:,0])
    coeffs = np.column_stack((alpha, beta))
    
    # Evaluate the function at the desired points
    y_eval = np.zeros_like(eval_pts)
    print(break_pts)
    for i, x in enumerate(eval_pts):
        # Find the appropriate interval
        idx = np.searchsorted(break_pts[:,0], x, side='right')-1 
        # print(x, idx)
        idx = np.clip(idx, 0, len(alpha) - 1)  # Ensure idx is within bounds
        
        # Compute the function value
        y_eval[i] = alpha[idx] * x + beta[idx]
    # import pdb; pdb.set_trace()
    
    return y_eval


def piecewise_linear_generation(n_break_pts=11, grid_resolution=50, n_samples=100, low=-1, high=1):
    break_pts_x = np.random.uniform(low=-1, high=1, size=(n_break_pts, 1))
    break_pts_y = np.random.uniform(low=low,high=high, size=(n_break_pts,1))
    break_pts_rand = np.concatenate((break_pts_x, break_pts_y), axis=1)
    break_pts_inner = break_pts_rand[break_pts_rand[:, 0].argsort()]
    break_pts = np.zeros((n_break_pts+2,2))
    break_pts[0] = np.array([-1, 0])
    break_pts[-1] = np.array([1, 0])
    break_pts[1:-1] = break_pts_inner
    grid = np.linspace(-1,1,grid_resolution+1)
    all_pts = np.zeros((n_samples, grid_resolution+1))
    for i in range(n_samples):
        y_eval = piecewise_linear_func(break_pts, grid)
        all_pts[i] = y_eval
    
    return all_pts

def chebyshev_poly_func(eval_pts, n=5):
    T0 = np.ones_like(eval_pts)
    T1 = eval_pts.copy()
    chebyshev_evals = np.zeros((len(eval_pts), n))
    chebyshev_evals[:, 0] = T0
    chebyshev_evals[:, 1] = T1
    for i in range(2, n):
        chebyshev_evals[:, i] = 2 * T1 *chebyshev_evals[:,i-1] - chebyshev_evals[:, i-2]
    return chebyshev_evals

def chebyshev_poly_generation(n=5, grid_resolution=50, n_sample=100, coeff_bound=0.1):
    random_coeff = np.random.uniform(low=-coeff_bound, high=coeff_bound, size=(n,))
    grid = np.linspace(-1,1,grid_resolution+1)
    chebyshev_evals = chebyshev_poly_func(grid, n)
    chebyshev_f = np.zeros(grid_resolution+1)
    data = np.zeros((n_sample, grid_resolution+1))
    for data_i in range(n_sample):
        for i in range(n):
            chebyshev_f += random_coeff[i] * chebyshev_evals[:, i]
        data[data_i] = chebyshev_f
    return data