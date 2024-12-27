import numpy as np
from scipy.integrate import solve_ivp
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

def generate_fourier_ic(x, n_modes=None, seed=None):
    """TODO: Generate random Fourier series initial condition.
    Hints:
    1. Use random coefficients for sin and cos terms
    2. Ensure the result is normalized to [-1, 1]
    3. Consider using np.random.normal for coefficients
    """
    if seed is not None:
        np.random.seed(seed)

    if n_modes is None:
        n_modes = np.random.randint(7, 9)
    
    # TODO: Generate coefficients for Fourier series
    a_coeffs = np.random.normal(size=n_modes)  # Cosine coefficients
    b_coeffs = np.random.normal(size=n_modes)  # Sine coefficients
    a0_coeff = np.random.normal(-0.5, 0.5)  # Constant term

    # TODO: Compute the Fourier series
    fourier_series = np.zeros_like(x, dtype=float)  # x = np.linspace(-1, 1, nx)
    P = 2
    for n in range(1, n_modes + 1):
        fourier_series += a_coeffs[n - 1] * np.cos(2 * np.pi * n * x / P)
        fourier_series += b_coeffs[n - 1] * np.sin(2 * np.pi * n * x / P)
    fourier_series += a0_coeff

    # TODO: Normalize to [-1, 1]
    max_val = np.max(np.abs(fourier_series))
    fourier_series = fourier_series / max_val
    assert np.isclose(fourier_series[0], fourier_series[-1]), "Fourier series is not periodic!"
    
    return fourier_series

def generate_gmm_ic(x, n_components=None, seed=None):
    """TODO: Generate Gaussian mixture model initial condition.
    Hints:
    1. Random number of components if n_components is None
    2. Use random means, variances, and weights
    3. Ensure result is normalized to [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_components is None:
        n_components = np.random.randint(7, 9)
    
    # TODO: Generate means, variances, and weights
    means = np.random.uniform(-1, 1, size=n_components)
    variances = np.random.uniform(0.01, 0.5, size=n_components)
    weights = np.random.dirichlet(np.ones(n_components))

    # extend grid to [-1.5T, 1.5T]
    x = np.concatenate([x - 2, x, x + 2])

    # TODO: Compute GMM
    gmm = np.zeros_like(x, dtype=float)
    for i in range(n_components):
        gmm += weights[i] * np.exp(-0.5 * ((x - means[i]) / variances[i])**2)
    
    for i in range(n_components):
        gmm += weights[i] * np.exp(-0.5 * ((x - means[i] - 2) / variances[i])**2)

    for i in range(n_components):
        gmm += weights[i] * np.exp(-0.5 * ((x - means[i] + 2) / variances[i])**2)

    gmm = gmm[len(x) // 3: 2 * len(x) // 3]

    # TODO: Normalize to [-1, 1]
    gmm = 2 * (gmm - np.min(gmm)) / (np.max(gmm) - np.min(gmm)) - 1
    assert np.isclose(gmm[0], gmm[-1], atol=1e-3), "GMM is not periodic!"

    return gmm

def generate_piecewise_ic(x, n_pieces=None, seed=None):
    """TODO: Generate piecewise linear initial condition.
    Hints:
    1. Generate random breakpoints
    2. Create piecewise linear function
    3. Add occasional discontinuities
    """
    if seed is not None:
        np.random.seed(seed)
    
    if n_pieces is None:
        n_pieces = np.random.randint(7, 9)
    
    # TODO: Generate breakpoints
    breakpoints = np.sort(np.random.uniform(-1, 1, size=n_pieces+1))
    breakpoints[0] = -1
    breakpoints[-1] = 1

    # TODO: Generate values at breakpoints
    values = np.random.uniform(-1, 1, size=breakpoints.shape[0])
    # make sure the periodic boundary condition
    values[-1] = values[0]

    # TODO: Create piecewise linear function
    piecewise = np.zeros_like(x, dtype=float)
    piecewise = np.interp(x, breakpoints, values)

    # normalize to [-1, 1]
    piecewise = 2 * (piecewise - np.min(piecewise)) / (np.max(piecewise) - np.min(piecewise)) - 1
    assert np.isclose(piecewise[0], piecewise[-1], atol=1e-3), "Piecewise function is not periodic!"

    return piecewise

def allen_cahn_rhs(t, u, epsilon, x_grid):
    """TODO: Implement Allen-Cahn equation RHS:
        ∂u/∂t = Δu - (1/ε²)(u³ - u)
    """
    dx = x_grid[1] - x_grid[0]
    
    # TODO: Compute Laplacian (Δu) with periodic boundary conditions
    laplacian = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx**2)
    # TODO: Compute nonlinear term -(1/ε²)(u³ - u)
    nonlin = -(1 / epsilon**2) * (u**3 - u)
    # TODO: Return full RHS
    return laplacian + nonlin
    # pass

def generate_dataset(n_samples, epsilon, x_grid, t_eval, ic_type='fourier', seed=None, dataset_type='train', n_modes=None):
    """Generate dataset for Allen-Cahn equation."""
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize dataset array
    dataset = np.zeros((n_samples, len(t_eval), len(x_grid)))   # (n_samples, n_timesteps, n_gridpoints)
    print (f"Generating {dataset_type} dataset with {n_samples} samples, epsilon = {epsilon}, IC type = {ic_type}")
    # Generate samples
    for i in range(n_samples):
        # Generate initial condition based on type
        if ic_type == 'fourier':
            u0 = generate_fourier_ic(x_grid, seed=seed+i if seed else None, n_modes=n_modes)
        elif ic_type == 'gmm':
            u0 = generate_gmm_ic(x_grid, seed=seed+i if seed else None, n_components=n_modes)
        elif ic_type == 'piecewise':
            u0 = generate_piecewise_ic(x_grid, seed=seed+i if seed else None, n_pieces=n_modes)
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")
        
        if (i % 20 == 0):
            print(f"Generating sample {i}")
            
        # Solve PDE using solve_ivp
        sol = solve_ivp(
            allen_cahn_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=t_eval,
            args=(epsilon, x_grid),
            method='RK45',
            rtol=1e-6,
            atol=1e-6
        )
        
        dataset[i] = sol.y.T

        os.makedirs('Foundational_Models_for_Allen_Cahn/vis_fig', exist_ok=True)
        if i == 0:
            plt.figure(figsize=(8, 6))
            plt.plot(x_grid, dataset[i, 0], label='Initial Condition')
            plt.plot(x_grid, dataset[i, 1], label='t=0.25')
            plt.plot(x_grid, dataset[i, 2], label='t=0.50')
            plt.plot(x_grid, dataset[i, 3], label='t=0.75')
            plt.plot(x_grid, dataset[i, 4], label='t=1.00')

            plt.legend()
            plt.xlabel('x')
            plt.ylabel('u(t=0, x)')
            # plt.show()
            plt.savefig(f'Foundational_Models_for_Allen_Cahn/vis_fig/initial_condition_{ic_type}_{epsilon}_{dataset_type}.png')
            
    
    return dataset

def main():
    """Generate all datasets."""
    # Set up spatial grid
    nx = 128
    x_grid = np.linspace(-1, 1, nx)
    
    # Set up temporal grid
    t_eval = np.linspace(0, 1e-4, 5)

    # remove the existing dataset folder
    os.system('rm -rf Foundational_Models_for_Allen_Cahn/vis_fig')
    os.system('rm -rf Foundational_Models_for_Allen_Cahn/dataset')
    
    # Parameters for datasets
    epsilons = [0.02, 0.01, 0.005]  # Different epsilon values
    epsilons_test = [0.02, 0.015, 0.01, 0.0075, 0.005]  # Different epsilon values for test
    n_train = 1000  # Number of training samples per configuration
    n_test = 200    # Number of test samples
    base_seed = 42  # For reproducibility
    
    # TODO: Generate training datasets for each epsilon and IC type
    # construct the dataset folder
    os.makedirs('Foundational_Models_for_Allen_Cahn/dataset', exist_ok=True)
    os.makedirs('Foundational_Models_for_Allen_Cahn/dataset/train', exist_ok=True)
    os.makedirs('Foundational_Models_for_Allen_Cahn/dataset/test', exist_ok=True)
    os.makedirs('Foundational_Models_for_Allen_Cahn/dataset/ood', exist_ok=True)
    os.makedirs('Foundational_Models_for_Allen_Cahn/dataset/high_freq', exist_ok=True)

    def run_gen_dataset(n_samples, epsilon, ic_type, base_seed, dataset_type, n_modes=None):
        data = generate_dataset(n_samples, epsilon, x_grid, t_eval, ic_type, seed=base_seed, dataset_type=dataset_type, n_modes=n_modes)
        np.save(f'Foundational_Models_for_Allen_Cahn/dataset/{dataset_type}/{dataset_type}_data_{epsilon}_{ic_type}.npy', data)
        return epsilon, data

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(delayed(run_gen_dataset)(n_train, epsilon, ic_type, base_seed, 'train') for epsilon in epsilons for ic_type in ['fourier', 'gmm', 'piecewise'])
    data = {}
    for epsilon in epsilons:
        for eps, d in results:
            if eps == epsilon:
                if str(eps) not in data:
                    data[str(eps)] = d
                else:
                    data[str(eps)] = np.concatenate([data[str(eps)], d], axis=0)
    np.savez(f'Foundational_Models_for_Allen_Cahn/dataset/train/train_data.npz', **data)

    # TODO: Generate standard test dataset
    results = Parallel(n_jobs=-1)(delayed(run_gen_dataset)(n_test, epsilon, ic_type, base_seed + n_train, 'test') for epsilon in epsilons_test for ic_type in ['fourier', 'gmm', 'piecewise'])
    data = {}
    for epsilon in epsilons_test:
        for eps, d in results:
            if eps == epsilon:
                if str(eps) not in data:
                    data[str(eps)] = d
                else:
                    data[str(eps)] = np.concatenate([data[str(eps)], d], axis=0)
    np.savez(f'Foundational_Models_for_Allen_Cahn/dataset/test/test_data.npz', **data)

    # TODO: Generate OOD test datasets (high frequency, sharp transitions)
    # with smaller epsilon values leading to sharper interfaces
    epsilons_ood = [0.05, 0.03, 0.025, 0.003, 0.002, 0.001]
    results = Parallel(n_jobs=-1)(delayed(run_gen_dataset)(n_test, epsilon, ic_type, base_seed + n_train + n_test, 'ood') for epsilon in epsilons_ood for ic_type in ['fourier', 'gmm', 'piecewise'])
    data = {}
    for epsilon in epsilons_ood:
        for eps, d in results:
            if eps == epsilon:
                if str(eps) not in data:
                    data[str(eps)] = d
                else:
                    data[str(eps)] = np.concatenate([data[str(eps)], d], axis=0)
    np.savez(f'Foundational_Models_for_Allen_Cahn/dataset/ood/ood_data.npz', **data)

    epsilons_hf = epsilons
    results = Parallel(n_jobs=-1)(delayed(run_gen_dataset)(n_test, epsilon, ic_type, base_seed + n_train + n_test, 'high_freq', 25) for epsilon in epsilons_hf for ic_type in ['fourier', 'gmm', 'piecewise'])
    data = {}
    for epsilon in epsilons_hf:
        for eps, d in results:
            if eps == epsilon:
                if str(eps) not in data:
                    data[str(eps)] = d
                else:
                    data[str(eps)] = np.concatenate([data[str(eps)], d], axis=0)
    np.savez(f'Foundational_Models_for_Allen_Cahn/dataset/high_freq/high_freq_data.npz', **data)

    # TODO: Save all datasets using np.save

if __name__ == '__main__':
    main()
