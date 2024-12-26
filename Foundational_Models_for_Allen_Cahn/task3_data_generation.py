"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
    f = -6
"""

import matplotlib.pyplot as plt
import fenics
import numpy as np
from task3_random_f import gaussian_process_generation, piecewise_linear_generation, chebyshev_poly_generation

def solve_poisson(f_data, resolution=50):
    # Create mesh and define function space
    mesh = fenics.IntervalMesh(resolution, -1, 1)

    V = fenics.FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    u_D = fenics.Expression('0', degree=1)

    def boundary(x, on_boundary):
        return on_boundary

    bc = fenics.DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = fenics.TrialFunction(V)
    v = fenics.TestFunction(V)
    f = fenics.Function(V)
    print(f_data.shape)
    f.vector().set_local(-f_data)

    a = fenics.dot(fenics.grad(u), fenics.grad(v))*fenics.dx
    L = f*v*fenics.dx

    # Compute solution
    u = fenics.Function(V)
    fenics.solve(a == L, u, bc)

    vertex_values_u = u.compute_vertex_values(mesh)

    return vertex_values_u

def data_generation(grid_resolution, mode='gaussian_process'):
    if mode == 'gaussian_process':
        data = gaussian_process_generation(grid_resolution=grid_resolution, sigma_f=1)
    elif mode == 'piecewise_linear':
        data = piecewise_linear_generation(grid_resolution=grid_resolution, n_break_pts=10, low=-3, high=3)
    elif mode == 'chebyshev':
        data = chebyshev_poly_generation(grid_resolution=grid_resolution,n=10, coeff_bound=0.05)
    else:
        raise Exception("Invalide mode")
    print(data.shape)

    n_sample, n_grid = data.shape
    u_data = np.zeros_like(data)
    print(u_data.shape)
    for i in range(n_sample):
        u_data[i, :] = solve_poisson(data[i, :], grid_resolution)
    np.save(f"sol_{mode}.npy", u_data)
    np.save(f"f_{mode}.npy", data)


if __name__ == '__main__':
    for mode in ['gaussian_process', 'piecewise_linear', 'chebyshev']:
    # for mode in ['chebyshev']:
    # for mode in ['piecewise_linear', 'chebyshev']:
        data_generation(50, mode)