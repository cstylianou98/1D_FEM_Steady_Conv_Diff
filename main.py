import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

from functions import *


def configure_simulation():
    # convection, diffusion coefficients

    print("\nWelcome to the steady 1D convection diffusion problem. We will be solving this problem with the standard galerking method and SUPG to illustrate its differences\n")
    a = get_float_input_with_default(">>> Please enter a convection speed (Press 'Enter' for default value) -----> ", 1)
    nu = get_float_input_with_default(">>> Please enter a value for kinematic viscosity (Press 'Enter' for default value) -----> ", 0.1)
    
    L = 1.0
    numel = 10
    h = L/numel
    numnp = numel + 1
    xnode = np.linspace(0, L, numnp)

    Pe = (a * h) / (2 * nu)
    print(f'\nThe problem you are solving for has a Peclet number = {Pe}')

    # Gauss points and weights for [-1, 1]
    xipg = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    wpg = np.array([1, 1])

    # Shape functions and their derivatives on reference element
    N_mef = np.array([(1 - xipg) / 2, (1 + xipg) / 2])
    Nxi_mef = np.array([[-1/2, 1/2], [-1/2, 1/2]])

    return {
        'a': a,
        'nu': nu,
        'L': L,
        'numel': numel,
        'h': h,
        'numnp': numnp,
        'xnode': xnode,
        'Pe': Pe,
        'xipg': xipg,
        'wpg': wpg,
        'N_mef': N_mef,
        'Nxi_mef': Nxi_mef,
    }


def run_simulation(config):
    K_galerkin, F_galerkin = assemble_matrices_galerkin(config['a'], config['nu'], config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['xipg'])
    K_supg, F_supg = assemble_matrices_SUPG(config['a'], config['nu'], config['numel'], config['xnode'], config['N_mef'], config['Nxi_mef'], config['wpg'], config['xipg'])
    
    u_standard_galerkin = solve(K_galerkin, F_galerkin)
    u_supg = solve(K_supg, F_supg)

    # exact solution
    x = np.linspace(0, config['L'], 100)
    u_analytical = exact_sol(x, config['a'], config['nu'])

    return x, u_analytical, u_standard_galerkin, u_supg

def main():
    config = configure_simulation()
    x, u_analytical, u_standard_galerkin, u_supg = run_simulation(config)
    plot_solution(x, config['xnode'], u_analytical, u_standard_galerkin, u_supg, config['Pe'])

if __name__ == "__main__":
    main()

