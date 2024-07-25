import numpy as np
import matplotlib.pyplot as plt
import os

def exact_sol (x, a, nu):
    return (x + (1 - np.exp(a/nu * x)) / (np.exp(a/nu) - 1)) / a

def get_float_input_with_default(prompt, default_value):
    user_input = (input(prompt))
    if user_input == '':
        user_val = default_value
    else:
        user_val = float(user_input)
    return user_val

def source_term (x):
    return 1

def assemble_matrices_galerkin(a, nu, numel, xnode, N_mef, Nxi_mef, wpg, xipg):

    numnp = numel + 1
    
    # Allocation
    K = np.zeros((numnp, numnp))
    F = np.zeros(numnp)

    # Loop on elements
    for i in range(numel):
        h = xnode[i+1] - xnode[i]
        xm = (xnode[i] + xnode[i+1]) / 2
        weight = wpg * h / 2
        isp = [i, i+1]  # Global number of the nodes of the current element
        
        ngaus = wpg.shape[0]
        #looping on the gauss points
        for ig in range (ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]
            x = xm + h / 2 * xipg[ig] # x-coordinate of the gauss point
            # matrix assembly
            K[np.ix_(isp,isp)] += w_ig * (a*np.outer(N,Nx) + np.outer(Nx, nu* Nx) )
            F [isp] +=  w_ig * ( N * source_term(x)) 

        # Applying Dirichlet boundary condition at the left boundary (x = 0)
    K[0, :] = 0
    K[:, 0] = 0
    K[0, 0] = 1
    F[0] = 0
    
    # Applying Dirichlet boundary condition at the right boundary (x = L)
    K[numnp-1, :] = 0
    K[:, numnp-1] = 0
    K[numnp-1, numnp-1] = 1
    F[numnp-1] = 0 
    return K, F 

def assemble_matrices_SUPG(a, nu, numel, xnode, N_mef, Nxi_mef, wpg, xipg):

    numnp = numel + 1
    
    # Allocation
    K = np.zeros((numnp, numnp))
    F = np.zeros(numnp)

    # Loop on elements
    for i in range(numel):
        h = xnode[i+1] - xnode[i]
        xm = (xnode[i] + xnode[i+1]) / 2
        weight = wpg * h / 2
        isp = [i, i+1]  # Global number of the nodes of the current element
        
        ngaus = wpg.shape[0]
        #looping on the gauss points
        for ig in range (ngaus):
            N = N_mef[ig, :]
            Nx = Nxi_mef[ig, :] * 2 / h
            w_ig = weight[ig]
            x = xm + h / 2 * xipg[ig] # x-coordinate of the gauss point

            Pe = (a * h) / (2 * nu)
            alfa = 1 / (np.tanh(Pe)) - 1/Pe
            tau = alfa*h/(2*a)

            # matrix assembly
            K[np.ix_(isp,isp)] += w_ig * (np.outer(N*a, Nx) + np.outer(nu*Nx, Nx) + np.outer(Nx*a*tau, a*Nx))
            F [isp] +=  w_ig * ( a * Nx * tau * source_term(x) + N * source_term(x))

        # Applying Dirichlet boundary condition at the left boundary (x = 0)
    K[0, :] = 0
    K[:, 0] = 0
    K[0, 0] = 1
    F[0] = 0
    
    # Applying Dirichlet boundary condition at the right boundary (x = L)
    K[numnp-1, :] = 0
    K[:, numnp-1] = 0
    K[numnp-1, numnp-1] = 1
    F[numnp-1] = 0 
    return K, F 


def plot_solution(x, xnode, u_analytical, u_galerkin, u_supg, Pe):
    # # Plot the exact solution

    # Plot the Galerkin solution as a dotted line
    plt.plot(xnode, u_galerkin, linestyle='--', marker = 'x',label='Galerkin')
    plt.plot(xnode, u_supg, linestyle ='-.', marker = '*', label = 'SUPG' )
    plt.plot(x, u_analytical, label='Analytical')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'1D Convection Diffusion Steady for Pe={Pe}')
    plt.legend()
    plt.ylim(-0.2, 1.2)

    folder_path = './results'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{folder_path}/Pe={Pe}.png')