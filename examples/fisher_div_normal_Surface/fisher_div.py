import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# true parameters
mu_d = 0.25
sigma_d2 = 1.2

# define loss functions
def L(mu, sigma2):
    return ((sigma2 - sigma_d2)**2) / (2 * sigma_d2 * sigma2**2) + (mu - mu_d)**2 / (2 * sigma2**2)

def J(mu, sigma2):
    return ((mu - mu_d)**2 + sigma_d2 - 2 * sigma2) / (2 * sigma2**2)

def J_hat(mu, sigma2, samples):
    x_bar = np.mean(samples)
    x_sq_bar = np.mean(samples**2)
    return -1/sigma2 + (mu**2 - 2*mu*x_bar + x_sq_bar) / (2 * sigma2**2)

# generate parameter grid (denser near true parameters)
mu = np.linspace(-1, 1, 100)  # focus near μ=0
sigma2 = np.linspace(0.5, 1.5, 100)  # focus near σ²=1
Mu, Sigma2 = np.meshgrid(mu, sigma2)

# Sample sizes to visualize
sample_sizes = [10, 100, 1000]

for N in sample_sizes:
    # generate data samples
    samples = np.random.normal(mu_d, np.sqrt(sigma_d2), N)
    
    # compute losses
    L_vals = L(Mu, Sigma2)
    J_vals = J(Mu, Sigma2)
    J_hat_vals = J_hat(Mu, Sigma2, samples)
    
    # find minima
    def find_min(grid, loss):
        min_idx = np.argmin(loss)
        return grid[0].flatten()[min_idx], grid[1].flatten()[min_idx]
    
    min_L = find_min((Mu, Sigma2), L_vals)
    min_J = find_min((Mu, Sigma2), J_vals)
    min_J_hat = find_min((Mu, Sigma2), J_hat_vals)
    
    # create figure with 3 subplots
    fig = plt.figure(figsize=(18, 6))
    #fig.suptitle(f'Sample Size N = {N}', fontsize=16, y=1.05)
    fig.text(
        0.5, 0.92,  # position (x=50%, y=92% of figure height)
        f'Sample Size: $N = {N}$', 
        ha='center', va='top', 
        fontsize=12, fontweight='light', color='black'
    )

    # plot 1: L(θ) vs reference
    ax1 = fig.add_subplot(131, projection='3d')
    # surface with contours
    surf_L = ax1.plot_surface(Mu, Sigma2, L_vals, cmap='viridis', alpha=0.8)
    ax1.contour(Mu, Sigma2, L_vals, zdir='z', offset=np.min(L_vals), cmap='viridis', linestyles='dashed')
    # highlight minima and true parameters
    ax1.scatter(mu_d, sigma_d2, L(mu_d, sigma_d2), color='red', s=200, marker='+', label='True Parameters')
    ax1.scatter(*min_L, L(*min_L), color='black', s=150, marker='o', label='Minima')
    ax1.view_init(elev=30, azim=-60)  # adjust camera angle
    ax1.set_title('True Loss $L(θ)$', pad=15)
    ax1.set_xlabel('μ')
    ax1.set_ylabel('σ²')
    ax1.legend()
    
    # plot 2: J(θ) vs reference
    ax2 = fig.add_subplot(132, projection='3d')
    surf_J = ax2.plot_surface(Mu, Sigma2, J_vals, cmap='plasma', alpha=0.8)
    ax2.contour(Mu, Sigma2, J_vals, zdir='z', offset=np.min(J_vals), cmap='plasma', linestyles='dashed')
    ax2.scatter(mu_d, sigma_d2, J(mu_d, sigma_d2), color='red', s=200, marker='+')
    ax2.scatter(*min_J, J(*min_J), color='black', s=150, marker='o')
    ax2.view_init(elev=30, azim=-60)
    ax2.set_title('Theoretical Loss $J(θ)$', pad=15)
    ax2.set_xlabel('μ')
    ax2.set_ylabel('σ²')
    ax2.legend()
    
    # plot 3: Ĵ(θ) vs reference
    ax3 = fig.add_subplot(133, projection='3d')
    surf_J_hat = ax3.plot_surface(Mu, Sigma2, J_hat_vals, cmap='coolwarm', alpha=0.8)
    ax3.contour(Mu, Sigma2, J_hat_vals, zdir='z', offset=np.min(J_hat_vals), cmap='coolwarm', linestyles='dashed')
    ax3.scatter(mu_d, sigma_d2, J_hat(mu_d, sigma_d2, samples), color='red', s=200, marker='+')
    ax3.scatter(*min_J_hat, J_hat(*min_J_hat, samples), color='black', s=150, marker='o')
    ax3.view_init(elev=30, azim=-60)
    ax3.set_title('Empirical Loss $Ĵ(θ)$', pad=15)
    ax3.set_xlabel('μ')
    ax3.set_ylabel('σ²')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()