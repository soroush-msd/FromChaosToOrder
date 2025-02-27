import numpy as np
import matplotlib.pyplot as plt

# define the true data distribution (Exponential) and model distribution
def fisher_divergence_loss(lambda_d, lambda_val):
    return 0.5 * (lambda_d - lambda_val)**2

def j_theta(lambda_val):
    return (0.5 * lambda_val**2)

def j_hat_theta(samples, lambda_val):
    # for Exponential distribution, score = -lambda (constant), so:
    trace_hessian = 0  # derivative of score w.r.t. x is 0
    squared_score = lambda_val**2  # (-lambda)^2
    return (trace_hessian + 0.5 * squared_score)

# parameters for the true distribution and samples
lambda_d = 1.0  # true rate parameter of data distribution
sample_sizes = [10, 50, 100, 1000]
lambda_values = np.linspace(0.1, 3.0, 100)  # range of lambda to plot

# generate plots for different sample sizes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, N in enumerate(sample_sizes):
    # generate samples from Exp(lambda_d)
    samples = np.random.exponential(scale=1/lambda_d, size=N)
    
    # Compute loss values
    L_vals = [fisher_divergence_loss(lambda_d, l) for l in lambda_values]
    J_vals = [j_theta(l) for l in lambda_values]
    J_hat_vals = [j_hat_theta(samples, l) for l in lambda_values]  # independent of samples
    
    # apply log(1 + loss) transformation for better visualization
    L_vals_log = np.log1p(L_vals)
    J_vals_log = np.log1p(J_vals)
    J_hat_vals_log = np.log1p(J_hat_vals)
    
    # find minimal points (for demonstration)
    min_L_log = np.min(L_vals_log)
    min_J_log = np.min(J_vals_log)
    min_J_hat_log = np.min(J_hat_vals_log)
    
    min_L_idx = np.argmin(L_vals_log)
    min_J_idx = np.argmin(J_vals_log)
    min_J_hat_idx = np.argmin(J_hat_vals_log)
    
    ax = axes[idx]
    ax.plot(lambda_values, L_vals_log, label="L(λ)", linestyle='-', color='blue')
    ax.plot(lambda_values, J_vals_log, label="J(λ)", linestyle='-', color='green')
    ax.plot(lambda_values, J_hat_vals_log, label="Ĵ(λ)", linestyle='-', color='red')
    
    ax.scatter(lambda_values[min_L_idx], min_L_log, color='blue', zorder=5)
    ax.scatter(lambda_values[min_J_idx], min_J_log, color='green', zorder=5)
    ax.scatter(lambda_values[min_J_hat_idx], min_J_hat_log, color='red', zorder=5)
    
    ax.set_title(f"Sample Size N={N}")
    ax.set_xlabel("λ (Model Parameter)")
    ax.set_ylabel("log(1 + Loss)")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()