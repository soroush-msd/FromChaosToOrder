import numpy as np
import matplotlib.pyplot as plt

# Define the true data distribution and model distribution
def fisher_divergence_loss(sigma_d2, sigma2):
    return 0.5 * sigma_d2 * ((1 / sigma_d2) - (1 / sigma2))**2

def j_theta(sigma_d2, sigma2):
    return ((sigma_d2**2) / (2 * sigma2**4)) - (1 / sigma2**2)

def j_hat_theta(samples, sigma2, num_runs=1):
    estimates = []
    for _ in range(num_runs):
        gradient_log_p = -samples / sigma2
        trace_hessian_log_p = -1 / sigma2
        term1 = trace_hessian_log_p
        term2 = 0.5 * np.mean(gradient_log_p**2)
        estimates.append(term1 + term2)
    return np.mean(estimates)  # Averaging over multiple runs


# Parameters for the true distribution and samples
sigma_d2 = 1.0
sample_sizes = [10, 50, 100, 1000]
sigma2_values = np.linspace(0.1, 3.0, 10000)

# Generate plots for different sample sizes
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, N in enumerate(sample_sizes):
    samples = np.random.normal(0, np.sqrt(sigma_d2), size=N)

    # Compute the original loss values
    L_vals = [fisher_divergence_loss(sigma_d2, sigma2) for sigma2 in sigma2_values]
    J_vals = [j_theta(sigma_d2, sigma2) for sigma2 in sigma2_values]
    J_hat_vals = [j_hat_theta(samples, sigma2) for sigma2 in sigma2_values]

    # Apply log(1 + loss) transformation
    L_vals_log = [np.log1p(L) for L in L_vals]
    J_vals_log = [np.log1p(J) for J in J_vals]
    J_hat_vals_log = [np.log1p(J_hat) for J_hat in J_hat_vals]

    # Find minimal points for transformed losses
    min_L_log = min(L_vals_log)
    min_J_log = min(J_vals_log)
    min_J_hat_log = min(J_hat_vals_log)

    min_L_idx = np.argmin(L_vals_log)
    min_J_idx = np.argmin(J_vals_log)
    min_J_hat_idx = np.argmin(J_hat_vals_log)

    ax = axes[idx]
    ax.plot(sigma2_values, L_vals_log, label=f"log(1 + L(\u03b8))", linestyle='--')
    ax.plot(sigma2_values, J_vals_log, label=f"log(1 + J(\u03b8))", linestyle='-')
    ax.plot(sigma2_values, J_hat_vals_log, label=f"log(1 + Ĵ(\u03b8))", linestyle='-.')

    # Add scatter points for minimal values
    ax.scatter(sigma2_values[min_L_idx], min_L_log, color='blue', label='Min log(1 + L(\u03b8))', zorder=5)
    ax.scatter(sigma2_values[min_J_idx], min_J_log, color='orange', label='Min log(1 + J(\u03b8))', zorder=5)
    ax.scatter(sigma2_values[min_J_hat_idx], min_J_hat_log, color='green', label='Min log(1 + Ĵ(\u03b8))', zorder=5)

    ax.set_title(f"Sample Size N={N}")
    ax.set_xlabel("\u03c3²")
    ax.set_ylabel("log(1 + Loss)")
    ax.legend()

plt.tight_layout()
plt.show()
