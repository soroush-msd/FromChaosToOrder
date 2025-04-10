import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class MoG(torch.nn.Module):
    def __init__(self, n_components):
        super().__init__()
        self.n_components = n_components
        # Initialize parameters
        self.weights = torch.nn.Parameter(torch.ones(n_components) / n_components)
        self.means = torch.nn.Parameter(torch.tensor([-2.0, 2.0]))  
        self.log_stds = torch.nn.Parameter(torch.zeros(n_components) - 0.5)  
        
    def forward(self, x):
        """Compute log probability density at x"""
        x = x.unsqueeze(-1)  # Shape: (batch_size, 1)
        stds = torch.exp(self.log_stds) + 1e-6
        variances = stds ** 2
        
        # Compute log density for each component
        log_densities = -0.5 * ((x - self.means) ** 2) / variances - 0.5 * torch.log(2 * np.pi * variances)
        
        # Add log mixture weights
        log_weights = torch.log_softmax(self.weights, dim=-1)
        log_mixture = log_densities + log_weights
        
        # Combine mixture components
        return torch.logsumexp(log_mixture, dim=-1)

# Training functions with separate score calculation
def score_matching_loss(model, x):
    """
    Calculate the score matching loss (J_hat) using a single batch
    This avoids computation graph issues by recalculating score and hessian directly
    """
    x = x.detach().clone().requires_grad_(True)
    
    # Forward pass to get log probabilities
    log_prob = model(x)
    
    # Compute score (first derivative)
    grad_outputs = torch.ones_like(log_prob)
    score = torch.autograd.grad(
        outputs=log_prob, 
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True
    )[0]
    
    # Compute Hessian (second derivative)
    grad_outputs_score = torch.ones_like(score)
    hessian = torch.autograd.grad(
        outputs=score,
        inputs=x,
        grad_outputs=grad_outputs_score,
        create_graph=True
    )[0]
    
    # J_hat = E[trace(∇_x s_m) + 0.5 ||s_m||^2]
    return hessian.mean() + 0.5 * (score**2).mean()

def compute_score(model, x):
    """Compute score function separately (for evaluation only)"""
    x_tensor = x.detach().clone().requires_grad_(True)
    log_p = model(x_tensor)
    
    grad_outputs = torch.ones_like(log_p)
    score = torch.autograd.grad(
        outputs=log_p, 
        inputs=x_tensor,
        grad_outputs=grad_outputs,
        create_graph=False  # No need for higher derivatives here
    )[0]
    
    return score.detach()

def compute_all_objectives(model, true_model, samples):
    """Compute all three objectives for monitoring"""
    # Compute both model scores
    model_score = compute_score(model, samples)
    true_score = compute_score(true_model, samples)
    
    # L(θ) = 0.5 * E[||s_m - s_d||^2]
    l_val = 0.5 * ((model_score - true_score) ** 2).mean()
    
    # For J(θ) and J_hat(θ), we recompute with the score matching loss function
    j_val = score_matching_loss(model, samples)
    
    # J_hat is same as J for this implementation
    j_hat_val = j_val
    
    return l_val.item(), j_val.item(), j_hat_val.item()

# Generate true distribution and sample data
true_weights = torch.tensor([0.3, 0.7])
true_means = torch.tensor([-2.0, 2.0])
true_stds = torch.tensor([0.5, 1.0])

# Create a "true" model with fixed parameters
true_model = MoG(n_components=2)
with torch.no_grad():
    true_model.weights.copy_(torch.log(true_weights))  # Use log weights since we use log_softmax
    true_model.means.copy_(true_means)
    true_model.log_stds.copy_(torch.log(true_stds))

# Fix the parameters of the true model
for param in true_model.parameters():
    param.requires_grad = False

# Sample data
n_samples = 1000
torch.manual_seed(42)
component = torch.multinomial(true_weights, n_samples, replacement=True)
samples = true_means[component] + true_stds[component] * torch.randn(n_samples)

# Initialize model and optimizer
model = MoG(n_components=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Initialize arrays to store metrics
epochs = []
L_values = []
J_values = []
J_hat_values = []
means_history = []
stds_history = []
weights_history = []

# Train the model with basic score matching
n_epochs = 3000
eval_every = 50  # How often to evaluate metrics
batch_size = 200  # Batch size for training

progress_bar = tqdm(range(n_epochs), desc="Training")
for epoch in progress_bar:
    # Sample a batch for training
    if batch_size < n_samples:
        idx = torch.randperm(n_samples)[:batch_size]
        batch = samples[idx]
    else:
        batch = samples
    
    optimizer.zero_grad()
    
    # Calculate loss (J_hat)
    loss = score_matching_loss(model, batch)
    
    # Backpropagate and update
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Evaluate metrics periodically
    if epoch % eval_every == 0:
        # Compute all three objectives
        l_val, j_val, j_hat_val = compute_all_objectives(model, true_model, samples)
        
        # Record metrics
        epochs.append(epoch)
        L_values.append(l_val)
        J_values.append(j_val)
        J_hat_values.append(j_hat_val)
        
        # Record parameter values
        with torch.no_grad():
            means = model.means.clone().numpy()
            stds = torch.exp(model.log_stds).clone().numpy()
            weights = torch.softmax(model.weights, dim=-1).clone().numpy()
            
            means_history.append(means)
            stds_history.append(stds)
            weights_history.append(weights)
        
        # Update progress bar
        progress_bar.set_postfix({
            'L': f"{l_val:.4f}",
            'J': f"{j_val:.4f}"
        })
    
    # Print detailed update occasionally
    if epoch % 500 == 0:
        with torch.no_grad():
            print(f"\nEpoch {epoch}")
            print(f"  Means: {model.means.numpy()}, True: {true_means.numpy()}")
            print(f"  Stds: {torch.exp(model.log_stds).numpy()}, True: {true_stds.numpy()}")
            print(f"  Weights: {torch.softmax(model.weights, dim=-1).numpy()}, True: {true_weights.numpy()}")
            if epochs:
                print(f"  L: {L_values[-1]:.4f}, J: {J_values[-1]:.4f}, J_hat: {J_hat_values[-1]:.4f}")

# Convert histories to numpy arrays
means_history = np.array(means_history)
stds_history = np.array(stds_history)
weights_history = np.array(weights_history)

# Plot metrics and parameter convergence
plt.figure(figsize=(15, 10))

# Plot all three objective functions
plt.subplot(2, 2, 1)
plt.plot(epochs, L_values, 'r-', label='L(θ) - Fisher Divergence')
plt.plot(epochs, J_values, 'g-', label='J(θ) - Theoretical')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Score Matching Objective Functions')
plt.legend()
plt.grid(alpha=0.3)

# Plot means convergence
plt.subplot(2, 2, 2)
plt.plot(epochs, means_history[:, 0], 'b-', label='Mean 1')
plt.plot(epochs, means_history[:, 1], 'r-', label='Mean 2')
plt.axhline(y=true_means[0].item(), color='b', linestyle='--', alpha=0.5, label='True Mean 1')
plt.axhline(y=true_means[1].item(), color='r', linestyle='--', alpha=0.5, label='True Mean 2')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Means Convergence')
plt.legend()
plt.grid(alpha=0.3)

# Plot stds convergence
plt.subplot(2, 2, 3)
plt.plot(epochs, stds_history[:, 0], 'b-', label='Std 1')
plt.plot(epochs, stds_history[:, 1], 'r-', label='Std 2')
plt.axhline(y=true_stds[0].item(), color='b', linestyle='--', alpha=0.5, label='True Std 1')
plt.axhline(y=true_stds[1].item(), color='r', linestyle='--', alpha=0.5, label='True Std 2')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Standard Deviations Convergence')
plt.legend()
plt.grid(alpha=0.3)

# Plot weights convergence
plt.subplot(2, 2, 4)
plt.plot(epochs, weights_history[:, 0], 'b-', label='Weight 1')
plt.plot(epochs, weights_history[:, 1], 'r-', label='Weight 2')
plt.axhline(y=true_weights[0].item(), color='b', linestyle='--', alpha=0.5, label='True Weight 1')
plt.axhline(y=true_weights[1].item(), color='r', linestyle='--', alpha=0.5, label='True Weight 2')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Weights Convergence')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('all_objectives.png')

# Plot learned distributions and scores
x_eval = torch.linspace(-6, 6, 1000)

# Compute densities
with torch.no_grad():
    log_probs_model = model(x_eval)
    log_probs_true = true_model(x_eval)
    
    model_density = torch.exp(log_probs_model).numpy()
    true_density = torch.exp(log_probs_true).numpy()

# Compute scores (gradients of log density)
model_score = compute_score(model, x_eval).numpy()
true_score = compute_score(true_model, x_eval).numpy()

# Plot densities and scores
plt.figure(figsize=(15, 10))

# Plot densities
plt.subplot(2, 1, 1)
plt.hist(samples.detach().numpy(), bins=50, density=True, alpha=0.4, label='Data')
plt.plot(x_eval.numpy(), model_density, 'r-', lw=2, label='Learned MoG')
plt.plot(x_eval.numpy(), true_density, 'g--', lw=2, label='True MoG')
plt.legend()
plt.title('Probability Densities')
plt.ylabel('Density')
plt.grid(alpha=0.3)

# Plot scores
plt.subplot(2, 1, 2)
plt.plot(x_eval.numpy(), model_score, 'r-', lw=2, label='Learned Score')
plt.plot(x_eval.numpy(), true_score, 'g--', lw=2, label='True Score')
plt.legend()
plt.title('Score Functions (∇_x log p(x))')
plt.xlabel('x')
plt.ylabel('Score')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_vs_true.png')

# Compute J-L relationship
J_minus_L = np.array(J_values) - np.array(L_values)
constant_estimate = np.mean(J_minus_L)
constant_std = np.std(J_minus_L)

# Plot J-L relationship
plt.figure(figsize=(10, 6))
plt.plot(epochs, J_minus_L, 'b-')
plt.axhline(y=constant_estimate, color='r', linestyle='--', 
            label=f'Mean: {constant_estimate:.4f} ± {constant_std:.4f}')
plt.xlabel('Epoch')
plt.ylabel('J(θ) - L(θ)')
plt.title('Difference Between J(θ) and L(θ) (Should be Constant)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('j_minus_l.png')

# Print final results
print("\nFinal values of objective functions:")
print(f"  L(θ) - Fisher Divergence: {L_values[-1]:.6f}")
print(f"  J(θ) - Theoretical Score Matching: {J_values[-1]:.6f}")
print(f"  Ĵ(θ) - Empirical Score Matching: {J_hat_values[-1]:.6f}")

print("\nFinal parameters:")
with torch.no_grad():
    print(f"  True means: {true_means.numpy()}, Learned means: {model.means.numpy()}")
    print(f"  True stds: {true_stds.numpy()}, Learned stds: {torch.exp(model.log_stds).numpy()}")
    print(f"  True weights: {true_weights.numpy()}, Learned weights: {torch.softmax(model.weights, dim=-1).numpy()}")

print(f"\nTheoretical relationship: J(θ) = L(θ) + C")
print(f"Estimated constant C: {constant_estimate:.6f}")
print(f"Std dev of estimates: {constant_std:.6f}")

# Show all plots
plt.show()