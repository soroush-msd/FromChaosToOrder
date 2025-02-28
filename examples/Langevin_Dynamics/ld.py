import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. we first define mixture parameters
# we use 2D Gaussians for visualization
weights = np.array([1/3, 1/3, 1/3])
means = [
    np.array([-3.0, 0.0]),
    np.array([3.0, 0.0]),
    np.array([0.0, 2.0])
]
covs = [
    0.5 * np.eye(2),
    0.5 * np.eye(2),
    0.5 * np.eye(2)
]

def gaussian_pdf(x, mean, cov):
    """
    Compute the value of a 2D Gaussian pdf N(x | mean, cov).
    x: shape (N, 2) or (2,)
    mean: shape (2,)
    cov: shape (2,2)
    """
    # make sure x is shape (N, 2)
    x = np.atleast_2d(x)
    dim = x.shape[1]
    cov_inv = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    norm_const = 1.0 / (np.sqrt((2*np.pi)**dim * det_cov))
    
    # (x - mean)
    diff = x - mean
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
    return norm_const * np.exp(exponent)

def mixture_pdf(x, weights, means, covs):
    """
    Mixture pdf = sum_i weights[i] * N(x|means[i], covs[i]).
    x can be shape (N,2) or (2,).
    """
    x = np.atleast_2d(x)
    pdf_vals = np.zeros(x.shape[0])
    for w, m, c in zip(weights, means, covs):
        pdf_vals += w * gaussian_pdf(x, m, c)
    return pdf_vals

def mixture_logpdf(x, weights, means, covs):
    """
    log of the mixture pdf = log( sum_i w_i N(...) ).
    For numerical stability, we can compute each Gaussian, sum, then take log.
    """
    return np.log(mixture_pdf(x, weights, means, covs) + 1e-15)  # add small epsilon

def mixture_score(x, weights, means, covs):
    """
    Gradient of log p(x). This is the sum of the weighted gradients of each Gaussian.
    x shape: (N,2) or (2,).
    Returns an array of the same shape as x.
    """
    x = np.atleast_2d(x)
    pdf_val = mixture_pdf(x, weights, means, covs).reshape(-1, 1)  # shape (N,1)
    
    # weighted sum of each component's pdf
    grads = np.zeros_like(x)
    for w, m, c in zip(weights, means, covs):
        c_inv = np.linalg.inv(c)
        # grad log N(x|m, c) = -c_inv (x - m)
        diff = (x - m)
        # pdf of that component
        comp_pdf = gaussian_pdf(x, m, c).reshape(-1, 1)
        # weighted gradient for that component
        grads_comp = (-comp_pdf * (diff @ c_inv.T))
        # accumulate
        grads += w * grads_comp
    
    # now divide by total mixture pdf to get gradient of log p
    # because: grad log p(x) = 1/p(x) * sum_i w_i * pdf_i * grad log(N_i)
    # we already multiplied each grad by w_i*pdf_i, so just divide by p(x).
    grads = grads / (pdf_val + 1e-15)
    return grads

# 2. now we set up the grid for visualization
# grid for contour and vector field
grid_size = 30  # for the vector field (fewer points = clearer arrows)
x_lin_vec = np.linspace(-6, 6, grid_size)
y_lin_vec = np.linspace(-6, 6, grid_size)
xx_vec, yy_vec = np.meshgrid(x_lin_vec, y_lin_vec)
xy_grid_vec = np.column_stack([xx_vec.ravel(), yy_vec.ravel()])

# precompute vector field on grid
grad_field = mixture_score(xy_grid_vec, weights, means, covs)
u = grad_field[:, 0].reshape(xx_vec.shape)
v = grad_field[:, 1].reshape(xx_vec.shape)

# calculate vector field magnitude for normalization
magnitude = np.sqrt(u**2 + v**2)
max_mag = np.max(magnitude)
u_norm = u / (max_mag + 1e-10)
v_norm = v / (max_mag + 1e-10)

# finer grid for the contour plot
grid_size_contour = 100
x_lin = np.linspace(-6, 6, grid_size_contour)
y_lin = np.linspace(-6, 6, grid_size_contour)
xx, yy = np.meshgrid(x_lin, y_lin)
xy_grid = np.column_stack([xx.ravel(), yy.ravel()])
z = mixture_pdf(xy_grid, weights, means, covs).reshape(xx.shape)

# 3. then it comes the Langevin sampling parameters
num_particles = 500    # how many particles we track
num_steps = 25        # increased for more convergence time
epsilon = 0.05         # step size for the gradient update

# initialize particles in a VERY spread out distribution
# covering the entire visible area from -6 to 6
np.random.seed(42)  # different seed for variety
# use uniform distribution to cover the entire space evenly
particles = np.random.uniform(-5.5, 5.5, size=(num_particles, 2))

# to store trajectories for animation
trajectories = [particles.copy()]

# 4. Finally, we run Langevin Dynamics
for step in range(num_steps):
    # compute gradient of log p at current positions
    grad_logp = mixture_score(particles, weights, means, covs)
    
    # langevin update:
    # x_{k+1} = x_k + epsilon * grad log p(x_k) + sqrt(2*epsilon) * noise
    noise = np.random.randn(num_particles, 2)
    particles = particles + epsilon * grad_logp + np.sqrt(2 * epsilon) * noise
    
    trajectories.append(particles.copy())

# 5. create animation
# set animation parameters for better compatibility
plt.rcParams['animation.html'] = 'jshtml'  # for notebook compatibility

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-6, 6)

# plot contour with the viridis colormap
contour = ax.contourf(xx, yy, z, levels=30, cmap='viridis')

# plot vector field with arrows
arrow_stride = 2  # only show arrows at every nth grid point for clarity
quiver = ax.quiver(xx_vec[::arrow_stride, ::arrow_stride], 
                  yy_vec[::arrow_stride, ::arrow_stride], 
                  u_norm[::arrow_stride, ::arrow_stride], 
                  v_norm[::arrow_stride, ::arrow_stride],
                  color='black', scale=25, width=0.003, alpha=0.7)

# plot initial particles
particles_scatter = ax.scatter([], [], c='white', edgecolors='black', s=20)

# add a frame counter for tracking progress
frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                    verticalalignment='top', fontsize=10)

# animation functions
def init():
    particles_scatter.set_offsets(np.empty((0, 2)))
    frame_text.set_text('')
    return particles_scatter, quiver, frame_text

def update(frame):
    particles_scatter.set_offsets(trajectories[frame])
    frame_text.set_text(f'Step: {frame}/{num_steps}')
    return particles_scatter, quiver, frame_text

# create animation
ani = animation.FuncAnimation(
    fig, update, frames=len(trajectories),
    init_func=init, blit=False, interval=200  # Faster animation
)

# set title
plt.title("Langevin Dynamics on 3-Gaussian Mixture")
plt.tight_layout()

# save the animation as a GIF
# ani.save('langevin_mixture.gif', writer='pillow', fps=1, dpi=100)

# display (this will show a static plot in most environments)
plt.show()

# keep a reference to prevent garbage collection
_ani_ref = ani