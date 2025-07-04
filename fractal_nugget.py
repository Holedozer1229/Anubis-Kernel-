import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import time

# Configuration
CONFIG = {
    "grid_size": 10,  # 10x10x10 grid
    "dx": 1.0,  # Spatial step (arbitrary units)
    "max_iterations": 100,  # Iterations for field evolution
}

# Setup logging
logging.basicConfig(level=logging.INFO, filename='fractal_set_plotter_log.txt')
logger = logging.getLogger(__name__)
logger.info(f"Starting Fractal Set Plotter at {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())}")

# Generate and Evolve Nugget Field
def generate_nugget_field(grid_size=CONFIG["grid_size"], dx=CONFIG["dx"], max_iterations=CONFIG["max_iterations"]):
    # Initialize random Nugget field
    phi_N = np.random.uniform(-0.1, 0.1, (grid_size, grid_size, grid_size))
    # Simple chaotic evolution (logistic map-like)
    for _ in range(max_iterations):
        phi_N_new = phi_N + 0.1 * np.sin(2 * np.pi * phi_N) * (1 - phi_N)
        phi_N = phi_N_new
    return phi_N

# Compute Fractal Dimension
def compute_fractal_dimension(phi_N, dx=CONFIG["dx"]):
    # Compute gradient using central differences
    grad_x = np.gradient(phi_N, dx, axis=0)
    grad_y = np.gradient(phi_N, dx, axis=1)
    grad_z = np.gradient(phi_N, dx, axis=2)
    # Magnitude of gradient squared
    grad_magnitude_squared = grad_x**2 + grad_y**2 + grad_z**2
    # Apply fractal dimension formula
    d_f = 1.7 + 0.4 * np.tanh(grad_magnitude_squared / 0.15)
    return d_f

# Fractal Set Plotter
class FractalSetPlotter:
    def __init__(self):
        self.phi_N = generate_nugget_field()
        self.d_f = compute_fractal_dimension(self.phi_N)

    def plot_fractal_set(self):
        # Create meshgrid for 3D plotting
        x, y, z = np.meshgrid(np.arange(CONFIG["grid_size"]), 
                              np.arange(CONFIG["grid_size"]), 
                              np.arange(CONFIG["grid_size"]), indexing='ij')
        
        # Plot Fractal Set (3D Scatter)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(x.flatten(), y.flatten(), z.flatten(), 
                        c=self.d_f.flatten(), cmap='plasma', alpha=0.6)
        plt.colorbar(sc, label='Fractal Dimension (d_f)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Fractal Set of Tetrahedral Lattice (d_f = 1.7 + 0.4 * tanh(|∇φ_N|^2 / 0.15))')
        plt.savefig('fractal_set.png')
        plt.close()
        
        logger.info(f"Fractal set plot generated at {time.strftime('%Y-%m-%d %H:%M:%S %Z', time.localtime())}")
        logger.info(f"Fractal dimension range: {self.d_f.min():.4f} to {self.d_f.max():.4f}")

if __name__ == "__main__":
    plotter = FractalSetPlotter()
    plotter.plot_fractal_set()