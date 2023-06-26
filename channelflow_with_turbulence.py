import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DYN_VISCOSITY_MOL = 0.01  # Dynamic viscosity
RHO = 1000
VON_KARMAN = 0.41

# Spatial constants and variables
aspect_ratio = 10                   # Aspect ratio between y and x direction
n_y = 40                            # Points in y direction
n_x = (n_y - 1) * aspect_ratio + 1  # Points in x direction
height = 1.0  
length = height * aspect_ratio 
delta_x = 1.0 / (n_x - 1)
delta_y = 1.0 / (n_y - 1)

x_range = np.linspace(0, length, n_x)
y_range = np.linspace(0, height, n_y)
coord_x, coord_y = np.meshgrid(x_range, y_range)
y = np.linspace(0, height, n_y - 1)

# Temporal constants and variables
delta_t = 3e-6                      # time step size
n_t = int(1e4)                      # number times steps
plot_frequency = 500

# X velocity
u_inlet = 100.0
# Initial Conditions
u_prev = np.ones((n_y + 1, n_x)) * u_inlet
u_prev[0, :] = - u_prev[1, :]
u_prev[-1, :] = - u_prev[-2, :]

# Y velocity
# Initial Conditions
v_prev = np.zeros((n_y, n_x + 1))
v_prev[0, :] = - v_prev[1, :]
v_prev[-1, :] = - v_prev[-2, :]

# Pressure initialization
pressure_prev = np.zeros((n_y + 1, n_x + 1))
n_poisson_pressure = 10                         # Pressure Poisson iterations

# Pre-allocate predictor velocity arrays
u_star = np.zeros_like(u_prev)
u_next = np.zeros_like(u_prev)
v_star = np.zeros_like(v_prev)
v_next = np.zeros_like(v_prev)

# Define function that check if y is at <10% or >90% of the width of the pipe, 
# then linear function is used, otherwise a constant value is used.
region_function = np.ones(n_y - 1) 
region_function[:int(.1 * n_y)] = np.array([(n - 1) / n_y * delta_y for n in range(1, int(.1 * height * n_y) + 1)])
region_function[int(.9 * n_y - 1) - 1:] = np.array([(height - n / n_y) * delta_y for n in range(int(.9 * height * n_y), int(height * n_y) + 1)])
region_function[int(.1 * n_y - 1):int(.9 * n_y)] = height * 0.1

for iter in tqdm(range(n_t)):
    # x velocity (u)
    diffusive_u = (DYN_VISCOSITY_MOL +  np.abs(u_prev[1:-1, 2:] - u_prev[1:-1, :-2])/ delta_y * \
                  (VON_KARMAN * region_function[:, np.newaxis])**2) * \
                                      (u_prev[1:-1, 2:] + \
                                       u_prev[1:-1, :-2] + \
                                       u_prev[2:, 1:-1] + \
                                       u_prev[:-2, 1:-1] - \
                                       4 * u_prev[1:-1, 1:-1]) / delta_x ** 2
    convective_u = (u_prev[1:-1, 2:]**2 - u_prev[1:-1, :-2]**2) / 2 / delta_x + \
                (v_prev[1:, 1:-2] + v_prev[1:, 2:-1] + v_prev[:-1, 1:-2] + v_prev[:-1, 2:-1]) / 4 * \
                (u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / 2 / delta_x
   
    pressure_gradient_x = (pressure_prev[1:-1, 2:-1] - pressure_prev[1:-1, 1:-2]) / delta_x
    u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + delta_t * (-pressure_gradient_x + diffusive_u - convective_u)

    # Boundary conditions
    u_star[1:-1, 0] = u_inlet
    u_star[1:-1, -1] = u_star[1:-1, -2]
    u_star[0, :] = - u_star[1, :]
    u_star[-1, :] = - u_star[-2, :]

    # y velocity (v)
    diffusive_v = DYN_VISCOSITY_MOL * (v_prev[1:-1, 2:] + \
                                       v_prev[1:-1, :-2] + \
                                       v_prev[2:, 1:-1] + \
                                       v_prev[:-2, 1:-1] - \
                                       4 * v_prev[1:-1,1:-1]) / delta_x ** 2
    convective_v = (v_prev[2:, 1:-1]**2 - v_prev[:-2, 1:-1]**2) / 2 / delta_x + \
                (u_prev[2:-1, 1:] + u_prev[2:-1, :-1] + u_prev[1:-2, 1:] + u_prev[1:-2, :-1]) / 4  * \
                (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / 2 / delta_x
    
    pressure_gradient_y = (pressure_prev[2:-1, 1:-1] - pressure_prev[1:-2, 1:-1]) / delta_x
    v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + delta_t * (-pressure_gradient_y + diffusive_v - convective_v)

    # Boundary Conditions
    v_star[1:-1, 0] = - v_star[1:-1, 1]
    v_star[1:-1, -1] = v_star[1:-1, -2]
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0

    # Righthandside poisson pressure
    poisson_pressure_rhs = (u_star[1:-1, 1:] - u_star[1:-1, :-1] + v_star[1:, 1:-1] - v_star[:-1, 1:-1]) / delta_x / delta_t

    # Pressure correction
    pressure_correction_prev = np.zeros_like(pressure_prev)
    for _ in range(n_poisson_pressure):
        pressure_correction_next = np.zeros_like(pressure_correction_prev)
        pressure_correction_next[1:-1, 1:-1] = (pressure_correction_prev[1:-1, 2:] + \
                                                pressure_correction_prev[1:-1, :-2] + \
                                                pressure_correction_prev[2:,1:-1] + \
                                                pressure_correction_prev[-2,1:-1] - \
                                                poisson_pressure_rhs * delta_x**2) / 4

        # Boundary Conditions, use Neumann every expect for outlet where we have Dirichlet
        pressure_correction_next[1:-1, 0] = pressure_correction_next[1:-1, 1]
        pressure_correction_next[1:-1, -1] = pressure_correction_next[1:-1, -2]
        pressure_correction_next[0, :] = -pressure_correction_next[1, :]
        pressure_correction_next[-1, :] = pressure_correction_next[-2, :]

        # Advance
        pressure_correction_prev = pressure_correction_next

    # Update Pressure
    pressure_next = pressure_prev + pressure_correction_next

    # Incompresibility
    pressure_correction_gradient_x = (pressure_correction_next[1:-1, 2:-1] - pressure_correction_next[1:-1, 1:-2]) / delta_x
    pressure_correction_gradient_y = (pressure_correction_next[2:-1, 1:-1] - pressure_correction_next[1:-2, 1:-1]) / delta_x

    u_next[1:-1, 1:-1] = u_star[1:-1, 1:-1] - delta_t * pressure_correction_gradient_x
    v_next[1:-1, 1:-1] = v_star[1:-1, 1:-1] - delta_t * pressure_correction_gradient_y

    # BC again
    u_next[1:-1, 0] = u_inlet
    inflow_flux = np.sum(u_next[1:-1, 0])
    outflow_flux = np.sum(u_next[1:-1, -2])
    u_next[1:-1, -1] = u_next[1:-1, -2] * inflow_flux / outflow_flux
    u_next[0, :] = - u_next[1, :]
    u_next[-1, :] = - u_next[-2, :]

    v_next[1:-1, 0] = - v_next[1:-1, 1]
    v_next[1:-1, -1] = v_next[1:-1, -2]
    v_next[0, :] = 0.0
    v_next[-1, :] = 0.0

    # Advance
    u_prev = u_next
    v_prev = v_next
    pressure_prev = pressure_next

    # Visualize simulation
    if iter % plot_frequency == 0:
        u_center = (u_next[1:, :] + u_next[:-1, :]) / 2
        v_center = (v_next[:, 1:] + v_next[:, :-1]) / 2
        
        plt.figure(dpi=200)
        plt.contourf(coord_x, coord_y, u_center, levels=10)
        plt.colorbar()

        plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

        # plt.plot(5 * delta_x + u_center[:, 5], coord_y[:, 5], linewidth=3)
        # plt.plot(20 * delta_x + u_center[:, 5], coord_y[:, 20], linewidth=3)
        # plt.plot(80 * delta_x + u_center[:, 5], coord_y[:, 80], linewidth=3)
        plt.draw()
        plt.pause(0.05)
        plt.clf()

plt.show()
plt.title("U velocity profile at exit of channel.")
plt.plot(coord_y[:, -1], u_center[:, -1])
