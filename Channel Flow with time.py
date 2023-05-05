import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

Aspect = 10  # Aspect ration between y and x direction
Ny = 15  # points in y direction
Nx = (Ny - 1) * Aspect + 1  # points in x direction
mu_mol = 0.01  # kinematic viscosity
dt = 1e-4  # time step size
N = 10000  # number times steps
Npp = 10  # Pressure Poisson iterations
Plot_Every = 500
dx = 1.0 / (Ny - 1)
H = 1.0  # channel hight
L = H * Aspect  # channel length
U_inlet = 1.0

x_range = np.linspace(0.0, L, Nx)
y_range = np.linspace(0.0, H, Ny)

coord_x, coord_y = np.meshgrid(x_range, y_range)

# Initial Conditions
u_prev = np.ones((Ny + 1, Nx)) * U_inlet
u_prev[0, :] = - u_prev[1, :]
u_prev[-1, :] = - u_prev[-2, :]

v_prev = np.zeros((Ny, Nx + 1))
v_prev[0, :] = - v_prev[1, :]
v_prev[-1, :] = - v_prev[-2, :]

P_prev = np.zeros((Ny + 1, Nx + 1))

# Pre-Allocate arrays
u_star = np.zeros_like(u_prev)
u_next = np.zeros_like(u_prev)
v_star = np.zeros_like(v_prev)
v_next = np.zeros_like(v_prev)

y = np.linspace(0, H, Ny - 1)
f_pos = 0.4 * y
f_neg = 0.4 * H - 0.4 * y
f_const = 0.1 * H * np.ones(len(y))
f_l = (np.minimum(np.minimum(f_pos, f_neg), f_const)) ** 2
# l = np.zeros((Ny,Nx))
# l[1:-1,1:-1] = f_l[:, np.newaxis]
# l = l**2


for iter in tqdm(range(N)):
    # u velocity
    diff_x = (mu_mol) * (u_prev[1:-1, 2:] + u_prev[1:-1, :-2] + u_prev[2:, 1:-1] + u_prev[:-2, 1:-1] - 4 * u_prev[1:-1,
                                                                                                           1:-1]) / dx ** 2
    conv_x = (u_prev[1:-1, 2:] ** 2 - u_prev[1:-1, :-2] ** 2) / (2 * dx) + (
                v_prev[1:, 1:-2] + v_prev[1:, 2:-1] + v_prev[:-1, 1:-2] + v_prev[:-1, 2:-1]) / 4 * (
                         u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / (2 * dx)
    p_grad_x = (P_prev[1:-1, 2:-1] - P_prev[1:-1, 1:-2]) / dx

    u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x)

    # BC
    u_star[1:-1, 0] = U_inlet
    u_star[1:-1, -1] = u_star[1:-1, -2]
    u_star[0, :] = - u_star[1, :]
    u_star[-1, :] = - u_star[-2, :]

    # v velocity
    diff_v = (mu_mol) * (v_prev[1:-1, 2:] + v_prev[1:-1, :-2] + v_prev[2:, 1:-1] + v_prev[:-2, 1:-1] - 4 * v_prev[1:-1,
                                                                                                           1:-1]) / dx ** 2
    conv_v = (v_prev[2:, 1:-1] ** 2 - v_prev[:-2, 1:-1] ** 2) / (2 * dx) + (
                u_prev[2:-1, 1:] + u_prev[2:-1, :-1] + u_prev[1:-2, 1:] + u_prev[1:-2, :-1]) / 4 * (
                         v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / (2 * dx)
    p_grad_v = (P_prev[2:-1, 1:-1] - P_prev[1:-2, 1:-1]) / dx

    v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v)

    # BC
    v_star[1:-1, 0] = - v_star[1:-1, 1]
    v_star[1:-1, -1] = v_star[1:-1, -2]
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0

    Pp_rhs = (u_star[1:-1, 1:] - u_star[1:-1, :-1] + v_star[1:, 1:-1] - v_star[:-1, 1:-1]) / dx / dt

    # Pressure correction
    P_correction_prev = np.zeros_like(P_prev)
    for _ in range(Npp):
        P_correction_next = np.zeros_like(P_correction_prev)
        P_correction_next[1:-1, 1:-1] = (P_correction_prev[1:-1, 2:] + P_correction_prev[1:-1, :-2] + P_correction_prev[
                                                                                                      2:,
                                                                                                      1:-1] + P_correction_prev[
                                                                                                              -2,
                                                                                                              1:-1] - dx ** 2 * Pp_rhs) / 4

        # BC, use Neumann every expect for outlet where we have Dirichlet
        P_correction_next[1:-1, 0] = P_correction_next[1:-1, 1]
        P_correction_next[1:-1, -1] = P_correction_next[1:-1, -2]
        P_correction_next[0, :] = -P_correction_next[1, :]
        P_correction_next[-1, :] = P_correction_next[-2, :]

        # Advance
        P_correction_prev = P_correction_next

    # Update Pressure
    P_next = P_prev + P_correction_next

    # Incompresibility
    P_correction_grad_x = (P_correction_next[1:-1, 2:-1] - P_correction_next[1:-1, 1:-2]) / dx
    P_correction_grad_y = (P_correction_next[2:-1, 1:-1] - P_correction_next[1:-2, 1:-1]) / dx

    u_next[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * P_correction_grad_x
    v_next[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * P_correction_grad_y

    # BC again
    u_next[1:-1, 0] = U_inlet
    Inflow_flux = np.sum(u_next[1:-1, 0])
    Outflow_flux = np.sum(u_next[1:-1, -2])
    u_next[1:-1, -1] = u_next[1:-1, -2] * Inflow_flux / Outflow_flux
    u_next[0, :] = - u_next[1, :]
    u_next[-1, :] = - u_next[-2, :]

    v_next[1:-1, 0] = - v_next[1:-1, 1]
    v_next[1:-1, -1] = v_next[1:-1, -2]
    v_next[0, :] = 0.0
    v_next[-1, :] = 0.0

    # Advance
    u_prev = u_next
    v_prev = v_next
    P_prev = P_next

    # Visualize simulation
    if iter % Plot_Every == 0:
        plt.figure(dpi=200)
        u_center = (u_next[1:, :] + u_next[:-1, :]) / 2
        v_center = (v_next[:, 1:] + v_next[:, :-1]) / 2
        plt.contourf(coord_x, coord_y, u_center, levels=10)

        plt.colorbar()

        plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

        plt.plot(5 * dx + u_center[:, 5], coord_y[:, 5], linewidth=3)
        plt.plot(20 * dx + u_center[:, 5], coord_y[:, 20], linewidth=3)
        plt.plot(80 * dx + u_center[:, 5], coord_y[:, 80], linewidth=3)
        plt.draw()
        plt.pause(0.05)
        plt.clf()

plt.show()