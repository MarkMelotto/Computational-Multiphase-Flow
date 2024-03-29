import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiphase_functions import *
import imageio
import os

Aspect = 10  # Aspect ratio between y and x direction
Ny = 30  # points in y direction
Nx = (Ny - 1) * Aspect + 1  # points in x direction
nu_mol = 1e-3  # kinematic viscosity
mu_mol = nu_mol * 1e3
dt = 1e-4  # time step size
N = int(9e4)  # number times steps
start_multi_phase = int(N * 0.3)  # start timestep of multiphase part
Npp = 10  # Pressure Poisson iterations
totalplots = 200
Plot_Every = int(N / totalplots)
dx = 0.1 / (Ny - 1)
H = 0.1  # channel height
L = H * Aspect  # channel length
U_inlet = 1
rho_1 = 1000
g = 9.81

x_range = np.linspace(0.0, L, Nx)
y_range = np.linspace(0.0, H, Ny)

coord_x, coord_y = np.meshgrid(x_range, y_range)

'''multiphase part'''
D_p = 1e-3  # diameter particles 1mm
rho_p = 1602  # density sand = 1602 kg/m3
V_p = (D_p/2)**3 * np.pi * 4/3  # volume particle
M_p = V_p * rho_p  # mass of the particle
initial_concentration = 0.01  # initial alpha 2 is set to be 0.01
T_p = M_p/(3*np.pi*mu_mol*D_p)  # Particle relaxation == 8.9e-5
T_p *= 100  # this fix works if nu =<1e-6 and dt =<1e-4
a_1 = 1  # safe to assume
angle = 0  # in streamwise direction

# Initial Conditions
u_prev = np.ones((Ny + 1, Nx)) * U_inlet
u_prev[0, :] = - u_prev[1, :]
u_prev[-1, :] = - u_prev[-2, :]
'''multiphase part'''
u_prev_2 = np.copy(u_prev)

v_prev = np.zeros((Ny, Nx + 1))
v_prev[0, :] = - v_prev[1, :]
v_prev[-1, :] = - v_prev[-2, :]
'''multiphase part'''
v_prev_2 = np.copy(v_prev)

a_2_x = initial_concentration * np.ones((Ny + 1, Nx))
a_2_x[0, :] = 0
a_2_x[-1, :] = 0


a_2_y = initial_concentration * np.ones((Ny, Nx + 1))
a_2_y[0, :] = 0.0
a_2_y[-1, :] = 0.0

P_prev = np.zeros((Ny + 1, Nx + 1))

# Pre-Allocate arrays
u_star = np.zeros_like(u_prev)
u_next = np.zeros_like(u_prev)
v_star = np.zeros_like(v_prev)
v_next = np.zeros_like(v_prev)

y = np.linspace(0, H, Ny)
f_pos = 0.4 * y
f_neg = 0.4 * H - 0.4 * y
f_const = 0.1 * H * np.ones(len(y))
f_l = (np.minimum(np.minimum(f_pos, f_neg), f_const)) ** 2
l_y = np.zeros((Ny,Nx+1))
l_y[:, :] = f_l[:, np.newaxis]

y_x = np.linspace(0, H, Ny+1)
f_pos_x = 0.4 * y_x
f_neg_x = 0.4 * H - 0.4 * y_x
f_const_x = 0.1 * H * np.ones(len(y_x))
f_l_x = (np.minimum(np.minimum(f_pos_x, f_neg_x), f_const_x)) ** 2
l_x = np.zeros((Ny+1,Nx))
l_x[:, :] = f_l_x[:, np.newaxis]

# Gravity
gravitational_particles_x = gravitational_force_particles(a_2_x, rho_1, rho_p, angle)
gravitational_fluid_x = gravitational_force_fluid(a_2_x, rho_1, rho_p, angle)

gravitational_particles_y = gravitational_force_particles(a_2_y, rho_1, rho_p, angle+np.pi/2)
gravitational_fluid_y = gravitational_force_fluid(a_2_y, rho_1, rho_p, angle+np.pi/2)


for iter in tqdm(range(N)):
    # u velocity
    diff_x = a_1 * ((nu_mol) * (u_prev[1:-1, 2:] + u_prev[1:-1, :-2] + u_prev[2:, 1:-1] + u_prev[:-2, 1:-1] - 4 * u_prev[1:-1,
                                                                                                           1:-1]) / dx ** 2)
    conv_x = a_1 * ((u_prev[1:-1, 2:] ** 2 - u_prev[1:-1, :-2] ** 2) / (2 * dx) + (
                v_prev[1:, 1:-2] + v_prev[1:, 2:-1] + v_prev[:-1, 1:-2] + v_prev[:-1, 2:-1]) / 4 * (
                         u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / (2 * dx))
    p_grad_x = a_1 * ((P_prev[1:-1, 2:-1] - P_prev[1:-1, 1:-2]) / dx)

    '''multiphase part'''
    if iter > start_multi_phase:

        U1mean_x = u_prev[1:-1, 1:-1]
        U2mean_x = u_prev_2[1:-1, 1:-1]
        interfacial_stress_x = get_F_i_fast_concentration(nu_mol, D_p, rho_p, a_2_x, U2mean_x, U1mean_x)
        u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x + interfacial_stress_x - gravitational_fluid_x[1:-1, 1:-1])
    else:
        u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x - gravitational_fluid_x[1:-1, 1:-1])


    # BC
    u_star[1:-1, 0] = U_inlet
    u_star[1:-1, -1] = u_star[1:-1, -2]
    u_star[0, :] = - u_star[1, :]
    u_star[-1, :] = - u_star[-2, :]

    # v velocity
    diff_v = a_1 * ((nu_mol) * (v_prev[1:-1, 2:] + v_prev[1:-1, :-2] + v_prev[2:, 1:-1] + v_prev[:-2, 1:-1] - 4 * v_prev[1:-1, 1:-1]) / dx ** 2)
    conv_v = a_1 * ((v_prev[2:, 1:-1] ** 2 - v_prev[:-2, 1:-1] ** 2) / (2 * dx) + (u_prev[2:-1, 1:] + u_prev[2:-1, :-1] + u_prev[1:-2, 1:] + u_prev[1:-2, :-1]) / 4 * (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / (2 * dx))
    p_grad_v = a_1 * ((P_prev[2:-1, 1:-1] - P_prev[1:-2, 1:-1]) / dx)

    '''multiphase part'''
    if iter > start_multi_phase:


        U1mean_y = v_prev[1:-1, 1:-1]
        U2mean_y = v_prev_2[1:-1, 1:-1]
        interfacial_stress_y = get_F_i_fast_concentration(nu_mol, D_p, rho_p, a_2_y, U2mean_y, U1mean_y)
        v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v + interfacial_stress_y - gravitational_fluid_y[1:-1, 1:-1])
    else:
        v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v - gravitational_fluid_y[1:-1, 1:-1])

    # BC
    v_star[1:-1, 0] = - v_star[1:-1, 1]
    v_star[1:-1, -1] = v_star[1:-1, -2]
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0

    '''multiphase part'''
    if iter > start_multi_phase:
        T_t = calc_T_t_new(u_star, l_x, dx)
        U_2i_U_2j = calc_U_2i_U_2j_new(T_t, T_p, u_star, l_x, dx)
        kinetic_stresses_x = a_2_x * rho_p * U_2i_U_2j
        kinetic_stresses_x[np.isnan(kinetic_stresses_x)] = 0
        kin_stress_x = (kinetic_stresses_x[1:-1, 2:] - kinetic_stresses_x[1:-1, 1:-1]) / dx

        u_prev_2[1:-1, 1:-1] = u_prev_2[1:-1, 1:-1] + dt * (-kin_stress_x - interfacial_stress_x - gravitational_particles_x[1:-1, 1:-1])

        '''BC'''
        u_prev_2[1:-1, 0] = U_inlet
        Inflow_flux = np.sum(u_prev_2[1:-1, 0])
        Outflow_flux = np.sum(u_prev_2[1:-1, -2])
        u_prev_2[1:-1, -1] = u_prev_2[1:-1, -2] * Inflow_flux / Outflow_flux

        T_t = calc_T_t_new(v_star, l_y, dx)
        U_2i_U_2j = calc_U_2i_U_2j_new(T_t, T_p, v_star, l_y, dx)
        kinetic_stresses_y = a_2_y * rho_p * U_2i_U_2j
        kinetic_stresses_y[np.isnan(kinetic_stresses_y)] = 0
        kin_stress_y = (kinetic_stresses_y[2:, 1:-1] - kinetic_stresses_y[1:-1, 1:-1]) / dx

        v_prev_2[1:-1, 1:-1] = v_prev_2[1:-1, 1:-1] + dt * (-kin_stress_y - interfacial_stress_y - gravitational_particles_y[1:-1, 1:-1])

        '''BC'''
        v_prev_2[1:-1, 0] = - v_prev_2[1:-1, 1]
        v_prev_2[1:-1, -1] = v_prev_2[1:-1, -2]
        v_prev_2[0, :] = - v_prev_2[1, :]
        v_prev_2[-1, :] = - v_prev_2[-2, :]

        u_prev_2[0, :] = u_prev_2[1, :]
        u_prev_2[-1, :] = u_prev_2[-2, :]

    Pp_rhs = (u_star[1:-1, 1:] - u_star[1:-1, :-1] + v_star[1:, 1:-1] - v_star[:-1, 1:-1]) / dx / dt

    # Pressure correction
    P_correction_prev = np.zeros_like(P_prev)
    for _ in range(Npp):
        P_correction_next = np.zeros_like(P_correction_prev)
        P_correction_next[1:-1, 1:-1] = (P_correction_prev[1:-1, 2:] + P_correction_prev[1:-1, :-2] + P_correction_prev[2:,1:-1] + P_correction_prev[-2,1:-1] - dx ** 2 * Pp_rhs) / 4

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

    a_2_x[1:-1, 0] = initial_concentration
    Inflow_flux_a2x = np.sum(a_2_x[1:-1, 0])
    Outflow_flux_a2x = np.sum(a_2_x[1:-1, -2])
    a_2_x[1:-1, -1] = a_2_x[1:-1, -2] * Inflow_flux_a2x / Outflow_flux_a2x
    a_2_x[0, :] = - a_2_x[1, :]
    a_2_x[-1, :] = - a_2_x[-2, :]

    a_2_y[1:-1, 0] = - a_2_y[1:-1, 1]
    a_2_y[1:-1, -1] = a_2_y[1:-1, -2]
    a_2_y[0, :] = 0.0
    a_2_y[-1, :] = 0.0

    '''update the concentration'''
    if iter > start_multi_phase:
        a_2_x = updated_a2(a_2_x, u_prev_2, dt, dx)
        a_2_y = updated_a2(a_2_y, v_prev_2, dt, dx)
        # print(a_2_x)

    # Advance
    u_prev = u_next
    v_prev = v_next
    P_prev = P_next


    # Visualize simulation
    if iter % Plot_Every == 0:
        plt.figure(dpi=50)
        u_center = (u_next[1:, :] + u_next[:-1, :]) / 2
        v_center = (v_next[:, 1:] + v_next[:, :-1]) / 2
        plt.contourf(coord_x, coord_y, u_center, levels=10)

        plt.colorbar(label="Velocity")

        plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

        if iter > start_multi_phase:
            plt.title(f"Continuous Phase, time: {iter*dt:.2f} s, Multiphase: on")
        else:
            plt.title(f"Continuous Phase, time: {iter * dt:.2f} s, Multiphase: off")
        plt.xlabel("Height (m)")
        plt.ylabel("Width (m)")
        plt.savefig(f'save_for_gif/img_{iter}.png',
                    transparent=False,
                    facecolor='white'
                    )
        plt.close

    if iter > start_multi_phase:
        if iter % Plot_Every == 0:
            plt.figure(dpi=50)
            u_center_2 = (u_prev_2[1:, :] + u_prev_2[:-1, :]) / 2
            v_center_2 = (v_prev_2[:, 1:] + v_prev_2[:, :-1]) / 2
            plt.contourf(coord_x, coord_y, u_center_2, levels=10)

            plt.colorbar()

            plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center_2[:, ::6], v_center_2[:, ::6], alpha=0.4)
            plt.title(f"Dispersed Phase, time: {iter*dt:.2f} s")
            plt.xlabel("Height (m)")
            plt.ylabel("Width (m)")
            plt.savefig(f'save_for_gif/img_mult_{iter}.png',
                        transparent=False,
                        facecolor='white'
                        )
            # plt.clf()
            plt.close

print("saving gif")
frames = []
frames_2 = []
for iter in range(N):

    if iter % Plot_Every == 0:
        image = imageio.v2.imread(f'save_for_gif/img_{iter}.png')
        frames.append(image)

        if iter > start_multi_phase:
            image = imageio.v2.imread(f'save_for_gif/img_mult_{iter}.png')
            frames_2.append(image)

imageio.mimsave(f'gifs/multiphase_continuous_concentration.gif',
                frames,
                duration=0.03
                )

height = np.linspace(0.0, H, Ny)
plt.close()
plt.figure()
u_center = (u_next[1:, :] + u_next[:-1, :]) / 2
directory = os.getcwd()
np.save(directory + "\\data\\u_center.npy", u_center)
np.save(directory + "\\data\\height.npy", height)
plt.plot(height, u_center[:,-5], label='Continuous phase')
plt.title("Steamwise Velocity profile at the end")
if start_multi_phase < N:
    imageio.mimsave(f'gifs/multiphase_dispersed_concentration.gif',
                    frames_2,
                    duration=0.03
                    )
    u_center_2 = (u_prev_2[1:, :] + u_prev_2[:-1, :]) / 2
    np.save(directory + "\\data\\u_center_2.npy", u_center_2)
    plt.plot(height, u_center_2[:,-5], label='Dispersed phase')

plt.ylabel("Velocity (m/s)")
plt.xlabel("Width (m)")
plt.legend()
plt.grid()
plt.savefig(f"plots/velocity_concentration.png")

print("gif saved")

plt.close
plt.clf()

a2_center_2x = (a_2_x[1:, :] + a_2_x[:-1, :]) / 2
a2_center_2y = (a_2_y[:, 1:] + a_2_y[:, :-1]) / 2
# plt.plot(height, a_2_x)
plt.contourf(coord_x, coord_y, a2_center_2x, levels=10)
plt.colorbar()
plt.title("a_2")
# plt.grid()
plt.savefig(f"plots/a_2_x_field.png")

plt.close
plt.clf()

plt.figure(dpi=200)
plt.plot(height, a2_center_2x[:,-5])
plt.title("a_2")
plt.grid()
plt.savefig(f"plots/a_2_x.png")