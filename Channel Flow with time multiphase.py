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
start_turb = int(N*0.3)  # start timestep of multiphase part
Npp = 10  # Pressure Poisson iterations
totalplots = 200
Plot_Every = int(N / totalplots)
dx = 1.0 / (Ny - 1)
H = 1.0  # channel height
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
a_2 = 0.01  # alpha 2 is set to be 0.01 for now
T_p = M_p/(3*np.pi*mu_mol*D_p)  # Particle relaxation == 8.9e-5
T_p *= 100  # this fix works if nu =<1e-6 and dt =<1e-4
a_1 = 1  # safe to assume
angle = 0  # in streamwise direction

# Gravity
gravitational_particles_x = gravitational_force_particles(a_2, rho_1, rho_p, angle)
gravitational_fluid_x = gravitational_force_fluid(a_1, rho_1, angle)

gravitational_particles_y = gravitational_force_particles(a_2, rho_1, rho_p, angle+np.pi/2)
gravitational_fluid_y = gravitational_force_fluid(a_1, rho_1, angle+np.pi/2)
# print(T_p)

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

P_prev = np.zeros((Ny + 1, Nx + 1))

# Pre-Allocate arrays
u_star = np.zeros_like(u_prev)
u_next = np.zeros_like(u_prev)
v_star = np.zeros_like(v_prev)
v_next = np.zeros_like(v_prev)
'''multiphase part'''
# u_star_2 = np.zeros_like(u_prev)
# u_next_2 = np.zeros_like(u_prev)
# v_star_2 = np.zeros_like(v_prev)
# v_next_2 = np.zeros_like(v_prev)

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


for iter in tqdm(range(N)):
    # u velocity
    diff_x = a_1 * ((nu_mol) * (u_prev[1:-1, 2:] + u_prev[1:-1, :-2] + u_prev[2:, 1:-1] + u_prev[:-2, 1:-1] - 4 * u_prev[1:-1,
                                                                                                           1:-1]) / dx ** 2)
    conv_x = a_1 * ((u_prev[1:-1, 2:] ** 2 - u_prev[1:-1, :-2] ** 2) / (2 * dx) + (
                v_prev[1:, 1:-2] + v_prev[1:, 2:-1] + v_prev[:-1, 1:-2] + v_prev[:-1, 2:-1]) / 4 * (
                         u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / (2 * dx))
    p_grad_x = a_1 * ((P_prev[1:-1, 2:-1] - P_prev[1:-1, 1:-2]) / dx)

    '''multiphase part'''
    if iter > start_turb:
        # U1mean_x = np.mean(u_prev[1:-1, 1:-1])
        U1mean_x = u_prev[1:-1, 1:-1]

        # print(f"mean x = {U1mean_x}")
        # U2mean_x = np.mean(u_prev_2[1:-1, 1:-1])
        U2mean_x = u_prev_2[1:-1, 1:-1]
        # print(f"mean x dispersed = {U2mean_x}")
        # U2_U1_avg = np.mean(u_prev_2[1:-1, 1:-1] - u_prev[1:-1, 1:-1])
        interfacial_stress_x = get_F_i(nu_mol, D_p, rho_p, a_2, U2mean_x, U1mean_x)
        # interfacial_stress_x = get_F_i_new(nu_mol, D_p, rho_p, a_2, U2_U1_avg)

        # print(f"interfacial stress in x {interfacial_stress_x}")
        u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x + interfacial_stress_x - gravitational_fluid_x)
    else:
        u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x - gravitational_fluid_x)


    # BC
    u_star[1:-1, 0] = U_inlet
    u_star[1:-1, -1] = u_star[1:-1, -2]
    u_star[0, :] = - u_star[1, :]
    u_star[-1, :] = - u_star[-2, :]

    '''multiphase part BC'''
    # if iter > start_turb:
    #     u_star_2[1:-1, 0] = U_inlet
    #     u_star_2[1:-1, -1] = u_star_2[1:-1, -2]
    #     u_star_2[0, :] = - u_star_2[1, :]
    #     u_star_2[-1, :] = - u_star_2[-2, :]

    # v velocity
    diff_v = a_1 * ((nu_mol) * (v_prev[1:-1, 2:] + v_prev[1:-1, :-2] + v_prev[2:, 1:-1] + v_prev[:-2, 1:-1] - 4 * v_prev[1:-1, 1:-1]) / dx ** 2)
    conv_v = a_1 * ((v_prev[2:, 1:-1] ** 2 - v_prev[:-2, 1:-1] ** 2) / (2 * dx) + (u_prev[2:-1, 1:] + u_prev[2:-1, :-1] + u_prev[1:-2, 1:] + u_prev[1:-2, :-1]) / 4 * (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / (2 * dx))
    p_grad_v = a_1 * ((P_prev[2:-1, 1:-1] - P_prev[1:-2, 1:-1]) / dx)

    '''multiphase part'''
    if iter > start_turb:
        # U1mean_y = np.mean(v_prev[1:-1, 1:-1])
        # U2mean_y = np.mean(v_prev_2[1:-1, 1:-1])

        U1mean_y = v_prev[1:-1, 1:-1]
        U2mean_y = v_prev_2[1:-1, 1:-1]
        # print(f"mean y = {U1mean_y}")
        interfacial_stress_y = get_F_i(nu_mol, D_p, rho_p, a_2, U2mean_y, U1mean_y)
        # print(f"interfacial stress in y {interfacial_stress_y}")

        v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v + interfacial_stress_y - gravitational_fluid_y)
    else:
        v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v - gravitational_fluid_y)

    # BC
    v_star[1:-1, 0] = - v_star[1:-1, 1]
    v_star[1:-1, -1] = v_star[1:-1, -2]
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0

    '''multiphase part BC'''
    # if iter > start_turb:
    #     v_star_2[1:-1, 0] = - v_star_2[1:-1, 1]
    #     v_star_2[1:-1, -1] = v_star_2[1:-1, -2]
    #     v_star_2[0, :] = 0.0
    #     v_star_2[-1, :] = 0.0

    '''multiphase part'''
    if iter > start_turb:
        T_t = calc_T_t_new(u_star, l_x, dx)
        U_2i_U_2j = calc_U_2i_U_2j_new(T_t, T_p, u_star, l_x, dx)
        kinetic_stresses = a_2 * rho_p * U_2i_U_2j
        kinetic_stresses[np.isnan(kinetic_stresses)] = 0
        # print(f"kinetic stress x: {kinetic_stresses}")
        u_prev_2[1:-1, 1:-1] = u_prev_2[1:-1, 1:-1] + dt * (kinetic_stresses[1:-1, 1:-1] - interfacial_stress_x - gravitational_particles_x)
        # u_prev_2[1:-1, 1:-1] = u_prev_2[1:-1, 1:-1] + dt * (interfacial_stress_x)

        '''BC'''
        u_prev_2[1:-1, 0] = U_inlet
        Inflow_flux = np.sum(u_prev_2[1:-1, 0])
        Outflow_flux = np.sum(u_prev_2[1:-1, -2])
        u_prev_2[1:-1, -1] = u_prev_2[1:-1, -2] * Inflow_flux / Outflow_flux
        # u_prev_2[0, :] = - u_prev_2[1, :]
        # u_prev_2[-1, :] = - u_prev_2[-2, :]


        T_t = calc_T_t_new(v_star, l_y, dx)
        U_2i_U_2j = calc_U_2i_U_2j_new(T_t, T_p, v_star, l_y, dx)
        kinetic_stresses = a_2 * rho_p * U_2i_U_2j
        # print(f"kinetic stress y: {kinetic_stresses}")
        kinetic_stresses[np.isnan(kinetic_stresses)] = 0
        v_prev_2[1:-1, 1:-1] = v_prev_2[1:-1, 1:-1] + dt * (kinetic_stresses[1:-1, 1:-1] - interfacial_stress_y - gravitational_particles_y)
        # print(f"u_2 velocity mean: {np.mean(u_star_2[1:-1, 1:-1] )}")

        '''BC'''
        v_prev_2[1:-1, 0] = - v_prev_2[1:-1, 1]
        v_prev_2[1:-1, -1] = v_prev_2[1:-1, -2]
        v_prev_2[0, :] = - v_prev_2[1, :]
        v_prev_2[-1, :] = - v_prev_2[-2, :]

        u_prev_2[0, :] =  u_prev_2[1, :]
        u_prev_2[-1, :] =  u_prev_2[-2, :]

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

    # Advance
    u_prev = u_next
    v_prev = v_next
    P_prev = P_next

    # '''multiphase part'''
    # if iter > start_turb:
    #     u_prev_2 = u_next_2
    #     v_prev_2 = v_next_2

    # Visualize simulation
    if iter % Plot_Every == 0:
        plt.figure(dpi=50)
        u_center = (u_next[1:, :] + u_next[:-1, :]) / 2
        v_center = (v_next[:, 1:] + v_next[:, :-1]) / 2
        plt.contourf(coord_x, coord_y, u_center, levels=10)

        plt.colorbar()

        plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

        # plt.plot(5 * dx + u_center[:, 5], coord_y[:, 5], linewidth=3)
        # plt.plot(20 * dx + u_center[:, 5], coord_y[:, 20], linewidth=3)
        # plt.plot(80 * dx + u_center[:, 5], coord_y[:, 80], linewidth=3)
        if iter > start_turb:
            plt.title(f"time: {iter*dt:.2f} s, Multiphase: on")
        else:
            plt.title(f"time: {iter * dt:.2f} s, Multiphase: off")
        # plt.draw()
        # plt.pause(0.05)
        plt.savefig(f'save_for_gif/img_{iter}.png',
                    transparent=False,
                    facecolor='white'
                    )
        # plt.clf()
        plt.close

    if iter > start_turb:
        if iter % Plot_Every == 0:
            plt.figure(dpi=50)
            u_center = (u_prev_2[1:, :] + u_prev_2[:-1, :]) / 2
            v_center = (v_prev_2[:, 1:] + v_prev_2[:, :-1]) / 2
            plt.contourf(coord_x, coord_y, u_center, levels=10)

            plt.colorbar()

            plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

            # plt.plot(5 * dx + u_center[:, 5], coord_y[:, 5], linewidth=3)
            # plt.plot(20 * dx + u_center[:, 5], coord_y[:, 20], linewidth=3)
            # plt.plot(80 * dx + u_center[:, 5], coord_y[:, 80], linewidth=3)
            plt.title(f"time: {iter*dt:.2f} s")
            # plt.draw()
            # plt.pause(0.05)
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

        if iter > start_turb:
            image = imageio.v2.imread(f'save_for_gif/img_mult_{iter}.png')
            frames_2.append(image)

imageio.mimsave(f'gifs/multiphase_continuous.gif',
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
plt.plot(height, u_center[:,-3], label='Continuous phase')
plt.title("Steamwise Velocity profile at the end")
if start_turb < N:
    imageio.mimsave(f'gifs/multiphase_dispersed.gif',
                    frames_2,
                    duration=0.03
                    )
    u_center_2 = (u_prev_2[1:, :] + u_prev_2[:-1, :]) / 2
    np.save(directory + "\\data\\u_center_2.npy", u_center_2)
    plt.plot(height, u_center_2[:,-3], label='Dispersed phase')

plt.ylabel("Velocity (m/s)")
plt.xlabel("Height (m)")
plt.legend()
plt.grid()
plt.savefig(f"plots/velocity.png")

print("gif saved")
# plt.show()