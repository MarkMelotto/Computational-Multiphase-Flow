import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiphase_functions import *
import imageio
import os

Aspect = 40  # Aspect ratio between y and x direction
Ny = 30  # points in y direction
Nx = (Ny - 1) * Aspect + 1  # points in x direction
H = 0.1  # channel height
L = H * Aspect  # channel length
dx = L / (Nx - 1)
dy = H / (Ny - 1)
DPI_for_figures = 50

x_range = np.linspace(0.0, L, Nx)
y_range = np.linspace(0.0, H, Ny)
coord_x, coord_y = np.meshgrid(x_range, y_range)

dt = 1e-5  # time step size
N = int(9e5)  # number times steps
start_multi_phase = int(N * 0.3)  # start timestep of multiphase part
Npp = 10  # Pressure Poisson iterations

totalplots = 100
Plot_Every = int(N / totalplots)

U_inlet = 1
jet_velocity = 1.8
start_jet = int(N * 0.4)

g = 9.81
rho_1 = 1000
mu_mol = 1e-3  # dynamic viscosity
nu_mol = mu_mol / rho_1


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
gravitational_fluid_x = gravitational_force_fluid(a_2, rho_1, rho_p, angle)

gravitational_particles_y = gravitational_force_particles(a_2, rho_1, rho_p, angle+np.pi/2)
gravitational_fluid_y = gravitational_force_fluid(a_2, rho_1, rho_p, angle+np.pi/2)
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

VON_KARMAN = 0.41
region_function = np.ones(Ny + 1) 
region_function[:int(.1 * Ny) + 1] = np.array([((n - 1) * H * dy)**2 for n in range(1, int(.1 * Ny) + 2)])
region_function[int(.9 * Ny):] = np.array([((Ny - n) * H * dy)**2 for n in range(int(.9 * Ny), int(Ny) + 1)])
region_function[int(.1 * Ny):int(.9 * Ny) + 1] = (H * 0.1) ** 2
region_function = region_function[:, np.newaxis] * VON_KARMAN**2

for iter in tqdm(range(N)):

    eddy_viscosity = np.abs(u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) * region_function[1:-1, :] / dy
    # u velocity
    diff_x = a_1 * ((nu_mol + eddy_viscosity) * (u_prev[1:-1, 2:] + u_prev[1:-1, :-2] + u_prev[2:, 1:-1] + u_prev[:-2, 1:-1] - 4 * u_prev[1:-1,
                                                                                                           1:-1]) / dy ** 2)
    conv_x = a_1 * ((u_prev[1:-1, 2:] ** 2 - u_prev[1:-1, :-2] ** 2) / (2 * dx) + (
                v_prev[1:, 1:-2] + v_prev[1:, 2:-1] + v_prev[:-1, 1:-2] + v_prev[:-1, 2:-1]) / 4 * (
                         u_prev[2:, 1:-1] - u_prev[:-2, 1:-1]) / (2 * dx))
    p_grad_x = a_1 * ((P_prev[1:-1, 2:-1] - P_prev[1:-1, 1:-2]) / dx)

    '''multiphase part'''
    if iter > start_multi_phase:
        U1mean_x = u_prev[1:-1, 1:-1]
        U2mean_x = u_prev_2[1:-1, 1:-1]
        interfacial_stress_x = get_F_i(nu_mol, D_p, rho_p, a_2, U2mean_x, U1mean_x)
        u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x + interfacial_stress_x - gravitational_fluid_x)
    else:
        u_star[1:-1, 1:-1] = u_prev[1:-1, 1:-1] + dt * (-p_grad_x + diff_x - conv_x - gravitational_fluid_x)


    # BC
    u_star[1:-1, 0] = U_inlet
    u_star[1:-1, -1] = u_star[1:-1, -2]
    u_star[0, :] = - u_star[1, :]
    u_star[-1, :] = - u_star[-2, :]

    if iter > start_jet:
        make_jet(u_star, jet_velocity)

    # v velocity
    diff_v = a_1 * ((nu_mol) * (v_prev[1:-1, 2:] + v_prev[1:-1, :-2] + v_prev[2:, 1:-1] + v_prev[:-2, 1:-1] - 4 * v_prev[1:-1, 1:-1]) / dx ** 2)
    conv_v = a_1 * ((v_prev[2:, 1:-1] ** 2 - v_prev[:-2, 1:-1] ** 2) / (2 * dx) + (u_prev[2:-1, 1:] + u_prev[2:-1, :-1] + u_prev[1:-2, 1:] + u_prev[1:-2, :-1]) / 4 * (v_prev[1:-1, 2:] - v_prev[1:-1, :-2]) / (2 * dx))
    p_grad_v = a_1 * ((P_prev[2:-1, 1:-1] - P_prev[1:-2, 1:-1]) / dx)

    '''multiphase part'''
    if iter > start_multi_phase:


        U1mean_y = v_prev[1:-1, 1:-1]
        U2mean_y = v_prev_2[1:-1, 1:-1]
        interfacial_stress_y = get_F_i(nu_mol, D_p, rho_p, a_2, U2mean_y, U1mean_y)
        v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v + interfacial_stress_y - gravitational_fluid_y)
    else:
        v_star[1:-1, 1:-1] = v_prev[1:-1, 1:-1] + dt * (-p_grad_v + diff_v - conv_v - gravitational_fluid_y)

    # BC
    v_star[1:-1, 0] = - v_star[1:-1, 1]
    v_star[1:-1, -1] = v_star[1:-1, -2]
    v_star[0, :] = 0.0
    v_star[-1, :] = 0.0

    '''multiphase part'''
    if iter > start_multi_phase:
        T_t = calc_T_t_new(u_star, l_x, dx)
        # U_2i_U_2j = calc_U_2i_U_2j_new(T_t, T_p, u_star, l_x, dx)
        # kinetic_stresses = a_2 * rho_p * U_2i_U_2j
        # kinetic_stresses[np.isnan(kinetic_stresses)] = 0
        kinetic_stresses = calculate_kinetic_stress_x(a_2, rho_p, T_t, T_p, u_star, region_function, dx)
        kin_stress_x = (kinetic_stresses[1:-1, 2:] - kinetic_stresses[1:-1, 1:-1]) / dx

        u_prev_2[1:-1, 1:-1] = u_prev_2[1:-1, 1:-1] + dt * (-kin_stress_x - interfacial_stress_x - gravitational_particles_x)

        '''BC'''
        u_prev_2[1:-1, 0] = U_inlet
        Inflow_flux = np.sum(u_prev_2[1:-1, 0])
        Outflow_flux = np.sum(u_prev_2[1:-1, -2])
        u_prev_2[1:-1, -1] = u_prev_2[1:-1, -2] * Inflow_flux / Outflow_flux

        T_t = calc_T_t_new(v_star, l_y, dx)
        # U_2i_U_2j = calc_U_2i_U_2j_new(T_t, T_p, v_star, l_y, dx)
        # kinetic_stresses = a_2 * rho_p * U_2i_U_2j
        # kinetic_stresses[np.isnan(kinetic_stresses)] = 0
        kinetic_stresses = calculate_kinetic_stress_x(a_2, rho_p, T_t, T_p, v_star, region_function, dx)
        kin_stress_y = (kinetic_stresses[2:, 1:-1] - kinetic_stresses[1:-1, 1:-1]) / dx

        v_prev_2[1:-1, 1:-1] = v_prev_2[1:-1, 1:-1] + dt * (-kin_stress_y - interfacial_stress_y - gravitational_particles_y)

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

    if iter > start_jet:
        make_jet(u_next, jet_velocity)


    # Advance
    u_prev = u_next
    v_prev = v_next
    P_prev = P_next

    # Visualize simulation
    if iter % Plot_Every == 0:
        plt.figure(dpi=DPI_for_figures)
        u_center = (u_next[1:, :] + u_next[:-1, :]) / 2
        v_center = (v_next[:, 1:] + v_next[:, :-1]) / 2
        plt.contourf(coord_x, coord_y, u_center, levels=10)

        plt.colorbar(label="Velocity")

        plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

        if iter > start_multi_phase:
            plt.title(f"SW Continuous Phase, time: {iter*dt:.2f} s, Multiphase: on")
        else:
            plt.title(f"SW Continuous Phase, time: {iter * dt:.2f} s, Multiphase: off")
        plt.xlabel("Length (m)")
        plt.ylabel("Width (m)")
        plt.savefig(f'save_for_gif/u_img_{iter}_with_turbulence.png',
                    transparent=False,
                    facecolor='white'
                    )
        plt.close

        plt.figure(dpi=DPI_for_figures)
        plt.contourf(coord_x, coord_y, v_center, levels=10)

        plt.colorbar(label="Velocity")

        plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

        if iter > start_multi_phase:
            plt.title(f"NSW Continuous Phase, time: {iter * dt:.2f} s, Multiphase: on")
        else:
            plt.title(f"NSW Continuous Phase, time: {iter * dt:.2f} s, Multiphase: off")
        plt.xlabel("Length (m)")
        plt.ylabel("Width (m)")
        plt.savefig(f'save_for_gif/v_img_{iter}_with_turbulence.png',
                    transparent=False,
                    facecolor='white'
                    )
        plt.close

    if iter > start_multi_phase:
        if iter % Plot_Every == 0:
            plt.figure(dpi=DPI_for_figures)
            u_center = (u_prev_2[1:, :] + u_prev_2[:-1, :]) / 2
            v_center = (v_prev_2[:, 1:] + v_prev_2[:, :-1]) / 2
            plt.contourf(coord_x, coord_y, u_center, levels=10)

            plt.colorbar()

            plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

            plt.title(f"SW Dispersed Phase, time: {iter*dt:.2f} s")
            plt.xlabel("Length (m)")
            plt.ylabel("Width (m)")
            plt.savefig(f'save_for_gif/u_img_mult_{iter}_with_turbulence.png',
                        transparent=False,
                        facecolor='white'
                        )
            plt.close
            plt.figure(dpi=DPI_for_figures)
            plt.contourf(coord_x, coord_y, v_center, levels=10)

            plt.colorbar()

            plt.quiver(coord_x[:, ::6], coord_y[:, ::6], u_center[:, ::6], v_center[:, ::6], alpha=0.4)

            plt.title(f"NSW Dispersed Phase, time: {iter * dt:.2f} s")
            plt.xlabel("Length (m)")
            plt.ylabel("Width (m)")
            plt.savefig(f'save_for_gif/v_img_mult_{iter}_with_turbulence.png',
                        transparent=False,
                        facecolor='white'
                        )
            plt.close

print("saving gif")
u_frames = []
v_frames = []
u_frames_2 = []
v_frames_2 = []
for iter in range(N):

    if iter % Plot_Every == 0:
        image = imageio.v2.imread(f'save_for_gif/u_img_{iter}_with_turbulence.png')
        u_frames.append(image)

        image = imageio.v2.imread(f'save_for_gif/v_img_{iter}_with_turbulence.png')
        v_frames.append(image)

        if iter > start_multi_phase:
            image = imageio.v2.imread(f'save_for_gif/u_img_mult_{iter}_with_turbulence.png')
            u_frames_2.append(image)
            image = imageio.v2.imread(f'save_for_gif/v_img_mult_{iter}_with_turbulence.png')
            v_frames_2.append(image)

imageio.mimsave(f'gifs/u_multiphase_continuous_with_turbulence.gif',
                u_frames,
                duration=0.03
                )
imageio.mimsave(f'gifs/v_multiphase_continuous_with_turbulence.gif',
                v_frames,
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
    imageio.mimsave(f'gifs/u_multiphase_dispersed_with_turbulence.gif',
                    u_frames_2,
                    duration=0.03
                    )
    imageio.mimsave(f'gifs/v_multiphase_dispersed_with_turbulence.gif',
                    v_frames_2,
                    duration=0.03
                    )
    u_center_2 = (u_prev_2[1:, :] + u_prev_2[:-1, :]) / 2
    np.save(directory + "\\data\\u_center_2.npy", u_center_2)
    plt.plot(height, u_center_2[:,-5], label='Dispersed phase')

plt.ylabel("Velocity (m/s)")
plt.xlabel("Width (m)")
plt.legend()
plt.grid()
plt.savefig(f"plots/velocity_with_turbulence.png")

print("gif saved")