import numpy as np

def calc_T_t_new(U,  l, dx):
    du_dy = calc_U_prime_new(U, l, dx)
    T_t = np.zeros_like(du_dy)
    T_t[1:-1, :] = 1/du_dy[1:-1, :]
    T_t[np.isnan(T_t)] = 0
    return T_t


def calc_U_prime_new(U, l, dx):
    du_dy = calc_dU_d_new(U, dx)
    return l*du_dy


def calc_dU_d_new(U, dx):
    du_dy = np.zeros_like(U)
    du_dx = np.zeros_like(U)

    du_dy[1:-1, 1:-1] = (U[2:, 1:-1] - U[:-2, 1:-1]) / dx

    du_dx[1:-1, 1:-1] = (U[1:-1, 2:] - U[1:-1, :-2]) / dx


    u_prime = np.sqrt((du_dx*du_dx + du_dy*du_dy + 2*du_dx*du_dy)/4)
    u_prime[np.isnan(u_prime)] = 0
    return u_prime

def calc_U_2i_U_2j_new(T_t, T_p, U, l, dx):
    # print(f"l is {l}")
    U_1i_U_1j = calc_U_prime_new(U, l, dx)**2
    U_1i_U_1j[np.isnan(U_1i_U_1j)] = 0
    return (T_t/(T_p+T_t))*U_1i_U_1j

def get_F_i(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    interfacial_stress = 18*nu_f * rho_2 * a_2 * U_2i_min_U_1i / D_p**2

    return interfacial_stress

def get_F_i_fast(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    interfacial_stress = 18*nu_f * rho_2 * a_2 * U_2i_min_U_1i

    return interfacial_stress
def get_F_i_fast_concentration(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    interfacial_stress = 18*nu_f * rho_2 * a_2[1:-1, 1:-1] * U_2i_min_U_1i

    return interfacial_stress

def reynolds_stress_x(U, eddy_viscosity_x, rey_x):
    rey_x[1:-1,1:-1] = eddy_viscosity_x*(U[2:, 1:-1] - U[:-2, 1:-1])
    return rey_x

def reynolds_stress_y(V, eddy_viscosity_y, rey_y):
    rey_y[1:-1,1:-1] = eddy_viscosity_y*(V[2:, 1:-1] - V[:-2, 1:-1])
    return rey_y

def calculate_kinetic_stress_x(a_2, rho_p, T_t, T_p, U, eddy_viscosity_x, rey_x):
    U_1i_U_1j = reynolds_stress_x(U, eddy_viscosity_x, rey_x)
    # U_1i_U_1j[np.isnan(U_1i_U_1j)] = 0  # operands could not be broadcast together with shapes (31,1161) (29,1159)
    # I added [1:-1, 1:-1] that should fix the error above, but that will lead to another error
    kin_stress = a_2 * rho_p * ((T_t / (T_p + T_t)) * U_1i_U_1j)
    kin_stress[np.isnan(kin_stress)] = 0
    return kin_stress

def calculate_kinetic_stress_y(a_2, rho_p, T_t, T_p, V, eddy_viscosity_y, rey_y):
    U_1i_U_1j = reynolds_stress_y(V, eddy_viscosity_y, rey_y)
    # U_1i_U_1j[np.isnan(U_1i_U_1j)] = 0
    kin_stress = a_2 * rho_p *((T_t / (T_p + T_t)) * U_1i_U_1j)
    kin_stress[np.isnan(kin_stress)] = 0
    return kin_stress


def gravitational_force_particles(a_2, rho_1, rho_2, angle):
    g = 9.81  # m/s^2
    return a_2*((rho_2 - rho_1)/rho_1) * g * np.cos(angle)
    # return a_2 * rho_2/(rho_1+rho_2) * g * np.cos(angle)
def gravitational_force_fluid(a_2, rho_1, rho_2, angle):
    g = 9.81  # m/s^2
    return -a_2*((rho_2 - rho_1)/rho_1) * g * np.cos(angle)
    # return a_1 * g * np.cos(angle)

def make_jet(U, velocity_of_jet):
    U[3:10, 150:161] = velocity_of_jet
    U[-10:-3, 150:161] = velocity_of_jet

def make_jet_pressure(P, pressure_of_jet):
    P[3:10, 150:161] /= pressure_of_jet
    P[-10:-3, 150:161] /= pressure_of_jet

def updated_a2(a_2, U, dt, dx):
    a_2[15, 200] = 0.01  # something that was suggested to us
    du_dxy = calc_dU_d_new(U, dx)
    da_2_dxy = calc_dU_d_new(a_2, dx)
    new_a_2 = a_2 - dt*(U*da_2_dxy + a_2*du_dxy)
    new_a_2[new_a_2 > 0.05] = 0.05
    new_a_2[new_a_2 < 0.00] = 0.00
    return new_a_2


if __name__ == "__main__":
    Aspect = 10  # Aspect ratio between y and x direction
    Ny = 15  # points in y direction
    Nx = (Ny - 1) * Aspect + 1  # points in x direction
    dx = 1.0 / (Ny - 1)
    H = 1.0  # channel height
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

    y = np.linspace(0, H, Ny)
    f_pos = 0.4 * y
    f_neg = 0.4 * H - 0.4 * y
    f_const = 0.1 * H * np.ones(len(y))
    f_l = (np.minimum(np.minimum(f_pos, f_neg), f_const)) ** 2
    l_y = np.zeros((Ny, Nx + 1))
    l_y[:, :] = f_l[:, np.newaxis]

    y_x = np.linspace(0, H, Ny + 1)
    f_pos_x = 0.4 * y_x
    f_neg_x = 0.4 * H - 0.4 * y_x
    f_const_x = 0.1 * H * np.ones(len(y_x))
    f_l_x = (np.minimum(np.minimum(f_pos_x, f_neg_x), f_const_x)) ** 2
    l_x = np.zeros((Ny + 1, Nx))
    l_x[:, :] = f_l_x[:, np.newaxis]

    # u_prime = calc_dU_dy(u_prev, v_prev, dx)
    # print(u_prime)

    uprime = calc_dU_d_new(u_prev, dx)
    print(uprime)

    vprime = calc_dU_d_new(v_prev, dx)
    print(vprime)

    # U_prime_x = calc_U_2i_U_2j_new(T_t, T_p, U, l, dx)
    # print(U_prime_x.shape)
    #
    # U_prime_y = calc_U_prime(u_prev, v_prev, l_y, dx)
    # print(U_prime_y.shape)