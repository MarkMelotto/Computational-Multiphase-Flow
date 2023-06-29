import numpy as np

def calc_T_t(U, V, l, dx):
    du_dy = calc_U_prime(U, V, l, dx)
    T_t = np.zeros_like(du_dy)
    T_t[1:-1, :] = 1/du_dy[1:-1, :]
    return T_t

def calc_T_t_new(U,  l, dx):
    du_dy = calc_U_prime_new(U, l, dx)
    T_t = np.zeros_like(du_dy)
    T_t[1:-1, :] = 1/du_dy[1:-1, :]
    T_t[np.isnan(T_t)] = 0
    return T_t

def calc_U_prime(U, V, l, dx):
    du_dy = calc_dU_dy(U, V, dx)
    return l*du_dy

def calc_U_prime_new(U, l, dx):
    du_dy = calc_dU_d_new(U, dx)
    return l*du_dy

def calc_dU_dy(U, V, dx):
    # du_dy = np.zeros((U.shape[0],V.shape[1]))
    # du_dx = np.zeros((U.shape[0],V.shape[1]))
    # dv_dy = np.zeros((U.shape[0],V.shape[1]))
    # dv_dx = np.zeros((U.shape[0],V.shape[1]))
    # du_dy = (U[2:-1, 1:-1] - U[1:-2, 1:-1]) / dx

    du_dx = (U[1:-1, 2:-1] - U[1:-1, 1:-2]) / dx

    dv_dy = (V[2:-1, 1:-1] - V[1:-2, 1:-1]) / dx
    # dv_dx = (V[1:-1, 2:-1] - V[1:-1, 1:-2]) / dx

    # u_prime = np.sqrt(du_dx*du_dx + dv_dy*dv_dy)/4
    u_prime = np.sqrt(np.mean(du_dx)**2+np.mean(dv_dy)**2)
    return u_prime

def calc_dU_d_new(U, dx):
    du_dy = np.zeros_like(U)
    du_dx = np.zeros_like(U)
    # dv_dy = np.zeros((U.shape[0],V.shape[1]))
    # dv_dx = np.zeros((U.shape[0],V.shape[1]))
    # print(f"u shape = {U.shape}")
    du_dy[2:-1, 1:-1] = (U[2:-1, 1:-1] - U[1:-2, 1:-1]) / dx
    du_dy[np.isnan(du_dy)] = 0
    # print(f"du_dy shape = {du_dy.shape}")
    du_dx[1:-1, 2:-1] = (U[1:-1, 2:-1] - U[1:-1, 1:-2]) / dx
    du_dx[np.isnan(du_dx)] = 0
    # print(f"du_dx shape = {du_dx.shape}")
    # dv_dy = (V[2:-1, 1:-1] - V[1:-2, 1:-1]) / dx
    # dv_dx = (V[1:-1, 2:-1] - V[1:-1, 1:-2]) / dx

    u_prime = np.sqrt((du_dx*du_dx + du_dy*du_dy + 2*du_dx*du_dy)/4)
    u_prime[np.isnan(u_prime)] = 0
    # u_prime = np.sqrt(np.mean(du_dx)**2+np.mean(dv_dy)**2)
    return u_prime

# def calc_dV_d_new(U, dx):
#     # du_dy = np.zeros((U.shape[0],V.shape[1]))
#     # du_dx = np.zeros((U.shape[0],V.shape[1]))
#     # dv_dy = np.zeros((U.shape[0],V.shape[1]))
#     # dv_dx = np.zeros((U.shape[0],V.shape[1]))
#     dv_dy = (U[2:-1, 1:-1] - U[1:-2, 1:-1]) / dx
#     print(f"dv_dy shape = {dv_dy.shape}")
#     dv_dx = (U[1:-1, 2:-1] - U[1:-1, 1:-2]) / dx
#
#     # dv_dy = (V[2:-1, 1:-1] - V[1:-2, 1:-1]) / dx
#     # dv_dx = (V[1:-1, 2:-1] - V[1:-1, 1:-2]) / dx
#
#     u_prime = np.sqrt(dv_dx*dv_dx + dv_dy*dv_dy + 2*dv_dx*dv_dy)/4
#     # u_prime = np.sqrt(np.mean(du_dx)**2+np.mean(dv_dy)**2)
#     return u_prime

def calc_U_2i_U_2j(T_t, T_p, U, V, l, dx):
    U_1i_U_1j = calc_U_prime(U, V, l, dx)**2

    return (T_t/(T_p+T_t))*U_1i_U_1j

def calc_U_2i_U_2j_new(T_t, T_p, U, l, dx):
    # print(f"l is {l}")
    U_1i_U_1j = calc_U_prime_new(U, l, dx)**2
    U_1i_U_1j[np.isnan(U_1i_U_1j)] = 0
    return (T_t/(T_p+T_t))*U_1i_U_1j

def get_F_i(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    # print(f'U2 min U1 = {U_2i_min_U_1i}')
    interfacial_stress = 18*nu_f * rho_2 * a_2 * U_2i_min_U_1i / D_p**2
    # interfacial_stress = 18*nu_f * rho_2 * a_2 * U_2i_min_U_1i

    return interfacial_stress


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

    u_prime = calc_dU_dy(u_prev, v_prev, dx)
    print(u_prime)

    uprime = calc_dU_d_new(u_prev, dx)
    print(uprime)

    vprime = calc_dU_d_new(v_prev, dx)
    print(vprime)

    # U_prime_x = calc_U_2i_U_2j_new(T_t, T_p, U, l, dx)
    # print(U_prime_x.shape)
    #
    # U_prime_y = calc_U_prime(u_prev, v_prev, l_y, dx)
    # print(U_prime_y.shape)