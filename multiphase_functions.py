import numpy as np

def calc_T_t(U, V, l, dx):
    du_dy = calc_U_prime(U, V, l, dx)
    T_t = np.zeros_like(du_dy)
    T_t[1:-1, :] = 1/du_dy[1:-1, :]
    return T_t

def calc_U_prime(U, V, l, dx):
    du_dy = calc_dU_dy(U, V, dx)
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

def calc_U_2i_U_2j(T_t, T_p, U, V, l, dx):
    U_1i_U_1j = calc_U_prime(U, V, l, dx)**2
    return (T_t/(T_p+T_t))*U_1i_U_1j

def get_F_i(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    return 18*nu_f * rho_2 * a_2 * U_2i_min_U_1i/D_p**2

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

    u_prime = calc_dU_dy(u_prev, v_prev, dx)
    print(u_prime)