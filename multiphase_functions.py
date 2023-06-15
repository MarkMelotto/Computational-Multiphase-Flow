import numpy as np

def calc_T_t(U, V, l, dx):
    du_dy = calc_U_prime(U, V, l, dx)
    if du_dy.any() != 0:
        return 1/du_dy
    else:
        return np.zeros_like(du_dy)

def calc_U_prime(U, V, l, dx):
    du_dy = np.zeros((U.shape[0],V.shape[1]))
    du_dx = np.zeros((U.shape[0],V.shape[1]))
    dv_dy = np.zeros((U.shape[0],V.shape[1]))
    dv_dx = np.zeros((U.shape[0],V.shape[1]))
    du_dy[1:-2,1:-2] = l[1:-1,1:-1] * (U[2:-1, 1:-1] - U[1:-2, 1:-1]) / dx
    du_dx[1:-1,2:-2] = l[1:,1:-2] * (U[1:-1, 2:-1] - U[1:-1,1:-2]) / dx
    dv_dy[2:-2,1:-1] = l[2:-1,1:] * (V[2:-1, 1:-1] - V[1:-2, 1:-1]) / dx
    dv_dx[1:-2,1:-2] = l[1:-1,1:-1] * (V[1:-1, 2:-1] - V[1:-1, 1:-2]) / dx

    u_prime = np.sqrt((du_dy*du_dy + 2*du_dy*du_dx + 2*du_dy*dv_dy + 2*du_dy*dv_dx +
                      du_dx*du_dx + 2*du_dx*dv_dy + 2*du_dx*dv_dx +
                      dv_dy*dv_dy * 2*dv_dy*dv_dx +
                      dv_dx*dv_dx)/16)
    return u_prime

def calc_U_2i_U_2j(T_t, T_p, U, V, l, dx):
    U_1i_U_1j = calc_U_prime(U, V, l, dx)**2
    return (T_t/(T_p+T_t))*U_1i_U_1j

def get_F_i(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    return 18*nu_f * rho_2 * a_2 * U_2i_min_U_1i/(D_p**2)