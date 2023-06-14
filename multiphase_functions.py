import numpy as np

def calc_T_t(U, dx):
    du_dy = (U[2:-1, 1:-1] - U[1:-2, 1:-1]) / dx
    return np.mean(1/du_dy)
def calc_U_2i_U_2j(T_t, T_p, U_1i_U_1j):
    return (T_t/(T_p+T_t))*U_1i_U_1j

def get_F_i(nu_f, D_p, rho_2, a_2, U2mean, U1mean):
    U_2i_min_U_1i = U2mean - U1mean
    return 18*nu_f/(D_p**2) * rho_2 * a_2 * U_2i_min_U_1i