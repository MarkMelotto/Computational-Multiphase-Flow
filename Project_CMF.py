'''Single phase 1D flow'''
import numpy as np
import matplotlib.pyplot as plt

# ny = 25 #nodes in y
# nx = 10 #nodes in x
# #nx= 1
# H = 0.05 #height in y
# L = 1 #length in x
# y = np.linspace(0,H,ny)
# x = np.linspace(0,L,nx)
# mu = 0.00105 #water
# Delta_P = -1 #some drop in presure
# dy = H/(ny-1) #Deltay

def initialize(ny, nx):
    H = 0.05  # height in y
    L = 1  # length in x
    y = np.linspace(0, H, ny)
    x = np.linspace(0, L, nx)
    mu = 0.00105  # water
    Delta_P = -1  # some drop in presure
    dy = H / (ny - 1)  # Deltay

    # Def matrix to solve, this is the A in Ax=b
    A = np.zeros([ny, ny])
    A[np.arange(ny), np.arange(ny)] = -2
    A[np.arange(ny - 1), np.arange(ny - 1) + 1] = 1
    A[np.arange(ny - 1) + 1, np.arange(ny - 1)] = 1
    A[-1, -1] = -3
    A[0, 0] = -3

    b = Delta_P * (dy) ** 2 / (mu) * np.ones(ny)

    return A, b, x, y

# N-S Eq for steady-state fully developed flow: mu * d^2/dy^2 u_x = dP/dx
# the continuity Eq gives: du_x/dx = 0

#Def b in Ax=b, this i a like source term
# b = Delta_P*(dy)**2/(mu)*np.ones(ny)
# b[0] = -10
#b[-1]=0

def boundary_conditions(b, bc_left=0, bc_right=0):
    b[0] = -bc_left
    b[-1] = bc_right
    return b


# Def matrix to solve, this is the A in Ax=b
# Matrix = np.zeros([ny,ny])
# Matrix[np.arange(ny),np.arange(ny)]=-2
# Matrix[np.arange(ny-1),np.arange(ny-1)+1] = 1
# Matrix[np.arange(ny-1)+1,np.arange(ny-1)] = 1
# Matrix[-1,-1] = -3
# Matrix[0,0] = -3

#invert matrix
# M_inv = np.linalg.inv(Matrix)
# U_num = np.dot(b,M_inv)

def solve(A,b):
    M_inv = np.linalg.inv(A)
    U_numerical = np.dot(b, M_inv)
    return U_numerical

#analytical solution to pipe flow
# U_an = -Delta_P /(2*mu) *y*(H-y)

#Cal the error and RMS
# Error = (U_num - U_an)
# RMS = np.sqrt(np.sum(Error**2)/len(Error))
# print('RMS=',RMS)

def error_RMS(numerical_solution, analytical_solution):
    error = (numerical_solution-analytical_solution)
    rms = np.sqrt(np.sum(error ** 2) / len(error))
    return rms

#print('Error=',Error)

#plot Num and Ana solutions
# plt.errorbar(y,U_num, Error, fmt ='*', label = 'Numerical Solution')
# plt.plot(y,U_an)
# plt.xlabel('channel height')
# plt.ylabel('velocity')
# plt.legend()
# plt.show()

def plot_flow(y, solution):
    plt.plot(y, solution)
    plt.xlabel('channel height')
    plt.ylabel('velocity')
    plt.grid()
    plt.show()

#for fun, plot the velocity profile in x-y direction
# u = U_num*np.ones([nx,ny])
# plt.contourf(u,ny)
# plt.xlabel('channel height')
# plt.ylabel('channel length')
# plt.colorbar()
# plt.show()

def plot_flow_contour(solution, nx,ny):
    u = solution * np.ones([nx, ny])
    plt.contourf(u, ny)
    plt.xlabel('channel height')
    plt.ylabel('channel length')
    plt.colorbar()
    plt.show()

#Calc the Reynolds number
# Re = np.max(U_num*H/mu)
# print('Re=',Re)


if __name__ == "__main__":
    nx = 10
    ny = 25

    A, b, x, y = initialize(ny, nx)
    b = boundary_conditions(b, bc_left=0, bc_right=0)
    numerical_solution = solve(A,b)
    plot_flow(y, numerical_solution)
    plot_flow_contour(numerical_solution, nx, ny)
