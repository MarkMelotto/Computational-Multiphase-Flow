import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg

aspect = 2
ny = 16
nx = ny * aspect
mu = 0.01
dt = 1e-6
N = 5
N_p = 10
H = 1.0
L = H * aspect
t = 0

dy = H/ny
dx = L/nx

x_range = np.linspace(0.0,L,nx)
y_range = np.linspace(0.0,H,ny)

x,y = np.meshgrid(x_range,y_range)

# initial conditions
u_i = np.ones((ny+2,nx+2))
u_i[0,:] = -u_i[1,:] #ghost boundary condition
u_i[-1,:] = -u_i[-2,:]

v_i = np.zeros((ny+2,nx+2))
p_i = np.zeros((ny+2,nx+2))

prhs = np.zeros([ny+2,nx+2])

# star arrays
u_t = np.zeros_like(u_i)
v_t = np.zeros_like(v_i)
p_t = np.zeros_like(p_i)

# final arrays
u = np.zeros_like(u_i)
v = np.zeros_like(v_i)
p = np.zeros_like(p_i)

# USE sparse solver
# build pressure coefficient matrix
Ap = np.zeros([ny,nx])
Ae = 1.0/dx/dx*np.ones([ny,nx])
As = 1.0/dy/dy*np.ones([ny,nx])
An = 1.0/dy/dy*np.ones([ny,nx])
Aw = 1.0/dx/dx*np.ones([ny,nx])
# set left wall coefs
Aw[:,0] = 0.0
# set right wall coefs
Ae[:,-1] = 0.0
# set top wall coefs
An[-1,:] = 0.0
# set bottom wall coefs
As[0,:] = 0.0
Ap = -(Aw + Ae + An + As)
n = nx*ny
d0 = Ap.reshape(n)
de = Ae.reshape(n)[:-1]
dw = Aw.reshape(n)[1:]
ds = As.reshape(n)[nx:]
dn = An.reshape(n)[:-nx]
A1 = scipy.sparse.diags([d0, de, dw, dn, ds], [0, 1, -1, nx, -nx], format='csr')

for iter in tqdm(range(N)):
    # ue = 0.5*(u_i[1:-2, 2:]+u_i[1:-2,1:-1])
    # uw = 0.5*(u_i[1:-2, :-2] +u_i[1:-2,1:-1])
    # un = 0.5*(u_i[2:-1,1:-1] + u_i[1:-2,1:-1])
    # us = 0.5*(u_i[1:-2 ,1:-1] + u_i[1:-2,1:-1])
    # #np = u_i[1:-1, 1:-1]
    #
    # ve = 0.5*(v_i[1:-2, 2:]+v_i[1:-2,1:-1])
    # vw = 0.5*(v_i[1:-2, :-2] +v_i[1:-2,1:-1])
    # vn = 0.5*(v_i[2:-1,1:-1] + v_i[1:-2,1:-1])
    # vs = 0.5*(v_i[1:-2 ,1:-1] + v_i[1:-2,1:-1])
    # #vp =
    # # bc

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            ue = 0.5 * (u_i[j,i+1] + u_i[j,i])
            uw = 0.5 * (u_i[j,i] + u_i[j,i-1])
            un = 0.5 * (u_i[j+1,i] + u_i[j,i])
            us = 0.5 * (u_i[j,i] + u_i[j-1,i])
            # np = u_i[1:-1, 1:-1]

            #ve = 0.5 * (v_i[j,i+1] + v_i[j,i])
            #vw = 0.5 * (v_i[j,i] + v_i[j,i-1])
            vn = 0.5 * (v_i[j+1,i] + v_i[j,i])
            vs = 0.5 * (v_i[j,i] + v_i[j-1,i])

            convection_x = - (ue*ue - uw*uw)/dx - (un*vn - us*vs)/dy
            diffusion_x = mu*( (u_i[j,i+1] - 2.0*u_i[j,i] + u_i[j,i-1])/dx/dx + (u_i[j+1,i] - 2.0*u_i[j,i] + u_i[j-1,i])/dy/dy )
            #p_x_grad = ((p_i[j,i+1]-p_i[j,i-1]) / (dy))
            u_t[j,i] = (u_i[j,i] + dt*( diffusion_x - convection_x)) #-p_x_grad +

    u_t[1:-2, 0] = 1.0  # inflow
    u_t[1:-2, -1] = u_t[1:-2, -2]  # continuous boundary
    u_t[0, :] = -u_t[1, :]  # ghost cel
    u_t[-1, :] = -u_t[-2, :]  # ghost cel

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            ue = 0.5 * (u_i[j, i + 1] + u_i[j, i])
            uw = 0.5 * (u_i[j, i] + u_i[j, i - 1])
            #un = 0.5 * (u_i[j + 1, i] + u_i[j, i])
            #us = 0.5 * (u_i[j, i] + u_i[j - 1, i])
            # np = u_i[1:-1, 1:-1]

            ve = 0.5 * (v_i[j, i + 1] + v_i[j, i])
            vw = 0.5 * (v_i[j, i] + v_i[j, i - 1])
            vn = 0.5 * (v_i[j + 1, i] + v_i[j, i])
            vs = 0.5 * (v_i[j, i] + v_i[j - 1, i])

            diffusion_y = mu*( (v_i[j,i+1] - 2.0*v_i[j,i] + v_i[j,i-1])/dx/dx + (v_i[j+1,i] - 2.0*v_i[j,i] + v_i[j-1,i])/dy/dy )
            convection_y = (ue*ve -uw*vw)/dx -(vn*vn -vs*vs)/dy
            #p_y_grad = ((p_i[j+1,i] - p_i[j-1,i]) / (dy))

            v_t[j,i] = (v_i[j,i] + dt * ( diffusion_y - convection_y)) #-p_y_grad

    #
    # # # Apply BC
    v_t[1:-2, 0] = - v_t[1:-2, 1]
    v_t[1:-2, -1] = v_t[1:-2, -2]
    v_t[0, :] = 0.0
    v_t[-1, :] = 0.0

    # compute pressure right hand side: prhs = 1/dt * div(ut)
    prhs[1:-1, 1:-1] = ((u_t[1:-1, 2:] - u_t[1:-1, 1:-1]) / dx + (v_t[2:, 1:-1] - v_t[1:-1, 1:-1]) / dy)/dt
    pt, info = scipy.sparse.linalg.bicg(A1, prhs[1:-1, 1:-1].ravel(), tol=1e-10)
    p_t[1:-1, 1:-1] = pt.reshape([ny, nx])

    p = p_i + p_t

    # Correct the velocities to be incompressible
    u[1:-1, 2:-1] = (u_t[1:-1, 2:-1] - dt * ((p[1:-1, 2:-1] - p[1:-1, 1:-2]) / (dx)))
    v[2:-1, 1:-1] = (v_t[2:-1, 1:-1] - dt * ((p[2:-1, 1:-1] - p[1:-2, 1:-1]) / (dy)))

    #u[1:-1,1:-1] =u_t[1:-1,2:] -dt*(p[1:-1,2:] -p[1:-1,:-2])/dx
    #v[1:-1, 1:-1] = v_t[2:, 1:-1] - dt * (p[2:, 1:-1] - p[:-2, 1:-1]) / dy

    # #bc again
    # # Again enforce BC
    # u[1:-1, 0] = 1.0
    # inflow_mass_rate_next = np.sum(u[1:-1, 0])
    # outflow_mass_rate_next = np.sum(u[1:-1, -2])
    # u[1:-2, -1] = u[1:-2, -2] * inflow_mass_rate_next / outflow_mass_rate_next
    # u[0, :] = - u[1, :]
    # u[-1, :] = - u[-2, :]
    #
    # v[1:-2, 0] = - v[1:-2, 1]
    # v[1:-2, -1] = v[1:-2, -2]
    # v[0, :] = 0.0
    # v[-1, :] = 0.0

    t +=dt
    u_i = u
    v_i = v
    p_i = p



plt.quiver(u,v)
plt.show()

#plt.plot(u[1:-1,6])
#plt.show()

