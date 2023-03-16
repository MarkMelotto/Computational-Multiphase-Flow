'''Single phase 1D flow'''

if __name__ == "__main__":
    pass

import numpy as np
import matplotlib.pyplot as plt

ny = 25 #nodes in y
nx = 10 #nodes in x
#nx= 1
H = 0.05 #height in y
L = 1 #length in x
y = np.linspace(0,H,ny)
x = np.linspace(0,L,nx)
mu = 0.00105 #water
Delta_P = -1 #some drop in presure
dy = H/(ny-1) #Deltay

# N-S Eq for steady-state fully developed flow: mu * d^2/dy^2 u_x = dP/dx
# the continuity Eq gives: du_x/dx = 0

#Def b in Ax=b, this i a like source term
b = Delta_P*(dy)**2/(mu)*np.ones(ny)
b[0] = 0
#b[-1]=0

# Def matrix to solve, this is the A in Ax=b
Matrix = np.zeros([ny,ny])
Matrix[np.arange(ny),np.arange(ny)]=-2
Matrix[np.arange(ny-1),np.arange(ny-1)+1] = 1
Matrix[np.arange(ny-1)+1,np.arange(ny-1)] = 1
Matrix[-1,-1] = -3
Matrix[0,0] = -3

#invert matrix
M_inv = np.linalg.inv(Matrix)
U_num = np.dot(b,M_inv)

#analytical solution to pipe flow
U_an = -Delta_P /(2*mu) *y*(H-y)

#Cal the error and RMS
Error = (U_num - U_an)
RMS = np.sqrt(np.sum(Error**2)/len(Error))
print('RMS=',RMS)
#print('Error=',Error)

#plot Num and Ana solutions
plt.errorbar(y,U_num, Error, fmt ='*', label = 'Numerical Solution')
plt.plot(y,U_an)
plt.xlabel('channel height')
plt.ylabel('velocity')
plt.legend()
plt.show()

#for fun, plot the velocity profile in x-y direction
u = U_num*np.ones([nx,ny])
plt.contourf(u,ny)
plt.xlabel('channel height')
plt.ylabel('channel length')
plt.colorbar()
plt.show()

#Calc the Reynolds number
Re = np.max(U_num*H/mu)
print('Re=',Re)
