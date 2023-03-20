##### ----- Single phase 1D flow ----- #####
import numpy as np
import matplotlib.pyplot as plt

class NumericalModel():
    def __init__(self,
                 length_x: float,
                 height_y: float,
                 nodes_x: int,
                 nodes_y: int,
                 bc_left: float,
                 bc_right: float,
                 pressure_difference: float,
                 kinematic_viscosity: float) -> None:
        self.length = length_x
        self.height = height_y
        self.nodes_x = nodes_x
        self.nodes_y = nodes_y
        self.x = np.linspace(0, length_x, nodes_x)
        self.y = np.linspace(0, height_y, nodes_y)
        self.delta_x = length_x / nodes_x
        self.delta_y = height_y / nodes_y

        self.delta_P = pressure_difference
        self.mu = kinematic_viscosity

        self.matrix, self.body_forces = self.setup_central_difference_matrix()
        self.body_forces = self.boundary_conditions(bc_left, bc_right)

    def setup_central_difference_matrix_equation(self):
        # Define matrix to solve, this is the A in Ax=b
        A = np.zeros([self.nodes_y, self.nodes_y])
        A[np.arange(self.nodes_y), np.arange(self.nodes_y)] = -2
        A[np.arange(self.nodes_y - 1), np.arange(self.nodes_y - 1) + 1] = 1
        A[np.arange(self.nodes_y - 1) + 1, np.arange(self.nodes_y - 1)] = 1
        A[-1, -1] = -3
        A[0, 0] = -3

        b = np.ones(self.nodes_y) * self.delta_P * self.delta_y**2 / self.mu
        return A, b

    def boundary_conditions(self, bc_left, bc_right):
        self.body_forces[0] = -bc_left
        self.body_forces[-1] = -bc_right
        
    def calculate_analytical_velocity(self):
        # Def b in Ax=b, this i a like source term
        return -self.delta_P * self.y * (self.height - self.y) / self.mu / 2

    def calculate_numerical_velocity(self):
        # N-S Eq for steady-state fully developed flow: mu * d^2/dy^2 u_x = dP/dx
        # the continuity Eq gives: du_x/dx = 0
        U_numerical = np.dot(self.body_forces, np.linalg.inv(self.matrix))
        return U_numerical

    #analytical solution to pipe flow
    # U_an = -Delta_P /(2*mu) *y*(H-y)

    def calculate_error_RMS(self, numerical_solution, analytical_solution):
        error = numerical_solution - analytical_solution
        rms = np.sqrt(np.sum(error ** 2) / len(error))
        return rms

    def plot_numerical_analytical_solutions(self,
                                            numerical_solution: np.ndarray,
                                            analytical_solution: np.ndarray,
                                            error: float) -> None:
        #plot Num and Ana solutions
        plt.errorbar(self.y, numerical_solution, error, fmt ='*', label = 'Numerical Solution')
        plt.plot(self.y, analytical_solution)
        plt.xlabel('Channel height')
        plt.ylabel('Velocity')
        plt.legend()
        plt.show()

    def plot_flow(self, solution):
        plt.plot(self.y, solution)
        plt.xlabel('Channel height')
        plt.ylabel('Velocity')
        plt.title("Velocity field for 1D channel flow")
        plt.grid()
        plt.show()

    def plot_flow_contour(self, solution):
        # for fun, plot the velocity profile in x-y direction
        U = solution * np.ones([self.nodes_y, self.nodes_y])
        plt.contourf(U, self.nodes_y)
        plt.title("Velocity field contour")
        plt.xlabel('channel height')
        plt.ylabel('channel length')
        plt.colorbar()
        plt.show()

#Calc the Reynolds number
# Re = np.max(U_num*H/mu)
# print('Re=',Re)


# Constants
LENGTH = 1
HEIGHT = 0.05
NODES_X = 10
NODES_Y = 25
BC_LEFT = 0
BC_RIGHT = 0

# Variables
pressure_drop = -1
kinematic_viscosity = 0.00105 # Water


if __name__ == "__main__":
    model = NumericalModel(
                length_x = LENGTH,
                height_y = HEIGHT,
                nodes_x = NODES_X,
                nodes_y = NODES_Y,
                bc_left = BC_LEFT,
                bc_right = BC_RIGHT,
                pressure_difference = pressure_drop,
                kinematic_viscosity = kinematic_viscosity)
    
    U_numerical = model.calculate_numerical_velocity()
    U_analytical = model.calculate_analytical_velocity()
    U_error = model.calculate_error_RMS(U_numerical, U_analytical)

    model.plot_numerical_analytical_solutions(U_numerical, U_analytical, U_error)
    model.plot_flow_contour(U_numerical)
    # plot_flow(y, numerical_solution)
