import numpy as np
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, rho, kinematic_viscosity):
        self.rho = rho
        self.kinematic_viscosity = kinematic_viscosity
        self.dynamic_viscosity = kinematic_viscosity * rho
        self.grid = None
        self.number_of_nodes = None
        self.boundary_conditions = {"bc_left": 0, "bc_right": 0}
        self.length = 0

    def make_grid(self, length, nodes, initial_velocity):
        self.number_of_nodes = nodes
        self.length = length
        self.dx = length / nodes
        boundary_node_position = length / nodes / 2

        position_array = np.linspace(boundary_node_position, length - boundary_node_position, nodes)
        gridpoints = np.zeros((), dtype=[
                            ("velocity", float, nodes),
                            ("position", float, nodes),
                            ("analytical_velocity", float, nodes),
                            ])
        for i, position in enumerate(position_array):
            gridpoints["velocity"][i] = initial_velocity
            gridpoints["position"][i] = position

        self.grid = gridpoints

    def analytical_sol_1d_pressure(self, x, bc_left, bc_right, pressure_gradient):
        c2 = bc_left
        c1 = (self.dynamic_viscosity / self.length) * bc_right -\
              self.length*pressure_gradient / 2 - self.dynamic_viscosity * bc_left / self.length
        return pressure_gradient * x**2 / (2 * self.dynamic_viscosity) + c1 * x / self.dynamic_viscosity + c2
    
    def laminar_velocity(self, bc_left, bc_right, pressure_gradient):
        """Calculates the laminar velocity with the Central difference method."""
        A = -2 * np.eye(self.number_of_nodes) \
            + np.eye(self.number_of_nodes, k=1) \
            + np.eye(self.number_of_nodes, k=-1)
        # dx = length / self.number_of_nodes
        # A /= dx
        A[0, 0] = -3
        A[-1, -1] = -3

        b = np.ones(self.number_of_nodes) * pressure_gradient * self.dx**2 / self.dynamic_viscosity
        b[0] -= 2 * bc_left
        # b[1] -= gradient
        b[-1] -= 2 * bc_right
        # print(b)
        velocity = np.linalg.solve(A, b)
        self.grid['velocity'] = velocity

        x = self.get_position()
        analytical = self.analytical_sol_1d_pressure(x, bc_left, bc_right, pressure_gradient)
        self.grid["analytical_velocity"] = analytical
        self.boundary_conditions["bc_left"] = bc_left
        self.boundary_conditions["bc_right"] = bc_right


    def turbulent_velocity(self, bc_left, bc_right, pressure_gradient):
        A = np.zeros([self.number_of_nodes, self.number_of_nodes])
        for j in range(self.number_of_nodes):
            for i in range(self.number_of_nodes):
                if i == j:
                    A[i, j] = -2 * self.dynamic_viscosity - self.rho * \
                        (self.eddy_viscosity(i) + self.eddy_viscosity(i + 1))
                elif (i + 1) == j:
                    A[i, j] = self.dynamic_viscosity + self.rho * self.eddy_viscosity(i)
                elif (i - 1) == j:
                    A[i, j] = self.dynamic_viscosity + self.rho * self.eddy_viscosity(i + 1)

# TODO: Kan nog fout zijn, metname A[-1, -1] moet gecontrolleerd worden
        A[0, 0] = -3 * self.dynamic_viscosity - self.rho * (self.eddy_viscosity(0) + self.eddy_viscosity(1))
        A[-1, -1] = -3 * self.dynamic_viscosity - self.rho * \
            (self.eddy_viscosity(self.number_of_nodes - 1) + self.eddy_viscosity(self.number_of_nodes - 2))

        b = np.ones(self.number_of_nodes) * pressure_gradient * self.dx**2
        b[0] -= 2 * bc_left
        b[-1] -= 2 * bc_right

        velocity = np.linalg.solve(A, b)
        self.grid['velocity'] = velocity

    def eddy_viscosity(self, i):
        kappa = 0.41
        kappa0 = 0.09
        boundary_layer = 0.1 * self.length
        pre_term = (kappa * boundary_layer)**2
        
        # TODO: if statement moet nog worden aangepast met juiste ghost cell check en return (weet niet 100% zeker of dit klopt)
        if (i >= (self.number_of_nodes - 1)) or (i <= 1):
            return 0
        else:
            velocity_difference = np.abs(self.grid["velocity"][i] - self.grid["velocity"][i - 1]) / self.dx

            if (i - 1) * self.dx < boundary_layer:
                return pre_term * ((i - 1) * self.dx)**2 * velocity_difference
            elif (i - 1) * self.dx > (self.length - boundary_layer):
                return pre_term * ((self.number_of_nodes - i + 1) * self.dx)**2 * velocity_difference
            
            return pre_term * velocity_difference * self.dx**2
    
    def get_velocities(self):
        return self.grid["velocity"]

    def get_analytical_solution(self):
        return self.grid["analytical_velocity"]

    def get_position(self):
        return self.grid["position"]

    def plot(self, analyicalTF):
        velocity = self.get_velocities()
        position = self.get_position()
        analytical = self.get_analytical_solution()
        plt.plot(position, velocity, label="numerical solution")
    
        if analyicalTF:
            plt.plot(position, analytical, label="analytical solution", ls='--')
            plt.title(f"bc_left = {self.boundary_conditions['bc_left']:.1f} m/s, bc_right = {self.boundary_conditions['bc_right']:.1f} m/s")
            plt.legend()

        plt.xlabel("x")
        plt.ylabel("velocity")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    rho = 1000  # water
    kinematic_viscosity = 0.0000105  # water
    length = 1
    nodes = 1000

    '''weird viscosity'''
    # x = np.linspace(0,length,nodes)
    # kinematic_viscosity = (1.5 + 1.4*np.sin(x*4))*1e-5

    pressure_gradient = -1e-1

    # Make our initial boy
    grid = Grid(rho, kinematic_viscosity)


    initial_velocity = 1
    # bc_velocity = 0
    grid.make_grid(length, nodes, initial_velocity)
    # |--------do simulations--------|
    """ first exercise wall boundary"""

    grid.laminar_velocity(bc_left=1,bc_right=0, pressure_gradient=pressure_gradient)
    grid.plot(False)
    grid.turbulent_velocity(bc_left=1,bc_right=0, pressure_gradient=pressure_gradient)
    grid.plot(False)