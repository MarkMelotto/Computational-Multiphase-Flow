import numpy as np
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, rho, kinematic_viscosity):
        self.rho = rho
        self.kinematic_viscosity = kinematic_viscosity
        self.dynamic_viscosity = kinematic_viscosity*rho
        self.grid = None
        self.number_of_nodes = None
        self.boundary_conditions = {"bc_left": 0, "bc_right": 0}
        self.length = 0

    def make_grid(self, length, nodes, initial_velocity):
        self.number_of_nodes = nodes
        self.length = length
        boundary_node_position = (length/nodes)/2
        position_array = np.linspace(boundary_node_position,length-boundary_node_position, nodes)
        gridpoints = np.zeros((), dtype=[
        ("velocity", float, nodes),
        ("position", float, nodes),
        ("analytical_velocity", float, nodes),
        ])
        for i,position in enumerate(position_array):
            gridpoints["velocity"][i] = initial_velocity
            gridpoints["position"][i] = position

        self.grid = gridpoints

    def analytical_sol_1d_pressure(self, x, bc_left, bc_right, pressure_gradient):
        c2 = bc_left
        c1 = (self.dynamic_viscosity/self.length)*bc_right - self.length*pressure_gradient/2 - self.dynamic_viscosity*bc_left/self.length
        return pressure_gradient*x**2/(2*self.dynamic_viscosity) + c1*x/self.dynamic_viscosity + c2
    def basic_velocity_CDM(self, bc_left, bc_right, pressure_gradient):
        A = -2*np.eye(self.number_of_nodes,self.number_of_nodes)\
            + np.eye(self.number_of_nodes,self.number_of_nodes, k=1) \
            + np.eye(self.number_of_nodes,self.number_of_nodes, k=-1)
        dx = length/self.number_of_nodes
        # A /= dx
        A[0,0] = -3
        A[-1,-1] = -3

        b = np.ones(self.number_of_nodes) * pressure_gradient * dx**2 / self.dynamic_viscosity
        # print(b)
        b[0] -= 2*bc_left
        b[-1] -= 2*bc_right
        # print(b)
        velocity = np.linalg.solve(A,b)
        self.grid['velocity'] = velocity

        x = self.get_position()
        analytical = self.analytical_sol_1d_pressure(x, bc_left, bc_right, pressure_gradient)
        self.grid["analytical_velocity"] = analytical
        self.boundary_conditions["bc_left"] = bc_left
        self.boundary_conditions["bc_right"] = bc_right

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
    kinematic_viscosity = 0.00000105  # water
    pressure_gradient = -1e-1

    # Make our initial boy
    grid = Grid(rho, kinematic_viscosity)

    length = 1
    nodes = 100
    initial_velocity = 1
    # bc_velocity = 0
    grid.make_grid(length, nodes, initial_velocity)
    # |--------do simulations--------|
    """ first exercise wall boundary"""
    grid.basic_velocity_CDM(3,0, pressure_gradient)
    grid.plot(True)