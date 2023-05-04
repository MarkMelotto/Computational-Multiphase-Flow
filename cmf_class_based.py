import numpy as np
import matplotlib.pyplot as plt

class Grid:

    def __init__(self, rho, kinematic_viscosity):
        self.rho = rho
        self.kinematic_viscosity = kinematic_viscosity
        self.dynamic_viscosity = kinematic_viscosity*rho
        self.grid = None
        self.number_of_nodes = None

    def make_grid(self, length, nodes, initial_velocity):
        self.number_of_nodes = nodes
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

    def basic_velocity_CDM(self, bc_left, bc_right, pressure_gradient):
        A = -2*np.eye(self.number_of_nodes,self.number_of_nodes)\
            + np.eye(self.number_of_nodes,self.number_of_nodes, k=1) \
            + np.eye(self.number_of_nodes,self.number_of_nodes, k=-1)

        A /= length/self.number_of_nodes
        A[0,0] = -3
        A[-1,-1] = -3

        b = 2*np.ones(self.number_of_nodes)*pressure_gradient/(self.dynamic_viscosity * self.rho)
        b[0] = bc_left
        b[-1] = bc_right
        velocity = np.linalg.solve(A,b)
        self.grid['velocity'] = velocity

        x = self.get_position()
        analytical = -(1/self.dynamic_viscosity)*pressure_gradient/2 * (x*(1-x))
        self.grid["analytical_velocity"] = analytical

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
            plt.legend()
        plt.xlabel("x")
        plt.ylabel("velocity")
        plt.grid()
        plt.show()






if __name__ == "__main__":
    rho = 1000  # water
    kinematic_viscosity = 0.00105  # water
    pressure_gradient = -1

    # Make our initial boy
    grid = Grid(rho, kinematic_viscosity)

    length = 1
    nodes = 1000
    initial_velocity = 1
    # bc_velocity = 0
    grid.make_grid(length, nodes, initial_velocity)
    # |--------do simulations--------|
    """ first exercise wall boundary"""
    grid.basic_velocity_CDM(0,0, pressure_gradient)
    grid.plot(True)