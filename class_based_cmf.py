import numpy as np


class GridPoint:
    def __init__(self, velocity, position, boundary=False):
        self.velocity = velocity
        self.previous_velocity = -1
        self.position = position
        self.previous_gridpoint = None
        self.next_gridpoint = None
        self.boundary = boundary

    def do_iteration(self):
        self.previous_velocity = self.velocity
        self.velocity = (self.previous_gridpoint.previous_velocity + self.next_gridpoint.velocity)/2

class Grid:

    def __init__(self, rho, kinematic_viscosity):
        self.rho = rho
        self.kinematic_viscosity = kinematic_viscosity
        # self.number_of_nodes = number_of_nodes

    def make_grid(self, length, nodes, initial_velocity):
        boundary_node_position = length/nodes/2
        position_array = np.linspace(boundary_node_position,length-boundary_node_position, nodes)
        for node_position in position_array:
            if node_position == boundary_node_position:

            elif node_position == length-boundary_node_position:

            else:

if __name__ == "__main__":
    rho = 1000  # water
    kinematic_viscosity = 0.00105  # water

    # Make our initial boy
    grid = Grid(rho, kinematic_viscosity)