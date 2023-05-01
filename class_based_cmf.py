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
        self.grid = None
        self.number_of_nodes = None

    def make_grid(self, length, nodes, initial_velocity):
        self.number_of_nodes = nodes
        boundary_node_position = (length/nodes)/2
        position_array = np.linspace(3*boundary_node_position,length-3*boundary_node_position, nodes-2)
        print(f"lol {position_array}")
        grid = GridPoint(initial_velocity, boundary_node_position, True)
        last_grid = GridPoint(initial_velocity, length - boundary_node_position, True)
        grid.next_gridpoint = last_grid
        last_grid.previous_gridpoint = grid
        self.grid = grid
        for node_position in position_array:
            current_grid = GridPoint(node_position, node_position)

            # fix the relation between previous grid point
            grid.next_gridpoint = current_grid
            current_grid.previous_gridpoint = grid
            # grid.previous_velocity = first_grid
            # first_grid.next_gridpoint = grid

            # fix the relation between the end of the grid point and the current one
            last_grid.previous_gridpoint = current_grid
            current_grid.next_gridpoint = last_grid

            grid = current_grid  # rename current grid to this

    def get_velocities(self):
        velocities = np.zeros(self.number_of_nodes)
        grid_point = self.grid
        velocities[0] = grid_point.velocity
        for i in range(self.number_of_nodes-1):
            grid_point = grid_point.next_gridpoint
            velocities[i+1] = grid_point.velocity

        return velocities




if __name__ == "__main__":
    rho = 1000  # water
    kinematic_viscosity = 0.00105  # water

    # Make our initial boy
    grid = Grid(rho, kinematic_viscosity)

    length = 1
    nodes = 5
    initial_velocity = 1
    # bc_velocity = 0
    grid.make_grid(length, nodes, initial_velocity)
    print(grid.get_velocities())