import numpy as np

class Grid:

    def __init__(self, rho, kinematic_viscosity):
        self.rho = rho
        self.kinematic_viscosity = kinematic_viscosity
        self.grid = None
        self.number_of_nodes = None

    def make_grid(self, length, nodes, initial_velocity):
        self.number_of_nodes = nodes
        boundary_node_position = (length/nodes)/2
        position_array = np.linspace(boundary_node_position,length-boundary_node_position, nodes)
        gridpoints = np.zeros((), dtype=[
        ("velocity", float, nodes),
        ("position", float, nodes),
        ])
        for i,position in enumerate(position_array):
            gridpoints["velocity"][i] = initial_velocity
            gridpoints["position"][i] = position

        self.grid = gridpoints

    def basic_velocity_CDM(self, bc_left, bc_right):
        A = -2*np.eye((self.number_of_nodes,self.number_of_nodes))\
            + np.eye((self.number_of_nodes,self.number_of_nodes), k=1) \
            + np.eye((self.number_of_nodes,self.number_of_nodes), k=-1)
        print(A)
        A[0,0] = bc_left
        A[-1,-1] = bc_right
        print(A)

    def get_velocities(self):
        return self.grid["velocity"]

    def get_position(self):
        return self.grid["position"]




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
    # |--------do simulations--------|
    grid.basic_velocity_CDM(-3,-3)


    # |--------do plots--------|
    x = grid.get_position()
    velocities = grid.get_velocities()