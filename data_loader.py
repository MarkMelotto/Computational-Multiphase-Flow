import numpy as np
import matplotlib.pyplot as plt
import os

def load_data_different_rho(rho, POI, pipe):
    directory = os.getcwd()
    folder = "p" + str(rho)
    x_axis = np.load(directory + "\\tests\\multiphase_tests\\" + pipe + "\\" + folder + "\\height.npy")
    data_fluid = np.load(directory + "\\tests\\multiphase_tests\\" + pipe + "\\" + folder + "\\u_center.npy")
    data_particles = np.load(directory + "\\tests\\multiphase_tests\\" + pipe + "\\" + folder + "\\u_center_2.npy")

    plt.plot(x_axis, data_fluid[:,-POI], ls='--', label=f"Continuous Phase, ρ = {rho} kg/m³")
    plt.plot(x_axis, data_particles[:,-POI], label=f"Dispersed Phase, ρ = {rho} kg/m³")
    print(f"Mean velocity of the continuous phase = {np.mean(data_fluid[:,-POI]):.4f}, ρ = {rho} kg/m³")
    print(f"Mean velocity of the dispersed phase = {np.mean(data_particles[:,-POI]):.4f}, ρ = {rho} kg/m³")


def load_data_different_a2(a2, POI, pipe):
    directory = os.getcwd()
    folder = "a00" + str(a2)
    x_axis = np.load(directory + "\\tests\\multiphase_tests\\" + pipe + "\\" + folder + "\\height.npy")
    data_fluid = np.load(directory + "\\tests\\multiphase_tests\\" + pipe + "\\" + folder + "\\u_center.npy")
    data_particles = np.load(directory + "\\tests\\multiphase_tests\\" + pipe + "\\" + folder + "\\u_center_2.npy")

    plt.plot(x_axis, data_fluid[:, -POI], ls='--', label=f"Continuous Phase, a_2 = 0.0{a2}")
    plt.plot(x_axis, data_particles[:, -POI], label=f"Dispersed Phase, a_2 = 0.0{a2}")
    print(f"Mean velocity of the continuous phase = {np.mean(data_fluid[:,-POI]):.4f}, a_2 = 0.0{a2}")
    print(f"Mean velocity of the dispersed phase = {np.mean(data_particles[:,-POI]):.4f}, a_2 = 0.0{a2}")

def load_data_dredging(adress, POI, label):
    directory = os.getcwd()
    x_axis = np.load(directory + "\\tests\\multiphase_tests\\dredging\\" + adress + "\\height.npy")
    data_fluid = np.load(directory + "\\tests\\multiphase_tests\\dredging\\" + adress + "\\u_center.npy")
    data_particles = np.load(directory + "\\tests\\multiphase_tests\\dredging\\" + adress + "\\u_center_2.npy")

    plt.plot(x_axis, data_fluid[:, -POI], ls='--', label=f"Continuous Phase, {label}")
    plt.plot(x_axis, data_particles[:, -POI], label=f"Dispersed Phase, {label}")
    print(f"Mean velocity of the continuous phase = {np.mean(data_fluid[:, -POI]):.4f}, {label}")
    print(f"Mean velocity of the dispersed phase = {np.mean(data_particles[:, -POI]):.4f}, {label}")


if __name__ == "__main__":

    POI = 5
    # pipe = "larger_pipe"
    pipe = "smaller_pipe_fixed gravity"
    load_data_different_rho(1602, POI, pipe)
    load_data_different_rho(2602, POI, pipe)
    load_data_different_rho(3602, POI, pipe)

    plt.title("Effects of gravity for different dispersed densities")
    plt.xlabel("Width (m)")
    plt.legend()
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.show()

    # load_data_different_a2(1, POI, pipe)
    # load_data_different_a2(3, POI, pipe)
    # load_data_different_a2(5, POI, pipe)
    #
    # plt.title("Effects of gravity for different dispersed volume fractions")
    # plt.xlabel("Width (m)")
    # plt.legend()
    # plt.ylabel("Velocity (m/s)")
    # plt.grid()
    # plt.show()
    #
    # adress_normal = "normal"
    # adress_14 = "jet 1.4\\3.3cm"
    # adress_18 = "jet 1.8\\3.3cm"
    #
    # label_normal = "no jets"
    # label_14 = "jet inlet = 1.4 m/s"
    # label_18 = "jet inlet = 1.8 m/s"
    #
    # load_data_dredging(adress_normal, POI, label_normal)
    # load_data_dredging(adress_14, POI, label_14)
    # load_data_dredging(adress_18, POI, label_18)
    #
    # plt.title("Effects of different jet velocities")
    # plt.xlabel("Width (m)")
    # plt.legend()
    # plt.ylabel("Velocity (m/s)")
    # plt.grid()
    # plt.show()