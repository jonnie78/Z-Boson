# -*- coding: utf-8 -*-
"""
Title: PHYS20161 – 2nd Assignment: Z0 boson

The function of this python script is to read in and validate data
gathered from two detectors that measure the cross-section of 
electron-positron collisions at different centre of mass energies.
From this data, the script calculates the mass, width and lifetime of
the Z0 boson, as well as the uncertainties on these values, by performing
a minimised chi-squared fit. The script also calculates the reduced chi-squared
of the fit, as well as producing plots of the data and the fit, the 
residuals and a contour plot to visualise the uncertainties.

b60879jm (11110928)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import hbar, electron_volt

FILE_NAME_1 = "z_boson_data_1.csv"
FILE_NAME_2 = "z_boson_data_2.csv"
PARTIAL_WIDTH = 83.91e-3
NATURAL_UNITS_CONVERSION = 0.3894e6

def read_data(file_name):
    """
    Reads in data file and sorts data by removing any data points in the
    second column that are not numbers, are negative or equal to zero,
    are more than 3 standard deviations away from the mean or 
    greater than the subsequent data point by 0.5.

    Parameters
    ----------
    file_name : string

    Returns
    -------
    np.ndarray

    """
    data = np.genfromtxt(file_name, delimiter=",", skip_header=1)
    data = data[~np.any(np.isnan(data), axis=1)]
    data = data[~np.any(data <= 0, axis=1)]
    mean = np.mean(data[:,1])
    std = np.std(data[:,1])
    data = data[(np.abs(mean - data[:,1]) <= (3 * std)), :]
    data = data[~np.append(np.abs(np.diff(data[:, 1])) > 0.5, False)]
    if np.size(data) == 0:
        return print("No valid data")
    return data

def sort_data(data_1, data_2):
    """
    Combines the two data sets and then extracts the centre of
    mass energy, cross section and uncertainty data from the combined
    data set.

    Parameters
    ----------
    data_1 : np.ndarray
    data_2 : np.ndarray

    Returns
    -------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    uncertainty : np.ndarray

    """
    data = np.vstack((data_1,data_2))
    centre_of_mass_energy = data[:,0]
    cross_section = data[:,1]
    uncertainty = data[:,2]
    return centre_of_mass_energy, cross_section, uncertainty

def cross_section_function(centre_of_mass_energy, mass, width,
                           partial_width = PARTIAL_WIDTH, conversion = NATURAL_UNITS_CONVERSION):
    """
    Defines the cross section of the interaction that creates
    a Z boson as a function of its mass, width and the different
    centre of mass energies of the colliding beams.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    mass : float
    width : float
    partial_width : float
    conversion : float

    Returns
    -------
    np.ndarray

    """
    natural_value=((12*np.pi/mass**2) * ((centre_of_mass_energy**2) /
    (((centre_of_mass_energy**2-mass**2)**2) + (mass**2*width**2)))) * partial_width**2
    return natural_value * conversion

def minimised_chi_squared_fit(centre_of_mass_energy, cross_section, uncertainty):
    """
    Performs a minimised chi squared fit by varying the Z boson mass
    and width simultaneously, and then returns the values of the mass 
    and width that resulted in the minimised chi squared.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    uncertainty : np.ndarray

    Returns
    -------
    parameters : list

    """
    initial_guess = [90, 3]
    parameters,_ = curve_fit(cross_section_function, centre_of_mass_energy, cross_section,
                              p0=initial_guess, sigma=uncertainty,)
    return parameters

def reduced_chi_squared_calculation(centre_of_mass_energy, cross_section, uncertainty, mass, width):
    """
    Calculates the reduced chi squared of the fitted function.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    uncertainty : np.ndarray
    mass : float
    width : float

    Returns
    -------
    value : float

    """
    predicition = cross_section_function(centre_of_mass_energy, mass, width)
    weighted_chi_squared = np.sum((cross_section - predicition)**2 / uncertainty**2)
    value = weighted_chi_squared / (len(cross_section) - 2)
    return value

def lifetime_calculation(width):
    """
    Calculates the life time of the Z boson using the value obtained
    for it's width.

    Parameters
    ----------
    width : float

    Returns
    -------
    value : float

    """
    value =  hbar / (width * 1e9 * electron_volt)
    return value

def chi_squared_calculation(centre_of_mass_energy, cross_section, uncertainty, mass, width):
    """
    Calculates the weighted chi squared of the function when compared
    with the data.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    uncertainty : np.ndarray
    mass : float
    width : float

    Returns
    -------
    value : float

    """
    predicition = cross_section_function(centre_of_mass_energy, mass, width)
    value = np.sum((cross_section - predicition)**2 / uncertainty**2)
    return value

def create_mesh_grid(centre_of_mass_energy, cross_section, uncertainty, mass, width):
    """
    Creates a mesh array of chi squared values with which 
    a contour plot can be created.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    uncertainty : np.ndarray
    mass : float
    width : float

    Returns
    -------
    mass_values : np.ndarray
    width_values : np.ndarray
    chi_squared_mesh : np.ndarray

    """
    mass_values = np.linspace(mass - 0.05, mass + 0.05, 100)
    width_values = np.linspace(width - 0.05, width + 0.05, 100)
    mass_mesh, _ = np.meshgrid(mass_values, width_values)
    chi_squared_mesh = np.zeros_like(mass_mesh)
    for i in range(100):
        for j in range(100):
            chi_squared_mesh[i, j] = chi_squared_calculation(centre_of_mass_energy, cross_section,
                                                             uncertainty,
                                                             mass_values[i], width_values[j])
    return chi_squared_mesh

def main_plot(centre_of_mass_energy, cross_section, uncertainty, mass, width):
    """
    Creates a graph that contains a plot of the recorded cross section data
    with uncertainties against centre of mass energy and a plot of the function 
    for the cross section, using the values for the mass and width of the Z boson
    obtained from the minimisation.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    uncertainty : np.ndarray
    mass : float
    width : float

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ax_1 = fig.add_subplot(111)
    new_energy = np.linspace(centre_of_mass_energy.min(), centre_of_mass_energy.max(),
                             len(centre_of_mass_energy))
    function = cross_section_function(new_energy, mass, width)
    ax_1.plot(new_energy, function, label = "Fitted Function", c = "r")
    ax_1.errorbar(centre_of_mass_energy, cross_section,
                 yerr = uncertainty, label = "Raw data", ecolor ="g", linestyle = "None")
    ax_1.set_xlabel("Centre of Mass Energy (GeV)")
    ax_1.set_ylabel("Cross-Section (nb)")
    plt.title("Plot of raw data and fitted function")
    plt.savefig("Plot of raw data.png")
    plt.legend()
    plt.show()

def residuals_plot(centre_of_mass_energy, cross_section, mass, width):
    """
    Creates a plot of the residuals of the fitted function
    when its compared to the actual data.

    Parameters
    ----------
    centre_of_mass_energy : np.ndarray
    cross_section : np.ndarray
    mass : float.
    width : float

    Returns
    -------
    None.

    """
    fig = plt.figure()
    ax_2 = fig.add_subplot(111)
    predicition = cross_section_function(centre_of_mass_energy, mass, width)
    residuals = cross_section - predicition
    ax_2.scatter(centre_of_mass_energy, residuals, s=10)
    ax_2.plot(centre_of_mass_energy, (centre_of_mass_energy * 0), c = "red")
    ax_2.set_xlabel("Centre of Mass Energy (GeV)")
    ax_2.set_ylabel("Residuals")
    plt.title("Plot of Residuals")
    plt.savefig("Plot of residuals.png")
    plt.show()

def error_ellipse_and_uncertainty(mass, width,chi_squared_mesh, minimum_chi_squared):
    """
    Creates a contour plot of the error ellipse which
    represents the minimum chi squared value plus one,
    as well as calculating the uncertainties on both the mass 
    and width of the Z boson
    

    Parameters
    ----------
    mass : float
    width : float.
    mass_values : np.ndarray
    width_values : np.ndarray
    chi_squared_mesh : np.ndarray
    minimum_chi_squared : float

    Returns
    -------
    mass_uncertainty : float
    width_uncertainty : float

    """
    fig = plt.figure()
    ax_3 = fig.add_subplot(111)
    mass_values = np.linspace(mass - 0.05, mass + 0.05, 100)
    width_values = np.linspace(width - 0.05, width + 0.05, 100)
    contour = plt.contour(mass_values, width_values,
                          chi_squared_mesh, levels=[minimum_chi_squared + 1])
    paths = contour.collections[0].get_paths()
    mass_points = []
    width_points= []
    for path in paths:
        mass_points.extend(path.vertices[:, 0])
        width_points.extend(path.vertices[:, 1])
    min_mass = min(mass_points)
    max_mass = max(mass_points)
    min_width = min(width_points)
    max_width = max(width_points)
    ax_3.contour(contour)
    ax_3.axvline(x = max_mass, linestyle = "dashed")
    ax_3.axvline(x = min_mass, linestyle = "dashed")
    ax_3.axhline(y = max_width, linestyle = "dashed")
    ax_3.axhline(y = min_width, linestyle = "dashed")
    ax_3.scatter(mass, width, color='red', label='Minimum Chi-Squared')
    ax_3.set_xlabel('Mass')
    ax_3.set_ylabel('Width')
    plt.title('Contour Plot of Chi-Squared + 1')
    plt.savefig("Contour plot.png")
    plt.legend()
    plt.show()
    mass_uncertainty = (max_mass - min_mass) / 2
    width_uncertainty = (max_width - min_width) / 2
    return mass_uncertainty, width_uncertainty

def main(file_name_1, file_name_2):
    """
    Executes the functions above and prints the values obtained
    for the Z boson mass and width with uncertainties as well as
    the values calculated for the reduced chi squared and lifetime
    of the Z boson

    Parameters
    ----------
    file_name_1 : string
    file_name_2 : string

    Returns
    -------
    None.

    """
    z_boson_data_1 = read_data(file_name_1)
    z_boson_data_2 = read_data(file_name_2)
    centre_of_mass_energy, cross_section, uncertainty = sort_data(z_boson_data_1, z_boson_data_2)
    z_boson_mass, z_boson_width = minimised_chi_squared_fit(centre_of_mass_energy,
                                                            cross_section, uncertainty)
    reduced_chi_squared = reduced_chi_squared_calculation(centre_of_mass_energy, cross_section,
                                                          uncertainty, z_boson_mass, z_boson_width)
    lifetime = lifetime_calculation(z_boson_width)
    minimum_chi_squared = chi_squared_calculation(centre_of_mass_energy, cross_section,
                                                  uncertainty, z_boson_mass, z_boson_width)
    chi_squared_mesh = create_mesh_grid(centre_of_mass_energy,
                                                                   cross_section, uncertainty,
                                                                   z_boson_mass, z_boson_width)
    main_plot(centre_of_mass_energy, cross_section, uncertainty, z_boson_mass, z_boson_width)
    residuals_plot(centre_of_mass_energy, cross_section, z_boson_mass, z_boson_width)
    mass_uncertainty, width_uncertainty = error_ellipse_and_uncertainty(z_boson_mass,
                        z_boson_width,chi_squared_mesh,
                        minimum_chi_squared)
    print(f"Mass: {z_boson_mass:.4g} ± {mass_uncertainty:.2g} GeV/c^2")
    print(f"Width: {z_boson_width:.3f} ± {width_uncertainty:.2g} GeV")
    print(f"Life time: {lifetime:.3g} ± {(width_uncertainty/z_boson_width)*lifetime:.2g} Seconds")
    print(f"Reduced Chi Squared: {reduced_chi_squared:.3f}")

main(FILE_NAME_1, FILE_NAME_2)
