import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, dblquad
from scipy.optimize import newton, brentq
from statsmodels.distributions.copula.api import GaussianCopula
from scipy.fftpack import dct, dctn, dst
from scipy.interpolate import interpn, interp1d, RegularGridInterpolator

import cProfile
import pstats

def sum_prime(N,summand):
    return 0.5 * summand(0) + sum(summand(index) for index in range(1,N))
def double_sum_prime(N_1,N_2,summand):
    inner_sum = lambda k_1: sum_prime(N_2,lambda k_2: summand(k_1,k_2))
    return sum_prime(N_1,inner_sum)

def multivariate_characteristic_function(t_1, t_2):
    # Kan ook met factor 0.5 erbij in de exponent! dan komt helemaal overeen met normale verdeling voor alpha=2
    return np.exp(- (sigma_1**2 * t_1**2 + 2 * rho * sigma_1 * sigma_2 * t_1 * t_2 + sigma_2**2 * t_2**2)**(alpha/2))
def univariate_characteristic_function_1(t):
    return multivariate_characteristic_function(t, 0)
def univariate_characteristic_function_2(t):
    return multivariate_characteristic_function(0, t)

def univariate_density_gridpoints(phi,N):
    k_values = np.arange(N)
    A = lambda k: 2 / (b - a) * np.real(phi(k * np.pi / (b - a)) * np.exp(-1j * k * np.pi * a / (b - a)))
    density_gridpoints = dct(A(k_values), type=3) / 2
    # return x_gridpoints, density_gridpoints
    return density_gridpoints

def F_plusminus(k_1, k_2,plusminus): # Here plusminus should be equal to either +1 or -1
    return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real(multivariate_characteristic_function(k_1*np.pi/(b_1-a_1), plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)))
def multivariate_density_gridpoints(N):
    k_values = np.arange(N)
    F = lambda k_1, k_2: 1 / 2 * (F_plusminus(k_1, k_2, plusminus=+1) + F_plusminus(k_1, k_2, plusminus=-1))
    density_gridpoints = dctn(F(k_values.reshape(N, 1), k_values.reshape(1, N)), type=3) / 4
    # return np.meshgrid(x_gridpoints, y_gridpoints), density_gridpoints
    return density_gridpoints

def univariate_cdf_gridpoints(phi, N):
    k_values = np.arange(1,N+1)
    A = lambda k: 2 / (b - a) * np.real(phi(k * np.pi / (b - a)) * np.exp(-1j * k * np.pi * a / (b - a)))
    coefficients = lambda k: (b - a) / (k * np.pi) * A(k)
    dst_coefficients = coefficients(k_values)
    summation_gridpoints = dst(coefficients(k_values), type=3) / 2
    linear_term = (x_gridpoints - a) / 2 * A(0)
    cdf_gridpoints = linear_term + summation_gridpoints
    return cdf_gridpoints

def quantile(u,cdf): # geschreven met behulp van gemini
    x_grid = np.linspace(a, b, 200)
    cdf_grid = cdf(x_grid)
    inverse_interpolator = interp1d(cdf_grid, x_grid, kind='linear', bounds_error=False, fill_value="extrapolate")
    x0 = inverse_interpolator(u)

    try:
        return newton(
            func=lambda x: cdf(x) - u,
            x0=x0,

            # onderstaande regel met de afgeleide weglaten zorgt ervoor dat
            # de secant method gebruikt wordt ipv de newton method

            # fprime=lambda x: univariate_density_COS(x, phi, N),
            maxiter=50,  # Should converge in < 5 iters now
            tol=1e-5  # Adjust precision as needed
        )
    except RuntimeError:
        # If Newton fails (e.g., in deep tails where density ~ 0),
        # fall back to the interpolation guess, which is usually decent.
        print("Warning: Newton failed to converge for some points. Returning interpolated guess.")

        # dit lijkt niet te werken? Mijn schript werkt niet meer goed als deze error voorkomt
        # Ik heb nog niet goed gekeken naar wat hier precies fout gaat

        return x0

def copula_density(u, v):
    u_quantile = quantile(u, marginal_cdf_x)
    v_quantile = quantile(v, marginal_cdf_y)
    quantiles = np.stack((u_quantile, v_quantile), axis=-1)
    return joint_density(quantiles) / (marginal_density_x(u_quantile) * marginal_density_y(v_quantile))

def contour_plot_COS(points, epsilon):
    u_values = v_values = np.linspace(epsilon, 1 - epsilon, points)
    U, V = np.meshgrid(u_values, v_values)
    W = copula_density(U, V)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(U, V, W, levels=20, cmap='viridis', extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label('Copula Density Value')
    plt.title(f"Copula density for alpha = {alpha:.1f}, L = {L}, N = {N_marginal}, M = {N_joint}")
    plt.xlabel('U = F^-1_X (X)')
    plt.ylabel('V = F^-1_Y (Y)')
    plt.show()
def contour_plot_Gaussian(points, epsilon):
    u_values = v_values = np.linspace(epsilon, 1 - epsilon, points)
    U, V = np.meshgrid(u_values, v_values)
    pos = np.dstack((U, V))
    datapoints = pos.reshape(-1, 2)
    W = gaussian_copula.pdf(datapoints).reshape(U.shape)

    plt.figure(figsize=(10, 8))
    contour = plt.contourf(U, V, W, levels=20, cmap='viridis', extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label('Copula Density Value')
    plt.title(f"Contour Plot of Gaussian copula density")
    plt.xlabel('U = F^-1_X (X)')
    plt.ylabel('V = F^-1_Y (Y)')
    plt.show()

def upper_tail_coefficient(u):
    coefficient = dblquad(copula_density, u, 1, u, 1)[0] / (1 - u)
    print(f"lambda_upper ({u}) = {dblquad(copula_density, u, 1, u, 1)[0]}")
    return dblquad(copula_density, u, 1, u, 1)[0] / (1 - u)

def plot_tail_coefficient():
    u_values = [0.9, 0.95, 0.97, 0.99]
    tail_coefficients = []
    for u in u_values:
        tail_coefficients.append(upper_tail_coefficient(u))
    tail_coefficient_estimate = tail_coefficients[-1]
    plt.plot(u_values, tail_coefficients, label = f"Tail dependence parameter", color = 'b')
    plt.plot(u_values, [tail_coefficient_estimate] * len(u_values), label = f"Limit = {tail_coefficient_estimate:.2f}", color = 'r', linestyle = '--')
    plt.title(f"Estimate of tail dependence parameter for alpha = {alpha}")
    plt.ylim(0,1)
    plt.xlabel('u')
    plt.ylabel('lambda_upper (u)')
    plt.legend()
    plt.show()

with cProfile.Profile() as profile:
    for alpha in [1.8, 1.6, 1.4, 1.2, 1.0, 0.8]:
        print("")
        print(f"alpha = {alpha}")

        rho = 0.3
        sigma_1 = 1
        sigma_2 = 0.5
        L = 10
        N_marginal = 10000  # of 128
        N_joint = 4000  # of 48
        epsilon = 1e-2

        a = -L
        b = L
        a_1 = a_2 = a
        b_1 = b_2 = b

        # cov_matrix = [[sigma_1 ** 2, rho * sigma_1 * sigma_2],
        #               [rho * sigma_1 * sigma_2, sigma_2 ** 2]]
        gaussian_copula = GaussianCopula(corr=rho, k_dim=2, allow_singular=False)

        x_index = np.arange(N_marginal)
        y_index = np.arange(N_marginal)
        x_gridpoints = a + (b - a) / N_marginal * (x_index + 0.5)
        y_gridpoints = a + (b - a) / N_marginal * (y_index + 0.5)

        x_index_joint = np.arange(N_joint)
        y_index_joint = np.arange(N_joint)
        x_gridpoints_joint = a_1 + (b_1 - a_1) / N_joint * (x_index_joint + 0.5)
        y_gridpoints_joint = a_2 + (b_2 - a_2) / N_joint * (y_index_joint + 0.5)

        marginal_density_x_gridpoints = univariate_density_gridpoints(univariate_characteristic_function_1, N_marginal)
        marginal_density_y_gridpoints = univariate_density_gridpoints(univariate_characteristic_function_2, N_marginal)
        joint_density_gridpoints = multivariate_density_gridpoints(N_joint)
        marginal_cdf_x_gridpoints = univariate_cdf_gridpoints(univariate_characteristic_function_1, N_marginal)
        marginal_cdf_y_gridpoints = univariate_cdf_gridpoints(univariate_characteristic_function_2, N_marginal)

        marginal_density_x = lambda x: np.interp(x, x_gridpoints, marginal_density_x_gridpoints, left=0, right=0)
        marginal_density_y = lambda y: np.interp(y, y_gridpoints, marginal_density_y_gridpoints, left=0, right=0)
        joint_density = RegularGridInterpolator((x_gridpoints_joint, y_gridpoints_joint),
                                                joint_density_gridpoints.reshape(N_joint, N_joint), method='linear',
                                                bounds_error=False, fill_value=0)
        marginal_cdf_x = lambda x: np.interp(x, x_gridpoints, marginal_cdf_x_gridpoints, left=0, right=1)
        marginal_cdf_y = lambda y: np.interp(y, y_gridpoints, marginal_cdf_y_gridpoints, left=0, right=1)

        #################################################################################
        # Bovenstaande is puur intialiseren voor verschillende waardes van alpha, hieronder wordt alles uitgevoerd wat ik wil doen voor elke alpha

        # plot_tail_coefficient() #deze functie maakt een plotje van de coefficient voor verschillende u, en ook worden de waardes van de coefficient in de console geprint

        contour_plot_COS(100, epsilon)
        # contour_plot_Gaussian(100, epsilon)
        # print(f"double integral of copula density = {dblquad(lambda x, y: copula_density(x, y), epsilon, 1 - epsilon, epsilon, 1 - epsilon)}")


results = pstats.Stats(profile)
results.sort_stats('tottime').print_stats()

