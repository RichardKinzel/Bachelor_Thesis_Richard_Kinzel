import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.optimize import brentq
from statsmodels.distributions.copula.api import GaussianCopula
from scipy.fftpack import dct, dctn, dst
from scipy.interpolate import interp1d, RegularGridInterpolator
from datetime import datetime

import cProfile
import pstats

def seconds(time):
    return time.days * 24 * 60 * 60 + time.seconds + time.microseconds / 1000000

def sum_prime(N,summand):
    return 0.5 * summand(0) + sum(summand(index) for index in range(1,N))
def double_sum_prime(N_1,N_2,summand):
    inner_sum = lambda k_1: sum_prime(N_2,lambda k_2: summand(k_1,k_2))
    return sum_prime(N_1,inner_sum)

def multivariate_characteristic_function(t_1, t_2):
    # Kan ook met factor 0.5 erbij in de exponent! Komt dan overeen met Gaussian voor alpha=2 maar heeft geen invloed op copula
    return np.exp(- (sigma_1**2 * t_1**2 + 2 * rho * sigma_1 * sigma_2 * t_1 * t_2 + sigma_2**2 * t_2**2)**(alpha/2))
def univariate_characteristic_function_1(t):
    return multivariate_characteristic_function(t, 0)
def univariate_characteristic_function_2(t):
    return multivariate_characteristic_function(0, t)

def univariate_density_gridpoints(phi,N):
    k_values = np.arange(N)
    A = lambda k: 2 / (b - a) * np.real(phi(k * np.pi / (b - a)) * np.exp(-1j * k * np.pi * a / (b - a)))
    density_gridpoints = dct(A(k_values), type=3) / 2
    return density_gridpoints

def F_plusminus(k_1, k_2,plusminus): # Here plusminus should be equal to either +1 or -1
    return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real(multivariate_characteristic_function(k_1*np.pi/(b_1-a_1), plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)))
def multivariate_density_gridpoints(N):
    k_values = np.arange(N)
    F = lambda k_1, k_2: 1 / 2 * (F_plusminus(k_1, k_2, plusminus=+1) + F_plusminus(k_1, k_2, plusminus=-1))
    density_gridpoints = dctn(F(k_values.reshape(N, 1), k_values.reshape(1, N)), type=3) / 4
    return density_gridpoints

def univariate_cdf_gridpoints(phi, N):
    k_values = np.arange(1,N+1)
    A = lambda k: 2 / (b - a) * np.real(phi(k * np.pi / (b - a)) * np.exp(-1j * k * np.pi * a / (b - a)))
    coefficients = lambda k: (b - a) / (k * np.pi) * A(k)
    summation_gridpoints = dst(coefficients(k_values), type=3) / 2
    linear_term = (x_gridpoints - a) / 2 * A(0)
    cdf_gridpoints = linear_term + summation_gridpoints
    return cdf_gridpoints

def quantile(u,cdf): # geschreven met behulp van gemini
    if u <= 0:
        return a
    if u >= 1:
        return b
    try:
        return brentq(f=lambda x: cdf(x) - u,a=a,b=b)
    except RuntimeError:
        print("Warning: Brentq failed to converge for some points.")
        return 0

def quantile_gridpoints(cdf, gridpoints):
    quantiles = np.array([])
    for u in gridpoints:
        quantiles = np.append(quantiles, quantile(u, cdf))
    return quantiles

def copula_density(u, v):
    u_clipped = np.clip(u,1e-10, 1 - 1e-10)
    v_clipped = np.clip(v,1e-10, 1 - 1e-10)
    u_quantile = quantile_x(u_clipped)
    v_quantile = quantile_y(v_clipped)
    quantiles = np.stack((u_quantile, v_quantile), axis=-1)

    num = joint_density(quantiles)
    denom = marginal_density_x(u_quantile) * marginal_density_y(v_quantile)
    # Perform division, replacing division-by-zero results with 0.0
    # "out" initializes the result array (zeros in this case)
    # "where" tells it to only divide where denom != 0
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 1e-50)

def contour_plot_COS(points, epsilon):
    u_values = v_values = np.linspace(epsilon, 1 - epsilon, points)
    U, V = np.meshgrid(u_values, v_values)
    W = copula_density(U, V)

    contour = plt.contourf(U, V, W, levels=np.linspace(0,10,21), cmap='viridis', extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label('Copula Density Value')
    plt.title(f"Numeric copula density for alpha = {alpha:.1f}, L = {L}, N = M = {N_marginal}")
    plt.xlabel('U = F_X (X)')
    plt.ylabel('V = F_Y (Y)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def contour_plot_Gaussian(points, epsilon):
    # deze functie geeft een foutmelding als u,v op de rand ligt. Daarom moet epsilon positief zijn
    if epsilon <= 0:
        u_values = v_values = np.linspace(1e-10, 1 - 1e-10, points)
    else:
        u_values = v_values = np.linspace(epsilon, 1 - epsilon, points)
    U, V = np.meshgrid(u_values, v_values)
    pos = np.dstack((U, V))
    datapoints = pos.reshape(-1, 2)
    W = gaussian_copula.pdf(datapoints).reshape(U.shape)

    contour = plt.contourf(U, V, W, levels=np.linspace(0,10,21), cmap='viridis', extend='both')
    cbar = plt.colorbar(contour)
    cbar.set_label('Copula Density Value')
    plt.title(f"Gaussian copula density")
    plt.xlabel('U = F_X (X)')
    plt.ylabel('V = F_Y (Y)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

def upper_tail_coefficient(u, epsilon):
    coefficient = dblquad(lambda x,y: copula_density(x,y), u, 1 - epsilon, u, 1 - epsilon)[0] / (1 - epsilon - u)
    print(f"lambda_upper ({u}) = {coefficient}")
    return coefficient

def plot_tail_coefficient(epsilon):
    u_values = [0.9, 0.95, 0.97, 0.99]
    tail_coefficients = []
    for u in u_values:
        tail_coefficients.append(upper_tail_coefficient(u, epsilon))
    tail_coefficient_estimate = tail_coefficients[-1]
    plt.plot(u_values, tail_coefficients, label = f"Tail dependence parameter", color = 'b')
    plt.plot(u_values, [tail_coefficient_estimate] * len(u_values), label = f"Limit = {tail_coefficient_estimate:.2f}", color = 'r', linestyle = '--')
    plt.title(f"Estimate of tail dependence parameter for alpha = {alpha}")
    plt.ylim(0,1)
    plt.xlabel('u')
    plt.ylabel('lambda_upper (u)')
    plt.legend()
    plt.show()

def theoretical_tail_dependence(rho, alpha):
    def integrand(t):
        return np.cos(t) ** alpha
    lower_limit = (np.pi / 2 - np.arcsin(rho)) / 2
    num = quad(integrand, lower_limit, np.pi / 2)[0]
    den = quad(integrand, 0, np.pi / 2)[0]

    tail_coefficient = num / den

    return tail_coefficient

rho = 0.3
sigma_1 = 1
sigma_2 = 0.5
L = 10
N_marginal = 10000
N_joint = 10000
epsilon = 1e-10

a = -L
b = L
a_1 = a_2 = a
b_1 = b_2 = b

gaussian_copula = GaussianCopula(corr=rho, k_dim=2, allow_singular=False)

x_index = np.arange(N_marginal)
y_index = np.arange(N_marginal)
x_gridpoints = a + (b - a) / N_marginal * (x_index + 0.5)
y_gridpoints = a + (b - a) / N_marginal * (y_index + 0.5)

x_index_joint = np.arange(N_joint)
y_index_joint = np.arange(N_joint)
x_gridpoints_joint = a_1 + (b_1 - a_1) / N_joint * (x_index_joint + 0.5)
y_gridpoints_joint = a_2 + (b_2 - a_2) / N_joint * (y_index_joint + 0.5)

# profiler meet welke functies het meeste tijd kosten. Heeft geen  invloed op functionaliteit van code, dus kan weggehaald worden
profiler = cProfile.Profile()
profiler.enable()

for alpha in [2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.8, 0.6, 0.4]:
    # print("")
    # print(f"alpha = {alpha}")

    marginal_density_x_gridpoints = univariate_density_gridpoints(univariate_characteristic_function_1, N_marginal)
    marginal_density_y_gridpoints = univariate_density_gridpoints(univariate_characteristic_function_2, N_marginal)
    joint_density_gridpoints = multivariate_density_gridpoints(N_joint)

    marginal_density_x = interp1d(x_gridpoints, marginal_density_x_gridpoints, kind='linear', fill_value='extrapolate')
    marginal_density_y = interp1d(y_gridpoints, marginal_density_y_gridpoints, kind='linear', fill_value='extrapolate')
    joint_density = RegularGridInterpolator((x_gridpoints_joint, y_gridpoints_joint),
                                            joint_density_gridpoints.reshape(N_joint, N_joint), method='linear',
                                            bounds_error=False, fill_value=None)

    marginal_cdf_x_gridpoints = univariate_cdf_gridpoints(univariate_characteristic_function_1, N_marginal)
    marginal_cdf_y_gridpoints = univariate_cdf_gridpoints(univariate_characteristic_function_2, N_marginal)
    marginal_cdf_x_interpolator = interp1d(x_gridpoints, marginal_cdf_x_gridpoints, kind='linear', fill_value='extrapolate')
    marginal_cdf_y_interpolator = interp1d(y_gridpoints, marginal_cdf_y_gridpoints, kind='linear', fill_value='extrapolate')
    marginal_cdf_x = lambda x: np.clip(marginal_cdf_x_interpolator(x), 0.0, 1.0)
    marginal_cdf_y = lambda y: np.clip(marginal_cdf_y_interpolator(y), 0.0, 1.0)

    u_gridpoints = v_gridpoints = np.linspace(0,1,N_marginal)

    quantile_x_gridpoints = quantile_gridpoints(marginal_cdf_x, u_gridpoints)
    quantile_y_gridpoints = quantile_gridpoints(marginal_cdf_y, v_gridpoints)
    quantile_x = interp1d(u_gridpoints, quantile_x_gridpoints, kind='linear', fill_value='extrapolate')
    quantile_y = interp1d(v_gridpoints, quantile_y_gridpoints, kind='linear', fill_value='extrapolate')

    # print("initializing complete")

    #################################################################################
    # Bovenstaande is puur intialiseren voor verschillende waardes van alpha, hieronder wordt alles uitgevoerd wat ik wil doen voor elke alpha


    # # voor het maken van contour plotjes
    # contour_plot_COS(1000, epsilon)
    # if alpha == 2.0:
    #     contour_plot_Gaussian(1000, epsilon)

    # # voor het uitrekenen van de tail dependence coefficient lambda
    # for epsilon in [1e-4, 1e-3]:
    #     # for u in [0.9, 0.95, 0.97, 0.99]:
    #         start = datetime.now()
    #         coefficient = dblquad(lambda x,y: copula_density(x,y), u, 1 - epsilon, u, 1 - epsilon)[0] / (1 - epsilon - u)
    #         end = datetime.now()
    #         print(f"alpha = {alpha} and epsilon = {epsilon:.0e} : lambda ({u}) = {coefficient:.5f}")
    #         print(f"alpha = {alpha} and epsilon = {epsilon:.0e} : computation time = {seconds(end-start):.5f} seconds")
    #         if alpha == 2.0:
    #             gaussian_coefficient = dblquad(lambda x, y: gaussian_copula.pdf((x, y)), u, 1 - epsilon, u, 1 - epsilon)[0] / (1 - epsilon - u)
    #             print(f"lambda_gaussian ({u}) = {gaussian_coefficient:.5f}")

    # # voor het uitrekenen van de theoretische waarde van lambda
    # print(f"Theoretical tail dependence parameter for alpha {alpha} is {theoretical_tail_dependence(rho, alpha):5f}")

    # voor het uitrekenen van de total probability mass
    for epsilon in [0, 1e-5, 1e-4, 1e-3]:
        start = datetime.now()
        integral = dblquad(lambda x, y: copula_density(x, y), epsilon, 1-epsilon, epsilon, 1-epsilon)[0]
        end = datetime.now()
        print(f"epsilon={epsilon:.0e}: double integral of copula density for alpha {alpha} is: {integral:.5f}")
        print(f"epsilon={epsilon:.0e}: computation time for alpha {alpha} is: {seconds(end - start):.5f} seconds")

# Ook onderstaande kan weggehaald worden. Heeft geen invloed op functionaliteit van code, maar meet hoe lang alle function calls duren.
profiler.disable()
results = pstats.Stats(profiler)
results.sort_stats('tottime').print_stats()