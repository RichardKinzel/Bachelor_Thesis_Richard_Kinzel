import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad, dblquad
from scipy.optimize import newton
from statsmodels.distributions.copula.api import GaussianCopula

alpha = 2
rho = 0.5
sigma_1 = 1
sigma_2 = 0.5
L = 6
N_marginal = 64 # of 128
N_joint = 32 # of 48

a = -L
b = L
a_1 = a_2 = a
b_1 = b_2 = b

cov_matrix = [[sigma_1**2, rho * sigma_1 * sigma_2],
       [rho * sigma_1 * sigma_2, sigma_2**2]]

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

# for checking the functions that we retreived above
# def univariate_characteristic_function_normal(t, sigma):
#     # return np.exp(-0.5 * sigma**2 * t**2)
#     return np.exp(- sigma**2 * t**2)
# def univariate_characteristic_function_alpha_stable(t, sigma):
#     return np.exp(-(sigma**2 * t**2)**(alpha/2))

# PLotting the univariate characteristic functions
# t_values = np.linspace(a,b,1000)
#
# plt.plot(t_values, univariate_characteristic_function_1(t_values), label = "Numeric marginal cf", color = 'r')
# plt.plot(t_values, univariate_characteristic_function_normal(t_values, sigma_1), label = "Normal cf", color = 'b')
# plt.plot(t_values, univariate_characteristic_function_alpha_stable(t_values, sigma_1), label = "alpha-stable cf", color = 'g', linestyle=':')
# plt.title(f"Univariate Characteristic Functions for sigma = {sigma_1}")
# plt.legend()
# plt.show()

def univariate_density_COS(x,phi, N):
    A = lambda k: 2 / (b - a) * np.real(phi(k * np.pi / (b - a)) * np.exp(-1j * k * np.pi * a / (b - a)))
    summand = lambda k: A(k) * np.cos(k * np.pi * (x - a) / (b - a))
    return sum_prime(N, summand)
# PLotting the univariate density functions.
# t_values = np.linspace(a,b,1000)
#
# plt.plot(t_values, univariate_density_COS(t_values, univariate_characteristic_function_1, N_marginal), label = f"Numeric marginal pdf", color = 'b')
# plt.plot(t_values, norm.pdf(t_values, loc=0, scale=sigma_1), label = f"Normal pdf", color = 'r')
# plt.title(f"Univariate Density Functions for sigma = {sigma_1}")
# plt.legend()
# plt.show()

#test for prob mass outside of truncation interval
# pdf = lambda x: univariate_density_COS(x, univariate_characteristic_function_1, N_marginal)
# print(f"prob_mass inside [a,b] = {quad(pdf, a, b)[0]}")
# print(f"estimated integration error = {quad(pdf, a, b)[1]}")
# print("")
#
# print(f"prob mass outside [a,b] = {quad(pdf, -100, a)[0] + quad(pdf, b, 100)[0]}")
# print(f"estimated integration error = {quad(pdf, -100, a)[1] +  + quad(pdf, b, 100)[1]}")

# print(quad(pdf, -100, a))

# the following function is used for calculating the fourier coefficients for the bivariate density function.
def F_plusminus(k_1, k_2,plusminus): # Here plusminus should be equal to either +1 or -1
    return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real(multivariate_characteristic_function(k_1*np.pi/(b_1-a_1), plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)))

def multivariate_density_COS(x,y,N):
    F = lambda k_1, k_2: 1/2 * (F_plusminus(k_1,k_2,plusminus=+1) + F_plusminus(k_1,k_2,plusminus=-1))
    # print(f"F(0,0) = {F(0,0)}")
    # print(f"4/((b_1-a_1)(b_2-a_2)) = {4/((b_1-a_1)*(b_2-a_2))}")
    # print("")
    summand = lambda k_1, k_2: F(k_1, k_2) * np.cos(k_1*np.pi*(x-a_1)/(b_1-a_1)) * np.cos(k_2*np.pi*(y-a_2)/(b_2-a_2))
    return double_sum_prime(N, N, summand)

# print statements in de functie worden uitgevoerd
# test = multivariate_density_COS(0,0,multivariate_characteristic_function, N_joint)


# countourplot maken van multivariate density

# points = 100
# x_values = np.linspace(a_1, b_1, points)
# y_values = np.linspace(a_2, b_2, points)
# X, Y = np.meshgrid(x_values, y_values)
# Z = multivariate_density_COS(X, Y, multivariate_characteristic_function, N_joint)
#
# plt.figure(figsize=(10, 8))
# # contourf fills the area (use plt.contour for lines only)
# # 'levels' determines how many distinct colors/lines there are
# # 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm')
# contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', extend='both')
#
# cbar = plt.colorbar(contour)
# cbar.set_label('Density Value')
# plt.title(f"Contour Plot of probability density approximation for alpha = {alpha}")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()
#
# # plot maken van normale verdeling pdf
# points = 100
# x_values = np.linspace(a_1, b_1, points)
# y_values = np.linspace(a_2, b_2, points)
# X, Y = np.meshgrid(x_values, y_values)
# pos = np.dstack((X, Y))
# normal_rv = multivariate_normal(mean=None, cov=cov_matrix)
# Z = normal_rv.pdf(pos)
#
# plt.figure(figsize=(10, 8))
# # contourf fills the area (use plt.contour for lines only)
# # 'levels' determines how many distinct colors/lines there are
# # 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm')
# contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', extend='both')
# cbar = plt.colorbar(contour)
# cbar.set_label('Density Value')
# plt.title('Contour Plot of normal probability density')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

def univariate_cdf(x, phi, N):
    A = lambda k: 2 / (b - a) * np.real(phi(k * np.pi / (b - a)) * np.exp(-1j * k * np.pi * a / (b - a)))
    summand = lambda k: A(k) * (b - a) / (k * np.pi) * np.sin(k * np.pi * (x - a) / (b - a))
    return (x - a) / 2 * A(0) + sum(summand(k) for k in range(1,N))

# # testing the univariate cdf
# x = np.array([a, -1, 0, 1, b])
# for value in x:
#     print(f"F_X ({value}) = {univariate_cdf(value, univariate_characteristic_function_1, N_marginal):.4}")

def quantile(u,phi):
    x0 = np.full_like(u, 0.5)
    return newton(lambda x: univariate_cdf(x, phi, N_marginal) - u, x0, fprime=lambda x: univariate_density_COS(x, phi, N_marginal))

# # testing the quantile function
# u = np.array([0, 0.25, 0.5, 0.75, 1])
# for value in u:
#     print(f"F^-1_X ({value}) = {quantile(value, univariate_characteristic_function_1):.4}")
# print(f"Array u als input geeft F^-1_X (u) = {quantile(u, univariate_characteristic_function_1)}")

def copula_density(u,v):
    u_quantile = quantile(u,univariate_characteristic_function_1)
    v_quantile = quantile(v,univariate_characteristic_function_2)
    return multivariate_density_COS(u_quantile, v_quantile, N_joint) / (univariate_density_COS(u_quantile, univariate_characteristic_function_1, N_marginal) * univariate_density_COS(v_quantile, univariate_characteristic_function_2, N_marginal))

# for (u,v) in [(0.5, 0.5), (0.5, 0.9), (0.9, 0.5), (0.9, 0.9), (0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.99, 0.99), (0.01, 0.01)]:
#     print(f"c({u}, {v}) = {copula_density(u,v):.4}")

points = 100
epsilon = 1e-2
u_values = v_values = np.linspace(epsilon, 1 - epsilon, points)
U, V = np.meshgrid(u_values, v_values)
W = copula_density(U, V)

plt.figure(figsize=(10, 8))
contour = plt.contourf(U, V, W, levels=20, cmap='viridis', extend='both')
cbar = plt.colorbar(contour)
cbar.set_label('Copula Density Value')
plt.title(f"Contour Plot of copula density approximation for alpha = {alpha}")
plt.xlabel('U = F^-1_X (X)')
plt.ylabel('V = F^-1_Y (Y)')
plt.show()

# # 1. Avoid 0 and 1. Use a small epsilon.
# points = 100
# epsilon = 1e-2
# u_values = v_values = np.linspace(epsilon, 1 - epsilon, points)
#
# # 2. Optimization: Calculate quantiles on 1D arrays first!
# # This reduces Newton solver calls from 10,000 to 200.
# u_quantiles = quantile(u_values, univariate_characteristic_function_1)
# v_quantiles = quantile(v_values, univariate_characteristic_function_2)
#
# # 3. Create the meshgrid based on the calculated quantiles (Input for Density)
# # Q_U corresponds to F^-1(u) and Q_V corresponds to F^-1(v)
# Q_U, Q_V = np.meshgrid(u_quantiles, v_quantiles)
#
# print("Calculating Joint Density...")
#
# # 4. Calculate the Numerator (Joint Density) using the Quantile Grid
# num = multivariate_density_COS(Q_U, Q_V, N_joint)
#
# print("Calculating Marginal Densities...")
#
# # 5. Calculate Denominators (Marginal Densities)
# # We can reuse the 1D calculations and broadcast, or just pass the 2D grid
# den_u = univariate_density_COS(Q_U, univariate_characteristic_function_1, N_marginal)
# den_v = univariate_density_COS(Q_V, univariate_characteristic_function_2, N_marginal)
#
# # 6. Compute Copula Density
# W = num / (den_u * den_v)
#
# # 7. Plotting
# # We use the original u_values, v_values for the axes (0 to 1),
# # but plot the W we calculated from the transformed Q_U, Q_V.
# U_grid, V_grid = np.meshgrid(u_values, v_values)
#
# plt.figure(figsize=(10, 8))
#
# # Copula densities often have high spikes at corners.
# # heavily restricting 'levels' allows you to see the structure in the middle.
# contour = plt.contourf(U_grid, V_grid, W, levels=20, cmap='viridis', extend='both')
#
# cbar = plt.colorbar(contour)
# cbar.set_label('Copula Density Value')
# plt.title(f"Contour Plot of Copula Density (alpha={alpha}, rho={rho})")
# plt.xlabel('u')
# plt.ylabel('v')
# plt.show()





# Gaussian copula density plotten
gaussian_copula = GaussianCopula(corr=rho, k_dim=2, allow_singular=False)

points = 100
u_values = v_values = np.linspace(0, 1, points)
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

# print(f"double integral of copula density = {dblquad(copula_density, epsilon, 1 - epsilon, epsilon, 1 - epsilon)}")