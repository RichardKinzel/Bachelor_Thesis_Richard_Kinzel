import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from scipy.integrate import quad

alpha = 2
rho = 0.5
sigma_1 = 1
sigma_2 = 0.5
L = 6
N_marginal = 64 # of 128
N_joint = 128 # of 48

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
    summand = lambda k: 2/(b-a) * np.real(phi(k * np.pi/(b-a)) * np.exp(-1j * k * np.pi * a/(b-a))) * np.cos(k * np.pi * (x-a)/(b-a))
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

def multivariate_density_COS(x,y,phi, N):
    def F_plusminus(k_1, k_2, plusminus): # Here plusminus should be equal to either +1 or -1
        return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real(phi(k_1*np.pi/(b_1-a_1), plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)))
    F = lambda k_1, k_2: 1/2 * (F_plusminus(k_1,k_2,plusminus=+1) + F_plusminus(k_1,k_2,plusminus=-1))
    print(f"F(0,0) = {F(0,0)}")
    print(f"4/((b_1-a_1)(b_2-a_2)) = {4/((b_1-a_1)*(b_2-a_2))}")
    print("")
    summand = lambda k_1, k_2: F(k_1, k_2) * np.cos(k_1*np.pi*(x-a_1)/(b_1-a_1)) * np.cos(k_2*np.pi*(y-a_2)/(b_2-a_2))
    return double_sum_prime(N, N, summand)

test = multivariate_density_COS(0,0,multivariate_characteristic_function, N_joint)


# countourplot maken van multivariate density

points = 100
x_values = np.linspace(a_1, b_1, points)
y_values = np.linspace(a_2, b_2, points)
X, Y = np.meshgrid(x_values, y_values)
Z = multivariate_density_COS(X, Y, multivariate_characteristic_function, N_joint)

plt.figure(figsize=(10, 8))
# contourf fills the area (use plt.contour for lines only)
# 'levels' determines how many distinct colors/lines there are
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm')
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', extend='both')

cbar = plt.colorbar(contour)
cbar.set_label('Density Value')
plt.title('Contour Plot of probability density approximation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# plot maken van normale verdeling pdf
points = 100
x_values = np.linspace(a_1, b_1, points)
y_values = np.linspace(a_2, b_2, points)
X, Y = np.meshgrid(x_values, y_values)
pos = np.dstack((X, Y))
normal_rv = multivariate_normal(mean=None, cov=cov_matrix)
Z = normal_rv.pdf(pos)

plt.figure(figsize=(10, 8))
# contourf fills the area (use plt.contour for lines only)
# 'levels' determines how many distinct colors/lines there are
# 'cmap' sets the color scheme (e.g., 'viridis', 'plasma', 'coolwarm')
contour = plt.contourf(X, Y, Z, levels=20, cmap='viridis', extend='both')
cbar = plt.colorbar(contour)
cbar.set_label('Density Value')
plt.title('Contour Plot of normal probability density')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()