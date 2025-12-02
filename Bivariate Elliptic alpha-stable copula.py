import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad

alpha = 2
rho = 0.5
sigma_1 = 1
sigma_2 = 0.5
L = 10
N = 500

a = -L
b = L

def sum_prime(N,summand):
    return 0.5 * summand(0) + sum(summand(index) for index in range(1,N))

def multivariate_characteristic_function(t_1, t_2):
    return np.exp(-(sigma_1**2 * t_1**2 + 2 * rho * sigma_1 * sigma_2 * t_1 * t_2 + sigma_2**2 * t_2**2)**(alpha/2))
def univariate_characteristic_function_1(t):
    return multivariate_characteristic_function(t, 0)
def univariate_characteristic_function_2(t):
    return multivariate_characteristic_function(0, t)

def univariate_characteristic_function_normal(t, sigma):
    return np.exp(-0.5 * sigma**2 * t**2)
def univariate_characteristic_function_alpha_stable(t, sigma):
    return np.exp(-(sigma**2 * t**2)**(alpha/2))


# PLotting the univariate characteristic functions
t_values = np.linspace(a,b,1000)

plt.plot(t_values, univariate_characteristic_function_1(t_values), label = "Numeric marginal cf", color = 'r')
plt.plot(t_values, univariate_characteristic_function_normal(t_values, sigma_1), label = "Normal cf", color = 'b')
plt.plot(t_values, univariate_characteristic_function_alpha_stable(t_values, sigma_1), label = "alpha-stable cf", color = 'g', linestyle=':')
plt.title(f"Univariate Characteristic Functions for sigma = {sigma_1}")
plt.legend()
plt.show()

# plt.plot(t_values, univariate_characteristic_function_2(t_values), label = 'numeric', color = 'r')
# plt.plot(t_values, univariate_characteristic_function_normal(t_values, sigma_2), label = "real", color = 'b')
# plt.title(f"Univariate Characteristic Functions for sigma = {sigma_2}")
# plt.legend()
# plt.show()

def univariate_density_COS(x,phi, N):
    summand = lambda k: 2/(b-a) * np.real(phi(k * np.pi/(b-a)) * np.exp(-1j * k * np.pi * a/(b-a))) * np.cos(k * np.pi * (x-a)/(b-a))
    return sum_prime(N, summand)

# PLotting the univariate density functions.
t_values = np.linspace(a,b,1000)

plt.plot(t_values, univariate_density_COS(t_values, univariate_characteristic_function_1, N), label = f"Numeric marginal pdf", color = 'b')
plt.plot(t_values, norm.pdf(t_values, loc=0, scale=sigma_1), label = f"Normal pdf", color = 'r')
plt.title(f"Univariate Density Functions for sigma = {sigma_1}")
plt.legend()
plt.show()

# plt.plot(t_values, univariate_density_COS(t_values, univariate_characteristic_function_2, N), label = f"Numeric pdf", color = 'b')
# plt.plot(t_values, norm.pdf(t_values, loc=0, scale=sigma_2), label = f"Real pdf", color = 'r')
# plt.title(f"Univariate Density Functions for sigma = {sigma_2}")
# plt.legend()
# plt.show()

#test for prob mass outside of truncation interval
pdf = lambda x: univariate_density_COS(x, univariate_characteristic_function_1, N)
print(f"prob_mass inside [a,b] = {quad(pdf, a, b)[0]}")
print(f"estimated integration error = {quad(pdf, a, b)[1]}")
print("")

print(f"prob mass outside [a,b] = {quad(pdf, -100, a)[0] + quad(pdf, b, 100)[0]}")
print(f"estimated integration error = {quad(pdf, -100, a)[1] +  + quad(pdf, b, 100)[1]}")

# print(quad(pdf, -100, a))