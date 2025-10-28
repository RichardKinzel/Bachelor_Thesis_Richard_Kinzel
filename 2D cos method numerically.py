#Here we will numerically compute the value of an option using the 2D cos method.

import cProfile
import pstats
import numpy as np
from scipy.integrate import quad, dblquad
from datetime import datetime

def double_integral(lower_limit_1,upper_limit_1,lower_limit_2,upper_limit_2,integrand):
    return dblquad(integrand, lower_limit_1, upper_limit_1, lower_limit_2, upper_limit_2)[0]

# Here we define the payoff function of a basket option at time T, depending on what type of option contract we use.
# def C_T(y_1, y_2):
#     #note that the option value should be max(value,0), but we eliminated the integration interval where value<0
#     return K * (weight_1*np.exp(y_1) + weight_2*np.exp(y_2) - 1)
# def P_T(y_1, y_2):
    #note that the option value should be max(value,0), but we eliminated the integration interval where value<0
    # return -K * (weight_1*np.exp(y_1) + weight_2*np.exp(y_2) - 1)

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u_1,u_2,t):
    #bivariate normal distribution:
    characteristic_value = np.exp(1j*t*(r*(u_1+u_2) - 0.5*(u_1 * sigma_1**2 + u_2 * sigma_2**2)) - 0.5*t*(u_1**2 * sigma_1**2 + 2*u_1*u_2*sigma_1*sigma_2*rho + u_2**2 * sigma_2**2))
    return characteristic_value

# This is the conditional characteristic function
def phi(u_1,u_2):
    return np.exp(1j*(u_1*x_1 + u_2*x_2)) * characteristic_function(u_1,u_2,T)

# cosine Fourier coefficients of payoff function
def H(k_1,k_2,option_type):
    #We eliminated the integration domain where V_T=0 by setting a variable integration limit for the inner integral.
    if option_type == 'call':
        # The following disabled line is more readable, but calling the function C_T made the code 10% slower.
        # integrand = lambda y_1, y_2: C_T(y_1, y_2) * np.cos(k_1 * np.pi * (y_1 - a_1) / (b_1 - a_1)) * np.cos(k_2 * np.pi * (y_2 - a_2) / (b_2 - a_2))
        integrand = lambda y_1, y_2: K * (weight_1 * np.exp(y_1) + weight_2 * np.exp(y_2) - 1) * np.cos(k_1 * np.pi * (y_1 - a_1) / (b_1 - a_1)) * np.cos(k_2 * np.pi * (y_2 - a_2) / (b_2 - a_2))
        return 2 / (b_1 - a_1) * 2 / (b_2 - a_2) * (double_integral(-np.log(weight_2), b_1, a_2, b_2, integrand) +
                                                    double_integral(a_1, -np.log(weight_2), lambda y_2: np.log((1-weight_2*np.exp(y_2))/weight_1), b_2, integrand))
    elif option_type == 'put':
        # The following disabled line is more readable, but calling the function P_T made the code significantly slower.
        # integrand = lambda y_1, y_2: P_T(y_1, y_2) * np.cos(k_1 * np.pi * (y_1 - a_1) / (b_1 - a_1)) * np.cos(k_2 * np.pi * (y_2 - a_2) / (b_2 - a_2))
        integrand = lambda y_1, y_2: -K * (weight_1 * np.exp(y_1) + weight_2 * np.exp(y_2) - 1) * np.cos(k_1 * np.pi * (y_1 - a_1) / (b_1 - a_1)) * np.cos(k_2 * np.pi * (y_2 - a_2) / (b_2 - a_2))
        return 2 / (b_1 - a_1) * 2 / (b_2 - a_2) * double_integral(a_1, -np.log(weight_2), a_2, lambda y_2: np.log((1-weight_2*np.exp(y_2))/weight_1), integrand)
    else:
        raise ValueError('option_type must be put or call')

def F(k_1,k_2,plusminus):
    return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real( phi(k_1*np.pi/(b_1-a_1),plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)) )

# The following two functions calculate a sum and a double sum respectively with the first term(s) halved
def sum_prime(N,summand):
    return 0.5 * summand(0) + sum([summand(index) for index in range(1,N)])

def double_sum_prime(N_1,N_2,summand):
    inner_sum = lambda k_1: sum_prime(N_2,lambda k_2: summand(k_1,k_2))
    return sum_prime(N_1,inner_sum)

# This is our main calculation. In it we use the functions defined above
def COS_formula(N_1,N_2,option_type):
    # option_type should be 'put' or 'call'
    # N_1 and N_2 are the summation limits

    summand = lambda k_1, k_2: 0.5 * (F(k_1,k_2,plusminus=+1) + F(k_1,k_2,plusminus=-1)) * H(k_1,k_2,option_type)
    return (b_1 - a_1)/2 * (b_2 - a_2)/2 * np.exp(-r * (T-t_0)) * double_sum_prime(N_1,N_2,summand)

T = 0.1 # time of maturity
t_0 = 0 # starting time of option contract
K = 110 # strike price for basket option
S_0 = np.array([100,100]) # asset prices at t=0
sigma_1 = 0.2 # volatility of first asset
sigma_2 = 0.5 # volatility of second asset
weight_1, weight_2 = 0.5, 0.5 # weights used for calculating the weighted arithmatic average of the asset prices
rho = 0.5 # correlation coefficient
r = 0.05 # risk-free interest rate
L = 8 # used for determining the size of truncation domain

a_1 = a_2 = -L * np.sqrt(T)
b_1 = b_2 = L * np.sqrt(T)
x_1, x_2 = np.log(S_0 / K)

#with cProfile.Profile() as profile:
    # numeric_valuation = COS_formula()
#results = pstats.Stats(profile)
#results.sort_stats('tottime').print_stats()

# converting datetime.timedelta object to minutes
def minutes(time):
    return time.days * 24 * 60 + time.seconds / 60 + time.microseconds / 60 / 1000000

# for computing and printing the valuations, and the duration of computation
def print_results(N,option_type):
    N_1 = N_2 = N
    start_time = datetime.now()
    numeric_value = COS_formula(N_1,N_2,option_type)
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"N_1 = N_2 = {N}")
    print(f"Value {option_type} option: {numeric_value}")
    print(f"Duration: {minutes(duration)} minutes")
    print("")
    return

for N in [4,8,12,16,24,32,48,64]:
    print_results(N,'call')

for N in [4,8,12,16,24,32,48,64]:
    print_results(N,'put')