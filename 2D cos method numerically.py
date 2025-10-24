#Here we will numerically compute the value of an option using the 2D cos method.

import numpy as np
from scipy.integrate import quad
import scipy.stats as st
from datetime import datetime

# Here we choose the way of numeric integration.
def integral(lower_limit,upper_limit,integrand):
    #using scipy.integrade.quad:
    integral_value = quad(integrand, lower_limit, upper_limit, limit=integration_subintervals)[0] #default limit of 50 leads to error for higher values of N. Hence the higher limit.

    #using scipy.integrate.simpson:
    # x_values = np.linspace(lower_limit,upper_limit,integration_subintervals)
    # integral_value = simpson(integrand(x_values), x=x_values)
    return integral_value

def double_integral(lower_limit_1,upper_limit_1,lower_limit_2,upper_limit_2,integrand):
    first_integral = lambda y_2: integral(lower_limit_2,upper_limit_2,lambda y_1: integrand(y_1,y_2))
    second_integral = integral(lower_limit_1, upper_limit_1, first_integral)
    return second_integral

# Here we define the payoff function of a basket option at time T, depending on what type of option contract we use.
def V_T(X_T):
    if option_type == 'call':
        option_value = np.maximum(K * (np.inner(weight, np.exp(X_T)) - 1), 0)
    elif option_type == 'put':
        option_value = np.maximum(-K * (np.inner(weight, np.exp(X_T)) - 1), 0)
    else:
        raise ValueError('option_type must be put or call')
    return option_value

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u_1,u_2,t):
    #bivariate normal distribution:
    characteristic_value = np.exp(1j*t*(r*(u_1+u_2) - 0.5*(u_1 * sigma_1**2 + u_2 * sigma_2**2)) - 0.5*t*(u_1**2 * sigma_1**2 + 2*u_1*u_2*sigma_1*sigma_2*rho + u_2**2 * sigma_2**2))
    return characteristic_value

# This is the conditional characteristic function
def phi(u_1,u_2):
    return np.exp(1j*(u_1*x_1 + u_2*x_2)) * characteristic_function(u_1,u_2,T)

# cosine Fourier coefficients of payoff function
def H(k_1,k_2):
    '''
    implement custom integration interval? -> remove part where payoff is zero
    '''

    integrand = lambda y_1, y_2: V_T(np.array([y_1, y_2])) * np.cos(k_1 * np.pi * (y_1 - a_1) / (b_1 - a_1)) * np.cos(k_2 * np.pi * (y_2 - a_2) / (b_2 - a_2))
    if option_type == 'call':
        return 2 / (b_1 - a_1) * 2 / (b_2 - a_2) * double_integral(a_2, b_2, lambda y_2: np.log((1-weight[1]*np.exp(y_2))/weight[0]), b_1, integrand)
    if option_type == 'put':
        return 2 / (b_1 - a_1) * 2 / (b_2 - a_2) * double_integral(a_2, b_2, a_1, b_1, integrand)
    else:
        raise ValueError('option_type must be put or call')

def F(k_1,k_2,plusminus):
    return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real( phi(k_1*np.pi/(b_1-a_1),plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)) )

# The following two functions calculate a sum and a double sum respectively with the first term(s) halved
def sum_prime(N,summand):
    return 0.5 * summand(0) + np.sum(np.array([summand(index) for index in range(1,N)]))
    # sum = 0.5 * summand(0)
    # for index in range(1,N):
    #     sum += summand(index)
    #     print(f"k = {index}, sum = {sum}")
    # return sum

def double_sum_prime(N_1,N_2,summand):
    inner_sum = lambda k_1: sum_prime(N_2,lambda k_2: summand(k_1,k_2))
    return sum_prime(N_1,inner_sum)

# This is our main calculation. In in we use the functions defined above
def COS_formula(N_1,N_2,T,t_0,K,S_0,sigma_1,sigma_2,rho,r,L,weight,option_type):
    summand = lambda k_1, k_2: 0.5 * (F(k_1,k_2,plusminus=+1) + F(k_1,k_2,plusminus=-1)) * H(k_1,k_2)
    return (b_1 - a_1)/2 * (b_2 - a_2)/2 * np.exp(-r * (T-t_0)) * double_sum_prime(N_1,N_2,summand)

# In order to check the accuracy of our numeric approximation, we also calculate the option value analytically using Black Scholes.
# def Black_Scholes(option_type,S_0,K,sigma,T,t_0,r):
#     d1 = 1/(sigma*np.sqrt(T-t_0)) * (np.log(S_0/K) + (r + 0.5 * sigma**2) * (T-t_0))
#     d2 = d1 - sigma * np.sqrt(T - t_0)
#     if option_type == 'call':
#         value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t_0))
#     elif option_type == 'put':
#         value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t_0)) - st.norm.cdf(-d1) * S_0
#     return value

N_1 = N_2 = 6 # number of summation terms
T = 0.1 # time of maturity
t_0 = 0 # starting time of option contract
K = 120 # strike price for basket option
S_0 = np.array([100,100]) # asset prices at t=0
sigma_1 = 0.2 # volatility of first asset
sigma_2 = 0.3 # volatility of second asset
weight = np.array([.5,.5]) # weights used for calculating the weighted arithmatic average of the asset prices
rho = 0.5 # correlation coefficient
r = 0.1 # risk-free interest rate
L = 8 # used for determining the size of truncation domain
option_type = 'call' # put or call
integration_subintervals = 100 #represents maximum number of subintervals for integrating numerically

a_1 = a_2 = -L * np.sqrt(T)
b_1 = b_2 = L * np.sqrt(T)
x_1, x_2 = np.log(S_0 / K)

# Main calculation
#start_time = datetime.now()
#numeric_valuation = COS_formula(N_1,N_2,T,t_0,K,S_0,sigma_1,sigma_2,rho,r,L,weight,option_type)
#end_time = datetime.now()
# black_scholes_valuation = Black_Scholes(option_type, S_0, K, sigma, T, t_0, r)

# Printing the results
#print(f"Numeric: {numeric_valuation}")
# print(f"Black Scholes: {black_scholes_valuation}")
# print(f"Error: {abs(numeric_valuation - black_scholes_valuation)}")
#print(f"Duration: {end_time - start_time}")

