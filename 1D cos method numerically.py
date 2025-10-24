#Here we will numerically compute the value of an option using the 1D cos method.

import numpy as np
from scipy.integrate import quad,simpson
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

# Here we define the payoff function of an option at time T, depending on what type of option contract we use. Note: we altered the integration interval so that V_T is always positive.
# Hence we eliminated the maximum functions for a faster computation
def V_T(X_T):
    if option_type == 'call':
        option_value = K * (np.exp(X_T)-1)
    elif option_type == 'put':
        option_value = -K * (np.exp(X_T)-1)
    else:
        raise ValueError('option_type must be put or call')
    return option_value

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u,t):
    #normal distribution:
    characteristic_value = np.exp(1j * u * (r - 0.5 * sigma**2) * t - 0.5 * sigma**2 * u**2 * t)

    return characteristic_value

# This is the conditional characteristic function
def phi(u):
    return np.exp(1j*u*X_0)*characteristic_function(u,T)

# cosine Fourier coefficients of payoff function
def H(k):
    integrand = lambda y: V_T(y) * np.cos(k * np.pi * (y - a) / (b - a))
    if option_type == 'call':
        #note that V_T(y)=0 if y<0, so we replace our lower integration limit with 0, assuming a<0 and b>0.
        H_value = 2/(b-a) * integral(0,b,integrand)
    elif option_type == 'put':
        # note that V_T(y)=0 if y>0, so we replace our upper integration limit with 0, assuming a<0 and b>0.
        H_value = 2/(b-a) * integral(a,0,integrand)
    else:
        raise ValueError('option_type must be put or call')
    return H_value

# The following function calculates a sum with the first term halved
def sum_prime(N,summand):
    return 0.5 * summand(0) + sum(summand(index) for index in range(1,N))

# This is the function used for our main calculation. In it we use the functions defined above
def COS_formula():
    summand = lambda k: np.real(phi(k*np.pi/(b-a)) * np.exp(-1j*k*np.pi*a/(b-a))) * H(k)
    return np.exp(-r * (T-t_0)) * sum_prime(N,summand)

# In order to check the accuracy of our numeric approximation, we also calculate the option value analytically using Black Scholes.
def Black_Scholes():
    d1 = 1/(sigma*np.sqrt(T-t_0)) * (np.log(S_0/K) + (r + 0.5 * sigma**2) * (T-t_0))
    d2 = d1 - sigma * np.sqrt(T - t_0)
    if option_type == 'call':
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t_0))
    elif option_type == 'put':
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t_0)) - st.norm.cdf(-d1) * S_0
    else:
        raise ValueError('option_type must be put or call')
    return value

# Variables that have to be set manually
N = 256     # Number of summation terms
T = 0.1     # Time of maturity
t_0 = 0     # Starting time of option contract
K = 120     # Strike price
S_0 = 100   # Asset value at t=0
sigma = 0.25# volatility
r = 0.1     # risk neutral interest rate
L = 8       # Used for determining the size of truncation domain
option_type = 'call' # put or call
integration_subintervals = 1000    #custom limit of maximum number of integration subintervals that scipy.integrate.quad uses

# Variables that are computed automatically
a = -L * np.sqrt(T)
b = L * np.sqrt(T)
X_0 = np.log(S_0 / K)

# Main calculation
start_time = datetime.now()
numeric_valuation = COS_formula()
end_time = datetime.now()
black_scholes_valuation = Black_Scholes()

# Printing the results
print(f"Numeric: {numeric_valuation}")
print(f"Black Scholes: {black_scholes_valuation}")
print(f"Error: {abs(numeric_valuation - black_scholes_valuation)}")
print(f"Duration: {end_time - start_time}")