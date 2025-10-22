#Here we will numerically compute the value of an option using the 1D cos method.

import numpy as np
from scipy.integrate import quad,simpson
import scipy.stats as st
from datetime import datetime

# Here we choose the way of numeric integration.
def integral(lower_limit,upper_limit,integrand,n):
    #using scipy.integrade.quad:
    integral_value = quad(integrand, lower_limit, upper_limit, limit=n)[0] #default limit of 50 leads to error for higher values of N. Hence the higher limit.

    #using scipy.integrate.simpson:
    # x_values = np.linspace(lower_limit,upper_limit,n)
    # integral_value = simpson(integrand(x_values), x=x_values)
    return integral_value

# Here we define the payoff function of an option at time T, depending on what type of option contract we use.
def V_T(X_T,K,option_type):
    if option_type == 'call':
        option_value = np.maximum(K * (np.exp(X_T)-1), 0)
    elif option_type == 'put':
        option_value = np.maximum(-K * (np.exp(X_T)-1),0)
    return option_value

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u,t,sigma,r):
    #normal distribution:
    characteristic_value = np.exp(1j * u * (r - 0.5 * sigma**2) * t - 0.5 * sigma**2 * u**2 * t)

    return characteristic_value

# This is the conditional characteristic function
def phi(u,X_0,T,sigma,r):
    return np.exp(1j*u*X_0)*characteristic_function(u,T,sigma,r)

# cosine Fourier coefficients of payoff function
def H(k,a,b,K,n,option_type):
    integrand = lambda y: V_T(y, K, option_type) * np.cos(k * np.pi * (y - a) / (b - a))
    if option_type == 'call':
        #note that V_T(y)=0 if y<0, so we replace our lower integration limit with 0, assuming a<0 and b>0.
        H_value = 2/(b-a) * integral(0,b,integrand,n)
    elif option_type == 'put':
        # note that V_T(y)=0 if y>0, so we replace our upper integration limit with 0, assuming a<0 and b>0.
        H_value = 2/(b-a) * integral(a,0,integrand,n)
    return H_value

def sum_prime(N,summand):
    return 0.5 * summand(0) + sum(summand(index) for index in range(1,N))

# This is our main calculation. In in we use the functions defined above
def COS_formula(N,T,t_0,K,S_0,sigma,r,L,n,option_type):
    a = -L * np.sqrt(T)
    b = L * np.sqrt(T)
    X_0 = np.log(S_0 / K)
    summand = lambda k: np.real(phi(k*np.pi/(b-a),X_0,T,sigma,r) * np.exp(-1j*k*np.pi*a/(b-a))) * H(k,a,b,K,n,option_type)
    return np.exp(-r * (T-t_0)) * sum_prime(N,summand)

# In order to check the accuracy of our numeric approximation, we also calculate the option value analytically using Black Scholes.
def Black_Scholes(option_type,S_0,K,sigma,T,t_0,r):
    d1 = 1/(sigma*np.sqrt(T-t_0)) * (np.log(S_0/K) + (r + 0.5 * sigma**2) * (T-t_0))
    d2 = d1 - sigma * np.sqrt(T - t_0)
    if option_type == 'call':
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t_0))
    elif option_type == 'put':
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t_0)) - st.norm.cdf(-d1) * S_0
    return value

# Values for all variables used
N = 256
T = 0.1
t_0 = 0
K = 120
S_0 = 100
sigma = 0.25
r = 0.1
L = 8
option_type = 'put'
n = 1000 #custom limit of maximum number of integration subintervals that scipy.integrate.quad uses

start_time = datetime.now()
numeric_valuation = COS_formula(N, T, t_0, K, S_0, sigma, r, L, n, option_type)
end_time = datetime.now()
black_scholes_valuation = Black_Scholes(option_type, S_0, K, sigma, T, t_0, r)

print(f"Numeric: {numeric_valuation}")
print(f"Black Scholes: {black_scholes_valuation}")
print(f"Error: {abs(numeric_valuation - black_scholes_valuation)}")
print(f"Duration: {end_time - start_time}")