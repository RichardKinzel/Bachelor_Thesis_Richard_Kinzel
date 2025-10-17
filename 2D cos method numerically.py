#Here we will numerically compute the value of an option using the 2D cos method.

import numpy as np
from scipy.integrate import dblquad
import scipy.stats as st
from datetime import datetime

# Here we choose the way of numeric integration.
def double_integral(lower_limit_1,upper_limit_1,lower_limit_2,upper_limit_2,integrand):
    #using scipy.integrade.quad:
    integral_value = dblquad(integrand, lower_limit_1, upper_limit_1, lower_limit_2, upper_limit_2)[0]
    return integral_value

# Here we define the payoff function of a basket option at time T, depending on what type of option contract we use.
def V_T(X_T,K,weight,option_type):
    if option_type == 'call':
        option_value = np.maximum(K * (np.inner(weight, np.exp(X_T)) - 1), 0)
    elif option_type == 'put':
        option_value = np.maximum(-K * (np.inner(weight, np.exp(X_T)) - 1), 0)
    return option_value

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u,t,sigma,r):
    #normal distribution:
    characteristic_value = np.exp(1j * u * (r - 0.5 * sigma**2) * t - 0.5 * sigma**2 * u**2 * t)

    return characteristic_value

def phi(u,X_0,T,sigma,r):
    return np.exp(1j*u*X_0)*characteristic_function(u,T,sigma,r)

def H(k,a,b,K,n,option_type):
    integrand = lambda y: V_T(y,K,option_type) * np.cos(k * np.pi * (y-a)/(b-a))
    return 2/(b-a) * integral(a,b,integrand,n)

def COS_formule(N,T,t_0,K,S_0,sigma,r,L,n,option_type):
    a = -L * np.sqrt(T)
    b = L * np.sqrt(T)
    X_0 = np.log(S_0 / K)
    somterm = lambda k:( phi(k*np.pi/(b-a),X_0,T,sigma,r) * np.exp(-1j*k*np.pi*a/(b-a)) ).real * H(k,a,b,K,n,option_type)
    return np.exp(-r * (T-t_0)) * (0.5*somterm(0) + sum(somterm(k) for k in range(1,N)))

def Black_Scholes(option_type,S_0,K,sigma,T,t_0,r):
    d1 = 1/(sigma*np.sqrt(T-t_0)) * (np.log(S_0/K) + (r + 0.5 * sigma**2) * (T-t_0))
    d2 = d1 - sigma * np.sqrt(T - t_0)
    if option_type == 'call':
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t_0))
    elif option_type == 'put':
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t_0)) - st.norm.cdf(-d1) * S_0
    return value

N = 128
T = 0.1
t_0 = 0
K = 120
S_0 = 100
sigma = 0.25
r = 0.1
L = 8
option_type = 'put'
n = 10000

start_time = datetime.now()
numeric_valuation = COS_formule(N, T, t_0, K, S_0, sigma, r, L, n, option_type)
end_time = datetime.now()
black_scholes_valuation = Black_Scholes(option_type, S_0, K, sigma, T, t_0, r)

print(f"Numeric: {numeric_valuation}")
print(f"Black Scholes: {black_scholes_valuation}")
print(f"Error: {abs(numeric_valuation - black_scholes_valuation)}")
print(f"Duration: {end_time - start_time}")