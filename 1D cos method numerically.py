#Here we will numerically compute the value of an option using the 1D cos method.

import numpy as np
from scipy.integrate import quad,simpson

# Here we choose the way of numeric integration.
def integral(lower_limit,upper_limit,integrand,n):
    #integral_value = quad(integrand, lower_limit, upper_limit)[0]
    x_values = np.linspace(lower_limit,upper_limit,n)
    integral_value = simpson(integrand(x_values), x=x_values)
    return integral_value

# Here we define the payoff function of an option at time T, depending on what type of option contract we use.
def V_T(X_T,K):
    #call option:
    #option_value = np.maximum(K * (np.exp(X_T)-1), 0)

    #put option:
    option_value = np.maximum(-K * (np.exp(X_T)-1),0)

    return option_value

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u,t,sigma,r):
    #normal distribution:
    characteristic_value = np.exp(1j * u * (r - 0.5 * sigma**2) * t - 0.5 * sigma**2 * u**2 * t)

    return characteristic_value

def phi(u,X_0,T,sigma,r):
    return np.exp(1j*u*X_0)*characteristic_function(u,T,sigma,r)

def H(k,a,b,K,n):
    integrand = lambda y: V_T(y,K) * np.cos(k * np.pi * (y-a)/(b-a))
    return 2/(b-a) * integral(a,b,integrand,n)

def COS_formule(N,T,t_0,K,S_0,sigma,r,L,n):
    a = -L * np.sqrt(T)
    b = L * np.sqrt(T)
    X_0 = np.log(S_0 / K)
    somterm = lambda k:( phi(k*np.pi/(b-a),X_0,T,sigma,r) * np.exp(-1j*k*np.pi*a/(b-a)) ).real * H(k,a,b,K,n)
    return (b-a)/2 * np.exp(-r * (T-t_0)) * (0.5*somterm(0) + sum(somterm(k) for k in range(1,N)))

print(COS_formule(N=156,T=0.1,t_0=0,K=120,S_0=100,sigma=0.25,r=0.1,L=8,n=1000))