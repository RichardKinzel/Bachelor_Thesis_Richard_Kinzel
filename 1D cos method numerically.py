#Here we will numerically compute the value of an option using the 1D cos method.

import numpy as np
import scipy.stats as st
from scipy.fftpack import dct
from datetime import datetime

def call_payoff(y):
    return np.maximum(K * (np.exp(y) - 1),0)
def put_payoff(y):
    return np.maximum(-K * (np.exp(y) - 1),0)

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u,t):
    #normal distribution:
    characteristic_value = np.exp(1j * u * (r - 0.5 * sigma**2) * t - 0.5 * sigma**2 * u**2 * t)

    return characteristic_value
# This is the conditional characteristic function
def conditional_characteristic_function(u):
    return np.exp(1j*u*X_0)*characteristic_function(u,T)

# The following function calculates a sum with the first term halved
def sum_prime(N,summand):
    return 0.5 * summand(0) + sum(summand(index) for index in range(1,N))

# This is the function used for our main calculation. In it we use the functions defined above
def COS_formula(N, Q, option_type):
    # option_type should be 'put' or 'call'
    # N is the summation limit
    y_index = np.arange(Q)
    y_midpoints = a + (b - a) / Q * (y_index + 0.5)
    if option_type == 'call':
        payoff_midpoints = call_payoff(y_midpoints)
    elif option_type == 'put':
        payoff_midpoints = put_payoff(y_midpoints)
    H_matrix = dct(payoff_midpoints, type=2) / Q
    summand = lambda k: np.real(conditional_characteristic_function(k*np.pi/(b-a)) * np.exp(-1j*k*np.pi*a/(b-a))) * H_matrix[k]
    return np.exp(-r * (T-t_0)) * sum_prime(N,summand)

# In order to check the accuracy of our numeric approximation, we also calculate the option value analytically using Black Scholes.
def Black_Scholes(option_type):
    d1 = 1/(sigma*np.sqrt(T-t_0)) * (np.log(S_0/K) + (r + 0.5 * sigma**2) * (T-t_0))
    d2 = d1 - sigma * np.sqrt(T - t_0)
    if option_type == 'call':
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T-t_0))
    elif option_type == 'put':
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T-t_0)) - st.norm.cdf(-d1) * S_0
    return value

# Variables that have to be set manually
T = 0.1     # Time of maturity
t_0 = 0     # Starting time of option contract
K = 90     # Strike price
S_0 = 100   # Asset value at t=0
sigma = 0.25# volatility
r = 0.05     # risk neutral interest rate
L = 8       # Used for determining the size of truncation domain

X_0 = np.log(S_0 / K)
def cumulant(i): # the i-th cumulant, assuming GBM
    if i == 1:
        return (r - 0.5 * sigma**2)*T
    elif i == 2:
        return sigma**2 * T
    elif i == 4:
        return 0
a = X_0 + cumulant(1) - L*np.sqrt(cumulant(2) + np.sqrt(cumulant(4)))
b = X_0 + cumulant(1) + L*np.sqrt(cumulant(2) + np.sqrt(cumulant(4)))

def seconds(time):
    return time.days * 24 * 60 * 60 + time.seconds + time.microseconds / 1000000
def print_results(N, Q, option_type):
    black_scholes_valuation = Black_Scholes(option_type)

    start_time = datetime.now()
    numeric_valuation = COS_formula(N, Q, option_type)
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"N = {N}, Q = {Q}")
    print(f"Value {option_type} option: {numeric_valuation}")
    print(f"Black Scholes: {black_scholes_valuation}")
    print(f"Error: {abs(numeric_valuation - black_scholes_valuation):.2e}")
    print(f"Duration: {seconds(duration):.2e} seconds")
    print("")
    return
def print_latex_table(option_type):
    black_scholes_valuation = Black_Scholes(option_type)
    error_list = np.array([])
    duration_list = np.array([])
    for Q in range(500, 3000, 500):
        for N in range(20,120,20):
            start_time = datetime.now()
            numeric_valuation = COS_formula(N, Q, option_type)
            end_time = datetime.now()
            duration = end_time - start_time
            error = abs(numeric_valuation - black_scholes_valuation)

            error_list = np.append(error_list, f"{error:.2e}")
            duration_list = np.append(duration_list, f"{seconds(duration):.2e}")
    print("Absolute error:")
    error_matrix = error_list.reshape(5,5)
    for row in range(5):
        Q = 500 * (row + 1)
        error_row_string = f"{Q} & {error_matrix[row][0]} & {error_matrix[row][1]} & {error_matrix[row][2]} & {error_matrix[row][3]} & {error_matrix[row][4]} \\\\"
        print(error_row_string)
    print("")
    print("Duration:")
    duration_matrix = duration_list.reshape(5, 5)
    for row in range(5):
        Q = 500 * (row + 1)
        duration_row_string = f"{Q} & {duration_matrix[row][0]} & {duration_matrix[row][1]} & {duration_matrix[row][2]} & {duration_matrix[row][3]} & {duration_matrix[row][4]} \\\\"
        print(duration_row_string)
    return

print_results(200, 5000, 'call')
print_latex_table('call')