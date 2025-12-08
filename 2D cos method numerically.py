#Here we will numerically compute the value of an option using the 2D cos method.

import numpy as np
from scipy.fftpack import dctn
from datetime import datetime

# Here we define the payoff function of an option at time T, depending on what type of option contract we use.
def basket_call_payoff(y_1, y_2):
    return np.maximum(K * (weight_1*np.exp(y_1) + weight_2*np.exp(y_2) - 1),0)
def basket_put_payoff(y_1, y_2):
    return np.maximum(-K * (weight_1*np.exp(y_1) + weight_2*np.exp(y_2) - 1),0)
def call_on_max_payoff(y_1, y_2):
    return np.maximum(K * (np.maximum(np.exp(y_1), np.exp(y_2)) - 1), 0)
def put_on_min_payoff(y_1, y_2):
    return np.maximum(-K * (np.minimum(np.exp(y_1), np.exp(y_2)) - 1), 0)

# Here we define our characteristic function, depending on the distribution of X we work with.
def characteristic_function(u_1,u_2,t):
    #bivariate normal distribution:
    characteristic_value = np.exp(1j*t*(r*(u_1+u_2) - 0.5*(u_1 * sigma_1**2 + u_2 * sigma_2**2)) - 0.5*t*(u_1**2 * sigma_1**2 + 2*u_1*u_2*sigma_1*sigma_2*rho + u_2**2 * sigma_2**2))
    return characteristic_value
def conditional_characteristic_function(u_1,u_2):
    return np.exp(1j*(u_1*x_1 + u_2*x_2)) * characteristic_function(u_1,u_2,T)
def F(k_1,k_2,plusminus):
    # Here plusminus should be equal to either +1 or -1
    return 2/(b_1-a_1) * 2/(b_2-a_2) * np.real( conditional_characteristic_function(k_1*np.pi/(b_1-a_1),plusminus*k_2*np.pi/(b_2-a_2)) * np.exp(-1j*np.pi*k_1*a_1/(b_1-a_1) - 1j*np.pi*plusminus*k_2*a_2/(b_2-a_2)) )

# The following two functions calculate a sum and a double sum respectively with the first term(s) halved
def sum_prime(N,summand):
    return 0.5 * summand(0) + sum([summand(index) for index in range(1,N)])
def double_sum_prime(N_1,N_2,summand):
    inner_sum = lambda k_1: sum_prime(N_2,lambda k_2: summand(k_1,k_2))
    return sum_prime(N_1,inner_sum)

# This is our main calculation. In it we use the functions defined above
def COS_formula(N,Q,option_type):
    # option_type should be 'basket call', 'basket put', 'call on max' or 'put on min'
    # N (= N_1 = N_2) are the summation limits
    # Q is number of integration subintervals
    y_index = np.arange(Q)
    y1_midpoints = a_1 + (b_1 - a_1) / Q * (y_index + 0.5)
    y2_midpoints = a_2 + (b_2 - a_2) / Q * (y_index + 0.5)
    if option_type == 'basket call':
        payoff_midpoints = basket_call_payoff(y1_midpoints.reshape(Q, 1), y2_midpoints.reshape(1, Q))
    elif option_type == 'basket put':
        payoff_midpoints = basket_put_payoff(y1_midpoints.reshape(Q, 1), y2_midpoints.reshape(1, Q))
    elif option_type == 'call on max':
        payoff_midpoints = call_on_max_payoff(y1_midpoints.reshape(Q, 1), y2_midpoints.reshape(1, Q))
    elif option_type == 'put on min':
        payoff_midpoints = put_on_min_payoff(y1_midpoints.reshape(Q, 1), y2_midpoints.reshape(1, Q))
    H_matrix = dctn(payoff_midpoints, type=2) / (Q ** 2)
    summand = lambda k_1, k_2: 0.5 * (F(k_1,k_2,plusminus=+1) + F(k_1,k_2,plusminus=-1)) * H_matrix[k_1,k_2]
    return (b_1 - a_1)/2 * (b_2 - a_2)/2 * np.exp(-r * (T-t_0)) * double_sum_prime(N,N,summand)

#parameter set basket call / basket put
T = 1 # time of maturity
t_0 = 0 # starting time of option contract
K = 100 # strike price for basket option
S_0 = np.array([90,110]) # asset prices at t=0
sigma_1 = 0.2 # volatility of first asset
sigma_2 = 0.3 # volatility of second asset
weight_1, weight_2 = 0.5, 0.5 # weights used, only relevant in the case of a basket option. Sum should be 1
rho = 0.25 # correlation coefficient
r = 0.04 # risk-free continuous interest rate
L = 8 # used for determining the size of truncation domain
reference_value_basket_call = 10.173230

#parameter set call on max / put on min
# T = 7/12 # time of maturity
# t_0 = 0 # starting time of option contract
# K = 40 # strike price for basket option
# S_0 = np.array([40,40]) # asset prices at t=0
# sigma_1 = 0.2 # volatility of first asset
# sigma_2 = 0.3 # volatility of second asset
# rho = 0.5 # correlation coefficient
# r = 0.048790 # risk-free continuous interest rate
# L = 8 # used for determining the size of truncation domain
# reference_value_call_on_max = 18.477 # uit Emma haar scriptie

x_1, x_2 = np.log(S_0 / K)
# the following functions define the i-th cumulant for first and second asset respectively
def cumulant_1(i):
    if i == 1:
        return (r - 0.5 * sigma_1**2)*T
    elif i == 2:
        return sigma_1**2 * T
    elif i == 4:
        return 0
def cumulant_2(i):
    if i == 1:
        return (r - 0.5 * sigma_2**2)*T
    elif i == 2:
        return sigma_2**2 * T
    elif i == 4:
        return 0
a_1 = a_2 = np.minimum(x_1 + cumulant_1(1) - L*np.sqrt(cumulant_1(2) + np.sqrt(cumulant_1(4))), x_2 + cumulant_2(1) - L*np.sqrt(cumulant_2(2) + np.sqrt(cumulant_2(4))))
b_1 = b_2 = np.maximum(x_1 + cumulant_1(1) + L*np.sqrt(cumulant_1(2) + np.sqrt(cumulant_1(4))), x_2 + cumulant_2(1) + L*np.sqrt(cumulant_2(2) + np.sqrt(cumulant_2(4))))

def seconds(time):
    return time.days * 24 * 60 * 60 + time.seconds + time.microseconds / 1000000
def print_results(N, Q, option_type):
    start_time = datetime.now()
    numeric_value = COS_formula(N, Q, option_type)
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"N_1 = N_2 = {N}, Q = {Q}")
    print(f"Value {option_type} option: {numeric_value}")
    if option_type == 'basket call':
        print(f"Error: {abs(numeric_value - reference_value_basket_call):.2e}")
    # if option_type == 'call on max':
    #     print(f"Error: {abs(numeric_value - reference_value_call_on_max):.2e}")
    print(f"Duration: {seconds(duration)} seconds")
    print("")
    return
def print_latex_table(option_type): #only supported for basket call, unless other reference values are added.
    error_list = np.array([])
    duration_list = np.array([])
    for Q in range(500, 3000, 500):
        for N in range(20,120,20):
            start_time = datetime.now()
            numeric_valuation = COS_formula(N, Q, option_type)
            end_time = datetime.now()
            duration = end_time - start_time
            if option_type == 'basket call':
                error = abs(numeric_valuation - reference_value_basket_call)

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

print_results(100, 2000, 'basket call')
print_latex_table('basket call')