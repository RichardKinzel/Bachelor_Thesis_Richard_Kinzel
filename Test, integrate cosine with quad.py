#in this file we will numerically integrate the cosine function, using the predefined scipy.integrate.quad function

import numpy as np
from datetime import datetime
from scipy.integrate import quad

start_time = datetime.now()

def integral(a,b,integrand):
    value = quad(integrand, a, b)[0]
    return value

def function(x):
    return np.cos(x)

n = 100
a = 0
b = 10
numeric_integral = integral(a, b, function)

end_time = datetime.now()

print(f"Numeric: {numeric_integral}")
print(f"Analytic: {np.sin(b)-np.sin(a)}")
print(f"Error: {np.abs(numeric_integral - np.sin(b)-np.sin(a))}")
print(f"Duration: {end_time - start_time}")
