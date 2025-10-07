#in this file we will numerically integrate the cosine function, and use the predefined scipy.integrate.simpson function

import numpy as np
from datetime import datetime
from scipy.integrate import simpson

start_time = datetime.now()

def integral(n,a,b,integrand):
    x = np.linspace(a,b,n)
    y = integrand(x)
    value = simpson(y, x=x)
    return value

def function(x):
    return np.cos(x)

n = 10000
a = 0
b = 10
numeric_integral = integral(n, a, b, function)

end_time = datetime.now()

print(f"Numeric: {numeric_integral}")
print(f"Analytic: {np.sin(b)-np.sin(a)}")
print(f"Error: {np.abs(integral(n, a, b, function) - np.sin(b)-np.sin(a))}")
print(f"Duration: {end_time - start_time}")
