#in this file we will numerically integrate the cosine function, and define the simpson rule ourselves

import numpy as np
from datetime import datetime
from scipy.integrate import simpson

start_time = datetime.now()

def integral(n,a,b,integrand):
    x = np.linspace(a,b,n+1)
    y = integrand(x)
    value = simpson(y, x=x)
    return value

def function(x):
    return np.cos(x)

n = 100
a = 0
b = 10

print(f"Numeric: {integral(n, a, b, function)}")
print(f"Analytic: {np.sin(b)-np.sin(a)}")
print(f"Error: {np.abs(integral(n, a, b, function) - np.sin(b)-np.sin(a))}")

end_time = datetime.now()
print(f"Duration: {end_time - start_time}")
