#in this file we will numerically integrate the cosine function, and define the simpson rule ourselves

import numpy as np
from datetime import datetime

start_time = datetime.now()

def integral(n,a,b,integrand):
    dx = (b-a)/n
    x = np.linspace(a,b,n+1)
    value = 0
    for i in range(1,int(n/2+1)):
        #Voor onderstaande identiteit, zie  wikipedia: https://en.wikipedia.org/wiki/Simpson%27s_rule#Composite_Simpson's_1/3_rule
        value += integrand(x[2*i-2]) + 4 * integrand(x[2*i-1]) + integrand(x[2*i])
    value = 1/3 * dx * value
    return value

def function(x):
    return np.cos(x)

n = 100
a = 0
b = 10
numeric_integral = integral(n, a, b, function)

end_time = datetime.now()

print(f"Numeric: {numeric_integral}")
print(f"Analytic: {np.sin(b)-np.sin(a)}")
print(f"Error: {np.abs(integral(n, a, b, function) - np.sin(b)-np.sin(a))}")
print(f"Duration: {end_time - start_time}")
