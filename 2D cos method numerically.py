#Here we will numerically compute the value of an option using the 2D cos method.

import numpy as np
from scipy.integrate import dblquad

# Here we choose the way of numeric integration. For now we use scipy.integrate.dblquad
def double_integral(a2,b2,a1,b1,integrand):
    integral_value = dblquad(integrand, a2, b2, a1, b1)[0]
    return integral_value

