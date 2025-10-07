import numpy as np
import math
from scipy.integrate import simpson
from datetime import datetime

#We bepalen de snelheid van deze code.
start_time = datetime.now()

#Variabelen.
sigma=0.25
r=0.1
delta=0
mu=r-1/2*sigma**2-delta

K=120
T=0.1
t_0=0

#We kiezen [a,b] zonder noemenswaardige nauwkeurigheid te verliezen, zie Hoofdstuk 4.
L=8
a=-L*np.sqrt(T)
b=L*np.sqrt(T)

#Bepaalt X(t_0) aan de hand van S(t_0), aannemende dat X=log(S/K).
S_0=100
X_0=np.log(S_0/K)

#Definieert de karakteristieke functie van een normale verdeling.
def karnormal(u,t):
    karnormal=np.exp(1j*u*mu*t-1/2*sigma**2*u**2*t)
    #Karakteristieke functie in COS-methode is net anders, zoals in het verslag:
        #phi=e^{iuX(t_0)}*varphi.
    return np.exp(1j*u*X_0)*karnormal

#Bepaalt de optiewaardering van een call-optie.
def Call(y,n):
    call=max(K*(np.exp(y)-1),0)
    return call

#Bepaalt de optiewaardering van een put-optie.
def Put(y,n):
    put=max(-K*(np.exp(y)-1),0)
    return put

#We definiëren hier de coëfficiënten H_n voor zowel de call- als put-optie.
def H(n, a, b, optie_type):
    if optie_type == 'call':
        y_values = np.linspace(a, b, 1000)  # Kies geschikte resolutie
        integrand_values = [Call(y, n)*math.cos(n*np.pi*(y-a)/(b-a)) for y in y_values]
        Hn = 2/(b-a)*simpson(integrand_values, x=y_values)
    elif optie_type == 'put':
        y_values = np.linspace(a, b, 1000)  # Kies geschikte resolutie
        integrand_values = [Put(y, n)*math.cos(n*np.pi*(y-a)/(b-a)) for y in y_values]
        Hn = 2/(b-a)*simpson(integrand_values, x=y_values)
    else:
        raise ValueError("Ongeldig optietype. Gebruik 'call' of 'put'.")
    return Hn

#We definiëren hier de coëfficiënten F_n, in de COS-formule herkennen we
#deze coëfficiënten aan Re{phi(...)*exp(...)}.
def F(kar,n):
    return np.real(kar(n*np.pi/(b-a),T)*np.exp(-1j*n*math.pi*a/(b-a)))

#Nu gebruiken we H_n en F_n om de 1-dimensionale COS-formule te geven.
def cos(kar,N,optie_type):
    somterm=(1/2*F(kar,0)*H(0,a,b,optie_type)+
             sum([F(kar,n)*H(n,a,b,optie_type) for n in range(1,N)]))
    V=np.exp(-r*(T-t_0))*(somterm)
    return V

#Bepaalt de eerlijke prijs voor een optiecontract op tijdstip t_0.
#Hier kunnen we N ook passend kiezen.
N=156
print(cos(karnormal,N,'put'))

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))