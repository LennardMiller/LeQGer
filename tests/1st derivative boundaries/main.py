import numpy as np
from leqger import schemes
import matplotlib.pyplot as plt

N = 50
L = 1

delta = L/N

psi = np.zeros([N, N])

der_b_an = np.zeros(4*N)

for n in range(N):
    for m in range(N):
        
        x = (n + 1/2)*delta
        y = (m + 1/2)*delta
        
        psi[n, m] =  x*(x-1)*y*(y-1)
        
for i in range(N):
    
    x = (i + 1/2)*delta
    
    der_b_an[i] = -x**2 + x
    der_b_an[i + N]  = -der_b_an[i]
    der_b_an[i + 2*N]  = -der_b_an[i]
    der_b_an[i + 3*N]  = der_b_an[i]
    
der_b_mat = schemes.der_b_mat(N, delta)

psi_vec = schemes.vectorise(psi)

der_b = der_b_mat@psi_vec

plt.plot(der_b)
plt.plot(der_b_an,'r--')
plt.title('first derivatives at boundary of sample function')
