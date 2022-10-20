''' tests the jacobian against a known analytical function '''

import numpy as np
from leqger import space_schemes
import matplotlib.pyplot as plt

N = 50
L = 1

delta = L/N

psi = np.zeros([N, N])
q = np.zeros([N, N])

J_ana = np.zeros([N,N])

for n in range(N):
    for m in range(N):
        
        x = (n + 1/2)*delta
        y = (m + 1/2)*delta
        
        psi[n, m] =  x*(x-1)*y*(y-1) #x*y   # np.sin(np.pi*x)*np.sin(np.pi*y)
        q[n,m] =    2*y*(y-1) + 2*x*(x-1) #2*(y**2 + x**2) # np.cos(np.pi*x)*np.cos(np.pi*y)
        
        # analytical solution

        J_ana[n,m] = (2*x*y**2 - y**2 - 2*x*y + y)*(4*y-2) - (2*x**2*y - 2*x*y - x**2 + x)*(4*x-2)  #y*4*y - x*4*x #-np.pi**2*(np.cos(np.pi*x)**2*np.sin(np.pi*y)**2 - np.sin(np.pi*x)**2*np.cos(np.pi*y)**2)

J_num = space_schemes.J(psi,q,delta)
Diff = J_num - J_ana



fig, (ax1, ax2) = plt.subplots(1, 2)
err1 = ax1.imshow(J_num.T,origin = 'lower')
plt.colorbar(err1, ax = ax1)
ax1.set_title('numerical Jacobian')
err2 = ax2.imshow(J_ana.T, origin = 'lower')
plt.colorbar(err2, ax = ax2)
ax2.set_title('analytical Jacobian')
plt.show()

plt.imshow(Diff[:,:].T, origin = 'lower')
plt.colorbar()

Int_circ = np.sum(np.sum(J_num))
Int_enst = np.sum(np.sum(J_num*q))
Int_en = np.sum(np.sum(J_num*psi))

print(f'circulation integral gives {Int_circ}')
print(f'enstrophy integral gives {Int_enst}')
print(f'energy integral gives {Int_en}')