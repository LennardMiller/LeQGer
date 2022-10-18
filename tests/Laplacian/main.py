''' tests laplacian functions and its inverse '''

import numpy as np
from leqger import space_schemes
import matplotlib.pyplot as plt
from scipy.linalg import solve

N = 64
L = 1

delta = L/(N - 1)

psi = np.zeros([N, N])
q = np.zeros([N, N])

for n in range(N):
    for m in range(N):
        
        x = n*delta
        y = m*delta
        
        # need to choose a function with zero normal derivatives at boundaries
        
        psi[n, m] = np.cos(2*np.pi*x)*np.cos(2*np.pi*y)
        q[n,m] = -8*np.pi**2*np.cos(2*np.pi*x)*np.cos(2*np.pi*y)

lap = space_schemes.lap_psi(N, delta)

psi_vec = space_schemes.vectorise(psi)
q_num_vec = lap@psi_vec
q_num = space_schemes.matricise(q_num_vec)

q_vec = space_schemes.vectorise(q)
psi_num_vec = solve(lap, q_vec)
psi_num = space_schemes.matricise(psi_num_vec)
psi_num += psi[0,0] - psi_num[0,0] 

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(psi)
ax1.set_title('psi (analytical)')
ax2.imshow(psi_num)
ax2.set_title('psi (numerical inverse)')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(q)
ax1.set_title('q (analytical)')
ax2.imshow(q_num)
ax2.set_title('q (numerical calculation)')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
err1 = ax1.imshow(q - q_num)
plt.colorbar(err1, ax = ax1)
ax1.set_title('error in q')
err2 = ax2.imshow(psi - psi_num)
plt.colorbar(err2, ax = ax2)
ax2.set_title('error in psi')
plt.show()

plt.imshow((space_schemes.matricise(lap@psi_num_vec) - q))
plt.title('error of applying laplacian and its inverse')
plt.colorbar()