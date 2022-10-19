''' tests laplacian functions and its inverse '''

import numpy as np
from leqger import space_schemes
import matplotlib.pyplot as plt

N = 100
L = 1

delta = L/N

psi = np.zeros([N, N])
q = np.zeros([N, N])

for n in range(N):
    for m in range(N):
        
        x = (n + 1/2)*delta
        y = (m + 1/2)*delta
        
        # need to choose a function with zero dirichlet at boundaries
        
        psi[n, m] = x*(x-1)*y*(y-1) 
        q[n,m] =  2*y*(y-1) + 2*x*(x-1)

lap = space_schemes.lap_psi_face(N, delta)

psi_vec = space_schemes.vectorise(psi)
q_num_vec = lap@psi_vec
q_num = space_schemes.matricise(q_num_vec)

q_vec = space_schemes.vectorise(q)
lap_inv = np.linalg.inv(lap)
psi_num_vec = lap_inv@q_vec
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

# show gauss integral equality (analytical value is 2/3)

Int_surf = np.sum(np.sum(q, axis = 0), axis = 0)*delta**2
Int_line = 0

for i in range(N):
    # southern and northern boundaries
    
    Int_line += -3*psi[i,0] + 1/3*psi[i,1]
    Int_line += -3*psi[i,-1] + 1/3*psi[i,-2]
    
    # western and eastern boundaries
    
    Int_line += -3*psi[0,i] + 1/3*psi[1,i]
    Int_line += -3*psi[-1,i] + 1/3*psi[-2,i]

print(f'Line integral of boundary flux = {Int_line}')
print(f'Surface integral of q = {Int_surf}')
print('analytic value is -2/3')
    

