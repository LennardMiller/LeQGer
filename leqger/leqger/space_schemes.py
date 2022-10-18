''' collection of spatial discretization schemes '''

import numpy as np

def lap(q, delta, BC = 'q_Bruno'):
    '''standard 2nd order Laplacian '''
    
    q_pad = pad(q, BC = 'q_Bruno')
    
    Lap = (q_pad[2:,1:-1] + q_pad[:-2,1:-1] + q_pad[1:-1,2:] + q_pad[1:-1,:-2] -4*q_pad[1:-1, 1:-1])/delta**2
    
    return Lap
    
def pad(q, BC):
    ''' function to pad fields with different boundary conditions'''
    
    if BC == 'psi':
        q = q
    return q
        

def vectorise(psi):
    
    '''vectorises field for x-coordinates to be grouped together, from 0 increasing '''
    
    N = np.shape(psi)[0]
    
    psi_vec = np.zeros(N**2)
    
    for n in range(N):
        psi_vec[(N*n):(N*(n+1))] = psi[:,n]
        
    return psi_vec

def matricise(psi_vec):
    
    ''' transforms vectorised field back into matrix format '''
    
    N = int(np.sqrt(len(psi_vec)))
    
    psi = np.zeros([N,N])
    
    for n in range(N):
        psi[:,n]  = psi_vec[(N*n):(N*(n+1))]
        
    return psi

def mat_coor(N, i,j,k,l):
    ''' gives back the matrix coordinates of the coefficient of (i,j)th element
    of the output field being influenced by the (k,l)th element of the input field'''
    
    if i < 0:
        i = N + i 
        
    if j < 0:
        j = N + j 
        
    if k < 0:
        k = N + k
        
    if l < 0:
        l = N + l
        
    m = i + N*j
    n = k + N*l
    
    return (m,n)

def lap_psi(N, delta): 
    ''' computes matrix laplacian to act on vectorised psi '''
    
    tlap = np.zeros([N**2, N**2])
    
    # scheme in the interior
    
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            tlap[mat_coor(N,i,j,i,j)] = -4
            tlap[mat_coor(N,i,j,i+1,j)] = 1
            tlap[mat_coor(N,i,j,i-1,j)] = 1
            tlap[mat_coor(N,i,j,i,j+1)] = 1
            tlap[mat_coor(N,i,j,i,j-1)] = 1
            
    #scheme on eastern and western boundaries
    
    for j in range(1, N - 1):
        # western
        tlap[mat_coor(N,0,j,0,j)] = -4
        tlap[mat_coor(N,0,j,1,j)] = 2
        tlap[mat_coor(N,0,j,0,j+1)] = 1
        tlap[mat_coor(N,0,j,0,j-1)] = 1
        
        # eastern
        tlap[mat_coor(N,-1,j,-1,j)] = -4
        tlap[mat_coor(N,-1,j,-2,j)] = 2
        tlap[mat_coor(N,-1,j,-1,j+1)] = 1
        tlap[mat_coor(N,-1,j,-1,j-1)] = 1
        
    # scheme on northern and southern boundaries
    
    for i in range(1, N - 1):
        # southern
        tlap[mat_coor(N,i,0,i,0)] = -4
        tlap[mat_coor(N,i,0,i,1)] = 2
        tlap[mat_coor(N,i,0,i+1,0)] = 1
        tlap[mat_coor(N,i,0,i-1,0)] = 1
    
        # northern
        tlap[mat_coor(N,i,-1,i,-1)] = -4
        tlap[mat_coor(N,i,-1,i,-2)] = 2
        tlap[mat_coor(N,i,-1,i+1,-1)] = 1
        tlap[mat_coor(N,i,-1,i-1,-1)] = 1
        
    # corners
    
    # south-west
    tlap[mat_coor(N,0,0,0,0)] = -4
    tlap[mat_coor(N,0,0,1,0)] = 2
    tlap[mat_coor(N,0,0,0,1)] = 2
    
    # north-west
    tlap[mat_coor(N,0,-1,0,-1)] = -4
    tlap[mat_coor(N,0,-1,1,-1)] = 2
    tlap[mat_coor(N,0,-1,0,-2)] = 2
    
    # south-east
    tlap[mat_coor(N,-1,0,-1,0)] = -4
    tlap[mat_coor(N,-1,0,-1,1)] = 2
    tlap[mat_coor(N,-1,0,-2,0)] = 2
    
    # north-east
    tlap[mat_coor(N,-1,-1,-1,-1)] = -4
    tlap[mat_coor(N,-1,-1,-2,-1)] = 2
    tlap[mat_coor(N,-1,-1,-1,-2)] = 2
    
    
    
    tlap /= delta**2
    
    return tlap
