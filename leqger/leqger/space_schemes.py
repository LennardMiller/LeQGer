''' collection of spatial discretization schemes '''

import numpy as np

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

def lap_psi_corner(N, delta): 
    ''' computes matrix laplacian to act on vectorised psi, with neumann 
    boundary conditions on an corner-centred scheme.'''
    
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

def lap_psi_face(N, delta): 
    ''' computes matrix laplacian to act on vectorised psi, with neumann 
    boundary conditions on an corner-centred scheme.'''
    
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
        tlap[mat_coor(N,0,j,0,j)] = -6
        tlap[mat_coor(N,0,j,1,j)] = 4/3
        tlap[mat_coor(N,0,j,0,j+1)] = 1
        tlap[mat_coor(N,0,j,0,j-1)] = 1
        
        # eastern
        tlap[mat_coor(N,-1,j,-1,j)] = -6
        tlap[mat_coor(N,-1,j,-2,j)] = 4/3
        tlap[mat_coor(N,-1,j,-1,j+1)] = 1
        tlap[mat_coor(N,-1,j,-1,j-1)] = 1
        
    # scheme on northern and southern boundaries
    
    for i in range(1, N - 1):
        # southern
        tlap[mat_coor(N,i,0,i,0)] = -6
        tlap[mat_coor(N,i,0,i,1)] = 4/3
        tlap[mat_coor(N,i,0,i+1,0)] = 1
        tlap[mat_coor(N,i,0,i-1,0)] = 1
    
        # northern
        tlap[mat_coor(N,i,-1,i,-1)] = -6
        tlap[mat_coor(N,i,-1,i,-2)] = 4/3
        tlap[mat_coor(N,i,-1,i+1,-1)] = 1
        tlap[mat_coor(N,i,-1,i-1,-1)] = 1
        
    # corners
    
    # south-west
    tlap[mat_coor(N,0,0,0,0)] = -8
    tlap[mat_coor(N,0,0,1,0)] = 4/3
    tlap[mat_coor(N,0,0,0,1)] = 4/3
    
    # north-west
    tlap[mat_coor(N,0,-1,0,-1)] = -8
    tlap[mat_coor(N,0,-1,1,-1)] = 4/3
    tlap[mat_coor(N,0,-1,0,-2)] = 4/3
    
    # south-east
    tlap[mat_coor(N,-1,0,-1,0)] = -8
    tlap[mat_coor(N,-1,0,-1,1)] = 4/3
    tlap[mat_coor(N,-1,0,-2,0)] = 4/3
    
    # north-east
    tlap[mat_coor(N,-1,-1,-1,-1)] = -8
    tlap[mat_coor(N,-1,-1,-2,-1)] = 4/3
    tlap[mat_coor(N,-1,-1,-1,-2)] = 4/3
    
    
    
    tlap /= delta**2
    
    return tlap
