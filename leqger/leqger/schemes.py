''' collection of spatial discretization schemes '''

import numpy as np
import matplotlib.pyplot as plt

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

def mat_coor_b(N, n,i,k,l):
    ''' gives back the matrix coefficients of the boundary derivative matrix.
    n is the number of boundary (0: western, 1: northern, 2: eastern, 3: south)
    and k,l is index of the input psi field.'''
        
    if k < 0:
        k = N + k
        
    if l < 0:
        l = N + l
        
    m = n*N + i
    n = k + N*l
    
    return (m,n)

def lap_psi(N, delta): 
    ''' computes matrix laplacian to act on vectorised psi, with dirichlet 
    boundary conditions on a face-centred scheme.'''
    
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



def J(psi, q, delta):
    ''' gives back the Jacobian between psi and q.'''
    
    N = np.shape(psi)[0]
    J = np.zeros([N,N])
    
    # compute Arakawa jacobian in the interior
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            J[i,j] += (psi[i+1,j] - psi[i-1,j])*(q[i,j+1] - q[i,j-1]) - (psi[i,j+1] - psi[i,j-1])*(q[i+1,j] - q[i-1,j]) # J++
            J[i,j] += psi[i+1,j]*(q[i+1,j+1] - q[i+1,j-1]) - psi[i-1,j]*(q[i-1,j+1] - q[i-1, j-1]) - psi[i,j+1]*(q[i+1,j+1] - q[i-1, j+1]) + psi[i,j-1]*(q[i+1,j-1] - q[i-1, j-1]) #J+x
            J[i,j] += -q[i+1,j]*(psi[i+1,j+1] -psi[i+1,j-1]) + q[i-1,j]*(psi[i-1,j+1] - psi[i-1, j-1]) + q[i,j+1]*(psi[i+1,j+1] - psi[i-1, j+1]) - q[i,j-1]*(psi[i+1,j-1] - psi[i-1, j-1]) #Jx+
            
    
    # apply off-centered scheme on the exterior
    
    # western and eastern boundaries
    for j in range(1, N - 1):
        # western
        J[0,j] += ((-3*psi[0,j] + 4*psi[1,j] - psi[2,j])*(q[0,j+1] - q[0,j-1]) - (psi[0,j+1] - psi[0,j-1])*(-3*q[0,j] + 4*q[1,j] - q[2,j])) #J++
        J[0,j] += (-3*psi[0,j]*(q[0,j+1] - q[0,j-1]) + 4*psi[1,j]*(q[1,j+1] - q[1,j-1]) - psi[2,j]*(q[2,j+1] - q[2,j-1]) - (psi[0,j+1]*(-3*q[0,j+1] + 4*q[1,j+1] - q[2,j+1]) - psi[0,j-1]*(-3*q[0,j-1] + 4*q[1,j-1] - q[2,j-1]))) # J+x
        J[0,j] += (q[0,j+1]*(-3*psi[0,j+1] + 4*psi[1,j+1] - psi[2,j+1]) - q[0,j-1]*(-3*psi[0,j-1] + 4*psi[1,j-1] - psi[2,j-1]) - (-3*q[0,j]*(psi[0,j+1] - psi[0,j-1]) + 4*q[1,j]*(psi[1,j+1] - psi[1,j-1]) - q[2,j]*(psi[2,j+1] - psi[2,j-1])))
        
        #eastern
        J[-1,j] += ((3*psi[-1,j] - 4*psi[-2,j] + psi[-3,j])*(q[-1,j+1] - q[-1,j-1]) - (psi[-1,j+1] - psi[-1,j-1])*(3*q[-1,j] - 4*q[-2,j] + q[-3,j])) #J++
        J[-1,j] += (3*psi[-1,j]*(q[-1,j+1] - q[-1,j-1]) - 4*psi[-2,j]*(q[-2,j+1] - q[-2,j-1]) + psi[-3,j]*(q[-3,j+1] - q[-3,j-1]) - (psi[-1,j+1]*(3*q[-1,j+1] - 4*q[-2,j+1] + q[-3,j+1]) - psi[-1,j-1]*(3*q[-1,j-1] - 4*q[-2,j-1] + q[-3,j-1]))) # J+x
        J[-1,j] += (-3*q[-1,j]*(psi[-1,j+1] - psi[-1,j-1]) + 4*q[-2,j]*(psi[-2,j+1] - psi[-2,j-1]) - q[-3,j]*(psi[-3,j+1] - psi[-3,j-1]) + (q[-1,j+1]*(3*psi[-1,j+1] - 4*psi[-2,j+1] + psi[-3,j+1]) - q[-1,j-1]*(3*psi[-1,j-1] - 4*psi[-2,j-1] + psi[-3,j-1]))) # Jx+
        
    # southern and northern
    for j in range(1, N - 1):
        # southern
        J[j,0] += (-(-3*psi[j,0] + 4*psi[j,1] - psi[j,2])*(q[j+1,0] - q[j-1,0]) + (psi[j+1,0] - psi[j-1,0])*(-3*q[j,0] + 4*q[j,1] - q[j,2])) #J++
        J[j,0] += -(-3*psi[j,0]*(q[j+1,0] - q[j-1,0]) + 4*psi[j,1]*(q[j+1,1] - q[j-1,1]) - psi[j,2]*(q[j+1,2] - q[j-1,2])) + (psi[j+1,0]*(-3*q[j+1,0] + 4*q[j+1,1] - q[j+1,2]) - psi[j-1,0]*(-3*q[j-1,0] + 4*q[j-1,1] - q[j-1,2])) # J+x
        J[j,0] += (-(3*q[j,0]*(psi[j+1,0] - psi[j-1,0]) - 4*q[j,1]*(psi[j+1,1] - psi[j-1,1]) + q[j,2]*(psi[j+1,2] - psi[j-1,2])) - (q[j+1,0]*(-3*psi[j+1,0] + 4*psi[j+1,1] - psi[j+1,2]) - q[j-1,0]*(-3*psi[j-1,0] + 4*psi[j-1,1] - psi[j-1,2]))) # Jx+
        
        #northern
        J[j,-1] += -(3*psi[j,-1] - 4*psi[j,-2] + psi[j,-3])*(q[j+1,-1] - q[j-1,-1]) + (psi[j+1,-1] - psi[j-1,-1])*(3*q[j,-1] - 4*q[j,-2] + q[j,-3]) #J++
        J[j,-1] += -(3*psi[j,-1]*(q[j+1,-1] - q[j-1,-1]) - 4*psi[j,-2]*(q[j+1,-2] - q[j-1,-2]) + psi[j,-3]*(q[j+1,-3] - q[j-1,-3])) + (psi[j+1,-1]*(3*q[j+1,-1] - 4*q[j+1,-2] + q[j+1,-3]) - psi[j-1,-1]*(3*q[j-1,-1] - 4*q[j-1,-2] + q[j-1,-3])) # J+x
        J[j,-1] += -(-3*q[j,-1]*(psi[j+1,-1] - psi[j-1,-1]) + 4*q[j,-2]*(psi[j+1,-2] - psi[j-1,-2]) - q[j,-3]*(psi[j+1,-3] - psi[j-1,-3])) - (q[j+1,-1]*(3*psi[j+1,-1] - 4*psi[j+1,-2] + psi[j+1,-3]) - q[j-1,-1]*(3*psi[j-1,-1] - 4*psi[j-1,-2] + psi[j-1,-3])) # Jx+
        
    # corners
    
    #SW
    J[0,0] += ((-3*psi[0,0] + 4*psi[1,0] - psi[2,0])*(-3*q[0,0] + 4*q[0,1] - q[0,2]) - (-3*psi[0,0] + 4*psi[0,1] - psi[0,2])*(-3*q[0,0] + 4*q[1,0] - q[2,0])) #J++
    J[0,0] += (-3*psi[0,0]*(-3*q[0,0] + 4*q[0,1] - q[0,2]) + 4*psi[1,0]*(-3*q[1,0] + 4*q[1,1] - q[1,2]) - psi[2,0]*(-3*q[2,0] + 4*q[2,1] - q[2,2]) - (-3*psi[0,0]*(-3*q[0,0] + 4*q[1,0] - q[2,0]) + 4*psi[0,1]*(-3*q[0,1] + 4*q[1,1] - q[2,1]) - psi[0,2]*(-3*q[0,2] + 4*q[1,2] - q[2,2])))
    J[0,0] += (-3*q[0,0]*(-3*psi[0,0] + 4*psi[1,0] - psi[2,0]) + 4*q[0,1]*(-3*psi[0,1] + 4*psi[1,1] - psi[2,1]) - q[0,2]*(-3*psi[0,2] + 4*psi[1,2] - psi[2,2]) - (-3*q[0,0]*(-3*psi[0,0] + 4*psi[0,1] - psi[0,2]) + 4*q[1,0]*(-3*psi[1,0] + 4*psi[1,1] - psi[1,2]) - q[2,0]*(-3*psi[2,0] + 4*psi[2,1] - psi[2,2])))
    
    #NW
    J[0,-1] += ((-3*psi[0,-1] + 4*psi[1,-1] - psi[2,-1])*(3*q[0,-1] - 4*q[0,-2] + q[0,-3]) - (3*psi[0,-1] - 4*psi[0,-2] + psi[0,-3])*(-3*q[0,-1] + 4*q[1,-1] - q[2,-1])) #J++
    J[0,-1] += (-3*psi[0,-1]*(3*q[0,-1] - 4*q[0,-2] + q[0,-3]) + 4*psi[1,-1]*(3*q[1,-1] - 4*q[1,-2] + q[1,-3]) - psi[2,-1]*(3*q[2,-1] - 4*q[2,-2] + q[2,-3]) - (3*psi[0,-1]*(-3*q[0,-1] + 4*q[1,-1] - q[2,-1]) - 4*psi[0,-2]*(-3*q[0,-2] + 4*q[1,-2] - q[2,-2]) + psi[0,-3]*(-3*q[0,-3] + 4*q[1,-3] - q[2,-3])))
    J[0,-1] += (-(-3*q[0,-1]*(3*psi[0,-1] - 4*psi[0,-2] + psi[0,-3]) + 4*q[1,-1]*(3*psi[1,-1] - 4*psi[1,-2] + psi[1,-3]) - q[2,-1]*(3*psi[2,-1] - 4*psi[2,-2] + psi[2,-3])) + (3*q[0,-1]*(-3*psi[0,-1] + 4*psi[1,-1] - psi[2,-1]) - 4*q[0,-2]*(-3*psi[0,-2] + 4*psi[1,-2] - psi[2,-2]) + q[0,-3]*(-3*psi[0,-3] + 4*psi[1,-3] - psi[2,-3])))
    
    #SE
    J[-1,0] += ((3*psi[-1,0] - 4*psi[-2,0] + psi[-3,0])*(-3*q[-1,0] + 4*q[-1,1] - q[-1,2]) - (-3*psi[-1,0] + 4*psi[-1,1] - psi[-1,2])*(3*q[-1,0] - 4*q[-2,0] + q[-3,0])) #J++
    J[-1,0] += (3*psi[-1,0]*(-3*q[-1,0] + 4*q[-1,1] - q[-1,2]) - 4*psi[-2,0]*(-3*q[-2,0] + 4*q[-2,1] - q[-2,2]) + psi[-3,0]*(-3*q[-3,0] + 4*q[-3,1] - q[-3,2]) - (-3*psi[-1,0]*(3*q[-1,0] - 4*q[-2,0] + q[-3,0]) + 4*psi[-1,1]*(3*q[-1,1] - 4*q[-2,1] + q[-3,1]) - psi[-1,2]*(3*q[-1,2] - 4*q[-2,2] + q[-3,2])))
    J[-1,0] += (-(3*q[-1,0]*(-3*psi[-1,0] + 4*psi[-1,1] - psi[-1,2]) - 4*q[-2,0]*(-3*psi[-2,0] + 4*psi[-2,1] - psi[-2,2]) + q[-3,0]*(-3*psi[-3,0] + 4*psi[-3,1] - psi[-3,2])) + (-3*q[-1,0]*(3*psi[-1,0] - 4*psi[-2,0] + psi[-3,0]) + 4*q[-1,1]*(3*psi[-1,1] - 4*psi[-2,1] + psi[-3,1]) - q[-1,2]*(3*psi[-1,2] - 4*psi[-2,2] + psi[-3,2])))
    
    #NE
    J[-1,-1] += ((3*psi[-1,-1] - 4*psi[-2,-1] + psi[-3,-1])*(3*q[-1,-1] - 4*q[-1,-2] + q[-1,-3]) - (3*psi[-1,-1] - 4*psi[-1,-2] + psi[-1,-3])*(3*q[-1,-1] - 4*q[-2,-1] + q[-3,-1])) #J++
    J[-1,-1] += 3*psi[-1,-1]*(3*q[-1,-1] - 4*q[-1,-2] + q[-1,-3]) - 4*psi[-2,-1]*(3*q[-2,-1] - 4*q[-2,-2] + q[-2,-3]) + psi[-3,-1]*(3*q[-3,-1] - 4*q[-3,-2] + q[-3,-3]) - (3*psi[-1,-1]*(3*q[-1,-1] - 4*q[-2,-1] + q[-3,-1]) - 4*psi[-1,-2]*(3*q[-1,-2] - 4*q[-2,-2] + q[-3,-2]) + psi[-1,-3]*(3*q[-1,-3] - 4*q[-2,-3] + q[-3,-3]))
    J[-1,-1] += -(3*q[-1,-1]*(3*psi[-1,-1] - 4*psi[-1,-2] + psi[-1,-3]) - 4*q[-2,-1]*(3*psi[-2,-1] - 4*psi[-2,-2] + psi[-2,-3]) + q[-3,-1]*(3*psi[-3,-1] - 4*psi[-3,-2] + psi[-3,-3])) + (3*q[-1,-1]*(3*psi[-1,-1] - 4*psi[-2,-1] + psi[-3,-1]) - 4*q[-1,-2]*(3*psi[-1,-2] - 4*psi[-2,-2] + psi[-3,-2]) + q[-1,-3]*(3*psi[-1,-3] - 4*psi[-2,-3] + psi[-3,-3]))
    
    J *= 1/(12*delta**2)
    
    return J
    
    
def der_b(N, delta):
    ''' gives back the matrix acting on psi to get an array containing all first
    derivatives at the boundaries (see bc_mat function for definition of output vector)'''
    
    der_b_mat = np.zeros([4*N, N**2])
    
    for i in range(N):
        # western 
        der_b_mat[mat_coor_b(N,0,i,0,i)] = 3
        der_b_mat[mat_coor_b(N,0,i,1,i)] = -1/3
        
        # northern
        der_b_mat[mat_coor_b(N,1,i,i,-1)] = -3
        der_b_mat[mat_coor_b(N,1,i,i,-2)] = 1/3
        
        # eastern
        der_b_mat[mat_coor_b(N,2,i,-1,i)] = -3
        der_b_mat[mat_coor_b(N,2,i,-2,i)] = 1/3
        
        # southern
        der_b_mat[mat_coor_b(N,3,i,i,0)] = 3
        der_b_mat[mat_coor_b(N,3,i,i,1)] = -1/3
        
    der_b_mat *= 1/delta
    
    return der_b_mat


def split_M(M, N):
    ''' splits the matrix to act on q into the parts acting on lap_q_(interior) and
    lap_q_(boundaries) '''
    
    # interior matrices are designed such that the input vector is counted along the
    # grid just like with an entire input field, but boundary values are ommited.
    
    # for second derivatives of q in x
    
    M_b_west = np.zeros([4*N, N])
    M_b_east = np.zeros([4*N, N])
    M_int_x = np.zeros([4*N, N**2 - 2*N])
    
    for i in range(N):
        M_b_west[:,i] = M[:,N*i]
        M_b_east[:,i] = M[:,N - 1 + N*i]
        M_int_x[:,i*(N-2):(i+1)*(N-2)] = M[:,N*i + 1:N - 1 + N*i]
        
    # for second derivatives of q in y
    
    M_b_south = M[:,:N]
    M_b_north = M[:,(N**2 - N):]
    M_int_y = M[:,N:(N**2 - N)]
    
    # stack matrices
    
    M_b = np.zeros([4*N, 4*N])
    
    M_b[:,:N] = M_b_west
    M_b[:,N:2*N] = M_b_north
    M_b[:,2*N:3*N] = M_b_east
    M_b[:,3*N:4*N] = M_b_south
    
    return M_b, M_int_x, M_int_y
        
    
    
def build_matrices(N, delta):
    ''' builds all matrices at the start of the simulation '''
    
    Lap_psi = lap_psi(N,delta)
    lap_psi_inv = np.linalg.inv(Lap_psi)
    del Lap_psi
    
    M = der_b(N, delta)@lap_psi_inv
    
    M_b, M_int_x, M_int_y = split_M(M,N)
    
    M_b_inv = np.linalg.inv(M_b)
    
    return M_b_inv, M, lap_psi_inv, M_int_x, M_int_y, M_b

def lap_q(N, delta, dt, q_v, Jac, tau, M, M_int_x, M_int_y, M_b_inv, M_b):
    ''' function to calculate the laplacian of q in the entire domain '''
    
    # interior
    
    q = matricise(q_v)
    
    lap_q_x = np.zeros([N,N])
    lap_q_y = np.zeros([N,N])
    
    for i in range(1, N-1):
        for j in range(N):
            lap_q_x[i,j] = q[i-1,j] - 2*q[i,j] + q[i+1,j]
            
    for i in range(N):
        for j in range(1, N-1):
            lap_q_y[i,j] = q[i,j-1] - 2*q[i,j] + q[i,j+1]
            
    lap_q_x /= delta**2
    lap_q_y /= delta**2
    
    lap_q_x_cut = np.zeros(N**2 - 2*N)
    lap_q_y_cut = np.zeros(N**2 - 2*N)
    
    for i in range(1, N-1):
        for j in range(N):
            lap_q_x_cut[(i-1) + (N-2)*j] = lap_q_x[i,j]
            
    for i in range(N):
        for j in range(1, N-1):
            lap_q_y_cut[i + N*(j-1)] = lap_q_y[i,j]
    
    
    # build right hand side to invert
    
    RHS = -M@(Jac + tau + q_v/dt) - M_int_x@lap_q_x_cut - M_int_y@lap_q_y_cut 
    
    # calculate the boundary values of the laplacian
    
    lap_q_b = np.linalg.solve(M_b,RHS)
    
    # fit them into the laplacian
    
    lap_q_x[0,:] = lap_q_b[N-1::-1]
    lap_q_y[:,-1] = lap_q_b[N:2*N]
    lap_q_x[-1,:] = lap_q_b[2*N:3*N]
    lap_q_y[:,0] = lap_q_b[-1:3*N-1:-1]
    
    lap_q = lap_q_x + lap_q_y
    
    lap_q = vectorise(lap_q)
    
    return lap_q
            
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    