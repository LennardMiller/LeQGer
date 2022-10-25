import matplotlib.pyplot as plt
import numpy as np
from leqger import utils
from leqger import schemes

# define physical parameters

beta = 1
nu = 1
tau0 = 0.0001
L = 10

# define grid parameters

N = 100 # number of points

delta = L/(N - 1)

# define stopping criterion (critical time vs. fixed number of time steps)

crit = 't_fin'
t = 0
t_fin = 1

# time step

dt = 0.1

# initiate q

q = np.zeros([N, N])

# initial conditions

for n in range(N):
    for m in range(N):
        
        x = n*delta
        y = m*delta
        
        q[n, m] = 0
    
q = schemes.vectorise(q)

# define forcing

def forcing(tau0, L, delta):
    
    ''' builds tau '''
        
    tau = np.zeros([N, N])
    
    for n in range(N):
        for m in range(N):
            
            y = m*delta
            
            tau[n,m] = tau0*np.pi/L*np.sin(np.pi*y/L)
            
    tau = schemes.vectorise(tau)
            
    return tau

# initiate output vector

q_out = [schemes.matricise(q)]


# start calculations

# build matrix to inverse q

M_b_inv, M, lap_psi_inv, M_int_x, M_int_y, M_b = schemes.build_matrices(N, delta)

# build forcing matrix

tau = forcing(tau0, L, delta)

while True: # main loop
    
    psi = lap_psi_inv@q
    
    Jac = schemes.vectorise(schemes.J(schemes.matricise(psi), schemes.matricise(q), delta))
    
    lap_q = schemes.lap_q(N, delta, dt, q, Jac, tau, M, M_int_x, M_int_y, M_b_inv, M_b)
    
    q = q + (-Jac + nu*lap_q + tau)*dt # time step q
    
    q_out = utils.output(q_out, q) # save output
    
    t = t + dt # advance time
    
    if utils.check_stop(crit, t = t + dt, t_fin = t_fin): # checks if the next time step will be computed or not
        t = t - dt
        break
    
    
q_out = np.array(q_out)

