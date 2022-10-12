''' 2D test of scheme with dq/dt = sin(2pi*t)*exp(-(x - x0)^2/(2*sigma^2)). 
All the solutions are negative cosines. '''

import matplotlib.pyplot as plt
import numpy as np

# define grid parameters

L = 10
N = 10 # number of grid cells, Number of points is this +1

delta = L/N

# define stopping criterion (critical time vs. fixed number of time steps)

crit = 't_fin'
t = 0
t_fin = 2

# time step

dt = 0.0243

# initial conditions

x0 = L/2
y0 = L/2
sigma = 1

q = np.zeros([N + 1, N + 1])
for n in range(N + 1):
    for m in range(N +1):
        
        x = n*delta
        y = m*delta
        
        q[n, m] = -1/(2*np.pi)*np.exp(-1/(2*sigma**2)*((x - x0)**2 + (y - y0)**2))
    

def forcing(q, t):
    ''' builds time derivative forcing '''
        
    F = np.zeros([N + 1, N + 1])
    
    for n in range(N + 1):
        for m in range(N +1):
            
            x = n*delta
            y = m*delta
            
            F[n,m] = np.sin(2*np.pi*t)*np.exp(-1/(2*sigma**2)*((x - x0)**2 + (y - y0)**2))
    
    return F

# initiate output vector

q_out = [q]


if __name__ == '__main__':
    
    ''' defining the forcing function outside the __main__ block lets me define it 
    in the same script as the input parameters and the main loop. This way I can 
    import it in the modules steps.py and schemes.py without causing 
    a circular import error (leqger functions importing main, main importing 
    leqger...)''' 

    from leqger import steps
    
    while True: # main loop
        
        tau = forcing(q, t)
    
        q = steps.advance(q, t, dt, scheme = 'euler') # time step q
        
        q_out = steps.output(q_out, q) # save output
        
        t = t + dt # advance time
        
        if steps.check_stop(crit, t = t + dt, t_fin = t_fin): # checks if the next time step will be computed or not
            t = t - dt
            break
        
        
    q_out = np.array(q_out)
    
    # plot solution    
        
    plt.plot(np.array(list(range(0, np.shape(q_out)[0])))*dt, q_out[:,5,5])
    plt.xlabel('t')
    plt.ylabel('q')

