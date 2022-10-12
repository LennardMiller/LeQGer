import matplotlib.pyplot as plt
import numpy as np
from leqger import space_schemes

# define grid parameters

L = 10
N = 3 # number of grid cells, Number of points is this value +1

delta = L/N

# define stopping criterion (critical time vs. fixed number of time steps)

crit = 't_fin'
t = 0
t_fin = 10

# time step

dt = 0.1

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
    
# define forcing

def forcing(q, t):
    
    ''' builds tau '''
        
    tau = np.zeros([N + 1, N + 1])
    
    for n in range(N + 1):
        for m in range(N +1):
            
            x = n*delta
            y = m*delta
            
            tau[n,m] = np.sin(2*np.pi*t)*np.exp(-1/(2*sigma**2)*((x - x0)**2 + (y - y0)**2)) #exponential envelope of sinusoidal oscillation in x/y
    
    return tau

# choose which spacial schemes to use in constructing F

def build_F(q, t):
    ''' builds time derivative forcing of dq/dt = F'''
    
    F = forcing(q,t)
    
    return F

# initiate output vector

q_out = [q]


# start calculations

if __name__ == '__main__':
    
    ''' defining the forcing function outside the __main__ block lets me define it 
    in the same script as the input parameters and the main loop. This way I can 
    import it in the modules steps.py and schemes.py without causing 
    a circular import error (leqger functions importing main, main importing 
    leqger...)''' 

    from leqger import steps
    
    while True: # main loop
        
        tau = forcing(q, t)
        
        q = steps.advance(q, t, dt, scheme = 'runge_kutta') # time step q
        
        q_out = steps.output(q_out, q) # save output
        
        t = t + dt # advance time
        
        if steps.check_stop(crit, t = t + dt, t_fin = t_fin): # checks if the next time step will be computed or not
            t = t - dt
            break
        
        
    q_out = np.array(q_out)
    
    # plot solution    
        
    plt.plot(np.array(list(range(0, np.shape(q_out)[0])))*dt, q_out[:,1,1])
    plt.xlabel('t')
    plt.ylabel('q')

