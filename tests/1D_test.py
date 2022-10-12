from leqger import steps
import matplotlib.pyplot as plt
import numpy as np

# define stopping criterion (critical time vs. fixed number of time steps)

crit = 't_fin'
t = 0
t_fin = 2*np.pi

# time step

dt = 0.1

# initial conditions

q = -1

# initiate output vector

q_out = np.array([q])


while True: # main loop
    
    q = steps.advance(q, t, dt, scheme = 'runge_kutta') # time step q
    
    q_out = steps.output(q_out, q) # save output
    
    t = t + dt # advance time
    
    if steps.check_stop(crit, t = t + dt, t_fin = t_fin): # checks if the next time step will be computed or not
        break
    
    
# plot solution    
    
plt.plot(np.linspace(0, t, int(t_fin/dt)+1), q_out)
plt.xlabel('t')
plt.ylabel('q')