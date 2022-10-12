''' collection of numerical schemes '''

from leqger import steps

def euler(q, t, dt):
    ''' euler time stepping '''
    
    q_n = q + steps.build_F(q, t)*dt
    
    return q_n

def runge_kutta(q, t, dt):
    ''' 4th order runge kutta method '''
    
    k1 = steps.build_F(q, t)*dt
    k2 = steps.build_F(q + k1/2, t + dt/2)*dt
    k3 = steps.build_F(q + k2/2, t + dt/2)*dt
    k4 = steps.build_F(q + k3, t + dt)*dt
    
    q_n = q + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return q_n