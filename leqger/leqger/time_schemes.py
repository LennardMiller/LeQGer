''' collection of numerical time stepping schemes '''

from main import build_F

def euler(q, t, dt):
    ''' euler time stepping '''
    
    q_n = q + build_F(q, t)*dt
    
    return q_n

def runge_kutta(q, t, dt):
    ''' 4th order runge kutta method '''
    
    k1 = build_F(q, t)*dt
    k2 = build_F(q + k1/2, t + dt/2)*dt
    k3 = build_F(q + k2/2, t + dt/2)*dt
    k4 = build_F(q + k3, t + dt)*dt
    
    q_n = q + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return q_n
