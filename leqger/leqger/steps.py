''' Functions that group steps in main routine '''

from leqger import schemes
import numpy as np

def build_F(q, t):
    ''' builds time derivative forcing '''
    
    return np.sin(t)

def advance(q, t, dt, scheme = 'euler'):
    ''' time steps state vector q according to the chosen scheme '''
    
    if scheme == 'euler':
        q_n = schemes.euler(q, t, dt)
        
    if scheme == 'runge_kutta':
        q_n = schemes.runge_kutta(q, t, dt)
        
    return q_n

def check_stop(criterion, t = 0, i = 0, i_tot = 0, t_fin = 0):
    ''' checks whether code has finished executing '''

    if criterion == 'i_tot':
        if i <= i_tot:
            c = False
        else:
            c = True
    
    if criterion == 't_fin':
        if t <= t_fin:
            c = False
        else:
            c = True
            
    return c

def output(q_output, q):
    
    ''' writes output into different vector q_output '''
    
    return np.append(q_output, q)