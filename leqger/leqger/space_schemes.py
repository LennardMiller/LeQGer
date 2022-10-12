''' collection of spatial discretization schemes '''

def laplacian_2nd(q, delta, BC = 'q_Bruno'):
    '''standard 2nd order Laplacian '''
    
    q_pad = pad(q, BC = 'q_Bruno')
    
    Lap = (q_pad[2:,1:-1] + q_pad[:-2,1:-1] + q_pad[1:-1,2:] + q_pad[1:-1,:-2] -4*q_pad[1:-1, 1:-1])/delta**2
    
    return Lap
    
def pad(q, BC):
    ''' function to pad fields with different boundary conditions'''
    
    if BC == 'q_Bruno':
        q = q
    return q
        