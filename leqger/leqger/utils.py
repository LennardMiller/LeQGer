''' Functions that group steps in main routine. 
The function forcing should be in the main script. '''

from leqger import schemes

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
    
    q_output.append(schemes.matricise(q))
    
    return q_output
