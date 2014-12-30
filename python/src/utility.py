__author__ = 'kjoseph'

import numpy as np
import signal

def epa_to_index(option):
    if option == 'e':
        return 0
    if option == 'p':
        return 1
    if option == 'a':
        return 2

def abo_to_index(option):
    if option == 'a':
        return 0
    if option == 'b':
        return 1
    if option == 'o':
        return 2
    return -1

def epa_abo_to_index(option):
    return epa_to_index(option[1])+3 * abo_to_index(option[0])


def draw_from_inverse_chi_square(degrees_of_freedom,scale, n=1):
    return (degrees_of_freedom * scale)/np.random.chisquare(degrees_of_freedom)

def stringify(data):
    return [str(x) for x in data]

def tab_stringify_newline(data,newline=True):
    to_return = "\t".join(stringify(data))
    if newline:
         return to_return + "\n"
    return to_return


#handle SIGINT from SyncManager object
def mgr_sig_handler(signal, frame):
    print 'not closing the mgr'

#initilizer for SyncManager
def mgr_init():
    signal.signal(signal.SIGINT, mgr_sig_handler)
    print 'initialized mananger'