# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 13:32:44 2019

@author: lawre
"""

import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
import MiscFunc as mf


#class FiniteSquareWell:
#    
#    def __init__(self, x, L, V0):
#        self.x = x
#        self.L = L
#        self.V0 = V0
#        
#    def potential(self):
#        potential = np.piecewise(self.x,
#                                 [np.any([self.x<=-self.L/2, self.x>=self.L/2], axis = 0), np.all([self.x>-self.L/2, self.x<self.L/2], axis = 0)],
#                                 [0, -self.V0])
#        return potential        
#    
#    def hamiltonian(self, wfunc, particle, finitediff_scheme = 'central', h_bar = 6.626e-34/(2*np.pi)):
#        kinetic = mf.kinetic_op(self.x, wfunc, particle, h_bar = h_bar, finitediff_scheme = finitediff_scheme)
#
#        potential = self.potential()*wfunc
#
#        new_wfunc = kinetic + potential
#
#        return new_wfunc
#
#    def eigenfunc(self, n):
#        return self.x        
        


def func_even(z, z0):
    return np.tan(z) - np.sqrt((z0/z)**2 - 1)

def func_odd(z, z0):
    return -1/np.tan(z) - np.sqrt((z0/z)**2 - 1)

def func_n(z, z0, n):
    if n%2 == 0:
        result = func_even(z, z0)
    else:
        result = func_odd(z, z0)
    
    return result
#
a = 1e-10
V0 = 1e-17

h_bar =  6.626e-34/(2*np.pi)
m = 9.11e-31
z0 = a*np.sqrt(2*m*V0)/h_bar
n = 4
res = opt.brentq(func_n, n*np.pi/2, (n+1)*np.pi/2, args = (z0, n))
print (res)
z = np.linspace(0, 10, 1000)
#    plt.plot(z, np.tan(z))
#plt.plot(z, -1/np.tan(z))
plt.plot(z, func_even(z, z0))
plt.plot(z, func_odd(z, z0))
plt.ylim(-10, 10)