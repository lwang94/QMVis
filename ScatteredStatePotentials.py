# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:09:17 2019

@author: lawre
"""

import numpy as np
import scipy.special as sp
import MiscFunc as mf
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

class FreePotential:
    
    def __init__(self, x):
        
        self.x = x
    
    def potential(self):
        return np.zeros(len(self.x))
    
    def hamiltonian(self, wfunc, particle, finitediff_scheme = 'central', h_bar = 6.626e-34/(2*np.pi)):
        
        kinetic = mf.kinetic_op(self.x, wfunc, particle, h_bar = h_bar, finitediff_scheme = finitediff_scheme)       
        
        return kinetic
    
    def unnormalized_eigenfunc(self, k, A = 1):
        """not physically realizable state. needs to be tested"""
        return A*np.exp(1j*k*self.x)
    
    def timedep_unnormalized_eigenfunc(self, t, k, particle, A = 1, h_bar = 6.626e-34/(2*np.pi)):
        """not physically realizable state. needs to be tested"""
        return A*np.exp(1j*(k*self.x - self.eigenvalue(k, particle, h_bar)*t))
    
    def eigenvalue(self, k, particle, h_bar = 6.626e-34/(2*np.pi)):
        """needs to be tested"""
        return (h_bar*k**2)/(2*particle.m)