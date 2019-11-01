# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:24:19 2019

@author: lawre
"""
import numpy as np
import scipy.integrate as integrate

def first_derivative(x, y, finitediff_scheme = 'central'):
    """only works if x is uniform"""
    if finitediff_scheme == 'central':
        return np.gradient(y, x[1]-x[0])
    
    elif finitediff_scheme == 'five point stencil':
        y = [0, 0] + list(y) + [0, 0]
        numerator = np.array([y[i-2] - 8*y[i-1] + 8*y[i+1] - y[i+2] for i in range(2, len(y)-2)])
        denominator = 12*(x[1] - x[0])        
        return numerator/denominator        
        
        
def second_derivative(x, y, finitediff_scheme = 'central'):
    """only works if x is uniform"""
    if finitediff_scheme == 'central':
        y = [0] + list(y) + [0]
        numerator = np.array([y[i-1] - 2*y[i] + y[i+1] for i in range(1, len(y)-1)])
        denominator = (x[1]-x[0])**2
        return numerator/denominator  
    
    elif finitediff_scheme == 'five point stencil':
        y = [0, 0] + list(y) + [0, 0]
        numerator = np.array([-y[i-2] + 16*y[i-1] - 30*y[i] + 16*y[i+1] - y[i+2] for i in range(2, len(y)-2)])
        denominator = 12*(x[1] - x[0])**2        
        return numerator/denominator  

def probdensity(wfunc):
    return wfunc*np.conj(wfunc)

def normalize(x, wfunc):
    return 1/np.sqrt(integrate.trapz(probdensity(wfunc), x = x))*wfunc      


#Observables
def momentum_op(x, wfunc, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
    """needs to be tested"""
    return -1j*h_bar*first_derivative(x, wfunc, finitediff_scheme = finitediff_scheme)

def position_op(x, wfunc):
    """needs to be tested"""
    return x*wfunc

def kinetic_op(x, wfunc, particle, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
    return (-h_bar**2/(2*particle.m))*second_derivative(x, wfunc, finitediff_scheme = finitediff_scheme)   