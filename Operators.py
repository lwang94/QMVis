# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:24:19 2019

@author: lawre
"""
import numpy as np
import scipy.integrate as integrate

def first_derivative(x, y, finitediff_scheme = 'central'):
    """
    Uses a finite difference method to approximate the first order derivative of a function.
    
    Parameters
    -------------------------------------------------
    x : array-like
        The x-values for the derivative ie. the independant variable
    y : array-like
        The y-values for the derviative ie. the dependant variable or f(x)
    finitediff_scheme : {'central', 'five point stencil'}, optional
        The finite difference scheme used to approximate the first derivative. Options are 'central' for
        the central differences method and 'five point stencil' for the five point stencil method. Defaults 
        to 'central'.
    
    Returns
    -----------------------------------------------------
    out : ndarray
        Returns the first order derivative of y(x).
    
    Examples 
    ----------------------------------------------------
    Return the first derivative of y = x**2. Note that the first and last values have errors due to the small length of the array.
    
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = x**2
    >>> first_derivative(x, y)
    array([ 1.  2.  4.  6.  8. 10. 12. 14. 16. 18. 19.])
    
    Notes
    -------------------------------------------------------
    The approximation works best when the length of the input arrays are large. The ends of the resulting output array also tend to have 
    larger associated errors
    """
    if len(y) < 100:
        print ("WARNING: Length of array may be too small to yield accurate results")
        
    if finitediff_scheme == 'central':
        return np.gradient(y, x[1]-x[0])
    
    elif finitediff_scheme == 'five point stencil':
        y = [0, 0] + list(y) + [0, 0]
        numerator = np.array([y[i-2] - 8*y[i-1] + 8*y[i+1] - y[i+2] for i in range(2, len(y)-2)])
        denominator = 12*(x[1] - x[0])        
        return numerator/denominator        
        
        
def second_derivative(x, y, finitediff_scheme = 'central'):
    """
    Uses a finite difference method to approximate the second order derivative of a function.
    
    Parameters
    -------------------------------------------------
    x : array-like
        The x-values for the derivative ie. the independant variable
    y : array-like
        The y-values for the derviative ie. the dependant variable or f(x)
    finitediff_scheme : {'central', 'five point stencil'}, optional
        The finite difference scheme used to approximate the second derivative. Options are 'central' for
        the central differences method and 'five point stencil' for the five point stencil method. Defaults 
        to 'central'.
    
    Returns
    -----------------------------------------------------
    out : ndarray
        Returns the second order derivative of y(x).
    
    Examples 
    ----------------------------------------------------
    Return the second derivative of y = x**3. Note that the first and last values have errors due to the small length of the array.
    
    >>> x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    >>> y = x**3
    >>> second_derivative(x, y)
    array([ 1.000e+00  6.000e+00  1.200e+01  1.800e+01  2.400e+01  3.000e+01
    3.600e+01  4.200e+01  4.800e+01  5.400e+01 -1.271e+03])
    
    Notes
    -------------------------------------------------------
    The approximation works best when the length of the input arrays are large. The ends of the resulting output array also tend to have 
    larger associated errors
    """
    if len(y) < 100:
        print ("WARNING: Length of array may be too small to yield accurate results")
        
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
    """
    Returns the probability density of an input wavefunction.
    
    Parameters
    --------------------------------------
    wfunc : complex ndarray
        Complex wavefunction the probability density function is based on.
        
    Returns
    -------------------------------------
    out : ndarray
        Probability density of the input wavefunction
    """
    return wfunc*np.conj(wfunc)

def normalize(x, wfunc):
    """
    Returns the normalized version of an input wavefunction.
    
    Parameters
    ------------------------------------------
    x : array-like
        The spatial coordinates used to define the wavefunction
    wfunc : complex ndarray
        Complex wavefunction to be normalized
    
    Returns
    --------------------------------------------------
    out : complex ndarray
        The normalized wavefunction
    
    Notes
    ------------------------------------------------
    The normalization operator uses to trapezoidal method to approximate the integral of the probability density. Therefore,
    to yield accurate results, a large array should be used.
    """
    if len(wfunc) < 100:
        print ("WARNING: Length of array may be too small to yield accurate results")
    return 1/np.sqrt(integrate.trapz(probdensity(wfunc), x = x))*wfunc      


#Observables
def momentum_op(x, wfunc, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
    """
    Returns the resulting wavefunction from the momentum operator acting on an input wavefunction
    
    Parameters
    -----------------------------------------------
    x : array-like
        The spatial coordinates used to define the wavefunction
    wfunc : complex ndarray
        Complex wavefunction for the momentum operator to act on
    h_bar : number, optional
        Tbe reduced Planck constant. For natural units, set to 1. Defaults to 6.626e-34 / (2 * np.pi)
    finitediff_scheme : {'central', 'five point stencil'}, optional
        The finite difference scheme used to approximate the first derivative. Options are 'central' for
        the central differences method and 'five point stencil' for the five point stencil method. Defaults 
        to 'central'.    
    
    Returns
    ------------------------------------------------
    out : complex ndarray
        The resulting wavefunction after the momentum operator has acted on the input wavefunction
    
    Notes
    -------------------------------------------------------
    The approximation works best when the length of the input arrays are large. The ends of the resulting output array also tend to have 
    larger associated errors
    """
    return -1j*h_bar*first_derivative(x, wfunc, finitediff_scheme = finitediff_scheme)

def position_op(x, wfunc):
    """
    Returns the resulting wavefunction from the position operator acting on an input wavefunction
    
    Parameters
    ---------------------------------------------------
    x : array-like
        The spatial coordinates used to define the wavefunction
    wfunc : complex ndarray
        Complex wavefunction for the position operator to act on    
        
    Returns
    ----------------------------------------------------
    out : ndarray
        The resulting wavefunction after the position operator has acted on the input wavefunction
    """
    return x*wfunc

def kinetic_op(x, wfunc, particle, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
    """
    Returns the resulting wavefunction from the kinetic energy operator acting on an input wavefunction
    
    Parameters
    -----------------------------------------------
    x : array-like
        The spatial coordinates used to define the wavefunction
    wfunc : complex ndarray
        Complex wavefunction for the kinetic energy operator to act on
    h_bar : number, optional
        Tbe reduced Planck constant. For natural units, set to 1. Defaults to 6.626e-34 / (2 * np.pi)
    finitediff_scheme : {'central', 'five point stencil'}, optional
        The finite difference scheme used to approximate the first derivative. Options are 'central' for
        the central differences method and 'five point stencil' for the five point stencil method. Defaults 
        to 'central'.    
    
    Returns
    ------------------------------------------------
    out : complex ndarray
        The resulting wavefunction after the kinetic energy operator has acted on the input wavefunction
    
    Notes
    -------------------------------------------------------
    The approximation works best when the length of the input arrays are large. The ends of the resulting output array also tend to have 
    larger associated errors
    """
    return (-h_bar**2/(2*particle.m))*second_derivative(x, wfunc, finitediff_scheme = finitediff_scheme)   