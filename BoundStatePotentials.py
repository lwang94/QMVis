# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:49:30 2019

@author: lawre
"""
import Operators as op

import numpy as np
import scipy.special as sp
import scipy.optimize as opt
import matplotlib.pyplot as plt

class InfSquareWell:
    """
    Simulates the Infinite Square Well potential. Also known as the Particle in a Box.
    
    Parameters
    --------------------
    x : array-like 
        The spatial coordinates used to define the potential, V(x)
    L : `float`, `int` 
        The width of the well/the size of the box. Note that the well is centered at 0 meaning L = 1 will produce 
        a well with with boundaries at x = -0.5 and x = 0.5
        
    Attributes
    ---------------------------
    Same as Parameters
    """
        
    def __init__(self, x, L):
        
        self.x = x
        self.L = L

        
    def potential(self, ylim = np.inf):
        """
        Generates the potential, V(x), for the Infinite Square Well/Particle in a Box.
        
        Parameters
        --------------------
        ylim : number, optional
            The value at the boundaries of the Infinite Square Well/Particle in a Box. Conventionally, the value is infinity but 
            if real numbers are needed, the value can be specified. Defaults to np.inf
        
        Returns
        --------------------
        out : `ndarray`
            The output is an array representing an infinite square well of size L. 
        
        Examples
        ----------------------
        Return V(x) for a well of size 4 with boundaries at x = -2 and x = 2.
        
        >>> x = np.linspace(-5, 5, 11)
        >>> infsquarewell = InfSquareWell(x, 4)
        >>> infsquarewell.potential(), x
        (array([inf, inf, inf, inf,  0.,  0.,  0., inf, inf, inf, inf]),
         array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.]))        
        
        Return V(x) for a well of size 4 with boundaries at x = -2 and x = 2. Let infinity be replaced by 10.

        >>> x = np.linspace(-5, 5, 11)
        >>> infsquarewell = InfSquareWell(x, 4)
        >>> infsquarewell.potential(ylim = 10), x
        (array([10., 10., 10., 10.,  0.,  0.,  0., 10., 10., 10., 10.]),
         array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.]))        
        
        """
        potential = np.piecewise(self.x,
                                 
                                 [np.any([self.x <= -self.L / 2, 
                                          self.x >= self.L / 2], axis = 0), 
                                  np.all([self.x > -self.L / 2, 
                                          self.x < self.L / 2], axis = 0)],
    
                                 [ylim, 
                                  0])
        return potential

    
    def hamiltonian(self, wfunc, particle, finitediff_scheme = 'central', h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the result of the Infinite Square Well/Particle in a Box Hamiltonian operator acting on a wavefunction. 

        Parameters
        -----------------------------------
        wfunc : array-like
            Wavefunction for the operator to act on. Should be the same size as x
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will be used in the kinetic energy portion of the hamiltonian
        finitediff_scheme : {'central', 'five point stencil'}, optional
            Method of finite difference approximation for the second order derivative in the kinetic energy operator. Options are 
            'central' for the central differences method or 'five point stencil' for the five point stencil method. 
            Defaults to 'central'.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        --------------------------------------
        out : `ndarray`
            The output is an array representing the result of the Infinite Square Well/Particle in a Box Hamiltonian acting on 
            the input wavefunction.
        
        Notes
        ------------------------------------------
        Since the function uses a finite difference method to approximate the second order derivative present in the kinetic energy
        operator, a large array shoud be used to yield the most accurate results.            
        """
        if len(wfunc) < 100:
            print ('WARNING: size of array may be too small to yield accurate results from finite difference approximation')  
            
        kinetic = op.kinetic_op(self.x, 
                                wfunc, 
                                particle, 
                                h_bar = h_bar, 
                                finitediff_scheme = finitediff_scheme)

        potential = self.potential() * wfunc

        new_wfunc = kinetic + potential
        
        new_wfunc[new_wfunc == np.inf] = 0 #converts all infinity values to zero
        new_wfunc = np.nan_to_num(new_wfunc) #converts all nan values to zero
        
        return new_wfunc
    
                
    def eigenfunc(self, n):
        """
        Returns the normalized eigenfunction for the Infinite Square Well/Particle in a Box. Does not include time dependance.
        This is equivalent to the eigenfunction at time t = 0. For the eigenfunction with the time dependance included, 
        see timedep_eigenfunc().
        
        Parameters
        ---------------------------------------------
        n : `int` starting from 1
            The order of the eigenfunction. Note that for the Infinite Square Well/Particle in a Box, n = 1, 2, 3.... and does
            not begin at 0.
            
        Returns
        ---------------------------------------------
        out : `ndarray`
            The output is an array representing the nth normalized eigenfunction for the Infinite Square Well/Particle in a Box.   
            
        Examples
        --------------------------------------------
        Return the ground state eigenfunction for a Particle in a Box of length L = 3.
        
        >>> x = np.linspace(-5, 5, 11)
        >>> infsquarewell = InfSquareWell(x, 3)
        >>> infsquarewell.eigenfunc(1)
        array([0.        , 0.        , 0.        , 0.        , 0.40824829,
               0.81649658, 0.40824829, 0.        , 0.        , 0.        ,
               0.        ])        
    
        """
        if n <= 0 or n%1 != 0:
            raise Exception('for the infinite square well, n must be a positive integer starting from 1')
                       
        if n%2 == 0:
            return np.piecewise(self.x, 
                                
                                [np.any([self.x <= -self.L / 2, 
                                         self.x >= self.L / 2], axis = 0), 
                                 np.all([self.x > -self.L / 2, 
                                         self.x < self.L / 2], axis = 0)], 
    
                                [0, 
                                 lambda x : np.sqrt(2 / self.L) * np.sin(n * np.pi * x / self.L)])
        else:
            return np.piecewise(self.x, 
                                
                                [np.any([self.x <= -self.L / 2, 
                                         self.x >= self.L / 2], axis = 0), 
                                 np.all([self.x > -self.L / 2, 
                                         self.x < self.L / 2], axis = 0)],
    
                                [0, 
                                 lambda x : np.sqrt(2 / self.L) * np.cos(n * np.pi * x / self.L)])

    
    def timedep_eigenfunc(self, t, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the normalized eigenfunction for the Infinite Square Well/Particle in a Box with time dependance included. 
        
        Parameters
        -------------------------------------------------------
        t : number
            The time at which the eigenfunction is to be evaluated.
        n : `int` starting from 1
            The order of the eigenfunction. Note that for the Infinite Square Well/Particle in a Box, n = 1, 2, 3.... and does
            not begin at 0.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the eigenfunction.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        ------------------------------------------------------
        out : `complex ndarray`
            The output is a complex array representing the nth normalized eigenfunction at time t for the
            Infinite Square Well/Particle in a Box.
            
        Examples
        ------------------------------------------------------
        Return the ground state eigenfunction for a Particle in a Box of length L = 3 at time t = 3. Units are natural units 
        such that the electron rest mass and the reduced planck constant is 1. 

        >>> import Particle as p
        >>> x = np.linspace(-5, 5, 11)
        >>> electron = p.Particle(1)
        >>> infsquarewell = InfSquareWell(x, 3)
        >>> infsquarewell.timedep_eigenfunc(3, 1, electron, h_bar = 1)
        array([ 0.        +0.j        ,  0.        +0.j        ,
        0.        +0.j        ,  0.        +0.j        ,
        -0.03023889-0.40712686j, -0.06047777-0.81425371j,
        -0.03023889-0.40712686j,  0.        +0.j        ,
        0.        +0.j        ,  0.        +0.j        ,
        0.        +0.j        ])
        
        """
        complex_x = np.array(self.x) + 0j #ensures input array is complex as np.piecewise returns an array of the same type as the input array
        
        if n <= 0 or n%1 != 0:
            raise Exception('for the infinite square well, n must be a positive integer starting from 1')
                       
        if n%2 == 0:
            return np.piecewise(complex_x, 
                                
                                [np.any([self.x <= -self.L / 2, 
                                         self.x >= self.L / 2], axis = 0), 
                                 np.all([self.x > -self.L / 2, 
                                         self.x < self.L / 2], axis = 0)],
    
                                [0, 
                                 lambda x : np.sqrt(2 / self.L)
                                          * np.sin(n * np.pi * x / self.L)
                                          * np.exp(-1j * self.eigenvalue(n, particle, h_bar = h_bar) * t / h_bar)])
        else:
            return np.piecewise(complex_x, 
                                
                                [np.any([self.x <= -self.L / 2, 
                                         self.x >= self.L / 2], axis = 0), 
                                 np.all([self.x > -self.L / 2, 
                                         self.x < self.L / 2], axis = 0)],
    
                                [0, 
                                 lambda x : np.sqrt(2 / self.L)
                                          * np.cos(n * np.pi * x / self.L)
                                          * np.exp(-1j * self.eigenvalue(n, particle, h_bar = h_bar) * t / h_bar)])        
        
            
    def eigenvalue(self, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the eigenvalue or energy of the Infinite Square Well/Particle in a Box. 
        
        Parameters
        ----------------------------------------------
        n : `int` starting from 1
            The order of the eigenvalue. Note that for the Infinite Square Well/Particle in a Box, n = 1, 2, 3.... and does
            not begin at 0.
        particle : Particle class (see Particle.py for more details)
            The particle whose mass determines the energy of the nth eigenstate.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        ----------------------------------------------
        out : `float`
            The output is the eigenvalue or the energy of the nth eigenstate for a particle trapped in the Infinite Square Well.
            
        Examples
        ------------------------------------------------
        Return the ground state energy of an electron trapped in an Infinite Square Well of length L = 1. Units are natural units
        such that the electron rest mass and reduced planck constant is 1.
        
        >>> import Particle as p 
        >>> x = np.linspace(-2, 2, 1000)
        >>> electron = p.Particle(1)
        >>> infsquarewell = InfSquareWell(x, 1)
        >>> infsquarewell.eigenvalue(1, electron, h_bar = 1)     
        4.934802200544679
        
        """
        return (h_bar * np.pi * n) ** 2 / (2 * particle.m * self.L ** 2)

class HarmonicOscillator:
    """
    Simulates the Harmonic Oscillator potential.
    
    Parameters
    -------------------
    x : array-like 
        The spatial coordinates used to define the potential, V(x)
    k : `float`, `int` 
        The force constant of the harmonic oscillator    
        
    Attributes
    --------------------
    Same as Parameters
    """
        
    def __init__(self, x, k):
        self.k = k
        self.x = x
        
        
    def potential(self):
        """
        Generates the potential, V(x), for the Harmonic Oscillator.
        """
        return (self.k * self.x ** 2) / 2
            
        
    def hamiltonian(self, wfunc, particle, finitediff_scheme = 'central', h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the result of the Harmonic Oscillator Hamiltonian operator acting on a wavefunction.
        
        Parameters
        ---------------------------------------------------
        wfunc : array-like
            Wavefunction for the operator to act on. Should be the same size as x
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will be used in the kinetic energy portion of the hamiltonian
        finitediff_scheme : {'central', 'five point stencil'}, optional
            Method of finite difference approximation for the second order derivative in the kinetic energy operator. Options are 
            'central' for the central differences method or 'five point stencil' for the five point stencil method. 
            Defaults to 'central'.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        --------------------------------------
        out : `ndarray`
            The output is an array representing the result of the Harmonic Oscillator Hamiltonian acting on 
            the input wavefunction.  
            
        Notes
        ------------------------------------------
        Since the function uses a finite difference method to approximate the second order derivative present in the kinetic energy
        operator, a large array shoud be used to yield the most accurate results.  
        """
        if len(wfunc) < 100:
            print ('WARNING: size of array may be too small to yield accurate results from finite difference approximation') 
            
        kinetic = op.kinetic_op(self.x, 
                                wfunc, 
                                particle, 
                                h_bar = h_bar, 
                                finitediff_scheme = finitediff_scheme)
        
        potential = self.potential() * wfunc
        
        return kinetic + potential


    def ladderup_op(self, wfunc, particle, finitediff_scheme = 'central', h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the result of the raising ladder operator acting on a wavefunction. The raising ladder operator increases the
        order of the Harmonic Oscillator eigenfunction by one (up to a normalization constant).
        
        Parameters
        ---------------------------------------------------------------------------------------------
        wfunc : array-like
            Wavefunction for the operator to act on. Should be the same size as x
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will be used with the force constant to determine the angular frequency
        finitediff_scheme : {'central', 'five point stencil'}, optional
            Method of finite difference approximation for the second order derivative in the kinetic energy operator. Options are 
            'central' for the central differences method or 'five point stencil' for the five point stencil method. 
            Defaults to 'central'.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.  
            
        Returns
        --------------------------------------
        out : `complex ndarray`
            The output is a complex array representing the result of the raising ladder operator acting on 
            the input wavefunction. Note that the resulting wavefunction has NOT been normalized.

        Notes
        ------------------------------------------
        Since the function uses a finite difference method to approximate the first order derivative present in the momentum
        operator, a large array shoud be used to yield the most accurate results. The output of the function is also NOT normalized.
        """
        if len(wfunc) < 100:
            print ('WARNING: size of array may be too small to yield accurate results from finite difference approximation')
            
        w = np.sqrt(self.k / particle.m)
        constant = 1 / np.sqrt(2 * h_bar * particle.m * w)
        
        return constant*(particle.m * w * op.position_op(self.x, wfunc)
                         - 1j * op.momentum_op(self.x, 
                                               wfunc, 
                                               h_bar = h_bar, 
                                               finitediff_scheme = finitediff_scheme))
        
        
    def ladderdown_op(self, wfunc, particle, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
        """
        Returns the result of the lowering ladder operator acting on a wavefunction. The lowering ladder operator decreases the
        order of the Harmonic Oscillator eigenfunction by one (up to a normalization constant).
        
        Parameters
        ---------------------------------------------------------------------------------------------
        wfunc : array-like
            Wavefunction for the operator to act on. Should be the same size as x
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will be used with the force constant to determine the angular frequency
        finitediff_scheme : {'central', 'five point stencil'}, optional
            Method of finite difference approximation for the second order derivative in the kinetic energy operator. Options are 
            'central' for the central differences method or 'five point stencil' for the five point stencil method. 
            Defaults to 'central'.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.  
            
        Returns
        --------------------------------------
        out : `complex ndarray`
            The output is a complex array representing the result of the lowering ladder operator acting on 
            the input wavefunction. Note that the resulting wavefunction has NOT been normalized.

        Notes
        ------------------------------------------
        Since the function uses a finite difference method to approximate the first order derivative present in the momentum
        operator, a large array shoud be used to yield the most accurate results. The output of the function is also NOT normalized.
        """
        if len(wfunc) < 100:
            print ('WARNING: size of array may be too small to yield accurate results from finite difference approximation')
            
        w = np.sqrt(self.k / particle.m)
        constant = 1 / np.sqrt(2 * h_bar * particle.m * w)
        
        return constant*(particle.m * w * op.position_op(self.x, wfunc)
                         + 1j*op.momentum_op(self.x, 
                                             wfunc, 
                                             h_bar = h_bar, 
                                             finitediff_scheme = finitediff_scheme))
        
        
    def eigenfunc(self, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the normalized eigenfunction for the Harmonic Oscillator. Does not include time dependance.
        This is equivalent to the eigenfunction at time t = 0. For the eigenfunction with the time dependance included, 
        see timedep_eigenfunc().
        
        Parameters
        ---------------------------------------------
        n : `int`
            The order of the eigenfunction.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the angular frequency from the force constant
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.        
            
        Returns
        ---------------------------------------------
        out : `ndarray`
            The output is an array representing the nth normalized eigenfunction for the Harmonic Oscillator.   
            
        Examples
        --------------------------------------------
        Return the ground state eigenfunction for a Harmonic Oscillator with force constant k = 1. Units are natural units such 
        that the electron rest mass and reduced Planck constant is 1.
        
        >>> import Particle as p
        >>> x = np.linspace(-5, 5, 10)
        >>> ho = HarmonicOscillator(x, 1)
        >>> electron = p.Particle(1)
        >>> ho.eigenfunc(0, electron, h_bar = 1)
        array([2.79918439e-06, 3.90567063e-04, 1.58560022e-02, 1.87294814e-01,
               6.43712257e-01, 6.43712257e-01, 1.87294814e-01, 1.58560022e-02,
               3.90567063e-04, 2.79918439e-06])        
    
        """        
        if n < 0 or n%1 != 0:
            raise Exception('for the harmonic oscillator, n must be a positive integer starting from 0')        
        n = int(n) #coverts n datatype to the native Python int type to ensure number of bits is enough for calculating the square root of large numbers
        
        w = np.sqrt(self.k / particle.m)
        alpha = particle.m * w / h_bar
        
        y = np.sqrt(alpha) * self.x
        Hermite = sp.eval_hermite(n, y)    
        
        C = (1 / np.sqrt(float(2 ** n) * np.math.factorial(n))) * (alpha / np.pi) ** (np.pi / 4)
        
        return C * np.exp(-y ** 2 / 2) * Hermite

    
    def timedep_eigenfunc(self, t, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the normalized eigenfunction for the Harmonic Oscillator with time dependance included. 
        
        Parameters
        -------------------------------------------------------
        t : number
            The time at which the eigenfunction is to be evaluated.
        n : `int`
            The order of the eigenfunction.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the eigenfunction.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        ------------------------------------------------------
        out : `complex ndarray`
            The output is a complex array representing the nth normalized eigenfunction at time t for the
            Harmonic Oscillator
            
        Examples
        ------------------------------------------------------
        Return the first excited state for a Harmonic Oscillator with k = 1 at time t = 3. Units are natural units 
        such that the electron rest mass and the reduced planck constant is 1. 

        >>> import Particle as p
        >>> x = np.linspace(-5, 5, 10)
        >>> electron = p.Particle(1)
        >>> ho = HarmonicOscillator(x, 1)
        >>> ho.timedep_eigenfunc(3, 0, electron, h_bar = 1)
        array([2.79918439e-06+0.j, 3.90567063e-04+0.j, 1.58560022e-02+0.j,
               1.87294814e-01+0.j, 6.43712257e-01+0.j, 6.43712257e-01+0.j,
               1.87294814e-01+0.j, 1.58560022e-02+0.j, 3.90567063e-04+0.j,
               2.79918439e-06+0.j])
        
        """
        if n < 0 or n%1 != 0:
            raise Exception('for the harmonic oscillator, n must be a positive integer starting from 0')
        n = int(n) #coverts n datatype to the native Python int type to ensure number of bits is enough for calculating the square root of large numbers
            
        w = np.sqrt(self.k / particle.m)
        alpha = particle.m * w / h_bar
        
        y = np.sqrt(alpha) * self.x
        Hermite = sp.eval_hermite(n, y)     
        
        C = (1 / np.sqrt(float(2 ** n) * np.math.factorial(n))) * (alpha / np.pi) ** (np.pi/4)
        
        return C * np.exp(-y ** 2 / 2) * Hermite * np.exp(-1j * self.eigenvalue(n, particle, h_bar = h_bar) * t / h_bar)     
        
    
    def eigenvalue(self, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the eigenvalue or energy of the Harmonic Oscillator. 
        
        Parameters
        ----------------------------------------------
        n : `int`
            The order of the eigenvalue.
        particle : Particle class (see Particle.py for more details)
            The particle whose mass determines the energy of the nth eigenstate.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        ----------------------------------------------
        out : `float`
            The output is the eigenvalue or the energy of the nth eigenstate for a particle trapped in the Harmonic Oscillator.
            
        Examples
        ------------------------------------------------
        Return the ground state energy of an electron trapped in a Harmonic Oscillator with force constant k = 1. Units are natural 
        units such that the electron rest mass and reduced planck constant is 1.
        
        >>> import Particle as p 
        >>> x = np.linspace(-2, 2, 1000)
        >>> electron = p.Particle(1)
        >>> ho = bsp.HarmonicOscillator(x, 1)
        >>> ho.eigenvalue(0, electron, h_bar = 1)   
        0.5
        
        """
        w = np.sqrt(self.k / particle.m)
        
        return h_bar * w * (n + 1/2)

class FiniteSquareWell:
    """
    Simulates the finite square well potential.
    
    Parameters
    -------------------
    x : array-like 
        The spatial coordinates used to define the potential, V(x)
    L : `float`, `int` 
        The length of the box
    V0 : `float`, `int`
        The 'depth' of the box
        
    Attributes
    --------------------
    Same as Parameters    
    """
    
    def __init__(self, x, L, V0):
        self.x = x
        self.L = L
        self.V0 = V0
   
     
    def potential(self):
        """
        Generates the potential, V(x), of the finite square well
        """
        potential = np.piecewise(self.x,
                                 
                                 [np.any([self.x <= -self.L / 2, 
                                          self.x >= self.L / 2], axis = 0), 
                                  np.all([self.x > -self.L / 2, 
                                          self.x < self.L / 2], axis = 0)],
    
                                 [0, 
                                  -self.V0])
    
        return potential        
    
    
    def hamiltonian(self, wfunc, particle, finitediff_scheme = 'central', h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the result of the Finite Square Potential Hamiltonian operator acting on a wavefunction.
        
        Parameters
        ---------------------------------------------------
        wfunc : array-like
            Wavefunction for the operator to act on. Should be the same size as x
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will be used in the kinetic energy portion of the hamiltonian
        finitediff_scheme : {'central', 'five point stencil'}, optional
            Method of finite difference approximation for the second order derivative in the kinetic energy operator. Options are 
            'central' for the central differences method or 'five point stencil' for the five point stencil method. 
            Defaults to 'central'.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        
        Returns
        --------------------------------------
        out : `ndarray`
            The output is an array representing the result of the Finite Square Potential Hamiltonian acting on 
            the input wavefunction.  
            
        Notes
        ------------------------------------------
        Since the function uses a finite difference method to approximate the second order derivative present in the kinetic energy
        operator, a large array shoud be used to yield the most accurate results.  
        """
        kinetic = op.kinetic_op(self.x, 
                                wfunc, 
                                particle, 
                                h_bar = h_bar, 
                                finitediff_scheme = finitediff_scheme)

        potential = self.potential() * wfunc

        new_wfunc = kinetic + potential

        return new_wfunc


    def eigenfunc(self, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        """
        Returns the normalized eigenfunction for the Finite Square Potential. Does not include time dependance.
        This is equivalent to the eigenfunction at time t = 0. For the eigenfunction with the time dependance included, 
        see timedep_eigenfunc().
        
        Parameters
        ---------------------------------------------
        n : `int`
            The order of the eigenfunction.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the Energy levels of the potential
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.        
            
        Returns
        ---------------------------------------------
        out : `ndarray`
            The output is an array representing the nth normalized eigenfunction for the Finite Square Potential.   
            
        Examples
        --------------------------------------------
        Return the ground state eigenfunction for a Finite Square Well with length, L = 1 and depth, V0 = 10. Units are natural units 
        that the electron rest mass and reduced Planck constant is 1.
        
        >>> import Particle as p
        >>> x = np.linspace(-1, 1, 10)
        >>> fsw = FiniteSquareWell(x, 1, 10)
        >>> electron = p.Particle(1)
        >>> fsw.eigenfunc(0, electron, h_bar = 1)
        array([0.07759989 0.18565918 0.44419303 0.87140321 1.12065552 1.12065552
               0.87140321 0.44419303 0.18565918 0.07759989])
        """
        #Finding roots of the transcendental equation: tan(z) = np.sqrt((z/z0) ** 2 - 1) for even functions or -cot(z) = np.sqrt((z/z0) ** 2 -1) for odd functions
        z0 = self.L * np.sqrt(2 * particle.m * self.V0) / (2 * h_bar)
        
        z = opt.brentq(self.transcendental_root, #uses brentq method of root finding
                       n * np.pi / 2,            #in the given region
                       (n + 1) * np.pi / 2, 
                       args = (z0, n)) 
        
        #catch any errors from root finding
        if z == 0 or (z0 ** 2 - z ** 2) < 0:
            raise Exception('Dimensions of the well do not permit bound states with this value of n. Please reconsider your parameters')
        
        #finds "Energy" constants from root, z
        l = 2 * z / self.L
        k = 2 * np.sqrt(z0 ** 2 - z ** 2) / self.L

        #construct bound state eigenfunctions
        #Even functions
        if n%2 == 0:
            const_factor = np.cos(z) / np.exp(-np.sqrt(z0 ** 2 - z ** 2)) #finds constant such that piecewise function is continuous at boundaries
            eigenfunction = np.piecewise(self.x,
                                         
                                         [self.x <= -self.L / 2, 
                                          np.all([self.x > -self.L / 2, 
                                                  self.x < self.L / 2], axis = 0), 
                                          self.x >= self.L / 2],
                                                  
                                         [lambda x : const_factor * np.exp(k * x), 
                                          lambda x : np.cos(l * x), 
                                          lambda x : const_factor * np.exp(-k * x)])
                                          
            eigenfunction = op.normalize(self.x, eigenfunction) #normalize eigenfunction
        #Odd functions    
        else:
            const_factor = np.sin(z) / np.exp(-np.sqrt(z0 ** 2 - z ** 2)) #finds constant such that piecewise function is continuous at boundaries
            eigenfunction = np.piecewise(self.x,
                                         
                                         [self.x <= -self.L / 2, 
                                          np.all([self.x > -self.L / 2, 
                                                  self.x < self.L / 2], axis = 0), 
                                          self.x >= self.L / 2],
                                                  
                                         [lambda x : -const_factor * np.exp(k * x), 
                                          lambda x : np.sin(l * x), 
                                          lambda x: const_factor * np.exp(-k * x)])
                                          
            eigenfunction = op.normalize(self.x, eigenfunction) #normalize eigenfunctions
            
        return eigenfunction


    def timedep_eigenfunc(self, t, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        z0 = self.L*np.sqrt(2*particle.m*self.V0)/(2*h_bar)
        
        z = opt.brentq(self.transcendental_root, n*np.pi/2, (n + 1)*np.pi/2, args = (z0, n))
        if z == 0:
            raise Exception('Dimensions of the well do not permit this value of n. Please reconsider your parameters')
            
        l = 2*z/self.L
        k = 2*np.sqrt(z0**2 - z**2)/self.L

        timefactor = np.exp(-1j*self.eigenvalue(n, particle, h_bar = h_bar)*t/h_bar) 
        if n%2 == 0:
            const_factor = np.cos(z)/np.exp(-np.sqrt(z0**2 - z**2))
            eigenfunction = np.piecewise(self.x, 
                                         [self.x <= -self.L/2, np.all([self.x > -self.L/2, self.x < self.L/2], axis = 0), self.x >= self.L/2],
                                         [lambda x: const_factor*np.exp(k*x), lambda x: np.cos(l*x), lambda x: const_factor*np.exp(-k*x)])           
            
        else:
            const_factor = np.sin(z)/np.exp(-np.sqrt(z0**2 - z**2))
            eigenfunction = np.piecewise(self.x, 
                                         [self.x <= -self.L/2, np.all([self.x > -self.L/2, self.x < self.L/2], axis = 0), self.x >= self.L/2],
                                         [lambda x: -const_factor*np.exp(k*x), lambda x: np.sin(l*x), lambda x: const_factor*np.exp(-k*x)])                    
        
        eigenfunction = op.normalize(self.x, eigenfunction)*timefactor
        
        
    def eigenvalue(self, n, particle, h_bar = 6.626e-34/(2*np.pi)):
        z0 = self.L*np.sqrt(2*particle.m*self.V0/(2*h_bar))
        
        z = opt.brentq(self.transcendental_root, n*np.pi/2, (n + 1)*np.pi/2, args = (z0, n))
        if z == 0:
            raise Exception('Dimensions of the well do not permit this value of n. Please reconsider your parameters')
        l = 2*z/self.L
        E = (l*h_bar)**2/(2*particle.m) - self.V0
        
        return E
        
        
    def transcendental_root(self, z, z0, n):
        if n%2 == 0:
            result = np.tan(z) - np.sqrt((z0/z)**2 - 1)
        else:
            result = -1/np.tan(z) - np.sqrt((z0/z)**2 - 1)
        
        return result

    def plot_transcendental(self, n, particle, h_bar = 6.626e-34/(2*np.pi), y_lim = 30):
        z0 = self.L*np.sqrt(2*particle.m*self.V0/(2*h_bar))   
        z = np.linspace(0, (n+1)*np.pi/2, 1000)
        dz = z[1] - z[0]
        
        even = np.tan(z)
        even[z % (np.pi/2) < dz*3] = np.nan

        odd = -1/np.tan(z)
        odd[z % (np.pi/2) < dz*3] = np.nan        
        
        plt.plot(z, np.sqrt((z0/z)**2 - 1))
        plt.plot(z, even, 'r')
        plt.plot(z, odd, 'r')
        plt.ylim(0, y_lim)
        
    
    