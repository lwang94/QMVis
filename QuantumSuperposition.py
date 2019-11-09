# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:08:29 2019

@author: lawre
"""
from flatten import flatten
import Operators as op

import numpy as np
import scipy.integrate as integrate
import matplotlib.animation as animation

class QuantumSuperposition:
    """
    Given a specific potential, construct a discrete linear combination or quantum superposition of the eigenstates of that potential.
    
    Parameters
    ------------------------------------
    potential : Potential class (see BoundStatePotentials or BoundScatteredPotentials for more details)
        The potential which defines the eigenstates to be used in the superposition. Note that since the QuantumSuperposition 
        class constructs a discrete sum of eigenstates, potentials with bound state solutions are best used to obtain accurate 
        results.   
    n : array-like
        The orders of the eigenstates to be used in the superposition. eg. n = [0, 1, 2, 4] construct a linear combination of the 
        first three eigenstates along with the fifth eigenstate of the given potential.
    coeff : array-like, {'uniform distribution'}, optional
        The coefficients for each eigenstate that makes up the linear combination. When initializing the instance, can use the 
        string 'uniform distribution' to make all coefficients the same while keeping the superposition normalized. This is the 
        default.
        
    Attributes
    -------------------------------------
    potential : Potential class (see BoundStatePotentials or BoundScatteredPotentials for more details)
        Same as in Parameters
    n : array-like
        Same as in Parameters
    coeff : array-like
        The coefficients for each eigenstate that makes up the linear combination.
    x : array-like
        The spatial coordinates used to define the potential
    """
    
    def __init__(self, potential, n, coeff = 'uniform distribution'):
        self.potential = potential
        self.x = potential.x
        self.n = n
    
        
        if coeff == 'uniform distribution':
            self.coeff = [1/np.sqrt(len(n))]*len(n) #sets the coefficients such that they are all equal and the superposition remains normalized (returns list type)
        else: 
            self.coeff = coeff

            
    def superposition(self, *args, **kwargs):
        """
        Generates the quantum superposition of eigenstates specified in the instance of the class.
        
        Paramters
        ---------------------------------------------------------
        *args
            Arguments (other than the order, n) to be passed into the eigenfunc method of potential used to initialize instance of class.
        **kwargs
            Keyword arguments to be passed into the eigenfunc method of potential used to initialize instance of class.
        
        Returns
        ---------------------------------------------------
        out : complex ndarray
            The output is a complex array representing the superposition of eigenstates specified in the instance of the class.
            
        Examples
        ------------------------------------------------------
        Return the superposition of the first two eigenstates of an Infinite Square Well of length L = 2 with coefficients c1 = c2 = 1/sqrt(2)
        
        >>> import BoundStatePotentials as bsp
        >>> x = np.linspace(-1, 1, 10)
        >>> basis = QuantumSuperposition(bsp.InfSquareWell(x, 2), [1, 2])
        >>> basis.superposition()
        array([ 0.        +0.j, -0.21267472+0.j, -0.24184476+0.j,  0.        +0.j,
                0.45451948+0.j,  0.938209  +0.j,  1.22474487+0.j,  1.15088372+0.j,
                0.69636424+0.j,  0.        +0.j])        
        
        Return the superposition of the first seven eigenstates of a Finite Square Well of length L = 7 and depth, V0 = 7,
        with coefficients c0 = c1 = .... = c6 = 1/sqrt(7). The eigenfunction of the Finite Square Well admits arguments other 
        than the order of the eigenfunction. Units are natural units such that the electron rest mass and the reduced planck constant is 1.
        
        >>> import BoundStatePotentials as bsp
        >>> import Particle as p
        >>> x = np.linspace(-7, 7, 10)
        >>> electron = p.Particle(1)
        >>> superposition = QuantumSuperposition(bsp.FiniteSquareWell(x, 7, 7), np.arange(0, 7))
        >>> superposition.superposition(electron, h_bar = 1)
        array([[-3.24291865e-05+0.j -6.81897652e-04+0.j  6.22211436e-02+0.j
                 2.60165430e-01+0.j -3.49052310e-01+0.j  6.08101425e-01+0.j
                 1.87817897e-01+0.j  1.28983816e-01+0.j  3.08711502e-04+0.j
                -1.85545355e-05+0.j]])
        
        """
        series = np.zeros(len(self.x), dtype = np.complex128)
        for i in range(len(self.n)):
            series += self.coeff[i]*self.potential.eigenfunc(self.n[i], *args, **kwargs)
            
        return series


    def superposition_timedep(self, t, particle, h_bar = 6.626e-34/(2*np.pi), args_list = [], kwargs = {}):
        """
        Generates the quantum superposition of eigenstates specified in the instance of the class at time, t.
        
        Parameters
        -----------------------------------------------------------
        t : number
            The time at which the superposition is evaluated.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the superposition.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        args_list : list
            List of extra arguments (other than the order, n, of the eigenfunction) to be passed into the eigenfunc method
            of the potential used to initialize the instance of the class. Defaults to []
        kwargs : dict
            Dictionary of keyword arguments to be passed into the eigenfunc method of the potential used to initialize the
            instance of the class. The keys should be strings which are the names of the keyword arguments while the values
            should be the arguments themselves. Defaults to {}.
        
        Returns
        ---------------------------------------------------
        out : complex ndarray
            The output is a complex array representing the superposition of eigenstates specified in the instance of the class
            at time, t.
        
        Examples
        -----------------------------------------------------
        Return the superposition of the first two eigenstates of the Infinite Square Well of length L = 2 with coefficients
        c1 = c2 = 1/sqrt(2) at time, t = 3.
        
        >>> import BoundStatePotentials as bsp
        >>> import Particle as p
        >>> x = np.linspace(-1, 1, 10)
        >>> electron = p.Particle(9.11e-31)
        >>> basis = QuantumSuperposition(bsp.InfSquareWell(x, 2), [1, 2])
        >>> basis.superposition_timedep(3, electron)      
        array([ 0.00000000e+00+0.j        , -2.12674070e-01+0.00067531j,
               -2.41843782e-01+0.00099865j,  8.43034180e-07+0.00078708j,
                4.54519769e-01+0.00011611j,  9.38208584e-01-0.0007128j ,
                1.22474392e+00-0.00131181j,  1.15088265e+00-0.00138812j,
                6.96363551e-01-0.00088254j,  0.00000000e+00+0.j        ])
    
        Return the superposition of the first four eigenstates of the Finite Square Well of length L = 4, depth, V0 = 7 with 
        coefficients c0 = c1 = c2 = c3 = 1/sqrt(4) at time, t=3. The Finite Square Well requires additional arguments other than
        the order, n and admits keyword arguments.

        >>> import BoundStatePotentials as bsp
        >>> import Particle as p
        >>> x = np.linspace(-4, 4, 10)
        >>> electron = p.Particle(1)
        >>> basis = qs.QuantumSuperposition(bsp.FiniteSquareWell(x, 4, 4), np.arange(0, 3))    
        >>> basis.superposition_timedep(3, electron, h_bar = 1, args_list = [electron], kwargs = {'h_bar' : 1})
        array([-0.00192514+0.00241917j -0.00763601+0.01156062j -0.00781711+0.03702718j
                0.1827927 -0.12018272j  0.46046032-0.58206209j  0.068171  -0.44538928j
               -0.50888665+0.12079698j -0.19776215+0.10320365j -0.02825697+0.01874492j
               -0.00416381+0.00319912j])
            
        """
        if len(args_list) == 1:
            args = (args_list[0], )
        else:
            args = tuple(args_list)
            
        time_series = np.zeros(len(self.x), dtype = np.complex128)
        for i in range(len(self.n)):
            time_series += self.coeff[i]*self.potential.eigenfunc(self.n[i], *args, **kwargs)*np.exp(-1j*self.potential.eigenvalue(self.n[i], particle, h_bar = h_bar)*t/h_bar)
        
        return time_series   

    
    def find_coeff(self, wfunc, *args, **kwargs):
        """
        Sets the coefficients of the superposition such that it approximates an input wavefunction.
        
        Parameters
        -----------------------------------------------------------------
        wfunc : array-like
            The wavefunction that the superposition is approximating.
        *args
            Arguments (other than the order, n) to be passed into the eigenfunc method of potential used to initialize instance of class.
        **kwargs
            Keyword arguments to be passed into the eigenfunc method of potential used to initialize instance of class.
            
        Notes
        ----------------------------------------------------------------
        Since the coefficients are found by approximating an integral using the trapezoidal method, a large array should be used
        to yield more accurate results.
        """
        if len(wfunc) < 100:
            print ('WARNING: size of array may be too small to yield accurate results from trapezoidal integral approximation')
            
        coeff = []
        for i in self.n:
            coeff += [integrate.trapz(np.conj(wfunc)*self.potential.eigenfunc(i, *args, **kwargs), x = self.x)] #finds the integral of the conjugate of the wavefunction multipled by the ith eigenfunction and puts it in a list.
                                                                                               #in bra-ket notation, this is c_i = <wfunc|eigenstate_i>
        
        self.coeff = coeff
   
     
    def timedependance_animation(self, fig, ax, particle,
                                plot_elements = {'animatedline': True, 'tracker': True, 'staticline': None}, 
                                ylim = [0, 5], frames = 200, speed = 50, h_bar = 6.626e-34/(2*np.pi), 
                                args_list = [], kwargs = {}):
        """
        Creates an animated plot showing how the probability density of the superposition changes with time.
        
        Parameters
        ----------------------------------------------------
        fig : matplotlib.figure.Figure
            The figure object that is used to get draw, resize, and any other needed events.
        ax : matplotlib.axes.Axes
            The axes object in which the plot elements are plotted in.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the superposition.
        plot_elements : `dict`, optional
            Dictionary that determines what will be plotted. The keys are:
                
                ``"animatedline"``
                Boolean determining whether an animated line will be plotted (`bool`). Defaults to True.
                
                ``"tracker"``
                Boolean determining whether to keep track of the dependance (`bool`). Defaults to True.
                
                ``"staticline"``
                Any arrays to be plotted along with the animation. To plot multiple arrays, 
                use a two dimensional nested array. Note that these arrays  will NOT be animated (`ndarray`, None). Defaults to None.
        ylim : `list` (must have two elements), optional
            List that determines the lower and upper bounds of the plot. The first element in the list is the lower bound and the 
            second element is the upper bound. Defaults to [0, 5].
        frames : `int`, optional
            The number of frames to be used in the animation. Defaults to 200. 
        speed : number, optional
            The time interval at which the animation updates. eg. For speed = 50, the animation will show the superposition at times,
            t = 0, 50, 100, 150... Defaults to 50.
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        args_list : list
            List of extra arguments (other than the order, n, of the eigenfunction) to be passed into the eigenfunc method
            of the potential used to initialize the instance of the class. Defaults to []
        kwargs : dict
            Dictionary of keyword arguments to be passed into the eigenfunc method of the potential used to initialize the
            instance of the class. The keys should be strings which are the names of the keyword arguments while the values
            should be the arguments themselves. Defaults to {}.
            
        Returns
        ------------------------------------------------------------------------
        out : `matplotlib.animation.FuncAnimation`
            The output is a matplotlib.animation.FuncAnimation object.
        
        Notes
        --------------------------------------------------------------------------
        The parameter, ax, must be in the same Figure object as fig. The resulting animation shows the time dependance of the 
        PROBABILITY DENSITY of the superposition, not the wavefunction itself. 
        """
        if ax not in fig.axes:
            raise Exception('Axes object (ax) must be in same Figure object as fig')
            
        plot_elements = plot_elements.copy() #copies the dictionary in case the user wants to use the function in a for loop
        
        #Goes through the dictionary input and determines what the elements the user would like to plot
        #Determine whether the user would like to plot an animation
        if plot_elements.get('animatedline') == True:
            plot_elements['animatedline'], = ax.plot([], [])
        else:
            plot_elements.pop('animatedline', None)
        
        #Determine whether the user would like to track the time
        if plot_elements.get('tracker') == True:
            plot_elements['tracker'] = ax.text(0.05, 0.9, '', transform = ax.transAxes, fontsize = 'x-large') #the tracker element will be displayed in the upper left corner
        else:
            plot_elements.pop('tracker', None)
        
        #Determine whether the user would like to include a static plot
        if np.all(plot_elements.get('staticline') == None) == False:
            if len(plot_elements['staticline'].shape) == 1: #if the user only want to include a single set of static data
                plot_elements['staticline'], = ax.plot(self.x, plot_elements['staticline'])
            else: #if the user wants more than one set of static data
                plot_elements['staticline'] = plot_elements['staticline'].tolist()
                for i in range(len(plot_elements['staticline'])):
                    plot_elements['staticline'][i], = ax.plot(self.x, plot_elements['staticline'][i]) 
        else:
            plot_elements.pop('staticline', None)

        #init function for blitting when calling FuncAnimation
        def init():
            pev = [element for element in flatten(plot_elements.values())] #returns one dimensional list containing the elements the user wants for the plot
            
            #ensure the return line returns an iterable of artists
            if len(pev) == 1: 
                pev = (pev[0],)
            else:
                pev = tuple(pev)
            
            return pev
            
        #perform the actual animation
        ani = animation.FuncAnimation(fig, 
                                      self.animate_time, 
                                      init_func= init, 
                                      frames = frames, 
                                      fargs = [plot_elements, ylim, particle, speed, h_bar, args_list, kwargs], 
                                      blit = True)                         
        return ani
    
    def animate_time(self, i, plot_elements, ylim, particle, speed, h_bar, args_list, kwargs):
        """
        Returns iterable of artist objects showing the probability density superposition at time, t = i for use in 
        matplotlib.animation.FuncAnimation().
        
        Parameters
        -----------------------------------------------------------------------
        i : number, iterable
            The frame of the animation. In this case, it represents the time.
        plot_elements : `dict`
            Dictionary containing matplotlib Artist objects. See timedependance_animation for more details. The keys are:
                
                ``"animatedline"``
                Matplotlib Line2D object to be animated. (`matplotlib.lines.Line2D`)
                
                ``"tracker"``
                Matplotlib Text object that displays the time of each frame in the animation. (`matplotlib.text.Text`)
        ylim : `list` (must have two elements)
            List that determines the lower and upper bounds of the plot. The first element in the list is the lower bound and the 
            second element is the upper bound.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the superposition.    
        speed : number
            The time interval at which the animation updates. eg. For speed = 50, the animation will show the superposition at times,
            t = 0, 50, 100, 150... 
        h_bar : number
            The reduced planck constant. 
        args_list : list
            List of extra arguments (other than the order, n, of the eigenfunction) to be passed into the eigenfunc method
            of the potential used to initialize the instance of the class.
        kwargs : dict
            Dictionary of keyword arguments to be passed into the eigenfunc method of the potential used to initialize the
            instance of the class. The keys should be strings which are the names of the keyword arguments while the values
            should be the arguments themselves.
            
        Returns
        -----------------------------------------------------------------------
        out : tuple of Artist objects
            The output is a tuple of matplotlib Artist objects that can be passed into matplotlib.animation.FuncAnimation to create
            an animated plot.
            
        Notes
        --------------------------------------------------------------------------
        The resulting Artist objects shows the time dependance of the PROBABILITY DENSITY of the superposition, not the wavefunction 
        itself.        
        """
        #sets the data for the animated plot
        if plot_elements.get('animatedline') != False and plot_elements.get('animatedline') != None:
            time_series = self.superposition_timedep(i*speed, particle, h_bar = h_bar, args_list = args_list, kwargs = kwargs) #finds the superposition at time, t = i*speed
            prob_density = op.probdensity(time_series) #find the probability density of the resulting superposition
            
            plot_elements['animatedline'].set_data(self.x, prob_density) #sets data of Line2D object to be probability density of wavefunction
            plot_elements['animatedline'].axes.axis([self.x[0], self.x[-1], ylim[0], ylim[1]])
        
        #sets text object displaying timing of each frame
        if plot_elements.get('tracker') != False and plot_elements.get('tracker') != None:
            plot_elements['tracker'].set_text(f'time = {int(i)}')
        
        #returns one dimensional list containing the elements the user wants for the plot
        pev = [element for element in flatten(plot_elements.values())]     

        #ensure the return line returns an iterable of artists
        if len(pev) == 1:
            pev = (pev[0],)
        else:
            pev = tuple(pev)
            
        return pev
 
    def basisdependance_animation(self, fig, ax, particle, wfunc, 
                                  plot_elements = {'animatedline': True, 'tracker': True, 'staticline': None}, 
                                  nbasis = None, t = 0, ylim = [0, 5], h_bar = 6.626e-34/(2*np.pi), interval = 200, 
                                  args_list = [], kwargs = {}):
        """
        Given a wavefunction, creates an animated plot showing how the probability density of the superposition approximates the 
        probability density of the wavefunction at time, t, with respect to the number of basis functions included.
        
        Parameters
        ----------------------------------------------------
        fig : matplotlib.figure.Figure
            The figure object that is used to get draw, resize, and any other needed events.
        ax : matplotlib.axes.Axes
            The axes object in which the plot elements are plotted in.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the superposition.
        wfunc : array-like
            The wavefunction that the superposition is approximating.            
        plot_elements : `dict`, optional
            Dictionary that determines what will be plotted. The keys are:
                
                ``"animatedline"``
                Boolean determining whether an animated line will be plotted (`bool`). Defaults to True.
                
                ``"tracker"``
                Boolean determining whether to keep track of the dependance (`bool`). Defaults to True.
                
                ``"staticline"``
                Any arrays to be plotted along with the animation. To plot multiple arrays, 
                use a two dimensional nested array. Note that these arrays  will NOT be animated (`ndarray`, None). Defaults to None.
        nbasis : int, None, optional
            The final number of basis included in the approximation. If None, nbasis is the length of the array, n, used to initialize
            the instance of the class. Defaults to None.
        t : number, optional
            The time at which the superposition is to be evaluated. Defaults to 0.
        ylim : `list` (must have two elements), optional
            List that determines the lower and upper bounds of the plot. The first element in the list is the lower bound and the 
            second element is the upper bound. Defaults to [0, 5].
        h_bar : number, optional
            The reduced planck constant. Defaults to 6.626e-34/2pi. For natural units, set to 1.
        interval : number, optional
            The time in milliseconds between each frame of the animation. Defaults to 200.
        args_list : list
            List of extra arguments (other than the order, n, of the eigenfunction) to be passed into the eigenfunc method
            of the potential used to initialize the instance of the class. Defaults to []
        kwargs : dict
            Dictionary of keyword arguments to be passed into the eigenfunc method of the potential used to initialize the
            instance of the class. The keys should be strings which are the names of the keyword arguments while the values
            should be the arguments themselves. Defaults to {}.
            
        Returns
        ------------------------------------------------------------------------
        out : `matplotlib.animation.FuncAnimation`
            The output is a matplotlib.animation.FuncAnimation object.
        
        Notes
        --------------------------------------------------------------------------
        The parameter, ax, must be in the same Figure object as fig. The resulting animation shows the basis dependance of the 
        PROBABILITY DENSITY of the superposition, not the wavefunction itself. 
        """
        if ax not in fig.axes:
            raise Exception('Axes object (ax) must be in same Figure object as fig')
        if nbasis == None:
            nbasis = len(self.n)
        if nbasis > len(self.n) or nbasis%1 != 0 or nbasis < 1:
            raise Exception('nbasis must be a positive integer less than or equal to the lengh of n in instance of QuantumSuperposition')
            
        plot_elements = plot_elements.copy() #copies the dictionary in case the user wants to use the function in a for loop
        
        #Goes through the dictionary input and determines what the elements the user would like to plot
        #Determine whether the user would like to plot an animation
        if plot_elements.get('animatedline') == True:
            plot_elements['animatedline'], = ax.plot([], [])
        else:
            plot_elements.pop('animatedline', None)
        
        #Determine whether the user would like to track the time
        if plot_elements.get('tracker') == True:
            plot_elements['tracker'] = ax.text(0.05, 0.9, '', transform = ax.transAxes, fontsize = 'x-large') #the tracker element will be displayed in the upper left corner
        else:
            plot_elements.pop('tracker', None)
        
        #Determine whether the user would like to include a static plot
        if np.all(plot_elements.get('staticline') == None) == False:
            if len(plot_elements['staticline'].shape) == 1: #if the user only want to include a single set of static data
                plot_elements['staticline'], = ax.plot(self.x, plot_elements['staticline'])
            else: #if the user wants more than one set of static data
                plot_elements['staticline'] = plot_elements['staticline'].tolist()
                for i in range(len(plot_elements['staticline'])):
                    plot_elements['staticline'][i], = ax.plot(self.x, plot_elements['staticline'][i]) 
        else:
            plot_elements.pop('staticline', None)

        #init function for blitting when calling FuncAnimation
        def init():
            pev = [element for element in flatten(plot_elements.values())] #returns one dimensional list containing the elements the user wants for the plot
            
            #ensure the return line returns an iterable of artists
            if len(pev) == 1: 
                pev = (pev[0],)
            else:
                pev = tuple(pev)
            
            return pev            

        #perform the actual animation
        ani = animation.FuncAnimation(fig, 
                                      self.animate_basis, 
                                      frames = nbasis, 
                                      init_func = init, 
                                      fargs = (plot_elements, wfunc, t, ylim, particle, h_bar, args_list, kwargs), 
                                      interval = interval, 
                                      blit = True)
        return ani
        
    def animate_basis(self, i, plot_elements, wfunc, t, ylim, particle, h_bar, args_list, kwargs):
        """
        Returns iterable of artist objects showing the probability density superposition at time, t, with number of basis, i,
        for use in matplotlib.animation.FuncAnimation().
        
        Parameters
        -----------------------------------------------------------------------
        i : number, iterable
            The frame of the animation. In this case, it represents the number of basis functions used to approximate a wavefunction.
        plot_elements : `dict`
            Dictionary containing matplotlib Artist objects. See timedependance_animation for more details. The keys are:
                
                ``"animatedline"``
                Matplotlib Line2D object to be animated. (`matplotlib.lines.Line2D`)
                
                ``"tracker"``
                Matplotlib Text object that displays the time of each frame in the animation. (`matplotlib.text.Text`)
        wfunc : array-like
            The wavefunction that the superposition is approximating.   
        t : number, optional
            The time at which the superposition is to be evaluated. Defaults to 0.            
        ylim : `list` (must have two elements)
            List that determines the lower and upper bounds of the plot. The first element in the list is the lower bound and the 
            second element is the upper bound.
        particle : Particle class (see Particle.py for more details)
            Particle whose mass will determine the energy and therefore the time dependance of the superposition.    
        h_bar : number
            The reduced planck constant. 
        args_list : list
            List of extra arguments (other than the order, n, of the eigenfunction) to be passed into the eigenfunc method
            of the potential used to initialize the instance of the class.
        kwargs : dict
            Dictionary of keyword arguments to be passed into the eigenfunc method of the potential used to initialize the
            instance of the class. The keys should be strings which are the names of the keyword arguments while the values
            should be the arguments themselves.
            
        Returns
        -----------------------------------------------------------------------
        out : tuple of Artist objects
            The output is a tuple of matplotlib Artist objects that can be passed into matplotlib.animation.FuncAnimation to create
            an animated plot.
            
        Notes
        --------------------------------------------------------------------------
        The resulting Artist objects shows the basis dependance of the PROBABILITY DENSITY of the superposition, not the superposition 
        itself.        
        """
        #sets the data for the animated plot
        if len(args_list) == 1:
            args = (args_list[0], )
        else:
            args = tuple(args_list)
            
        if plot_elements.get('animatedline') != False and plot_elements.get('animatedline') != None:
            basis = QuantumSuperposition(self.potential, self.n[:int(i + 1)]) #create new superposition with specified number of basis functions
            basis.find_coeff(wfunc, *args, **kwargs) #finds coefficients to approximate wavefunction
            
            prob_density = op.probdensity(basis.superposition_timedep(t, particle, h_bar = h_bar, args_list = args_list, kwargs = kwargs)) #finds probability density of resulting superposition
            
            plot_elements['animatedline'].set_data(self.x, prob_density) #sets data of Line2D object to be probability density of wavefunction
            plot_elements['animatedline'].axes.axis([self.x[0], self.x[-1], ylim[0], ylim[1]])            

        #sets text object displaying number of basis for each frame
        if plot_elements.get('tracker') != False and plot_elements.get('tracker') != None:
            plot_elements['tracker'].set_text(f'last n-value = {self.n[int(i)]}')           

        #returns one dimensional list containing the elements the user wants for the plot
        pev = [element for element in flatten(plot_elements.values())]     

        #ensure the return line returns an iterable of artists
        if len(pev) == 1:
            pev = (pev[0],)
        else:
            pev = tuple(pev)
            
        return pev            
            
