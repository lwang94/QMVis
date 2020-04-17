# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:38:16 2019

@author: lawre
"""
import numpy as np
from flatten import flatten
import matplotlib.animation as animation

def plot_animation(component, fig, ax, x, func, args, kwargs,
                      plot_elements = {'animatedline': True, 'tracker': True, 'staticline': None},
                      ylim = [0, 5], frames = 200, interval = 200, speed = 100):
    """
    """
    if ax not in fig.axes:
        raise Exception('Axes object (ax) must be in same Figure object as fig')
    
    if component == 'r': 
        function = animate_time_real
    elif component == 'i':
        function = animate_time_imag
    else:
        raise Exception('Component must be either \'r\' or \'i\'')
        
    plot_elements = plot_elements.copy()
    
    if plot_elements.get('animatedline') == True:
        plot_elements['animatedline'], = ax.plot([], [])
    else:
        plot_elements.pop('animatedline', None)
        
    if plot_elements.get('tracker') == True:
        plot_elements['tracker'] = ax.text(0.05, 0.9, '', transform = ax.transAxes, fontsize = 'x-large')
    else:
        plot_elements.pop('tracker', None)

    if np.all(plot_elements.get('staticline') == None) == False:
        if len(plot_elements['staticline'].shape) == 1: #if the user only want to include a single set of static data
            plot_elements['staticline'], = ax.plot(x, plot_elements['staticline'])
        else: #if the user wants more than one set of static data
            plot_elements['staticline'] = plot_elements['staticline'].tolist()
            for i in range(len(plot_elements['staticline'])):
                plot_elements['staticline'][i], = ax.plot(x, plot_elements['staticline'][i]) 
    else:
        plot_elements.pop('staticline', None)


        
    def init():
        pev = [element for element in flatten(plot_elements.values())]  
        if len(pev) == 1:
            pev = (pev[0],)
        else:
            pev = tuple(pev)
        return pev

    ani = animation.FuncAnimation(fig, 
                                  function, 
                                  init_func= init, 
                                  frames = frames, 
                                  fargs = [plot_elements, x, func, args, kwargs, ylim, speed], 
                                  interval = interval,
                                  blit = True)                         
    return ani    
    

def animate_time_real(i, plot_elements, x, func, args, kwargs, ylim, speed):
    if plot_elements.get('animatedline') != False and plot_elements.get('animatedline') != None:
        real_wfunc = np.real(func(i*speed, *args, **kwargs))
        
        plot_elements['animatedline'].set_data(x, real_wfunc)
        plot_elements['animatedline'].axes.axis([x[0], x[-1], ylim[0], ylim[1]]) 
    
    if plot_elements.get('tracker') != False and plot_elements.get('tracker') != None:
        plot_elements['tracker'].set_text(f'time = {int(i)}')

    pev = [element for element in flatten(plot_elements.values())]     

    if len(pev) == 1:
        pev = (pev[0],)
    else:
        pev = tuple(pev)
        
    return pev

def animate_time_imag(i, plot_elements, x, func, args, kwargs, ylim, speed):
    if plot_elements.get('animatedline') != False and plot_elements.get('animatedline') != None:
        imag_wfunc = np.imag(func(i*speed, *args, **kwargs))
        
        plot_elements['animatedline'].set_data(x, imag_wfunc)
        plot_elements['animatedline'].axes.axis([x[0], x[-1], ylim[0], ylim[1]]) 
    
    if plot_elements.get('tracker') != False and plot_elements.get('tracker') != None:
        plot_elements['tracker'].set_text(f'time = {int(i)}')

    pev = [element for element in flatten(plot_elements.values())]     

    if len(pev) == 1:
        pev = (pev[0],)
    else:
        pev = tuple(pev)
        
    return pev
    