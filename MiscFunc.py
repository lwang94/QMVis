# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:59:25 2019

@author: lawre
"""

import numpy as np
import scipy.integrate as integrate
import BoundStatePotentials as bsp
import collections 
import matplotlib.animation as animation

#def is_sorted(a):
#    for i in range(a.size-1):
#         if a[i+1] < a[i] :
#               return False
#    return True

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
            
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
    
def momentum_op(x, wfunc, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
    """needs to be tested"""
    return -1j*h_bar*first_derivative(x, wfunc, finitediff_scheme = finitediff_scheme)

def position_op(x, wfunc):
    """needs to be tested"""
    return x*wfunc

def kinetic_op(x, wfunc, particle, h_bar = 6.626e-34/(2*np.pi), finitediff_scheme = 'central'):
    return (-h_bar**2/(2*particle.m))*second_derivative(x, wfunc, finitediff_scheme = finitediff_scheme)   

def probdensity(wfunc):
    return wfunc*np.conj(wfunc)

def normalize(x, wfunc):
    return 1/np.sqrt(integrate.trapz(probdensity(wfunc), x = x))*wfunc

def plot_animate_real(fig, ax, x, func, args,
                      plot_elements = {'animatedline': True, 'tracker': True, 'staticline': None},
                      ylim = [0, 5], speed = 100):
    if ax not in fig.axes:
        raise Exception('Axes object (ax) must be in same Figure object as fig')
        
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
        if len(plot_elements['staticline'].shape) == 1:
            plot_elements['staticline'], = ax.plot(x, plot_elements['staticline'])
        else:
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
        

    ani = animation.FuncAnimation(fig, animate_time_real, init_func= init, fargs = [plot_elements, x, func, args, ylim, speed], blit = True)                         
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

def animate_time_imag(i, plot_elements, x, func, args, ylim, speed):
    if plot_elements.get('animatedline') != False and plot_elements.get('animatedline') != None:
        imag_wfunc = np.imag(func(i*speed, *args))
        
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
    

def custom_func(x):
    y = np.exp(-x**2)
#    y = -(x+0.5)**2*(0.5-x)**2
    return  normalize(x, y)


def custom_func2(x):
    infsquarewell = bsp.InfSquareWell(x, 7)
    return (1/np.sqrt(3))*infsquarewell.eigenfunc(1) + (1/np.sqrt(3))*infsquarewell.eigenfunc(3) + (1/np.sqrt(3))*infsquarewell.eigenfunc(4)