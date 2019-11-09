# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:23:58 2019

@author: lawre
"""
import Operators as op

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Particle:
    
    def __init__(self, m):
        self.m = m

        self._gauss = None
    
    def get_gauss(self):
        return self._gauss
            
    def set_gauss(self, args):
        """give lots of warnings about approximations and such. Also give credit to the person on github. test this out.
        
        """
        x, a, x0, k0 = args #unpack arguments
        self._gauss = ((a * np.sqrt(np.pi)) ** (-0.5)
                                * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x *k0))
        
        if np.any(np.diff(x) == x[1] - x[0]) == False:
            raise Exception('x must be a uniform array')             
        self._x = x
    
    gauss = property(get_gauss, set_gauss)


    def xspace_params(self):
        N = len(self._x)
        dx = self._x[1] - self._x[0]
        dk = 2*np.pi / (N * dx)
        
        return N, dx, dk
        
    def timestep_ssfm(self, kmin, Vx, t, dt, h_bar = 6.626e-34/(2*np.pi)):
        N, dx, dk = self.xspace_params()
        
        Nsteps = int(t / dt)
        k = kmin + dk * np.arange(N)   
        
        psi_xt_dft = self.gauss * np.exp(-1j * kmin * self._x) * (dx / np.sqrt(2 * np.pi))
        psi_xt_dft = psi_xt_dft * np.exp(-1j * Vx * dt / (2 * h_bar))
        
        for i in range(Nsteps - 1):
            psi_kt_dft = fft(psi_xt_dft)
            psi_kt_dft = psi_kt_dft * np.exp(-1j * h_bar *k * k * dt / (2 * self.m))
            
            psi_xt_dft = ifft(psi_kt_dft)
            psi_xt_dft = psi_xt_dft * np.exp(-1j * Vx * dt / h_bar)
        
        psi_kt_dft = fft(psi_xt_dft)
        psi_kt_dft = psi_kt_dft * np.exp(-1j * h_bar * k * k * dt / (2 * self.m)) 
        
        psi_xt_dft = ifft(psi_kt_dft)
        psi_xt_dft = psi_xt_dft * np.exp(-1j * Vx * dt / (2 * h_bar))
        self._gauss = psi_xt_dft * np.exp(1j * kmin * self._x) * (np.sqrt(2 * np.pi) / dx)
    
    
    def animate_ssfm(self, fig, ax, kmin, Vx, t, dt, ylim, h_bar = 6.626e-34/(2*np.pi)):
        line, = ax.plot([], [])
        
        def init():
            line, = ax.plot([], [])
            
            return line,
    
        ani = animation.FuncAnimation(fig, 
                                      self.animate, 
                                      init_func = init, 
                                      fargs = (kmin, Vx, t, dt, ylim, line, h_bar), 
                                      interval = 30, 
                                      blit = True)
        return ani
    

    def animate(self, i, kmin, Vx, t, dt, ylim, line, h_bar):
        self.timestep_ssfm(kmin, Vx, t, dt, h_bar = h_bar)
        wfunc = self.gauss
        
        data = 4 * abs(wfunc)
#        data = op.probdensity(wfunc)
        
        line.set_data(self._x, data)
        line.axes.axis([self._x[0], self._x[-1], 0, ylim])
        
        return line,
    
    
#electron = Particle(1.9)
#N = 2 ** 11
#dx = 0.1
#x = dx * (np.arange(N) - 0.5 * N)
#V0 = 1.
#L = 1/np.sqrt(2*1.9*V0)
#a = 3*L
#p0 = np.sqrt(2*1.9*0.2*V0)
#dp2 = p0*p0*1./80
#d = 3/np.sqrt(2*dp2)
#
#
#Vx = np.piecewise(x,
#                         
#                         [np.any([x <= 0, 
#                                  x >= a], axis = 0), 
#                          np.all([x > 0, 
#                                  x < a], axis = 0)],
#
#                         [0, 
#                          V0])
#
#Vx[x < -98] = 1E6
#Vx[x > 98] = 1E6
#    
#k0 = p0
##electron.set_gauss(x, d, -60, k0)
###electron.gauss_wavepacket = [0, 1, 2]
##print (electron.gauss_wavepacket)
##print (electron.dx)
#
#electron.gauss = (x, d, -60*L, k0)
##electron.timestep_ssfm(-28, Vx, 0.5, 0.01, h_bar = 1)
##electron.set_gauss(x, d, -60*L, k0)
#
##electron.timestep_ssfm(Vx, x, -28., 0.5, 0.01, h_bar = 1)
##electron.timestep_ssfm(Vx, x, -28., 0.5, 0.01, h_bar = 1)
##print (electron.timestep_ssfm(Vx, x, -28., 1.5, 0.01, h_bar = 1)[1024])
#
#
#fig = plt.figure()
#ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#ani = electron.animate_ssfm(fig, ax, -28, Vx, 0.5, 0.01, 1.5, h_bar = 1.)
#        
#    
##timestep = electron.timestep_ssfm(Vx, x, k0, 1, 0.01, h_bar = 1.)
#
#plt.plot(x, Vx)

        
        