# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:23:58 2019

@author: lawre
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
import matplotlib.animation as animation
import matplotlib.pyplot as plt

class Particle:
    
    def __init__(self, m):
        self.m = m

        self.gauss_wavepacket = None
        
    def set_gauss(self, x, a, x0, k0):
        """give lots of warnings about approximations and such. Also give credit to the person on github. test this out."""
        self.gauss_wavepacket = ((a*np.sqrt(np.pi))**(-0.5)
                                *np.exp(-0.5*((x - x0)*1./a)**2 + 1j*x*k0))
    
    def timestep_ssfm(self, Vx, x, k0, t, dt, h_bar = 6.626e-34/(2*np.pi)):
        if np.any(np.diff(x) == x[1] - x[0]) == False:
            raise Exception('x must be a uniform array')
        
        nsteps = int(t/dt)
        dx = x[1] - x[0]
        dk = 2*np.pi/(x[-1] - x[0])
        k = k0 + dk*np.arange(len(x))
        psi_xt = self.gauss_wavepacket
        for i in range(nsteps):
            psi_xt = psi_xt*np.exp(-1j*Vx*dt/(2*h_bar))
            psi_kt = fft(psi_xt)
            psi_kt = psi_kt*np.exp(-1j*h_bar*k**2/(2*self.m))
            psi_xt = ifft(psi_kt)
            psi_xt = psi_xt*np.exp(-1j*Vx*dt/(2*h_bar))
            
        return psi_xt
    
    def animate_ssfm(self, fig, ax, Vx, x, k0, h_bar = 6.626e-34/(2*np.pi)):
        if np.any(np.diff(x) == x[1] - x[0]) == False:
            raise Exception('x must be a uniform array')
            
        line, = ax.plot([], [])
            
        dx = x[1] - x[0]
        dk = 2*np.pi/(x[-1] - x[0])
        k = k0 + dk*np.arange(len(x))
        self.psi_xt = self.gauss_wavepacket
        
        def init():
            line, = ax.plot([], [])
            
            return line,
        
        def animate(i, dt = 0.01):
            i = i*dt
            self.psi_xt = self.psi_xt*np.exp(-1j*Vx*i/(2*h_bar))
            psi_kt = fft(self.psi_xt)
            psi_kt = psi_kt*np.exp(-1j*h_bar*k**2/(2*self.m))
            self.psi_xt = ifft(psi_kt)
            self.psi_xt = self.psi_xt*np.exp(-1j*Vx*i/(2*h_bar))
            
            line.set_data(x, self.psi_xt*np.conj(self.psi_xt))
            line.axes.axis([x[0], x[-1], 0, 0.08])
            return line,
        
        ani = animation.FuncAnimation(fig, animate, init_func = init, blit = True)
        return ani
    
#electron = Particle(1)
#x = np.linspace(-100, 100, 5000)
#Vx = np.zeros(5000)
#p0 = np.sqrt(0.4*1.5)
#dp2 = p0**2/80
#d = 1/np.sqrt(2*dp2)
#k0 = p0
#electron.set_gauss(x, d, 0, k0)
#
#fig = plt.figure()
#ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#ani = electron.animate_ssfm(fig, ax, Vx, x, k0, h_bar = 1)
#        
#    
#timestep = electron.timestep_ssfm(Vx, x, k0, 1, 0.01, h_bar = 1)

#plt.plot(x, electron.gauss_wavepacket*np.conj(electron.gauss_wavepacket))
#plt.plot(x, timestep*np.conj(timestep))

        
        