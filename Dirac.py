# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:09:44 2019

@author: lawre
"""
import numpy as np


class Dirac:
    
    def __init__(self, wfunc):
        self.bra = np.array(wfunc)
        self.ket = np.conj(wfunc)

class Basis:
    
    def __init__(self, list_):
        self.basis = {i : Dirac(wfunc) for wfunc, i in enumerate(list_)}
        
        self.bbra = {key : self.basis[key].bra for key in self.basis}
        self.bket = {key : self.basis[key].ket for key in self.basis}