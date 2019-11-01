# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:29:28 2019

@author: lawre
"""
import collections

def flatten(l):
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el