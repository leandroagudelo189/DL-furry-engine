#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 23:35:31 2017

@author: leandro.agudelo
"""

import numpy as np

import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print(c)
print("vector:" + str(1000*(toc-tic)) + "ms")



c = 0
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("for loop:" + str(1000*(toc-tic)) + "ms")
