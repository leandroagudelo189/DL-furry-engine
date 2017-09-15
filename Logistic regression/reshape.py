#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:26:57 2017

@author: Leo
"""
import numpy as np

cal = [59,2,2,2,3]
cal1 = np.array(cal)

cal1.reshape(1,5) # one row 5 columns

# we use the reshape command to make sure your matrix is in the right dimension