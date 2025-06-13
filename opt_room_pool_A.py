#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 11:44:26 2021

@author: thkam
"""
import numpy as np
from libow8 import sensor_net
import matplotlib.pyplot as plt
import pickle
import mixed_ga as ga
import owutils as ut
from designs import designs
from multiprocessing import Pool
from time import sleep

N_CPUS = 4
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})


params_d = designs['A']  
l = sensor_net(**params_d) 
l.calch()
l.light_sim()
x = sensor_net(**params_d) 
x.calch()
x.light_sim()
x.calc_noise()
x.calc_rq() 

