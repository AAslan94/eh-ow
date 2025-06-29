#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from libow8 import sensor_net
import matplotlib.pyplot as plt
import owutils as ut
from designs import designs
from nrg_harvesting import *

N_CPUS = 4
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 14,
    "lines.linewidth" : 2,
})

ehx = EH()
params_d = designs['A']  
x = sensor_net(**params_d) 
x.calch()
x.light_sim()
x.calc_noise(np.sum(ehx.l.i_sm_tot)) #need to include thermal noise
x.calc_rq() 


