#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:46:45 2022

@author: thkam
"""

import numpy as np
from defaults import constants

designs = {
  'A' :  {
    'room_L' : 10,
    'room_W' : 10,
    'room_H' : 3,
    'refl_north' : 0.6,
    'refl_south' : 0.6,
    'refl_east' : 0.6,
    'refl_west' : 0.6,
    'refl_ceiling' : 0.6,
    'refl_floor' : 0.3,
    'm_sensor' : 1,
    'r_sensor' : np.array([[5, 5, 0], [2, 2, 0]]),
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 2.0,
    'amb_L2' : 1.0,
    'nR_sensor' : constants.ez,
    'nS_sensor' : constants.ez,
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 250e3,
    'Rb_sensor' : 250e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 6,
    'A_master' : 1e-4,
    'A_sensor' : 27e-4,
    'pv': True,
    'Vcharge': 4.2,
    'Isc':0.165, #Short-circuit current of PV panel in Amperes
    'Voc': 2.79, #Open-circuit voltage of PV panel in Volts
    'Imp': 0.14, #Maximum point current of PV panel in Amperes
    'Vmp': 2.24, #Maximum point voltage of PV panel in Volts
    'N': 4, #Number of PV cells connected in series
    'Pmax': 0.14*2.24,
    'A': 1, #first approximation of ideality factor,
    'r_lights': np.array([[2,5,3], [4,4,3]]),    
    'PT_lights': 6
    }
}
  