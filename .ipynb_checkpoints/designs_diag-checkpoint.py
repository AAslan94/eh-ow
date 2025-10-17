#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from defaults import constants
from create_points import *
import owutils as ut

#scenario B
f_b_opt = np.load('no_sun_to_use21.npz') #optimization results - optimized area / orientation
f_b_def = np.load('nosun_noor_21.npz') # optimization results - optimized area / default orientation
A_b_d = f_b_def['orientations'].flatten() #optimized area for default orientation
nz = f_b_opt['orientations'][:,0:2] # optimized orientation - cartesian
nR_b = ut.spher_to_cart_ar(1,nz[:,0],nz[:,1]).T  #optimized orientation - spherical
A_b_opt = f_b_opt['orientations'][:,2] # optimized area for optimized orientation 

#scenario A
f_a_opt = np.load('SUN_TO_USE_21.npz') #optimization results - optimized area / orientation
f_a_def = np.load('sun_noor_21.npz') # optimization results - optimized area / default orientation
A_a_d = f_a_def['orientations'].flatten() #optimized area for default orientation
nz = f_a_opt['orientations'][:,0:2] # optimized orientation - cartesian
nR_a = ut.spher_to_cart_ar(1,nz[:,0],nz[:,1]).T  #optimized orientation - spherical
A_a_opt = f_a_opt['orientations'][:,2] # optimized area for optimized orientation 


def align_receiver_to_transmitter(r_rec, r_tra):
    """
    Calculates the unit vector direction from the receiver's position (r_rec) 
    to the transmitter's position (r_tra).

    Parameters:
    r_rec (np.ndarray): Receiver's position(s). Can be a 1D vector (N,) 
                        or a 2D array of vectors (M, N).
    r_tra (np.ndarray): Transmitter's position. Must be a 1D vector (N,).

    Returns:
    np.ndarray: The normalized unit vector(s) representing the alignment direction.
    """
    # 1. Calculate the displacement vector: (Transmitter - Receiver)
    # This vector points from r_rec to r_tra.
    # Broadcasting handles the subtraction for single or multiple receiver positions.
    displacement = r_tra - r_rec

    # 2. Calculate the magnitude (L2 norm) of the displacement vector(s)
    # axis=-1 ensures the norm is calculated along the vector components (the last axis).
    # keepdims=True ensures the norm can be correctly broadcast back for division.
    norm = np.linalg.norm(displacement, axis=-1, keepdims=True)

    # Handle the zero-length vector case to prevent division by zero
    # np.where is used for a safe division.
    # It returns [0., 0., 0.] for any vector with a magnitude of 0.
    unit_vector = np.where(norm == 0, 0, displacement / norm)

    return unit_vector

arr = gen_points(1,9,1,9,3,5,5,False) #positions of lighting LEDs 
arr_m = np.array([5,5,3])
diag = np.round(diagonal_points(0, 10, 0, 10, 0,21),2) #positions of sensors
#u = np.round(align_receiver_to_transmitter(np.array([6,6,0]), arr_m), 2)

designs = {
  'A' :  {
    'room_L' : 10,
    'room_W' : 10,
    'room_H' : 3,
    'refl_north' : 0.7,
    'refl_south' : 0.7,
    'refl_east' : 0.7,
    'refl_west' : 0.7,
    'refl_ceiling' : 0.7,
    'refl_floor' : 0.3,
    'm_sensor' : 1,
    'r_sensor' : diag,
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 2,
    'amb_L2' : 1,
    'nR_sensor' : nR_a,
    'nS_sensor' : align_receiver_to_transmitter(diag, arr_m),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 6,
    'A_master' : 1e-4,
    'A_sensor' : A_a_opt,
    'pv': True,
    'r_lights': arr,    
    'PT_lights': 6
    },
  'B' :  {
    'room_L' : 10,
    'room_W' : 10,
    'room_H' : 3,
    'refl_north' : 0.7,
    'refl_south' : 0.7,
    'refl_east' : 0.7,
    'refl_west' : 0.7,
    'refl_ceiling' : 0.7,
    'refl_floor' : 0.3,
    'm_sensor' : 1,
    'r_sensor' : diag,
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' : nR_b,
    'nS_sensor' : np.round(align_receiver_to_transmitter(diag, arr_m),2),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 6,
    'A_master' : 1e-4,
    'A_sensor' : A_b_opt,
    'pv': True,
    'r_lights': arr,    
    'PT_lights': 6
    },
}
  