#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from defaults import constants
from create_points import *
import owutils as ut

def led_pow(light,mn,p):
    is_index = np.where(np.all(light == mn, axis=1))
    index = is_index[0][0] #extraxt int from np array
    size = light.shape[0]
    power = np.full((size,), p)
    power[index] = power[index] * 0.707 
    return power
    

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


led_pos_A = gen_points(1,9,1,9,3,5,5,False) #positions of Lighting LEDs
sn_pos_A = gen_points(0.4,9.6,0.4,9.6,0,10,10,False) #SNs positions
m_pos_A = np.array([5,5,3]) #MN position

sn_pos_B = gen_points(0,8,0,6,0,32,24,False)
led_pos_B = np.array([[2,1,2.8], [2,5,2.8],[6,1,2.8], [6,5,2.8], [4,3,2.8]])
m_pos_B = np.array([4,3,2.8])

sn_pos_C = gen_points(0,8,0,1,0,24,6,False)
led_pos_C = np.array([
           [0.8, 0.5, 4.0],  
           [2.4, 0.5, 4.0],  
           [5.6, 0.5, 4.0],  
           [7.2, 0.5, 4.0],
           [4,0.5,4]
           ])
m_pos_C = np.array([4,0.5,4])


designs = {
  'A2' :  {
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
    'r_sensor' : sn_pos_A,
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 2,
    'amb_L2' : 1,
    'nR_sensor' : np.round(align_receiver_to_transmitter(sn_pos_A, m_pos_A),2),
    'nS_sensor' : align_receiver_to_transmitter(sn_pos_A, m_pos_A),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : 10e-4,
    'pv': True,
    'r_lights': led_pos_A,  
    'm_lights': 1,  
    'PT_lights': led_pow(led_pos_A,m_pos_A,4)
    },
  'A1' :  {
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
    'r_sensor' : sn_pos_A,
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' : np.round(align_receiver_to_transmitter(sn_pos_A, m_pos_A),2),
    'nS_sensor' : align_receiver_to_transmitter(sn_pos_A, m_pos_A),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : 10e-4,
    'pv': True,
    'r_lights': led_pos_A,    
    'm_lights': 1, 
    'PT_lights': led_pow(led_pos_A,m_pos_A,4)
    },
    
    'B' :  {
    'room_L' : 8,
    'room_W' : 6,
    'room_H' : 2.8,
    'refl_north' : 0.5,
    'refl_south' : 0.5,
    'refl_east' : 0.5,
    'refl_west' : 0.5,
    'refl_ceiling' : 0.8,
    'refl_floor' : 0.2,
    'm_sensor' : 1,
    'r_sensor' : sn_pos_B,
    'm_master' : 1,
    'r_master' : m_pos_B,
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' : np.round(align_receiver_to_transmitter(sn_pos_B, m_pos_B),2),
    'nS_sensor' : align_receiver_to_transmitter(sn_pos_B, m_pos_B),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : 10e-4,
    'pv': True,
    'r_lights': led_pos_B,    
    'm_lights': 3, 
    'PT_lights': led_pow(led_pos_B,m_pos_B,4)
    },
    
    'C' :  {
    'room_L' : 8,
    'room_W' : 1,
    'room_H' : 4,
    'refl_north' : 0.35,
    'refl_south' : 0.35,
    'refl_east' : 0.0,
    'refl_west' : 0.0,
    'refl_ceiling' : 0.6,
    'refl_floor' : 0.4,
    'm_sensor' : 1,
    'r_sensor' : sn_pos_C,
    'm_master' : 1,
    'r_master' : m_pos_C,
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' : np.round(align_receiver_to_transmitter(sn_pos_C, m_pos_C),2),
    'nS_sensor' : align_receiver_to_transmitter(sn_pos_C, m_pos_C),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 5,
    'A_master' : 1e-4,
    'A_sensor' : 10e-4,
    'pv': True,
    'r_lights': led_pos_C,    
    'm_lights': 1, 
    'PT_lights': led_pow(led_pos_C,m_pos_C,5)
    },
}
  
