import numpy as np
from defaults import constants
from create_points import *
import owutils as ut


#area only A1
d = np.load('area_only_cma_results_A1_eight.npy', allow_pickle=True)
areas_a1_def = np.array([item['area_cm2'] / 1e4 for item in d])

#area only A2
d2 = np.load('area_only_cma_results_A2.npy', allow_pickle=True)
areas_a2_def = np.array([item['area_cm2'] / 1e4 for item in d2])

#area only B1
d = np.load('area_only_cma_results_B1_eight.npy', allow_pickle=True)
areas_b1_def = np.array([item['area_cm2'] / 1e4 for item in d])


def led_pow(light,mn,p):
    is_index = np.where(np.all(light == mn, axis=1))
    index = is_index[0][0] #extraxt int from np array
    size = light.shape[0]
    power = np.full((size,), p)
    power[index] = power[index] * 0.707 
    return power

#load results for B1
#load results for C
data = np.load('cma_simple_C_eight.npy', allow_pickle=True)
areas = np.array([d['area_cm2'] for d in data])
thetas = np.array([d['theta'] for d in data])
phis = np.array([d['phi'] for d in data])

area_B1 = areas*1e-4
theta_B1 = np.deg2rad(thetas)
phi_B1 = np.deg2rad(phis)
nR_B1 = ut.spher_to_cart_ar(1,theta_B1,phi_B1).T

data = np.load('robust_minimax_results_B1.npy')

# Extract parameters
area_B1= data[:, 5]
theta_B1 = data[:, 3]
phi_B1   = data[:, 4]

# Receiver normals
nR_B1 = ut.spher_to_cart_ar(1, theta_B1, phi_B1).T



start1 = np.array([0.2, 3, 0])
end1   = np.array([7.8, 3, 0])
line1 = np.linspace(start1, end1, 11)


start2 = np.array([0.2, 1.5, 0])
end2   = np.array([7.8, 1.5, 0])
line2 = np.linspace(start2, end2, 11)

# Combine into one array
pos_opt_B  = np.vstack((line1, line2))


#load results for A1
data = np.load('robust_minimax_results_A1.npy')

# Extract parameters
area_A1= data[:, 5]
theta_A1 = data[:, 3]
phi_A1   = data[:, 4]

# Receiver normals
nR_A1 = ut.spher_to_cart_ar(1, theta_A1, phi_A1).T


#load results for A2
data = np.load('robust_minimax_results_A2.npy')

# Extract parameters
area_A2= data[:, 5]
theta_A2 = data[:, 3]
phi_A2   = data[:, 4]

# Receiver normals
nR_A2 = ut.spher_to_cart_ar(1, theta_A2, phi_A2).T


#load results for B2
data = np.load('robust_minimax_results_B2.npy')

# Extract parameters
area_B2= data[:, 5]
theta_B2 = data[:, 3]
phi_B2   = data[:, 4]

# Receiver normals
nR_B2 = ut.spher_to_cart_ar(1, theta_B2, phi_B2).T


#load results for C
data = np.load('robust_minimax_results_C.npy')

# Extract parameters
area_C= data[:, 5]
theta_C = data[:, 3]
phi_C   = data[:, 4]

# Receiver normals
nR_C = ut.spher_to_cart_ar(1, theta_C, phi_C).T

start1 = np.array([0.2, 0.1, 1])
end1   = np.array([7.8, 0.1, 1])
line1 = np.linspace(start1, end1, 7)

# Second line (z = 3)
start2 = np.array([0.2, 0.1, 3])
end2   = np.array([7.8, 0.1, 3])
line2 = np.linspace(start2, end2, 7)

# Combine into one array
pos_opt_C  = np.vstack((line1, line2))

diag_A = np.round(diagonal_points(0, 10, 0, 10, 0,21),2) #positions of sensors

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
    'r_sensor' : diag_A,
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 2,
    'amb_L2' : 1,
    'nR_sensor' : nR_A2,
    'nS_sensor' : align_receiver_to_transmitter(diag_A, m_pos_A),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : area_A2,
    'pv': True,
    'r_lights': led_pos_A,  
    'm_lights': 1,  
    'PT_lights': 4
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
    'r_sensor' : diag_A,
    'm_master' : 1,
    'r_master' : np.array([5, 5, 3]),
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' : nR_A1,
    'nS_sensor' : align_receiver_to_transmitter(diag_A, m_pos_A),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : area_A1,
    'pv': True,
    'r_lights': led_pos_A,    
    'm_lights': 1, 
    'PT_lights': 4
    },
    
    'B1' :  {
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
    'r_sensor' : pos_opt_B,
    'm_master' : 1,
    'r_master' : m_pos_B,
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' :np.round( align_receiver_to_transmitter(pos_opt_B, m_pos_B),2),
    'nS_sensor' : align_receiver_to_transmitter(pos_opt_B, m_pos_B),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : areas_b1_def,
    'pv': True,
    'r_lights': led_pos_B,    
    'm_lights': 3, 
    'PT_lights': led_pow(led_pos_B,m_pos_B,4)
    },
    
    'B2' :  {
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
    'r_sensor' : pos_opt_B,
    'm_master' : 1,
    'r_master' : m_pos_B,
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 1,
    'amb_L2' : 1,
    'nR_sensor' : nR_B2,
    'nS_sensor' : align_receiver_to_transmitter(pos_opt_B, m_pos_B),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 4,
    'A_master' : 1e-4,
    'A_sensor' : area_B2,
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
    'r_sensor' : pos_opt_C,
    'm_master' : 1,
    'r_master' : m_pos_C,
    'FOV_master' : np.pi / 2.0,
    'FOV_sensor' : np.pi / 2.0,
    'amb_L1' : 0,
    'amb_L2' : 0,
    'nR_sensor' : nR_C,
    'nS_sensor' : align_receiver_to_transmitter(pos_opt_C, m_pos_C),
    'nR_master' : -constants.ez,
    'nS_master' : -constants.ez,
    'no_bounces' : 4,
    'Rb_master' : 10e3,
    'Rb_sensor' : 10e3,  
    'PT_sensor' : 25e-3,
    'PT_master' : 5,
    'A_master' : 1e-4,
    'A_sensor' : area_C,
    'pv': True,
    'r_lights': led_pos_C,    
    'm_lights': 1, 
    'PT_lights': led_pow(led_pos_C,m_pos_C,5)
    },
}
