#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from libow8 import sensor_net
import matplotlib.pyplot as plt
import owutils as ut
from panel_ow import Panel
from designs_diag import designs
from scipy.interpolate import griddata



N_CPUS = 4
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size" : 18,
    "lines.linewidth" : 2,
})

#run for communications
params_d = designs['A']

#to calculate optical power from ambient LEDs
params_amb = designs['A'].copy()
params_amb['r_master'] = params_d['r_lights']
params_amb['PT_master'] = params_d['PT_lights']
l_amb = sensor_net(**params_amb) 
l_amb.calch()
l_amb.light_sim()


#to calculate optical power from master led and uplink  
x = sensor_net(**params_d) 
x.calch()
x.light_sim()
x.calc_noise() #useless
x.calc_rq() 

p_ac_dif = np.sum(np.sum(x.Pin_sm_diff,axis = 0),axis = 1)

#connect
p_all = np.sum(np.sum(l_amb.Pin_sm_diff,axis = 0),axis = 1) + np.sum(l_amb.Pin_sm,axis = 0) + l_amb.Pin_sa #LOS + Diffuse + Ambient
p_no_sun = np.sum(np.sum(l_amb.Pin_sm_diff,axis = 0),axis = 1) + np.sum(l_amb.Pin_sm,axis = 0)
p_los = np.sum(l_amb.Pin_sm,axis = 0)
p_diff = np.sum(np.sum(l_amb.Pin_sm_diff,axis = 0),axis = 1)
p_amb = l_amb.Pin_sa
G_all = p_all/l_amb.A_sensor
G_los = p_los/l_amb.A_sensor
G_ac = x.Pin_sm_tot/l_amb.A_sensor
G_ac = G_ac.flatten()
G_no_sun = p_no_sun/l_amb.A_sensor
G_ac_diff = p_ac_dif/l_amb.A_sensor
panels = []
pmax = np.zeros(G_los.shape)
ind = np.zeros(G_los.shape, dtype = int)
bw = np.zeros(G_los.shape)
snr = np.zeros(G_los.shape)
signal = np.zeros(G_los.shape)
noise = np.zeros(G_los.shape)
snr_dB = np.zeros(G_los.shape)
v = np.zeros(G_los.shape)
C = np.zeros(G_los.shape)
req = np.zeros(G_los.shape)
rx = np.zeros(G_los.shape)
shot = np.zeros(G_los.shape)

sigma_rc = np.zeros(G_los.shape)
sigma_rs = np.zeros(G_los.shape)
sigma_rsh = np.zeros(G_los.shape)
sigma_r = np.zeros(G_los.shape)

los = x.Pin_sm.flatten()
g_los = los/l_amb.A_sensor
g_los[g_los ==0] = 1e-15

freq = np.linspace(100,30e3,400)

for i in range(p_all.shape[0]):
    #print(G_all[i])
    panels.append(Panel(l_amb.A_sensor[i]*1e4, rs=1, rsh=1000, n=1.6, voc=0.64, isc=35e-3, G=G_all[i], Gac = G_ac[i]))
    panels[i].run(False)
    pmax[i] = panels[i].Pmax
    ind[i] = int(panels[i].ind)    
    panels[i].calc_capacitance()
    panels[i].set_circuit(Rc = 10, Lo = 5, Co = 220e-6)
    panels[i].find_bw()     
    bw[i] = panels[i].BW[int(ind[i])]
    freq = np.linspace(100,np.floor(bw[i]),400)
    panels[i].tf(freq)
    panels[i].thermal_noise()
    panels[i].all_thermal_noise(freq)
    panels[i].shot_noise(freq)
    panels[i].vp2p(freq)
    signal[i] = panels[i].vac[int(ind[i])]
    noise[i] = 4 *(panels[i].th_noise[int(ind[i])] + panels[i].sh_noise[int(ind[i])])
    shot[i] = panels[i].sh_noise[int(ind[i])]
    sigma_rc[i] = panels[i].int_rc[ind[i]]
    sigma_rs[i] = panels[i].int_rs[ind[i]]
    sigma_rsh[i] = panels[i].int_rsh[ind[i]]
    sigma_r[i] = panels[i].int_r[ind[i]]
    snr[i] = signal[i]**2/noise[i]
    snr_dB[i] = 10*np.log10(snr[i])
    v[i] = panels[i].V[int(ind[i])]
    C[i] = panels[i].C[int(ind[i])]
    req[i] = panels[i].req[int(ind[i])]
    rx = panels[i].rx

sigma = sigma_rc + sigma_rs + sigma_rsh + sigma_r 

x.calc_tbattery(br = np.floor(bw)*0.4)


e_day = pmax*3600*8*0.8
V = 3.3 #Volt
cycle_p = np.array([c.calc_cycle_consumption() for c in x.cycles]) * V
day_s = 3600*24
cycles_day = day_s/x.Tcycle
e_day_c = cycles_day * cycle_p
    
    
