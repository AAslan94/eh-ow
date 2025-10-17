#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from libow8 import sensor_net
import matplotlib.pyplot as plt
import owutils as ut
from nrg_harvesting import EH
from panel_ow import Panel
from designs import designs
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
params_d = designs['B']

#to calculate optical power from ambient LEDs
params_amb = designs['B'].copy()
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


freq = np.linspace(100,30e3,400)

for i in range(p_all.shape[0]):
    #print(G_all[i])
    panels.append(Panel(l_amb.A_sensor*1e4, rs=1, rsh=1000, n=1.6, voc=0.64, isc=35e-3, G=G_all[i], Gac = G_ac[i]))
    panels[i].run(False)
    pmax[i] = panels[i].Pmax
    ind[i] = int(panels[i].ind)    
    panels[i].calc_capacitance()
    panels[i].set_circuit(Rc = 10, Lo = 5, Co = 220e-6)
    panels[i].find_bw()    
    bw[i] = panels[i].BW[int(ind[i])]
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
    
    
#plot pt min    
sensor_coords = x.r_sensor[:, :2] 
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, x.PT_rq_s_tot.flatten()*1000, (grid_x, grid_y), method='cubic')

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='viridis',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$P_\mathrm{T}$ [mW]')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
plt.show()


ui = -45 #to find the solar panel in the middle

#plot noise vs freq for one panel 
nrc = panels[ui].n_rc[ind[ui]]
nrsh = panels[ui].n_rsh[ind[ui]]
nrs = panels[ui].n_rs[ind[ui]]
nr = panels[ui].n_r[ind[ui]]

plt.figure(figsize=(10, 6))

# Plot thermal noise contributions
plt.plot(freq, nrc, label=r'$R_\mathrm{c}$', color='#1f77b4', linewidth=2)
plt.plot(freq, nrsh, label=r'$R_\mathrm{sh}$', color='#ff7f0e', linewidth=2)
plt.plot(freq, nrs, label=r'$R_\mathrm{s}$', color='#2ca02c', linewidth=2)
plt.plot(freq, nr, label=r'$r_\mathrm{d}$', color='#d62728', linewidth=2)

# Axes labels
plt.xlabel('Frequency [Hz]', fontsize=12)
plt.ylabel('Noise spectral density [V$^2$/Hz]', fontsize=12)

# Log-log scale
plt.xscale('log')
plt.yscale('log')

# Grid and ticks
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()
plt.tick_params(direction='in', length=6, width=1)

# Legend
plt.legend(loc='best', frameon=False)

# Layout
plt.tight_layout()
plt.savefig("noise_freq_false.pdf", format="pdf", bbox_inches="tight", dpi=300)

plt.show()

print("Integrated noise Rc" + str(panels[ui].int_rc[ind[ui]]))
print("Integrated noise Rsh " + str(panels[ui].int_rsh[ind[ui]]))
print("Integrated noise Rs" + str(panels[ui].int_rs[ind[ui]]))
print("Integrated noise r " + str(panels[ui].int_r[ind[ui]]))





#plot irradiance

sensor_coords = x.r_sensor[:, :2]  # shape (N,2)
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, G_all, (grid_x, grid_y), method='cubic')

# Plot G
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='viridis',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$G$ [W/$\mathrm{m^2}$]')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
#plt.savefig("g_sun_false.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()





#plot Gac 

sensor_coords = x.r_sensor[:, :2]  # shape (N,2)
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds (not fixed 0–10)
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, G_ac*1e3, (grid_x, grid_y), method='cubic')

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='plasma',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$G$ [mW/$\mathrm{m^2}$]')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
#plt.savefig("gac_false.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()






#plto Gnosu
sensor_coords = x.r_sensor[:, :2]  # shape (N,2)
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds (not fixed 0–10)
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, G_no_sun, (grid_x, grid_y), method='cubic')

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='viridis',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$G$ [W/$\mathrm{m^2}$]')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
#plt.savefig("g_nsun_false.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()



#plot consumed energy per day
sensor_coords = x.r_sensor[:, :2]  # shape (N,2)
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds (not fixed 0–10)
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, e_day_c, (grid_x, grid_y), method='cubic')

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='viridis',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$E_\mathrm{cons}$ ($T_\mathrm{cycle} = 10$ s) [J]')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
#plt.savefig("e_cons_t=10.Nsun_FALSE.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()



#plot harvested energy per day
sensor_coords = x.r_sensor[:, :2]  # shape (N,2)
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds (not fixed 0–10)
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, e_day, (grid_x, grid_y), method='cubic')

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='viridis',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$E_\mathrm{harv} [J]$')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
#plt.savefig("E_harv.Nsun.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()


#bw
sensor_coords = x.r_sensor[:, :2]  # shape (N,2)
x_coords, y_coords = sensor_coords[:, 0], sensor_coords[:, 1]

# Create grid based on sensor bounds (not fixed 0–10)
grid_x, grid_y = np.linspace(x_coords.min(), x_coords.max(), 200), np.linspace(y_coords.min(), y_coords.max(), 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)


snr_grid_interp = griddata(sensor_coords, bw/1e3, (grid_x, grid_y), method='cubic')

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(snr_grid_interp, origin='lower', cmap='viridis',
           extent=(x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()),
           aspect='auto')
plt.colorbar(label='$B_\mathrm{PV}$ [kHz]')
plt.xlabel('$L$ [m]')
plt.ylabel('$W$ [m]')
plt.tight_layout()
#plt.savefig("bw_nsun_up.pdf", format="pdf", bbox_inches="tight", dpi=300)
plt.show()