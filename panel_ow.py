#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alx
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

class Panel:
    def __init__(self, A=10, rs=1, rsh=1000, n=2, voc=0.64, isc=35e-3, G=1000, Gac = 1): #A in cm^2
        # Constants
        self.k = 1.380649e-23      # Boltzmann constant (J/K)
        self.q = 1.602176634e-19   # Electron charge (C)
        self.T = 298.15            # Temperature (K) ~ 25 °C
        Gstc = 1000           # Standard test irradiance
        self.I = None
        self.V = None
        self.i = None
        self.v = None
        # Inputs
        self.A, self.rs, self.rsh, self.n, self.Voc, self.isc, self.G= A*1e-4, rs, rsh, n, voc, isc, G
        self.Vt = self.k * self.T / self.q
        self.P = None
        self.Pmax = None
        self.Gm = self.G + Gac
        
        # Scaled parameters for panel area
        self.Rs = rs / A 
        self.Rsh = rsh / A
        self.Isc = isc * A
        self.Iph = self.Isc

        # Diode saturation current
        self.I0 = (self.Iph - self.Voc / self.Rsh) / (np.exp(self.Voc / (self.n * self.Vt)) - 1)

        # Irradiance correction
        if G != Gstc:
            self.iac = self.Iph * Gac/Gstc
            self.Iph *= self.G / Gstc
            
            self.Isc *= self.G / Gstc
            self.Voc += self.Vt * np.log(G / Gstc)
            self.Rsh *= Gstc / self.G
            #self.Rs  *= Gstc / self.G
            self.I0 = (self.Isc - self.Voc / self.Rsh) / (np.exp(self.Voc / (self.n * self.Vt)) - 1)  
        
        
        
    

    def pv_current(self, V):
        func = lambda I: self.Iph \
            - self.I0 * (np.exp((V + I * self.Rs) / (self.n * self.Vt)) - 1) \
            - (V + I * self.Rs) / self.Rsh \
            - I
        return fsolve(func, self.Iph)[0]
    
    def iv_curv(self):
        self.V = np.linspace(0, self.Voc, 200)
        self.I = np.array([self.pv_current(v) for v in self.V])
        self.I[self.I < 0] = 1e-20
        self.P = self.I * self.V
        self.ind = np.argmax(self.P)
        self.Pmax = self.P[self.ind]
        self.ID = self.I0 * (np.exp((self.V + self.I * self.Rs) / (self.n * self.Vt)))
        self.r = (self.n * self.Vt) / self.ID 
        
   
    def calc_load(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            self.Rl = np.where(self.I != 0, self.V / self.I, 1e10)

    def plot_IV(self):        
        plt.figure(figsize=(7,5))
        plt.plot(self.V, self.I, label="I-V Curve")
        plt.xlabel("Voltage (V)")
        plt.ylabel("Current (A)")
        plt.title("I-V Curve")
        plt.grid(True)
        plt.legend()
        plt.show()

    def run(self, a = True):
        self.iv_curv()
        self.calc_load()
        if a: self.plot_IV()


    def calc_capacitance(self):
        Na = 1e16 * 1e6
        Nd = 1e19 *1e6
        L = 300e-6
        er = 11.68
        eo = 8.854e-12
        es = er*eo
        T = 298.15
        ni = 1e10 * 1e6
        q = 1.6e-19
        ktq = 0.0259
        no = ni**2/Na #cm-3
        vbi = ktq * np.log(Na*Nd/ni**2)
        num = q * es * Na * Nd
        denom = 2 * (Na + Nd) * (vbi - self.V)
        c_dep = self.A * np.sqrt(num/ denom)
        c_dif = self.A * q * L * no * np.exp(self.V/ktq) /ktq
        self.C =  c_dep + c_dif

    def set_circuit(self,Rc,Lo,Co):
        self.Rc = Rc
        self.Lo = Lo
        self.Co = Co

    def find_bw(self,a = False):
        r_eq = 1/(1/self.Rsh + 1/self.r)
        r_x = self.Rs + self.Rc
        r_eq = 1/(1/r_eq + 1/r_x)
        self.BW = 1/(2*np.pi*r_eq*self.C) 
        self.req = r_eq
        self.rx = r_x
        if a: print("BW is " + str(self.BW))
    
    
    
    def thermal_noise(self):
        self.No_r = 4*self.k*self.T*self.r
        self.No_Rs = 4*self.k*self.T*self.Rs
        self.No_Rsh = 4*self.k*self.T*self.Rsh
        self.No_Rl = 4*self.k*self.T*self.Rl
        self.No_Rc = 4*self.k*self.T*self.Rc

    
    def tf(self, f):
        w = 2 * np.pi * f[None, :]    # shape (1, M)
        r  = self.r[:, None]          # shape (N, 1)
        C  = self.C[:, None]
        Rl = self.Rl[:, None]
        Zp   = 1 / (1/self.Rsh + 1/r + 1j*w*C)
        Zdc  = 1j*w*self.Lo + Rl
        Zac  = self.Rc + 1/(1j*w*self.Co)
        Zout = 1 / (1/Zac + 1/Zdc) + self.Rs
        h1 = Zp / (Zp + Zout)
        h2 = Zdc / (Zac + Zdc)
        self.hpv = np.abs(h1 * h2 * self.Rc)   # shape (N, M)

    def noise_psd_rc(self,f):
        w = 2*np.pi*f
        r = self.r[:,None]
        C = self.C[:,None]      
        Rl = self.Rl[:,None]
        J_p = 1/r + 1j*w*C + 1/self.Rsh
        Z_p = 1/J_p
        Z_source = self.Rs + Z_p 
        Z_EH = Rl + 1j*w*self.Lo 
        Z_sp = 1/ (1/Z_source + 1/Z_EH)
        Z_C0 = 1/(1j*w*self.Co)
        Z_Comm = Z_C0 + self.Rc
        den = Z_Comm + Z_sp 
        self.n_rc = np.abs(self.Rc/den)**2 * self.No_Rc

    def noise_psd_rsh(self,f):
        w = 2*np.pi*f
        r = self.r[:,None]
        C = self.C[:,None]
        Rl = self.Rl[:,None]
        h1 = 1/(self.Rc + (1/(1j*w*self.Co)))
        h2 = 1/(Rl + 1j*w*self.Lo)
        r1sh = 1/(h1+h2)
        r2sh = 1/(1/(self.Rs + r1sh) + 1/r + 1j*w*C )
        u1 = r2sh/(self.Rsh + r2sh)
        u2 = r1sh/(r1sh + self.Rs)
        u3 = self.Rc/(self.Rc + 1/(1j*w*self.Co))
        self.n_rsh = np.abs(u1*u2*u3)**2 * self.No_Rsh

    def noise_psd_r(self,f):
        r = self.r[:,None]
        C = self.C[:,None]
        Rl = self.Rl[:,None]
        nor = self.No_r[:,None]
        w = 2*np.pi*f
        h1 = 1/(self.Rc + 1/(1j*w*self.Co)) 
        h2 = 1/(Rl + 1j*w*self.Lo)
        r1r = 1/(h1+h2)
        r2r = 1/(1/(self.Rs + r1r) + 1/(self.Rsh) + 1j*w*C)
        u1 = r2r/(r + r2r)
        u2 = r1r/(r1r + self.Rs)
        u3 = self.Rc/(self.Rc + 1/(1j*w*self.Co))
        self.n_r = np.abs(u1*u2*u3)**2* nor

    def noise_psd_rl(self,f):
        w = 2*np.pi*f
        r = self.r[:,None]
        C = self.C[:,None]
        Rl = self.Rl[:,None]
        norl = self.No_Rl[:,None]
        r1l = 1/(1/r + 1j*w*C + 1/self.Rsh) + self.Rs 
        r2l = self.Rc + 1/(1j*w*self.Co)
        r3l = 1/(1/r1l + 1/r2l)
        u1 = self.Rc / (self.Rc + 1/(1j*w*self.Co))
        u2 = r3l/(Rl + r3l + 1j*w*self.Lo)
        self.n_rl = np.abs(u1*u2)**2 * norl

    def noise_psd_rs(self,f):
        w = 2*np.pi*f
        r = self.r[:,None]
        C = self.C[:,None]     
        Rl = self.Rl[:,None]
        r1s = 1/(1/r + 1j*w*C + 1/self.Rsh)
        Z_Comm = self.Rc + 1/(1j*w*self.Co) # self.Co is C0
        Z_EH = 1j*w*self.Lo + Rl
        r2s = 1/(1/Z_EH + 1/Z_Comm) 
        u1 = r2s / (self.Rs + r1s + r2s)
        u2 = self.Rc / Z_Comm
        self.n_rs = np.abs(u1*u2)**2 * self.No_Rs

    def all_thermal_noise(self,f):
        self.thermal_noise()
        self.noise_psd_rc(f)
        self.noise_psd_rs(f)
        self.noise_psd_rsh(f)
        self.noise_psd_rl(f)
        self.noise_psd_r(f)
        self.int_rc = np.trapz(self.n_rc, f, axis=1)
        self.int_rs = np.trapz(self.n_rs, f, axis=1)
        self.int_rsh = np.trapz(self.n_rsh, f, axis=1)
        self.int_rl = np.trapz(self.n_rl, f, axis=1)
        self.int_r = np.trapz(self.n_r, f, axis=1)
        self.th_noise = self.int_rc + self.int_rs + self.int_rsh + self.int_rl + self.int_r

    def shot_noise(self,f):
        t = np.abs(self.hpv)**2
        self.sh_noise = 2*self.q * self.Iph * np.trapz(t,f,axis = 1)

    def vp2p(self,f):
        self.vac = np.max(self.hpv,axis =1) * self.iac
