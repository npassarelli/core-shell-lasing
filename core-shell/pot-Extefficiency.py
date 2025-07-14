# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:33:37 2024

saturable gain + Purcell

@author: npas8772
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams.update({'font.size': 6}) #all plots



c=299792458
eps0=8.8541878176e12
mu0=np.pi*4e-7
hbar=1.054571817e-34# js
Na=6.023e23
gammadrude=0.06*241.8e12*2*np.pi	


r0=3.6e-9*5
r1=r0+2.1e-9*5


gfrom='Gxx.txt'

wltarget=600e-9

#####################################

wlmin=300e-9
wlmax=900e-9
wlplot1,wlplot2=0.510,0.620
kmin,kmax=2*np.pi/wlmax, 2*np.pi/wlmin

k0s=np.linspace(2*np.pi/wlmax,2*np.pi/wlmin , 3001 )
wls=2*np.pi/k0s

ws=2*np.pi*c/wls
wtgt=2*np.pi*c/wltarget

dw=ws[1]-ws[0]

nk=np.genfromtxt("Jiang.txt") #silver
nshell=np.interp(wls, nk[:,0]*1e-6, nk[:,1])
kshell=np.interp(wls, nk[:,0]*1e-6, nk[:,2])
epsAg=nshell**2-kshell**2+2j*nshell*kshell

nm=1.33
epsm=nm**2
nh=1.5 #ref index host
epsh=nh**2
ks=nh*k0s


#lattice
L=590e-9/nh/0.866
#L=2*r1
S_L=L**2*3**.5/2


#rhodamine
wabs=3.32e15
gabs=0.22e15
wlabs=2*np.pi*c/wabs

wems=3.16e15
gems=0.21e15
wlems=2*np.pi*c/wems
trad=0.5e-9 
Qeff=0.95

molarabs=1.16e7  #1 per molar  per metre
conc=0.5  #moles per liter = molar


#pump
wpump= c/532e-9*2*np.pi
gpump=0.002*wpump


Imax=4.75
Imin=5.25

Imax=14
Imin=Imax/1000


#unst_min=19
#unst_max=20
#I1=np.linspace(Imin*1e12,unst_min*1e12,30)
#I2=np.linspace(unst_min*1e12,unst_max*1e12,31)
#I3=np.linspace(unst_max*1e12,Imax*1e12,30)
#I0pump=np.concatenate((I1,I2,I3))




Ipumpf= (gpump/2)**2 / ( (ws-wpump)**2 + (gpump/2)**2 )

Iticks=[1,5,9,13]
Imticks=[0,2,3,4,6,7,8,10,11,12,14]


wlticks=[0.51,0.54,0.57,0.60]
wlmticks=[0.52,0.53,0.55,.56,.58,.59,.61,.62]
#following illy reference
#20 kW/mm2*1.5ns=  30 uJ/mm2 

#0.3mJ/mm2 /1.5nm = 200 kW/mm2 

# other rhod6g laser in the same numbers Functionalized polymer nanofibers: A versatile platform for manipulating light at the nanoscale
#20 kW/mm2 --> 20 1e3 /1e-6= 2e10

Qext=np.genfromtxt('Qext.txt')
I0pump=np.genfromtxt('I0pump.txt')


fact=Ipumpf[ np.argmin( np.abs( wtgt - ws ) ) ]

fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.5/2, 3.5/1.618/2), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-12)
Z=Qext.T*1e-12

print('min', np.min(Z))
print('max', np.max(Z))

mask=wlplot1*1e-6<wls
mask2=wlplot2*1e-6>wls
maxi=np.max(Z)
norm = colors.LogNorm()

maxi=np.quantile(np.abs(Z), 1)
xxx=np.quantile(np.abs(Z) , 0.75)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )

norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)


ax1.text(1.3, -0.12, '(f)', transform=ax1.transAxes,
      fontsize=8, fontweight='bold', va='top', ha='right')

c = ax1.pcolormesh(X,Y , Z, cmap="seismic_r", norm=norm )#)seismic_r #PuBu

tks=[linthresh*1000,-linthresh*1000,linthresh,-linthresh]

cbar=fig.colorbar(c, ax=ax1, ticks=tks )
cbar.ax.minorticks_on()

#tks=[ '$10^{-1}$','$-10^{-1}$', '$10^{-3}$','$-10^{-3}$']
#ax1.set_yticks([.2,.6,1])

#cbar.ax.set_yticklabels(tks)
cbar.ax.tick_params(axis='y', direction='in', which ='both')

q2r=lambda x: x
r2q=lambda x: x
#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
#secax_x.set_ticks([0.532,  wlems*1e6 ])
labels=[ "$\lambda_{pump}$" ,  "$\lambda_{ems}$" ]
#secax_x.set_xticklabels(labels)

cbar.set_label(r'$\bf{ I_{ext}}$ ($W/\mu m^2 $)',labelpad=-1.1)
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($W/\mu m^2 $)',labelpad=-1.7)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)',labelpad=-1.7)
#ax1.minorticks_on()
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which='both')
#plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)
plt.axvline(wltarget*1e6, ls='--', c='chocolate', lw=0.5)
plt.tight_layout(pad=0,w_pad=0 ,h_pad=0)
plt.savefig( 'far field.png' )


index=np.argmin(np.abs( wls-wltarget  ))
print('index',index)

maskk=I0pump*1e-12 > 10
coefs=np.polyfit(I0pump[maskk]*1e-12, -Z[maskk,index], 1)
fit=np.polyval(coefs, I0pump*1e-12)  # 3 * 5**2 + 0 * 5**1 + 1		
		

fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.5/2, 3.5/1.618/2), dpi=200)

ax1.plot(I0pump*1e-12,I0pump*1e-12*fact,label=r'$\bf{I_{source}}$')

ax1.plot(I0pump*1e-12,-Z[:,index],label=r'$\bf{-I_{ext}}$')
ax1.plot(I0pump*1e-12,fit,'--')

#ax1.plot(I0pump*1e-12,I0pump*1e-12, label=r'$\bf{  I_{p}}$')
ax1.set_xlabel(r'$\bf{  I_{p}}$ ($W/\mu m^2 $)',labelpad=0)
ax1.set_ylabel(r'$\bf{ I}$ ($W/\mu m^2 $)',labelpad=0)
plt.legend()
#ax1.set_xlim([10,14])
#ax1.set_ylim([0.02,0.025])

ax1.tick_params( direction='in', which='both')
plt.tight_layout(pad=0,w_pad=0 ,h_pad=0)
plt.savefig( 'ext532.png' )


fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(3.5/2, 3.5/1.618/2), dpi=200)
ax1.plot(I0pump*1e-12,-Z[:,index]/(I0pump*1e-12) /fact)
ax1.set_xlabel(r'$\bf{  I_{p}}$ ($W/\mu m^2 $)',labelpad=0)
ax1.set_ylabel('efficiency',labelpad=0)
#ax1.set_ylim([0,0.3])
ax1.tick_params( direction='in', which='both')
plt.tight_layout(pad=0,w_pad=0 ,h_pad=0)
plt.savefig( 'ext532-relative.png' )

