
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:33:37 2024

@author: npas8772
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns

cmap1=sns.color_palette("vlag", as_cmap=True).reversed()
cmap2=sns.color_palette("blend:#004020,#FFC", as_cmap=True).reversed()
plt.rcParams.update({'font.size': 6}) #all plots


#constants
c=299792458
eps0=8.8541878176e12
mu0=np.pi*4e-7
hbar=1.054571817e-34# js
Na=6.023e23
#####################################

#spectral range
wlmin=300e-9
wlmax=900e-9
wlplot1,wlplot2=0.510,0.650

wlticks=[0.51,0.55,0.59,0.63]
wlmticks=[0.53,0.57,0.61,.65]

Imax=10
Iticks=[3,6,9]
Imticks=[1,2,4,5,7,8,10]



k0s=np.linspace(2*np.pi/wlmax,2*np.pi/wlmin , 1001 )
wls=2*np.pi/k0s
ws=2*np.pi*c/wls
dw=ws[1]-ws[0]


#wls=np.linspace(wlmin,wlmax , 1001 )
#dwl=wls[1]-wls[0]
#k0s=2*np.pi/wls
#ws=2*np.pi*c/wls

#Geometry
nh=1.5 
epsh=nh**2
ks=nh*k0s
path=10e-9


#rhodamine parameters
wabs=3.32e15
gabs=0.22e15
wlabs=2*np.pi*c/wabs
wems=3.16e15
gems=0.21e15
wlems=2*np.pi*c/wems
trad=0.5e-9 
molarabs=1.16e7  #1 per molar  per metre
conc=1 #moles per liter = molar


# source of pump
wpump= c/532e-9*2*np.pi
gpump=0.002*wpump

#kW/mm2 = 1e3 * 1e6 W/m2 

I0pump=np.linspace(Imax*1e9,Imax*1e12,1001)
Ipumpf= (gpump/2)**2 / ( (ws-wpump)**2 + (gpump/2)**2 )


#********************************** initialize


abscoeff= molarabs*conc # 1 per m
N0=conc*Na*1e3 #molecules per m3
sigma_abs=abscoeff/N0*2.303#
I_ab =gabs/2* gabs/2 / ( (ws-wabs)**2 + (gabs/2)**2 )
sigma_abs_spec=sigma_abs*I_ab
I_em =gems* gems/4 / ( (ws-wems)**2 + (gems/2)**2 )
sigma_ems=c**2/epsh/trad/wems**2/gems
sigma_ems_spec=sigma_ems*I_em
pump_spec=Ipumpf#*I0pump[-1]


#plot init
fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
ax2 = ax1.twinx()
ax1.plot( 1e6*wls,sigma_abs_spec*1e20 ,label='$\sigma_{abs}$',c='teal')
ax1.plot( 1e6*wls,sigma_ems_spec*1e20,label='$\sigma_{ems}$',c='goldenrod')
norma=np.max(pump_spec)
#ax2.plot( 1e6*wls,pump_spec/(Imax*1e10),c='red')
ax2.plot( 1e6*wls,pump_spec,c='tomato',label='$I_{pump}$')

ax1.arrow(0.532, 4.75, 0.025, 0, color='tomato',head_width=0.1, head_length=0.01)
		
		
# ask matplotlib for the plotted objects and their labels
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2,
		   loc='upper right')

#ax1.legend(loc='upper right',fontsize=5)

ax2.set_yscale('log')
ax1.set_ylim([0 , 5.5])
ax2.set_ylim([1e-5 , 1.2])
ax1.set_yticks([1, 3, 5])
ax1.set_yticks([2, 4], minor=True)

ax2.set_yticks([1e0, 1e-2, 1e-4])
ax2.set_yticks([1e-1, 1e-3],labels=['',''], minor=True)

#ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks([.55,.65], minor=True)
#ax1.set_yticks(Iticks, minor=False)

ax1.set_ylabel(r'$\bf{\sigma_t}$ ($10^{-20} m^2$)')
#ax1.set_ylabel(r'$\bf{\sigma}$ ($Å^{2}$)')
ax2.set_ylabel(r'$\bf{I_{pump}/I_{p}}$')
ax1.set_xlabel(r"$\mathbf{\lambda}$ ($\mu m$)")
ax1.set_xlim(.510,.650)


ax1.tick_params( direction='in', which="both")
ax2.tick_params( direction='in', which="both")


#ax1.text(1.3, -0.14, '(a)', transform=ax1.transAxes,
 #     fontsize=8, fontweight='bold', va='top', ha='right')

plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(a)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')

#transform=fig.transFigure
#transform=ax1.transAxes


plt.savefig( 'sigma+pump.eps' )





#******************************main

N2=np.zeros(len(I0pump))

Wabs_aprx=sigma_abs_spec[ np.argmin(np.abs(wpump- ws) )]/hbar/wabs
Wems_aprx=sigma_ems_spec[ np.argmin(np.abs(wpump- ws) )]/hbar/wabs
polariz= 3*eps0* (epsh-1) / (2+epsh )


def rates(Ipump):
	Ipump=np.abs(Ipump)
	
	deriv=np.gradient(Ipump,dw)
	Wabs =np.trapz(deriv/ws*sigma_abs_spec,dx=dw)/hbar
	
	deriv=np.gradient(Ipump,dw)
	Wems =np.trapz(deriv/ws*sigma_ems_spec,dx=dw)/hbar	
#spectral irradiance
#	Ispec=np.gradient(Ipump,dw)
	
	#Wabs=np.convolve(Ispec/ws/hbar, sigma_abs_spec  )
	#Wabs =np.trapz(Wabs,dx=dw)
	#Wems=np.convolve(Ispec/ws/hbar, sigma_ems_spec  )	
	#Wems =np.trapz(Wems,dx=dw)	
	
#	Wabs =np.trapz( (Ipump*sigma_abs_spec)**2, x=ws  )/hbar/wabs 
#	Wabs/=np.trapz( (Ipump*sigma_abs_spec)   , x=ws  )	
#	Wems =np.trapz( (Ipump*sigma_ems_spec)**2, x=ws  )/hbar/wems 
#	Wems/=np.trapz( (Ipump*sigma_ems_spec)   , x=ws  )	
	N2=Wabs/(Wabs+1/trad+Wems)
	return N2

def	epsilon():	
	#spectral
	int30 = np.trapz( (epsh/nh*sigma_abs_spec/ws**2), x=ws  )
	int12 = np.trapz( (epsh/nh*sigma_ems_spec/ws**2), x=ws  )
	musq30=3/2*gabs*hbar*c*eps0 #Cm
	musq12=3/2*gems*hbar*c*eps0 #Cm
	musq30*=int30
	musq12*=int12
	alpha_abs=musq30/hbar/gabs #F/m
	alpha_ems=musq12/hbar/gems #F/m
	epsLabs = alpha_abs*N0/eps0
	epsLems = alpha_ems*N0/eps0
	return epsLabs,epsLems

def chi(N2):   ############     here modified  v8


	S=(1-N2)/N2
	R=N2/(1-N2)
#	gems2=gems*(1+S)**.5
	chi = (1+R)*((wabs-ws+1j*gabs/2)*gabs/2) / ((ws-wabs)**2+(1+R)*(gabs/2)**2)
	chi2= (1+S)*((wems-ws+1j*gems/2)*gems/2) / ((ws-wems)**2+(1+S)*(gems/2)**2)
#	fac=1+S*(gems/2)**2/((ws-wems)**2+(gems/2)**2)

	epsi=epsh+chi*(1-N2)*epsLabs-chi2*epsLems*N2
	return epsi
	
	
def nearfield(epsi,Isource):	
	Epump = (2*Isource/(c*eps0*nh))**.5	
	polariz= 3*eps0*(epsi-1)/(2+epsi)	
	P=Epump*polariz
	D=eps0*Epump+P
	return	np.abs(D/epsi/eps0)
	
#main
eps=np.zeros((len(ws),len(I0pump)),dtype='complex')
Egain=np.zeros((len(ws),len(I0pump)),dtype='complex')
Qabs=np.zeros((len(ws),len(I0pump)))
N2=np.zeros(len(I0pump))
epsLabs,epsLems=epsilon()

for i, I0pumpi in enumerate(I0pump):
	Isource=I0pumpi*Ipumpf		
	N2i=rates(Isource)	
	epsi=chi(N2i)	
	Egaini=nearfield(epsi, Isource)	
	
	if True:
	#start interating
		converg=1.5
		counter=0
		while converg>1e-6:
			
			Igain=c*eps0/2*nh*Egaini**2 ## is it nh real?
			old=N2i
			N2i=rates(Igain)	
			epsi=chi(N2i)		
			Egaini=nearfield(epsi, Isource)		
			converg= np.abs(1-(N2i/ old)**2)
			
			if False:
				fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
				ax1.plot( 1e6*wls,Igain*1e-12,c='blue',label='$I_{pump}$')
				ax1.plot( 1e6*wls,Isource*1e-12,'--',c='red',label='$I_{pump}$')
				ax1.set_ylim([0,np.max(Igain*1e-12)])
				ax1.set_ylabel('$I_{p}$ ($MW/mm^2 $)')
				ax1.set_xlabel("$\lambda$ ($\mu m$)")
				ax1.set_xlim(wlplot1,wlplot2)
				ax1.tick_params( direction='in', which="both")
				plt.tight_layout()
				plt.show()
				pause

			counter+=1

	abs_coef= 4*np.pi*np.imag( epsi**0.5 )/wls	#per meter
	N2[i]=N2i
	eps[:,i]=epsi
	Egain[:,i]=Egaini/(2* Isource / (c*eps0*nh))**.5
	Qabs[:,i]=Isource-Isource *np.exp(-abs_coef*path)


#plots
##########


fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-12)
Z=np.imag(eps**.5).T
maxi=np.max(np.abs(Z))

print( maxi,  np.min(Z))

norm = colors.SymLogNorm(1e-1, vmin=-maxi, vmax=maxi)
norm = colors.Normalize( vmin=-maxi, vmax=maxi)
c = ax1.pcolormesh(X,Y , Z, cmap=cmap1,  norm=norm, rasterized=True)
cbar=fig.colorbar(c, ax=ax1, orientation='horizontal', location='top')
cbar.ax.tick_params(direction='in', which="both")

#cbar.ax.yaxis.set_ticks([0,0.04,-0.04,0.08,-0.08], minor=False)
#cbar.ax.yaxis.set_ticks([0.02,-0.02,0.06,-0.06], minor=True)
#cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='y', direction='in', which="both")

ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
#ax1.set_yticks([2,4,6,8])
cbar.set_label(r'$\bf{Im\{n\}}$',labelpad=5)	
ax1.set_ylabel(r'I0pump')
ax1.set_ylabel(r'$\bf{ I_{p}}$ ($MW/mm^2 $)',labelpad=0)
ax1.set_xlabel(r"$\mathbf{\lambda}$ ($\mu m$)")
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which="both")
ax2.tick_params( direction='in', which="both")



plt.tight_layout(pad=0.,w_pad=0.0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(c)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')


plt.savefig( 'Im(n).eps' )


fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-12)
Z=np.imag(eps**.5).T
maxi=np.max(np.abs(Z))

print( maxi,  np.min(Z))

xxx=np.quantile(np.abs(Z) , 0.75)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )

norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)
c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm, rasterized=True)
#ax1.set_yticks([2,4,6,8])
tks=[linthresh,-linthresh,1e1*linthresh,-1e1*linthresh]
cbar=fig.colorbar(c, ax=ax1, ticks=tks , orientation='horizontal', location='top')

cbar.ax.tick_params(direction='in', which="both")
cbar.ax.minorticks_on()

cbar.ax.tick_params(axis='y', direction='in', which="both")

ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
#ax1.set_yticks([2,4,6,8])
cbar.set_label(r'$\bf{Im\{n\}}$',labelpad=5)	
ax1.set_ylabel(r'I0pump')
ax1.set_ylabel(r'$\bf{ I_{p}}$ ($MW/mm^2 $)',labelpad=0)
ax1.set_xlabel(r"$\mathbf{\lambda}$ ($\mu m$)")
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which="both")
ax2.tick_params( direction='in', which="both")



plt.tight_layout(pad=0.0,w_pad=0.0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(c)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')


plt.savefig( 'Im(n) -log.eps' )



##########
fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-12)
Z=np.abs(Egain**.5).T**2

maxi=np.quantile(np.abs(Z), 1)
mini=0#np.quantile(np.abs(Z), 0.03)
norm = colors.LogNorm(vmin=mini,vmax=maxi)
c = ax1.pcolormesh(X,Y , Z, cmap=cmap2, rasterized=True)
cbar=fig.colorbar(c, ax=ax1, orientation='horizontal', location='top')#,ticks=[0.82,0.83,0.84,0.85])
cbar.ax.tick_params(direction='in', which="both")
cbar.ax.xaxis.set_ticks([0.80,0.83,0.86], minor=False)
#cbar.ax.yaxis.set_ticks([0.81,0.83,0.85,0.87], minor=True)

cbar.ax.tick_params(axis='y', direction='in', which="both")
#cbar.ax.minorticks_on()
#cbar.ax.yaxis.set_ticks(ticks=[0.75,.85,.95], minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
#ax1.set_yticks([2,4,6,8])
cbar.set_label(r'$\bf{ |E/E_0|}$',labelpad=5)
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($MW/mm^2 $)',labelpad=0)
ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)')

ax1.set_xlim(wlplot1,wlplot2)

if True: #inset n2
	# These are in unitless percentages of the figure size. (0,0 is bottom left)
	left, bottom, width, height = [0.62, 0.53, 0.3, 0.2]
	ax2 = fig.add_axes([left, bottom, width, height])
	plt.scatter( I0pump*1e-12,N2, c='tomato',s=0.01)
	ax2.set_xlabel('I0pump')
	ax2.set_xticks([3,6,9])	
	ax2.set_yticks([0,1])	
	ax2.set_yticks([0,0.2,0.4,0.6,0.8], minor=True)		
	ax2.set_xlabel(r'$\bf{  I_{p}}$', fontsize=5,  labelpad=0)
	ax2.set_ylabel(r"$\bf{ N'_2}$", fontsize=5, labelpad=0)		
	ax2.set_ylim([0,1])		
	ax2.set_xlim([0,Imax])	
	#ax2.minorticks_on()	
	ax2.tick_params( direction='in', which="both")	
	ax2.tick_params(axis="both", labelsize=5)	
	
	ax2.xaxis.set_tick_params(labelsize=5)
	ax2.yaxis.set_tick_params(labelsize=5)	
	ax2.patch.set_alpha(0.5)

ax1.tick_params( direction='in', which="both")


plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(b)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')

plt.savefig( 'E2.eps' )






##########
fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-12)
Z=Qabs.T*1e-12
xxx=np.quantile(np.abs(Z) , 0.75)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )

norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)
c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm, rasterized=True)
#ax1.set_yticks([2,4,6,8])
tks=[linthresh,-linthresh,1e3*linthresh,-1e3*linthresh]
cbar=fig.colorbar(c, ax=ax1, ticks=tks , orientation='horizontal', location='top')
cbar.ax.tick_params(direction='in', which="both")
cbar.ax.minorticks_on()

cbar.ax.tick_params(axis='y', direction='in', which="both")


cbar.set_label(r'$\bf{ I_{ext}}$ ($MW/mm^2 $)',labelpad=5)
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($MW/mm^2 $)',labelpad=0)
ax1.set_xlabel(r'$\mathbf{ \lambda}$ ($\mu m$)')
#ax1.set_yticks([2,4,6,8])
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which="both")
ax2.tick_params( direction='in', which="both")

ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)



plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(d)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')


plt.savefig( 'far field.eps' )

index=np.argmin(np.abs( wls-600e-9  ))
print('index',index)
fac=Ipumpf[index]


if False:
	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	
	ax1.plot(I0pump*1e-12,-Z[:,index],label=r'$\bf{-I_{ext}}$')
	#ax1.plot(I0pump*1e-12,I0pump*1e-12*fac, label=r'$\bf{  I_{p}}$')
	ax1.set_xlabel(r'$\bf{  I_{p}}$ ($MW/mm^2 $)',labelpad=0)
	ax1.set_ylabel(r'$\bf{ I}$ ($MW/mm^2 $)',labelpad=0)
	plt.legend()
	ax1.tick_params( direction='in', which="both")
	plt.tight_layout(pad=0,w_pad=0 ,h_pad=0)
	plt.savefig( 'ext532.eps' )
	
	
	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	ax1.plot(I0pump*1e-12,-Z[:,index]/(I0pump*1e-12))
	ax1.set_xlabel(r'$\bf{  I_{p}}$ ($MW/mm^2 $)',labelpad=0)
	ax1.set_ylabel('efficiency',labelpad=0)
	#ax1.set_ylim([0,0.3])
	ax1.tick_params( direction='in', which="both")
	plt.tight_layout(pad=0,w_pad=0 ,h_pad=0)
	plt.savefig( 'ext532-relative.eps' )
	


