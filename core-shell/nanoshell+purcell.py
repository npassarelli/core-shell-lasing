# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 14:33:37 2024

saturable gain + Purcell

@author: npas8772
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from tqdm import tqdm


cmap1=sns.color_palette("vlag", as_cmap=True).reversed()
cmap2=sns.color_palette("blend:#004020,#FFC", as_cmap=True).reversed()
plt.rcParams.update({'font.size': 6}) #all plots



c=299792458
eps0=8.8541878176e12
mu0=np.pi*4e-7
hbar=1.94571817e-34# js
Na=6.023e23
gammadrude=0.06*241.8e12*2*np.pi	


r0=4.0e-9
r1=r0+1e-9



#####################################

wlmin=400e-9
wlmax=750e-9
wlplot1,wlplot2=0.500,0.650



k0s=np.linspace(2*np.pi/wlmax,2*np.pi/wlmin , 250001 )
wls=2*np.pi/k0s
wlticks=[0.51,0.55,0.59,0.63]
wlmticks=[0.53,0.57,0.61]

ws=2*np.pi*c/wls
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
conc=1  #moles per liter = molar


#pump
wpump= c/532e-9*2*np.pi
gpump=0.002*wpump





#unst_max=.7

#I1=np.linspace(Imin*1e9,unst_min*1e9,30)
#I2=np.linspace(unst_min*1e9,unst_max*1e9,31)
#I3=np.linspace(unst_max*1e9,Imax*1e9,30)
#I0pump=np.concatenate((I1,I2,I3))




Imax=600
Imin=Imax/1000
unst_min=100
unst_max=580
Iticks=[100,300,500]
Imticks=[200,400,600]



I0pump=np.linspace(Imin*1e9,Imax*1e9,51) # W/m2 
Ipumpf= (gpump/2)**2 / ( (ws-wpump)**2 + (gpump/2)**2 )

#following illy reference

#20 kW/mm2= 2 10**(1+3+6) W/m2= 2 10(1+3+6-12) W/mm2 = 0.02 W/mm2

#20 kW/mm2*1.5ns=  30 uJ/mm2 



#0.3mJ/mm2 /1.5nm = 200 kW/mm2 

# other rhod6g laser in the same numbers Functionalized polymer nanofibers: A versatile platform for manipulating light at the nanoscale
#20 kW/mm2 --> 20 1e3 /1e-6= 2e10 W/m
#	 	 	20 1e3*1e-6 W/um2 = 0.02


#********************************** initialize
np.savetxt('wls.txt',wls)
np.savetxt('I0pump.txt',I0pump)

abscoeff= molarabs*conc # 1 per m
N0=conc*Na*1e3 #molecules per m3
sigma_abs=abscoeff/N0*2.303#


I_ab =gabs/2* gabs/2 / ( (ws-wabs)**2 + (gabs/2)**2 )
sigma_abs_spec=sigma_abs*I_ab

I_em =gems* gems/4 / ( (ws-wems)**2 + (gems/2)**2 )
sigma_ems=c**2/epsh/trad/wems**2/gems
sigma_ems_spec=sigma_ems*I_em

pump_spec=Ipumpf*I0pump[-1]



#plot init
fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
ax2 = ax1.twinx()
ax1.plot( 1e6*wls,sigma_abs_spec*1e20 ,label='$\sigma_a$')
ax1.plot( 1e6*wls,sigma_ems_spec*1e20,label='$\sigma_e$')
ax2.plot( 1e6*wls,pump_spec*1e-9,c='red',label='$I_{pump}$')
ax1.set_ylim([0 , np.max(sigma_abs_spec)*1e20])
ax2.set_ylim([0 , np.max(pump_spec)*1e-9])
ax1.set_ylabel('$\sigma$ ($Å^{2}$)')
ax2.set_ylabel('$I_{p}$ ($ kW/mm^2 $)')
ax1.set_xlabel("$\lambda$ ($\mu m$)")
ax1.set_xlim(wlplot1,wlplot2)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.tick_params( direction='in', which='both')
ax2.tick_params( direction='in', which='both')

plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
plt.savefig( 'sigma+pump.eps' )

#spectral integral
int30 = np.trapz( (epsh/nh*sigma_abs_spec/ws**2), x=ws  )
#int30/= np.trapz( (epsh/nh*sigma_abs_spec/ws)   , x=ws  )
int12 = np.trapz( (epsh/nh*sigma_ems_spec/ws**2), x=ws  )
#int12/= np.trapz( (epsh/nh*sigma_ems_spec/ws)   , x=ws  )	
musq30=3/2*gabs*hbar*c*eps0 #Cm
musq12=3/2*gems*hbar*c*eps0 #Cm	
musq30*=int30
musq12*=int12	
alpha_abs=musq30/hbar/gabs #F/m
alpha_ems=musq12/hbar/gems #F/m	
epsLabs = alpha_abs*N0/eps0
epsLems = alpha_ems*N0/eps0



def polariz_QSshelli(e0,e1,e2,r0,r1):	
	
	alpha0 =r0**3*(e0-e1)/(e0+2*e1)	
	f0=(1-alpha0/r1**3)/(1+2*alpha0/r1**3)	
	alpha1=r1**3*(e1-f0*e2 ) /(e1+2*f0*e2)

	return alpha1

	
def InnerField(e0,e1,e2,r0,r1, alpha,Epump):	
	DD=-1
	EE=alpha/4/np.pi 	
	CC=r1**3 *  DD*(e1-e2)/e1/3 + EE* (e1+2*e2)/3/e1 
	BB= DD* ((2*e1+e2)/e1/3) + EE/r1**3 * ((2*e1-2*e2)/3/e1 )
	AA= BB*(e0+e1)/e0/2 + CC/r0**3 * (e0-2*e1)/2/e0
	return AA*Epump#/(4*np.pi*eps0)
	

def modevolume(e0,e1,e2,r0,r1,alphamax):
	
	times=2
	x=np.linspace(-times*r1,times*r1,200)
	y=np.linspace(-times*r1,times*r1,200)
	z=np.linspace(-times*r1,times*r1,200)	
	Voltot=8*(times*r1)**3

	
	X,Y,Z=np.meshgrid(x,y,z, indexing='ij')
	dV=(x[1]-x[0])*(y[1]-y[0])*(z[1]-z[0])	
	R=np.sqrt(X**2+Y**2+Z**2)
	Rho=np.sqrt(X**2+Y**2)
	Theta=np.arctan2(Rho,Z)
#	Phi=np.arctan2(Y,X)
	E=np.zeros(R.shape)
	Emode=np.zeros(R.shape)	
	Endens=np.zeros(R.shape)		
	Eps=np.zeros(R.shape,dtype='complex')
	
	Ndif=R.shape[0]*R.shape[1]*R.shape[2]
	
	DD=-1
	EE=alphamax/4/np.pi 	
	CC=r1**3 * DD*(e1-e2)/e1/3 + EE* (e1+2*e2)/3/e1 
	BB= DD* ((2*e1+e2)/e1/3) + EE/r1**3 * ((2*e1-2*e2)/3/e1 )
	
	AA= BB*(e0+e1)/e0/2 + CC/r0**3 * (e0-2*e1)/2/e0
			
	Er= AA *np.cos(Theta)	
	Et=-AA *np.sin(Theta)
	Eri= -1*np.cos(Theta) #incident
	Eti= 1*np.sin(Theta)
	E2 =np.abs(Et-Eti)**2 + np.abs(Er-Eri)**2 
	E3 =np.abs(Et    )**2 + np.abs(Er    )**2 	
	mask=R<=r0   #core
	E[mask]=E3[mask]
	Emode[mask]=E2[mask]
	Eps[mask]=e0*np.ones(R.shape)[mask]
	Endens[mask]=Emode[mask]*np.real(e0) + w_vol*np.imag(e0)/gabs*2
	
	

	Er= (DD-2*EE/R**3)*np.cos(Theta)
	Et=-(DD+1*EE/R**3)*np.sin(Theta)
	Eri= -1*np.cos(Theta) #incident
	Eti= 1*np.sin(Theta)	
	E2 =np.abs(Et-Eti)**2 + np.abs(Er-Eri)**2 
	E3 =np.abs(Et    )**2 + np.abs(Er    )**2 		
	mask= R>=r1  	 #outside
	E[mask]=E3[mask]	
	Emode[mask]=E2[mask]	
	Eps[mask]=e2*np.ones(R.shape)[mask]	
	Endens[mask]=Emode[mask]*np.real(e2)

	

	Er= (BB-2*CC/R**3)*np.cos(Theta)
	Et=-(BB+1*CC/R**3)*np.sin(Theta)
	Eri= -1*np.cos(Theta) #incident
	Eti= 1*np.sin(Theta)	
	E2 =np.abs(Et-Eti)**2 + np.abs(Er-Eri)**2 
	E3 =np.abs(Et    )**2 + np.abs(Er    )**2 		
	mask= np.logical_and(R<r1 , R>r0 )
	E[mask]=E3[mask]	
	Emode[mask]=E2[mask]
	Eps[mask]=e1*np.ones(R.shape)[mask]	
	Endens[mask]=Emode[mask]*np.real(e0) + w_vol*np.imag(e1)/gammadrude*2		
		
	ef2=Emode
	ef4=(Emode)**2
	
	modevolume=np.sum(ef2.flatten())**2/np.sum(ef4.flatten())*dV
	print('mod vol', modevolume)
	

	EnDensity= Endens #Emode * (np.real(Eps) + 2* wdrude/gammadrude* np.imag(Eps))
	norm=np.max(EnDensity)
	modevolume=np.sum(EnDensity.flatten())/norm*dV
	print('mod vol2', modevolume)	
		
	
	vol=4/3*np.pi*r0**3
	print('core vol', vol)	
	
	if True: #plot NF		
		import matplotlib.patches as patches
		indx=int(len(X)/2)		
		#print(E.shape)		
		Xp= X[:,indx,:]*1e9
		Yp= Z[:,indx,:]*1e9
		Zp= np.abs(Emode[:,indx,:]) #w/o incident field
		Zp= np.abs(E[:,indx,:])		#w/  incident field
#		Xp= X[:,:,175]*1e9
#		Yp= Y[:,:,175]*1e9
#		Zp= np.abs(E[:,:,175])		
		
		#Zp= Zp/np.max(E)
		fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.25, 1.9), dpi=200)
#		norm = colors.LogNorm()
		norm = colors.Normalize()#vmin=0,vmax=1)	
		c = ax1.pcolormesh(Xp,Yp , Zp**.5, cmap=cmap2,norm=norm, rasterized=True)	
		
#		cbar=fig.colorbar(c, ax=ax1,fraction=0.046, pad=0.04)#,ticks=[2,4,6])
		cbar=fig.colorbar(c, ax=ax1, orientation='horizontal', location='top')#,fraction=0.046, pad=0.04)#,ticks=[2,4,6])
		cbar.ax.tick_params(axis='x', direction='in', which ='both')		
		ax1.add_patch(patches.Circle((0,0),radius=r0*1e9,ec='y',fill=None))
		ax1.add_patch(patches.Circle((0,0),radius=r1*1e9,ec='y',fill=None))		
		
		ax1.set_ylim(-2e9*r1,2e9*r1)
		ax1.set_xlim(-2e9*r1,2e9*r1)
		ax1.set_ylabel(r'$\bf{y}$ ($nm$)', labelpad=0)
		ax1.set_xlabel(r'$\bf{x}$ ($nm$)', labelpad=0)		
		cbar.set_label(r'$\bf{ |E/E_0|}$', labelpad=5)

		ax1.tick_params( direction='in', which='both')

		
		plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
		coords = ax1.get_position().get_points()
		ax1.text(0, 1, '(a)',transform=fig.transFigure,
			va='top', ha='left', fontsize=8, fontweight='bold')
		
		
		plt.savefig( 'E-nearfield.eps' )

		np.savetxt('x.txt',x)
		np.savetxt('z.txt',z)
		np.savetxt('E_near.txt',E[:,indx,:])		
	
	return np.abs(modevolume)


from scipy.optimize import minimize
def Qfactor(alpha):
	
	indx=np.argmax(np.abs(alpha))	
	maxi=np.max(np.abs(alpha))
	om=ws[indx]
	f= lambda Q:  maxi*((om-ws) +1j*om/Q/2 )*om/Q/2/((ws-om)**2 +(om/Q/2)**2 )
	opt= lambda Q: np.sum( np.abs( f(Q)-alpha )/maxi)
	res=minimize(opt, 50 )
	
	Qfac=res.x[0]
	Lor=f(Qfac )
	#print('Qfac',Qfac)
	
	if False:
		fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
		
		ax1.plot( ws,np.real(alpha) ,label='Re alfa')
		ax1.plot( ws,np.imag(alpha) ,label='Im alfa')		
		ax1.plot( ws,np.real(Lor),'--' ,label='Re L')
		ax1.plot( ws,np.imag(Lor),'--' ,label='Im L')
		
		#ax1.set_ylim([0 , np.max(sigma_abs_spec)*1e20])
		#ax1.set_xlim(wlplot1,wlplot2)
		ax1.set_ylabel('$\\alpha$ ($m^{3}$)')		
		ax1.set_xlabel("$\omega$ ($s^{-1}$)")
		ax1.legend(loc='upper right')		
		ax1.tick_params( direction='in', which='both')		
		plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
		plt.savefig( 'Qfact.eps' )
		
			
	return Qfac
	
	

chi = ((wabs-ws+1j*gabs/2)*gabs/2) / ((ws-wabs)**2+(gabs/2)**2)
chi2= ((wems-ws+1j*gems/2)*gems/2) / ((ws-wems)**2+(gems/2)**2)
epsinic=epsh+chi*(1)*epsLabs


Epump	= (2* pump_spec / (c*eps0*nh))**.5

xx=polariz_QSshelli(epsinic,epsAg,epsm,r0,r1)
index=np.argmax(np.abs( xx ))

w_vol=ws[index]



print('pol max', xx[index])

modvol=modevolume(epsinic[index],epsAg[index],epsm,r0,r1,xx[index])
QF=Qfactor(xx)


Purcell=1/np.pi *QF/modvol *(wls[index]/nh/2/np.pi)**3

print('Purcell',Purcell)

alphaCS=polariz_QSshelli(epsinic,epsAg,epsm,r0,r1)

Qabs=ks*np.imag( 4*np.pi*epsm* alphaCS) / np.pi/ r1**2
if False:
	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	ax2 = ax1.twinx()
	ax1.plot( 1e6*wls,Qabs,label='$Q_a$')
	ax2.plot( 1e6*wls,pump_spec*1e-9,c='red',label='$I_{pump}$')
	ax1.set_ylabel('$Qabs$')
	ax2.set_ylabel('$I_{p}$ ($ kW/mm^2 $)')
	ax1.set_xlabel("$\lambda$ ($\mu m$)")
	ax1.set_xlim(wlplot1,wlplot2)
	ax1.legend(loc='upper left')
	ax2.legend(loc='upper right')
	ax2.set_ylim([0 , np.max(pump_spec)*1e-9])
	
	ax1.tick_params( direction='in', which='both')
	ax2.tick_params( direction='in', which='both')
	plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
	plt.savefig( 'spectra_init.eps' )
	


	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	#ax2 = ax1.twinx()
	ax1.plot( 1e6*wls,np.real(alphaCS),label='Re')
	ax1.plot( 1e6*wls,np.imag(alphaCS),label='Im')
	#ax2.plot( 1e6*wls,pump_spec*1e-9,c='red',label='$I_{pump}$')
	ax1.set_ylabel('$alpha$')
	#ax2.set_ylabel('$I_{p}$ ($ MW/mm^2 $)')
	ax1.set_xlabel("$\lambda$ ($\mu m$)")
	ax1.set_xlim(wlplot1,wlplot2)
	ax1.legend(loc='upper left')
	#ax2.legend(loc='upper right')
	#ax2.set_ylim([0 , np.max(pump_spec)*1e-9])
	
	ax1.tick_params( direction='in', which='both')
	#ax2.tick_params( direction='in', which='both')
	plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
	plt.savefig( 'alpha.eps' )




#******************************main
#N2zero=np.zeros(len(I0pump))
N2=np.zeros(len(I0pump))


#approx	monochrom
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
	
#	Wabs=np.convolve(Ispec/ws/hbar, sigma_abs_spec  )
#	Wabs =np.trapz(Wabs,dx=dw)
#	Wems=np.convolve(Ispec/ws/hbar, sigma_ems_spec  )	
#	Wems =np.trapz(Wems,dx=dw)	
	
#	Wabs =np.trapz( (Ipump*sigma_abs_spec)**2, x=ws  )/hbar/wabs 
#	Wabs/=np.trapz( (Ipump*sigma_abs_spec)   , x=ws  )
	
#	Wems =np.trapz( (Ipump*sigma_ems_spec)**2, x=ws  )/hbar/wems 
#	Wems/=np.trapz( (Ipump*sigma_ems_spec)   , x=ws  )
	
	N2=Wabs/(Wabs+Purcell/trad+Purcell*Wems)

	return N2

def	epsilon():
	
	#spectral integral
	int30 = np.trapz( (epsh/nh*sigma_abs_spec/ws**2), x=ws  )
	#int30/= np.trapz( (epsh/nh*sigma_abs_spec/ws)   , x=ws  )
	int12 = np.trapz( (epsh/nh*sigma_ems_spec/ws**2), x=ws  )
	#int12/= np.trapz( (epsh/nh*sigma_ems_spec/ws)   , x=ws  )

	
	musq30=3/2*gabs*hbar*c*eps0 #Cm
	musq12=3/2*gems*hbar*c*eps0 #Cm
	
	musq30*=int30
	musq12*=int12
	
	alpha_abs=musq30/hbar/gabs #F/m
	alpha_ems=musq12/hbar/gems #F/m
	
	epsLabs = alpha_abs*N0/eps0
	epsLems = alpha_ems*N0/eps0

	return epsLabs,epsLems

def chi(N2):

	S=(1-N2)/N2
	R=N2/(1-N2)
	chi = (1+R)*((wabs-ws+1j*gabs/2)*gabs/2) / ((ws-wabs)**2+(1+R)*(gabs/2)**2)
	chi2= (1+S)*((wems-ws+1j*gems/2)*gems/2) / ((ws-wems)**2+(1+S)*(gems/2)**2)
	epsi=epsh+chi*(1-N2)*epsLabs-chi2*N2*epsLems*Purcell
	#eps[:,i]=epsi
	return epsi
	
	
def nearfield(epsi,Isource,alphai):	
	Epump	= (2* Isource / (c*eps0*nh))**.5	
	
	
#	return	polariz_bulk(epsi,Epump)
#	return	polariz_QSsphere(epsi,epsm,Epump)
	return  InnerField(epsi,epsAg,epsm,r0,r1,alphai,Epump)
	


#main

eps=np.zeros((len(ws),len(I0pump)),dtype='complex')
Egain=np.zeros((len(ws),len(I0pump)),dtype='complex')
alpha=np.zeros((len(ws),len(I0pump)),dtype='complex')

Qext=np.zeros((len(ws),len(I0pump)))
N2=np.zeros(len(I0pump))
PF=np.zeros(len(I0pump))
epsLabs,epsLems=epsilon()

N2i_old=1e-11
for i, I0pumpi in tqdm(enumerate(I0pump)):
	
	Isource=I0pumpi*Ipumpf		
#	N2i=N2i_old	#rates(Isource)
	N2i=rates(Isource)	
	
	epsi=chi(N2i)	
	
	alphai=polariz_QSshelli(epsi,epsAg,epsm,r0,r1)
	index=np.argmax(np.abs( xx ))		
	QF=Qfactor(alphai)
	Purcell=1/np.pi *QF/modvol *(wls[index]/nh/2/np.pi)**3
	
	Egaini=nearfield(epsi, Isource,alphai)	
	

	#start iterating
	converg=1.5
	counter=0
	while converg>1e-6:
		
		Igain=c*eps0/2*nh*Egaini**2 ## is it nh real?
		old=N2i
		N2i=rates(Igain)	
		epsi=chi(N2i)		
		alphai=polariz_QSshelli(epsi,epsAg,epsm,r0,r1)	
		index=np.argmax(np.abs( xx ))
		QF=Qfactor(alphai)
		Purcell=1/np.pi *QF/modvol *(wls[index]/nh/2/np.pi)**3
		Egaini=nearfield(epsi, Isource,alphai)		
		converg= np.abs(1-(N2i/ old)**2)
		
		#W/m2 -->  1e-11 kW/mm
		
		if False:
			fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
			ax1.plot( 1e6*wls,Igain*1e-9,c='blue',label='$I_{pump}$')
			ax1.plot( 1e6*wls,Isource*1e-9,'--',c='red',label='$I_{pump}$')
			ax1.set_ylim([0,np.max(Igain*1e-9)])
			ax1.set_ylabel('$I_{p}$ ($ kW/mm^2 $)')
			ax1.set_xlabel("$\lambda$ ($\mu m$)")
			ax1.set_xlim(wlplot1,wlplot2)
			ax1.tick_params( direction='in', which='both')
			plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
			plt.show()
			
		counter+=1
		if counter>20: converg=0
			
		
	print(counter)
	alphaCSi=polariz_QSshelli(epsi,epsAg,epsm,r0,r1)

	Qabsi=ks*np.imag( 4*np.pi*epsm* alphaCSi) / np.pi/ r1**2
	
	Qsca=ks**4/6/np.pi*np.abs(4*np.pi*epsm* alphaCSi)**2 / np.pi/ r1**2
	
	N2[i]=N2i
	PF[i]=Purcell
	N2i_old=N2i
	eps[:,i]=epsi		
	alpha[:,i]=alphaCSi			
	Egain[:,i]=Egaini/(2* Isource / (c*eps0*nh))**.5
	Qext[:,i]=(Qabsi+Qsca)*Isource
	

		

alpha=alpha*4*np.pi*eps0	
if False:
	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	ax1.scatter( I0pump*1e-9,N2,s=0.1)#,label='w/ saturation' )
	ax2 = ax1.twinx()
	ax2.scatter( I0pump*1e-9,PF,s=0.1,c='r')#,label='w/ saturation' )
	ax1.set_xlabel('I0pump')
	ax1.set_xlabel('$I_{p}$ ($ kW/mm^2 $)')
	ax1.set_ylabel("$N_2$")		
	ax2.set_ylabel("Purcell factor")		
	
	ax1.tick_params( direction='in', which='both')
	ax2.tick_params( direction='in', which='both')
	plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
	plt.savefig( 'N2.eps' )	
	np.savetxt('N2.txt',np.vstack(N2).T)
	



fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(2 , 1.9) , dpi=200)

#ax1.scatter( I0pump*1e-9,N2*PF, c='brown',s=0.1,marker='*')#,label='w/ saturation' )
ax1.plot( I0pump*1e-9,N2*PF, c='brown')#,label='w/ saturation' )
plt.axvline(unst_min, ls='--', c='chocolate')
#plt.axvline(unst_max, ls='--', c='chocolate', lw=0.5)
ax1.set_xlabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)',  labelpad=0)
ax1.set_ylabel(r"$\bf{ N_{2,r}}$  $\bf{P_f}$",  labelpad=0)	
ax1.set_xlim(0,Imax)	
ax1.set_xticks(Iticks)		
ax1.set_xticks(Imticks, minor=True)	
ax1.set_ylim(-0.01,1)	
ax1.set_yticks([0.5,1,1.5,])	
ax1.set_yticks([0.25,0.75,1.25,1.75], minor=True)	
		
ax1.tick_params( direction='in', which='both')

if True:
	left, bottom, width, height = [0.3, 0.65, 0.30, 0.30]	
	ax2 = fig.add_axes([left, bottom, width, height])
	#ax2.scatter( I0pump*1e-9,N2*1e7,s=0.05)
	ax2.plot( I0pump*1e-9,N2*1e5,c='b',lw=0.5)
	ax3 = ax2.twinx()
	ax3.plot( I0pump*1e-9,PF*1e-5,c='r',lw=0.5)
	#ax3.scatter( I0pump*1e-9,PF*1e-7,s=0.05,c='r')
	ax2.set_xlabel(r'$\bf{ I_p}$', fontsize=5,  labelpad=-1)
	ax2.set_ylabel(r"$\bf{ N_{2,r}}$ $\cdot 10^5$", fontsize=5,   labelpad=-1)		
	ax3.set_ylabel(r"$\bf{P_f}$ $\cdot 10^{-5}$", fontsize=5,   labelpad=0)		

	ax2.yaxis.label.set_color('blue')
	ax3.yaxis.label.set_color('red')

	ax3.tick_params(axis='both', which='major', labelsize=5)
	ax2.tick_params(axis='both', which='major', labelsize=5)
	#ax3.set_xticks([0,14])		
	#ax3.set_yticks([1,2])	
	ax3.set_ylim([-1,3])	
	ax2.set_ylim([-1,10])		
	#ax2.set_yticks([1,2])
	
	plt.axvline(unst_min, ls='--', c='chocolate')
#	plt.axvline(unst_max, ls='--', c='chocolate', lw=0.5)
	ax2.tick_params( direction='in', which='both')
	ax3.tick_params( direction='in', which='both')
	

plt.tight_layout(pad=0.0,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(b)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')



plt.savefig( 'N2xPf.eps' )	



np.savetxt('N2.txt',N2)
np.savetxt('PF.txt',PF)
#pause


#plot


fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
Z=np.imag(eps**.5).T
maxi=np.quantile(np.abs(Z), 1)
xxx=np.quantile(np.abs(Z) , 0.1)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )
maxi=10**np.ceil( np.log10(maxi) )
maxi=np.max([maxi,10* linthresh ])
norm = colors.SymLogNorm( linthresh, vmin=-maxi, vmax=maxi)
c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm, rasterized=True)
cbar=fig.colorbar(c, ax=ax1, orientation='horizontal', location='top')
cbar.ax.tick_params(axis='x', direction='in', which='both')
plt.axvline(0.532, ls='--', c='chocolate', lw=0.75)
plt.axvline(wlabs*1e6, ls='--', c='chocolate', lw=0.75)
plt.axvline(wlems*1e6, ls='--', c='chocolate', lw=0.75)

cbar.set_label(r'$\bf{Im\{n\}}$', labelpad=5)	
ax1.set_ylabel('I0pump')
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)')
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)

ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)')
#ax1.minorticks_on()
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which='both')
ax2.tick_params( direction='in', which='both')
plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)
plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)
plt.savefig( 'Im(n).eps' )

np.savetxt('eps.txt',eps)


fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
Z=(np.abs(Egain**.5)**2).T
mask=np.logical_and( X<wlplot2,  X>wlplot1)
maxi=np.max(np.abs(Z[mask]))
mini=np.min(np.abs(Z[mask]))
norm = colors.LogNorm(vmin=mini,vmax=maxi)

c = ax1.pcolormesh(X,Y , Z, cmap=cmap2,norm=norm, rasterized=True)
cbar=fig.colorbar(c, ax=ax1,ticks=[100,10,1,0.1], orientation='horizontal', location='top')
cbar.ax.minorticks_on()
cbar.ax.tick_params(axis='x', direction='in', which ='both')


q2r=lambda x: x
r2q=lambda x: x
#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
#secax_x.set_ticks([])
#labels=[ "$\lambda_{pump}$" , "$\lambda_{abs}$" , "$\lambda_{ems}$" ]
##secax_x.set_xticklabels(labels)
cbar.set_label(r'$\bf{ |E_{core}/E_0|}$', labelpad=5)
#cbar.ax.minorticks_on()
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)', labelpad=0)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)', labelpad=0)
##ax1.minorticks_on()
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which='both')
ax2.tick_params( direction='in', which='both')

if False:
	# These are in unitless percentages of the figure size. (0,0 is bottom left)
	left, bottom, width, height = [0.27, 0.57, 0.2, 0.2]
	ax2 = fig.add_axes([left, bottom, width, height])
	plt.scatter( I0pump*1e-9,N2, c='g',s=0.1)
	ax2.set_xlabel('I0pump')
	#ax2.set_xticks([0,Imax])	
	ax2.set_yticks([0,1])	
	ax2.tick_params(axis='x', which='minor', bottom=True)
	
	ax2.set_xlabel('$I_{p}$', fontsize=4,  labelpad=0)
	ax2.set_ylabel("$N_2$", fontsize=4, labelpad=0)		
	ax2.set_ylim([0,1])		
	ax2.set_xlim([Imin,Imax])		
	ax2.tick_params(axis='both', labelsize=9)	
	ax2.tick_params( direction='in', which='both')
	plt.tight_layout(pad=0.5,w_pad=0 ,h_pad=0)
	ax2.patch.set_alpha(0.5)
	
plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)	

plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(e)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')


plt.savefig( 'E2.eps' )

np.savetxt('modE.txt',np.abs(Egain**.5))


fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
Z=Qext.T*1e-9

mask=wlplot1*1e-6<wls
mask2=wlplot2*1e-6>wls
maxi=np.max(Z)
norm = colors.LogNorm()

maxi=np.quantile(np.abs(Z), 1)
xxx=np.quantile(np.abs(Z) , 0.75)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )

norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)



c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm , rasterized=True)#)seismic_r #PuBu

tks=[linthresh*100,-linthresh*100,linthresh,-linthresh]

cbar=fig.colorbar(c, ax=ax1, ticks=tks, orientation='horizontal', location='top' )
cbar.ax.minorticks_on()

#tks=[ '$10^{-1}$','$-10^{-1}$', '$10^{-3}$','$-10^{-3}$']
#ax1.set_yticks([.2,.6,1])

#cbar.ax.set_yticklabels(tks)
cbar.ax.tick_params(axis='x', direction='in', which ='both')

q2r=lambda x: x
r2q=lambda x: x
#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
#secax_x.set_ticks([0.532,  wlems*1e6 ])
labels=[ "$\lambda_{pump}$" ,  "$\lambda_{ems}$" ]
#secax_x.set_xticklabels(labels)

cbar.set_label(r'$\bf{ I_{ext}}$ ($kW/mm^2 $)',labelpad=5)
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)',labelpad=0)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)',labelpad=0)
#ax1.minorticks_on()
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which='both')
ax2.tick_params( direction='in', which='both')
plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)
plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)

coords = ax1.get_position().get_points()
ax1.text(0, 1, '(f)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')

plt.savefig( 'far field.eps' )



np.savetxt('Qext.txt',Qext)



fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
Z=np.real(alpha).T 

mask=wlplot1*1e-6<wls
mask2=wlplot2*1e-6>wls
maxi=np.max(np.abs(Z))


maxi=np.quantile(np.abs(Z), 1)
xxx=np.quantile(np.abs(Z) , 0.75)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )

norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)


c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm , rasterized=True)#)seismic_r #PuBu

tks=[linthresh*100,-linthresh*100,linthresh,-linthresh]
cbar=fig.colorbar(c, ax=ax1 , ticks=tks , orientation='horizontal', location='top')
#tks=[ '$10^{-1}$','$-10^{-1}$', '$10^{-3}$','$-10^{-3}$']
#ax1.set_yticks([.2,.6,1])

cbar.ax.tick_params(axis='x', direction='in', which ='both')
cbar.ax.minorticks_on()


q2r=lambda x: x
r2q=lambda x: x
#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
#secax_x.set_ticks([])
#labels=[ "$\lambda_{pump}$" , "$\lambda_{abs}$" , "$\lambda_{ems}$" ]
##secax_x.set_xticklabels(labels)
cbar.set_label(r'$\bf{4 \pi \varepsilon_0 Re\{ \alpha \}}$ ($F m^2$)',labelpad=5)

ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)',labelpad=0)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)',labelpad=0)
#ax1.minorticks_on()
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which='both')
ax2.tick_params( direction='in', which='both')
plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)


plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(c)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')

plt.savefig( 'Polariz-R.eps' )




fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
Z=np.imag(alpha).T

mask=wlplot1*1e-6<wls
mask2=wlplot2*1e-6>wls
maxi=np.max(np.abs(Z))

maxi=np.quantile(np.abs(Z), 1)
xxx=np.quantile(np.abs(Z) , 0.75)  
maxi=np.quantile(np.abs(Z) , 1) 
linthresh= 10**np.ceil( np.log10(xxx) )

norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)


c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm , rasterized=True)#)seismic_r #PuBu

tks=[linthresh*1000,-linthresh*1000,linthresh,-linthresh]
cbar=fig.colorbar(c, ax=ax1, ticks=tks , orientation='horizontal', location='top')
cbar.ax.minorticks_on()

#tks=[ '$10^{-1}$','$-10^{-1}$','$10^{-2}$','$-10^{-2}$']
#ax1.set_yticks([.2,.6,1])

#cbar.ax.set_yticklabels(tks)
cbar.ax.tick_params(axis='x', direction='in', which ='both')

q2r=lambda x: x
r2q=lambda x: x
#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
#secax_x.set_ticks([wlems*1e6 ])
labels=[ "$\lambda_{ems}$" ]
#secax_x.set_xticklabels(labels)

cbar.set_label(r'$\bf{4 \pi \varepsilon_0 Im\{ \alpha \}}$ ($F m^2$)',labelpad=5)
ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)',labelpad=0)
ax1.set_yticks(Iticks, minor=False)
ax1.set_yticks(Imticks, minor=True)
ax1.set_xticks(wlticks, minor=False)
ax1.set_xticks(wlmticks, minor=True)
ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)',labelpad=0)
#ax1.minorticks_on()
ax1.set_xlim(wlplot1,wlplot2)
ax1.tick_params( direction='in', which='both')
ax2.tick_params( direction='in', which='both')
plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)

plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)
coords = ax1.get_position().get_points()
ax1.text(0, 1, '(d)',transform=fig.transFigure,
	va='top', ha='left', fontsize=8, fontweight='bold')

plt.savefig( 'Polariz-I.eps' )
np.savetxt('alpha.txt',alpha.T)



if False:	
	
	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
	Z=np.imag(alpha).T*1e12
	
	mask=wlplot1*1e-6<wls
	mask2=wlplot2*1e-6>wls
	maxi=np.max(np.abs(Z))
	
	maxi=np.quantile(np.abs(Z), 1)
	xxx=np.quantile(np.abs(Z) , 0.75)  
	maxi=np.quantile(np.abs(Z) , 1) 
	linthresh= 10**np.ceil( np.log10(xxx) )
	
	norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)
	
	
	c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm , rasterized=True)#)seismic_r #PuBu
	
	tks=[linthresh*1000,-linthresh*1000,linthresh,-linthresh]
	cbar=fig.colorbar(c, ax=ax1, ticks=tks , orientation='horizontal', location='top')
	cbar.ax.minorticks_on()
	
	#tks=[ '$10^{-1}$','$-10^{-1}$','$10^{-2}$','$-10^{-2}$']
	#ax1.set_yticks([.2,.6,1])
	
	#cbar.ax.set_yticklabels(tks)
	cbar.ax.tick_params(axis='x', direction='in', which ='both')
	
	q2r=lambda x: x
	r2q=lambda x: x
	#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
	#secax_x.set_ticks([wlems*1e6 ])
	labels=[ "$\lambda_{ems}$" ]
	#secax_x.set_xticklabels(labels)
	
	cbar.set_label(r'$\bf{4 \pi \varepsilon_0 Im\{ \alpha \}}$ ($F \mu m^2$)',labelpad=0)
	ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)',labelpad=0)
	ax1.set_yticks(Iticks, minor=False)
	ax1.set_yticks(Imticks, minor=True)
	#ax1.set_xticks(wlticks, minor=False)
	#ax1.set_xticks(wlmticks, minor=True)
	ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)',labelpad=0)
	#ax1.minorticks_on()
	ax1.set_xlim(0.59,0.605)
	ax1.tick_params( direction='in', which='both')
	ax2.tick_params( direction='in', which='both')
	plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
	#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)
	
	plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)
	coords = ax1.get_position().get_points()
	ax1.text(0, 1, '(d)',transform=fig.transFigure,
		va='top', ha='left', fontsize=8, fontweight='bold')
	
	plt.savefig( 'Polariz-I-zoom.eps' )
	
	
	
	
	fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(1.625, 1.9), dpi=200)
	X,Y=np.meshgrid(wls*1e6,I0pump*1e-9)
	Z=np.real(alpha).T *1e12
	
	mask=wlplot1*1e-6<wls
	mask2=wlplot2*1e-6>wls
	maxi=np.max(np.abs(Z))
	
	
	maxi=np.quantile(np.abs(Z), 1)
	xxx=np.quantile(np.abs(Z) , 0.75)  
	maxi=np.quantile(np.abs(Z) , 1) 
	linthresh= 10**np.ceil( np.log10(xxx) )
	
	norm = colors.SymLogNorm(linthresh, vmin=-maxi,vmax=maxi)
	
	
	c = ax1.pcolormesh(X,Y , Z, cmap=cmap1, norm=norm , rasterized=True)#)seismic_r #PuBu
	
	tks=[linthresh*100,-linthresh*100,linthresh,-linthresh]
	cbar=fig.colorbar(c, ax=ax1 , ticks=tks, orientation='horizontal', location='top' )
	#tks=[ '$10^{-1}$','$-10^{-1}$', '$10^{-3}$','$-10^{-3}$']
	#ax1.set_yticks([.2,.6,1])
	
	cbar.ax.tick_params(axis='x', direction='in', which ='both')
	cbar.ax.minorticks_on()
	
	
	q2r=lambda x: x
	r2q=lambda x: x
	#secax_x = ax1.secondary_xaxis('top', xlabel = None, functions = (q2r, r2q))
	#secax_x.set_ticks([])
	#labels=[ "$\lambda_{pump}$" , "$\lambda_{abs}$" , "$\lambda_{ems}$" ]
	##secax_x.set_xticklabels(labels)
	cbar.set_label(r'$\bf{Re\{4 \pi \varepsilon_0 \alpha \}}$ ($F \mu m^2$)',labelpad=5)
	
	ax1.set_ylabel(r'$\bf{  I_{p}}$ ($kW/mm^2 $)',labelpad=0)
	ax1.set_yticks(Iticks, minor=False)
	ax1.set_yticks(Imticks, minor=True)
	#ax1.set_xticks(wlticks, minor=False)
	#ax1.set_xticks(wlmticks, minor=True)
	
	ax1.set_xlim(0.59,0.605)
	ax1.set_xlabel(r'$\mathbf{\lambda}$ ($\mu m$)',labelpad=0)
	#ax1.minorticks_on()
	#ax1.set_xlim(wlplot1,wlplot2)
	ax1.tick_params( direction='in', which='both')
	ax2.tick_params( direction='in', which='both')
	plt.axhline(unst_min, ls='--', c='chocolate', lw=0.75)
	#plt.axhline(unst_max, ls='--', c='chocolate', lw=0.75)
	
	
	plt.tight_layout(pad=0.3,w_pad=0 ,h_pad=0)
	coords = ax1.get_position().get_points()
	ax1.text(0, 1, '(c)',transform=fig.transFigure,
		va='top', ha='left', fontsize=8, fontweight='bold')
	
	plt.savefig( 'Polariz-R-zoom.eps' )
	
	
	
	

