# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:06:58 2024

dipole sums following 
Evlyukhin PHYSICAL REVIEW B 82, 045404 2010, Optical response features of Si-nanoparticle arrays


@author: npas8772
"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 6}) #all plots

from matplotlib.patches import Circle
from tqdm import tqdm
#from definitions import *



def angle(vers,ref):
	ref=ref/np.sqrt(np.dot(ref,ref))
	
	vsq=np.dot(vers,vers)
	if vsq!=0.:
		vers=vers/np.sqrt(vsq)
		norm=np.cross(vers,ref)
	else:
		vers=0
		norm=0		
	
	yt=np.dot(norm,norm)**.5
	xt=np.dot(vers,ref)**.5
	theta=np.arctan2(yt,xt)
	return np.cos(theta), np.sin(theta)


def posic(m,n,L):
	xs=(m+n)*L/2
	ys=(m-n)*L*3**.5/2
	zs=xs*0	
	return xs,ys,zs

def qter_hx_lattice(nmax,L):
	ms=np.array(list(range(0,nmax+1)))
	ns=np.array(list(range(0,nmax+1)))
	Ms,Ns=np.meshgrid(ms,ns)
	mns=np.vstack([ Ms.flatten(), Ns.flatten() ]).T
	return mns

def hx_lattice(nmax,L):
	ms=np.array(list(range(-nmax,nmax+1)))
	ns=np.array(list(range(-nmax,nmax+1)))
	Ms,Ns=np.meshgrid(ms,ns)
	mns=np.vstack([ Ms.flatten(), Ns.flatten() ]).T
	return mns


def dyadic(a,b):
	a1,a2,a3=a[0],a[1],a[2]
	b1,b2,b3=b[0],b[1],b[2]	
	matrix= np.array([ [a1*b1,a1*b2,a1*b3],[a2*b1,a2*b2,a2*b3],[a3*b1,a3*b2,a3*b3] ])
	return matrix
	

def Green_tensor(kd,ri,mn,doquarter):
	
	m,n=mn	


	if doquarter:
		x,y,z=posic(m,n,L)
		rj=np.array([x,y,z])
		rij=ri-rj
		Rij=np.sqrt(np.dot(rij,rij))
		e1=rij/Rij
		m1=np.identity(3)
		m2=dyadic(e1,e1)
		f1=np.exp(1j*kd*Rij)/4/np.pi
		t1= 1/Rij + 1j/(kd*Rij**2) - 1/(kd**2*Rij**3)
		t2=-1/Rij - 3j/(kd*Rij**2) + 3/(kd**2*Rij**3)	
		G_tens =f1*(t1*m1 + t2*m2)
			
		x,y,z=posic(-m,n,L)
		rj=np.array([x,y,z])
		rij=ri-rj
		Rij=np.sqrt(np.dot(rij,rij))
		e1=rij/Rij
		m1=np.identity(3)
		m2=dyadic(e1,e1)
		f1=np.exp(1j*kd*Rij)/4/np.pi
		t1= 1/Rij + 1j/(kd*Rij**2) - 1/(kd**2*Rij**3)
		t2=-1/Rij - 3j/(kd*Rij**2) + 3/(kd**2*Rij**3)	
		G_tens+=f1*(t1*m1 + t2*m2)		
		
		
		x,y,z=posic(m,-n,L)
		rj=np.array([x,y,z])
		rij=ri-rj
		Rij=np.sqrt(np.dot(rij,rij))
		e1=rij/Rij
		m1=np.identity(3)
		m2=dyadic(e1,e1)
		f1=np.exp(1j*kd*Rij)/4/np.pi
		t1= 1/Rij + 1j/(kd*Rij**2) - 1/(kd**2*Rij**3)
		t2=-1/Rij - 3j/(kd*Rij**2) + 3/(kd**2*Rij**3)	
		G_tens+=f1*(t1*m1 + t2*m2)

		x,y,z=posic(-m,-n,L)
		rj=np.array([x,y,z])
		rij=ri-rj
		Rij=np.sqrt(np.dot(rij,rij))
		e1=rij/Rij
		m1=np.identity(3)
		m2=dyadic(e1,e1)
		f1=np.exp(1j*kd*Rij)/4/np.pi
		t1= 1/Rij + 1j/(kd*Rij**2) - 1/(kd**2*Rij**3)
		t2=-1/Rij - 3j/(kd*Rij**2) + 3/(kd**2*Rij**3)	
		G_tens+=f1*(t1*m1 + t2*m2)
		
		if (m==0 or n==0):	G_tens/=2
			
	else:	
		x,y,z=posic(m,n,L)
		rj=np.array([x,y,z])
		rij=ri-rj
		Rij=np.sqrt(np.dot(rij,rij))
		e1=rij/Rij
		m1=np.identity(3)
		m2=dyadic(e1,e1)
		f1=np.exp(1j*kd*Rij)/4/np.pi
		t1= 1/Rij + 1j/(kd*Rij**2) - 1/(kd**2*Rij**3)
		t2=-1/Rij - 3j/(kd*Rij**2) + 3/(kd**2*Rij**3)	
		G_tens =f1*(t1*m1 + t2*m2)

	t3= 1j*kd/Rij - 1/(Rij**2)
	f2=np.exp(1j*kd*Rij)/4/np.pi/Rij
	g_vec= f2*t3*rij	
	return G_tens, g_vec



def dipolesum(mns,kd,ri,doquarter):
	G0t=np.zeros((3,3),dtype='complex')
	g0v=np.zeros(3,dtype='complex')	
	
	for j in range(len(mns)):
		mn=mns[j]
		G_tens, g_vec=Green_tensor(kd,ri,mn,doquarter)#				
		G0t += G_tens
		g0v += g_vec
	return G0t,g0v

def func(m,n,n_inc,n_dif,sinT,cosP,sinP,L):
	A= 4/3*(m**2+n**2+n*m)
	B=-2*n_inc/n_dif*sinT
	B*=(m+n)*cosP+ 3**.5/3 * (m-n)*sinP
	C=(n_inc/n_dif*sinT)**2-1	
	r1=(-B+(B**2-4*A*C)**.5)/2/A
	r2=(-B-(B**2-4*A*C)**.5)/2/A
	return np.real(r1*n_dif*L),np.real(r2*n_dif*L)






c=299792458
eps0=8.8541878176e12
mu0=np.pi*4e-7

#parameters input
doquarter=True
nh=1.5
wlmin=200e-9
wlmax=200000e-9

k0s=np.linspace( 2*np.pi/wlmax, 2*np.pi/wlmin , 200 )
wls=2*np.pi/k0s
ws=2*np.pi*c/wls
ks=nh*k0s


rad1=15e-9#L
L= ((wlmax-wlmin)/2+wlmin )/nh


L=600e-9/nh/0.866


print('lattice parameter', L*1e9)
S_L=L**2*3**.5/2
nlp=200 #number of x-lattice points

#   full lattice   (2*nlp+1)**2 - 1 =  4*nlp**2 + 4*nlp   points
#quarter lattice   nlp**2+nlp



#incident plane wave
E0=1
#dirE=np.array([3,3**.5,0]) #polarization direction
ref=np.array([0,0,0]) #central particle position
dirE=np.array([1,0,0]) #polarization direction
dirk=np.array([0,0,1]) #incident wavevector direction
cosT,sinT=angle(dirk, [0,0,1]) #theta polar
cosP,sinP=angle([dirk[0],dirk[1],0],[1,0,0]) #psi azimuth
if sinT==0: cosP,sinP = 0,0
cosA,sinA=angle(dirE, [1,0,0])#pol angle
dirE=dirE/np.sqrt(np.dot(dirE,dirE)) 
dirk=dirk/np.sqrt(np.dot(dirk,dirk)) 
cos2tethas=np.linspace(0,1,1000)


#############################################################






file1=np.genfromtxt('Gxx.txt')#, np.vstack([k0s*L,np.real(Gxx),np.imag(Gxx)]).T)
#file1=np.genfromtxt('Gxx-small-40400.txt')


f1r=np.interp(nh*k0s*L, file1[:,0], file1[:,1])
f1i=np.interp(nh*k0s*L, file1[:,0], file1[:,2])	
Gxx=f1r+1j*f1i


#Gxx2=(-Gxx*k0s**2)**-1 #polarizability of neighbours
app=  (2*S_L*ks)**-1 #approx 

app2= (-Gxx *k0s**2)**-1 # alpha_o
app3=  app2 / (2*nh*S_L/k0s) # alpha_o relative 

#app2=(app*k0s**2)**-1
#    =( k0s**2 /2 / S / k0s)**-1
#    =( k0s /2 / S )**-1
#    =2S/k0s 

#app2=S_L/ k0s





#rwas

ms=np.arange(-10,10)
ns=ms
Ns,Ms=np.meshgrid(ms,ns)
Ns,Ms=Ns.flatten(),Ms.flatten()
	
X=[] # s pol
for ii, cos2tetha  in enumerate(cos2tethas):
	dirk=np.array([0,(1-cos2tetha)**.5,cos2tetha**0.5]) #incident wavevector direction
	cosT,sinT=angle(dirk, [0,0,1]) #theta polar
	cosP,sinP=angle([dirk[0],dirk[1],0],[1,0,0]) #psi azimuth

	divs2=[]
	for i in range(len(Ns)):
		if not (Ms[i]==0 and Ns[i]==0):
			dic_lambda=func(Ms[i],Ns[i],nh,nh,sinT,cosP,sinP,L)
			divs2.append( [dic_lambda[0],dic_lambda[1]] )
	divs2=nh*L/np.array(divs2)
	divs2=np.sort(divs2)	
	divs3= np.unique(divs2)
	divs3=divs3[ divs3 > 0 ]
	X.append(divs3)

Y=[] # p pol
for ii, cos2tetha  in enumerate(cos2tethas):
	dirk=np.array([(1-cos2tetha)**.5,0,cos2tetha**0.5]) #incident wavevector direction
	cosT,sinT=angle(dirk, [0,0,1]) #theta polar
	cosP,sinP=angle([dirk[0],dirk[1],0],[1,0,0]) #psi azimuth

	divs2=[]
	for i in range(len(Ns)):
		if not (Ms[i]==0 and Ns[i]==0):
			dic_lambda=func(Ms[i],Ns[i],nh,nh,sinT,cosP,sinP,L)
			divs2.append( [dic_lambda[0],dic_lambda[1]] )
	divs2=nh*L/np.array(divs2)
	divs2=np.sort(divs2)	
	divs3= np.unique(divs2)
	divs3=divs3[ divs3 > 0 ]
	Y.append(divs3)


dirk=np.array([0,0,1]) #incident wavevector direction
cosT,sinT=angle(dirk, [0,0,1]) #theta polar
cosP,sinP=angle([dirk[0],dirk[1],0],[1,0,0]) #psi azimuth

divs2=[]
for i in range(len(Ns)):
	if not (Ms[i]==0 and Ns[i]==0):
		dic_lambda=func(Ms[i],Ns[i],nh,nh,sinT,cosP,sinP,L)
		divs2.append( [dic_lambda[0],dic_lambda[1]] )
divs2=nh*L/np.array(divs2)
divs2=np.sort(divs2)	
divs3= np.unique(divs2)
divs3=divs3[ divs3 > 0 ]
X.append(divs3)






fig, ax1 =plt.subplots(2,1, sharex=True ,figsize=(3.25, 1.4), dpi=200)


maskx= nh*k0s*L/np.pi/2 < 1.15

plt.subplots_adjust( hspace=0)
ax1[0].axhline(linewidth=0.5, color='k',zorder=0)


ax1[0].plot(nh*k0s*L/np.pi/2,np.real(Gxx)*1e-6,label='$Real$',zorder=0)
ax1[0].plot(nh*k0s*L/np.pi/2,np.imag(Gxx)*1e-6,ls='-',label='$Imaginary$',zorder=0)
ax1[0].plot(nh*k0s[maskx]*L/np.pi/2,app[maskx]*1e-6,ls='--',label='',zorder=10)
ax1[0].plot(nh*k0s*L/np.pi/2,np.abs(Gxx)*1e-6,ls=':',label='$Abs. value$',zorder=0)

ax1[0].legend(loc='upper right')
ax1[0].set_ylim(-2,8)

ymin, ymax = ax1[0].get_ylim()
ax1[0].vlines(divs3,ymin,ymax,colors='k',  linestyles='dashed',linewidths=0.5)
ax1[0].set_ylabel(r'$\bf{G_{xx0}}$ ($\mu m^{-1}$)')
ax1[0].tick_params( direction='in',which='both')
#ax1[0].set_yticks([0,5,10])
ax1[1].axhline(linewidth=0.5, color='k',zorder=0)


ax1[1].plot(nh*k0s*L/np.pi/2,np.real(app3),label='',zorder=0)
ax1[1].plot(nh*k0s*L/np.pi/2,np.imag(app3),ls='-',label='',zorder=0)
ax1[1].plot(nh*k0s*L/np.pi/2,np.zeros(app3.size),lw=0,label='',zorder=0)
ax1[1].plot(nh*k0s*L/np.pi/2,np.abs(app3),ls=':',label='',zorder=0)
ax1[1].set_xlim(nh*L/wlmax,nh*L/wlmin)
#ax1[1].set_yticks([0,0.5])

ymin, ymax = ax1[1].get_ylim()
ax1[1].vlines(divs3,ymin,ymax,colors='k',  linestyles='dashed',linewidths=0.5)
#ax1[1].legend(loc='upper right')
ax1[1].tick_params( direction='in',which='both')
ax1[1].set_ylabel(r'$\bf{ \alpha^\prime_{hl} }$ ')
ax1[1].set_xlabel(r'$\bf{k_s L/(2 \pi)}$ ')

ax1[1].minorticks_on()
ax1[0].minorticks_on()
plt.tight_layout(pad=0,w_pad=0 ,h_pad=0)
plt.savefig('lattice polarizab.eps')	