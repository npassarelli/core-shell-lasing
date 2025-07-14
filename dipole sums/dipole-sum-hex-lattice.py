# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:06:58 2024

dipole sums following 
Evlyukhin PHYSICAL REVIEW B 82, 045404 2010, Optical response features of Si-nanoparticle arrays


@author: npas8772
"""


import numpy as np
import matplotlib.pyplot as plt
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


rad1=5e-9  # only for plot
L=((wlmax-wlmin)/2+wlmin )/nh

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


def main():
	
	#lattice definition
	if doquarter:
		mns=qter_hx_lattice(nlp,L)
		print('Summing over ', len(mns)-1 , ' particles' )	
	else:
		mns=hx_lattice(nlp,L)	
		print('Summing over ', len(mns)-1 , ' particles' )	

		
	maskx=mns[:,0]==0
	masky=mns[:,1]==0
	maskxy=np.logical_and(maskx,masky)
	indx=np.squeeze(np.argwhere(maskxy))
	mns=np.delete(mns,indx,axis=0)

	
	if False: #plot
		fig, ax1 =plt.subplots(nrows=1, ncols=1, sharex=True)
		for mn in mns:
			x,y,z=posic(mn[0],mn[1],L)			
			ci=Circle( (x*1e9,y*1e9),rad1*1e9,fc='b')
			ax1.add_patch(ci)
			
			
		xmin,xmax= -nlp*L -rad1,nlp*L +rad1 #(np.min(rjs[:,0])-rad1)*1e9,(np.max(rjs[:,0])+rad1)*1e9
		ymin,ymax= -nlp*L -rad1,nlp*L +rad1#(np.min(rjs[:,1])-rad1)*1e9,(np.max(rjs[:,1])+rad1)*1e9
		ax1.set_xlim(xmin*1e9,xmax*1e9)
		ax1.set_ylim(ymin*1e9,ymax*1e9)
		fig.tight_layout()
		plt.savefig('lattice.png')
		plt.show()
	
	
	#dipole sums
	
	print('K iterations', len(k0s))
	Gxx=np.zeros(len(k0s),dtype='complex')
	Gyy=np.zeros(len(k0s),dtype='complex')	
	for	k,ki in tqdm(enumerate(k0s)):		
		G0t,g0v=dipolesum(mns,ki*nh,ref,doquarter)
		Gxx[k]=G0t[0,0]
		Gyy[k]=G0t[1,1]		
		#if doqter:
		#	Gxx[k]*=4
		#	Gyy[k]*=4		
			
	ms=np.arange(-10,10)
	ns=ms
	Ns,Ms=np.meshgrid(ms,ns)
	Ns,Ms=Ns.flatten(),Ms.flatten()

	divs2=[]
	for i in range(len(Ns)):
		if not (Ms[i]==0 and Ns[i]==0):
			dic_lambda=func(Ms[i],Ns[i],nh,nh,sinT,cosP,sinP,L)
			divs2.append( dic_lambda[0] )
			divs2.append( dic_lambda[1])
		
		
	#divs2= lambda
	
	#quiero k_0L/2pi = L/lambda = 2/x

	
	divs2=nh*L/np.asarray(divs2)
	divs2=np.sort(divs2)	
	divs3= np.unique(divs2)
	divs3=divs3[ divs3 > 0 ]
	
	
	
	
	#approx
	
	app=1/(2*S_L*ks)
	
	#plot
	if True:
		fig, ax1 =plt.subplots(nrows=2, ncols=1, sharex=True)
		ax1[0].axhline(linewidth=0.5, color='k',zorder=0)
		ax1[1].axhline(linewidth=0.5, color='k',zorder=0)
				
		ax1[0].plot(nh*k0s*L/np.pi/2,np.real(Gxx),label='Re',zorder=0)
		ax1[0].plot(nh*k0s*L/np.pi/2,np.imag(Gxx),label='Im',zorder=0)
		ax1[0].plot(nh*k0s*L/np.pi/2,app,ls='--',label='$(2 S_L k_s)^{-1}$',zorder=0)			
				
		ax1[1].plot(nh*k0s*L/np.pi/2,np.real(Gyy),label='Re',zorder=0)
		ax1[1].plot(nh*k0s*L/np.pi/2,np.imag(Gyy),label='Im',zorder=0)
		ax1[1].plot(nh*k0s*L/np.pi/2,app,ls='--',label='$(2 S_L k_s)^{-1}$',zorder=0)			
				
		ax1[0].scatter(divs3,divs3*0,label='RW anomaly',zorder=10)
		ax1[1].scatter(divs3,divs3*0,label='RW anomaly',zorder=10)
		#ax1[0].scatter(divs2/2,divs2*0,label='divs2',zorder=5)
		#ax1[1].scatter(divs2/2,divs2*0,label='divs2',zorder=5)
		ax1[0].set_xlim(0,nh*L/wlmin)
		ax1[1].set_xlim(0,nh*L/wlmin)
#		ax1[0].set_ylim(-0.2e7,0.5e7)
#		ax1[1].set_ylim(-0.2e7,0.5e7)
		ax1[0].legend(loc='upper right')
		ax1[1].legend(loc='upper right')
		ax1[0].set_ylabel('sums Gxx')
		ax1[1].set_ylabel('sums Gyy')
		ax1[0].set_xlabel("$k_s L/2 pi$")
		ax1[1].set_xlabel("$k_s L/2 pi$")
		plt.savefig('dipole sums.png')	

		
	print('saving...')
	np.savetxt('Gxx.txt', np.vstack([nh*k0s*L,np.real(Gxx),np.imag(Gxx)]).T)
	np.savetxt('Gyy.txt', np.vstack([nh*k0s*L,np.real(Gyy),np.imag(Gyy)]).T)

 
	#end main

if __name__ == "__main__":
    main()




