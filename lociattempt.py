from matplotlib import pyplot as plt
from matplotlib import style
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

import numpy as np
import pandas as pd


A = np.zeros(200)
B = np.zeros(200)
C = np.zeros(200)

Vbn = np.linspace(0,2,200)
Vcn = np.linspace(0,2,200)
Van = np.linspace(0.9,1.1,100)

X,Y = np.meshgrid(Vbn,Vcn)

a = np.zeros(200)

phA = np.zeros(200)
phB = np.linspace(-120,-120,200)
phC = np.linspace(110,130,20)

##col=('blue','magenta','red','yellow','green')

K = 0.02

i = 0
ax=plt.figure().add_subplot(projection='3d')
for i in range(10):

    A[i] = (2*(np.cos(np.deg2rad((phA[i]-phB[i])+120))-K**2*np.cos(np.deg2rad((phA[i]-phB[i])-120))))/(1-K**2)
    B[i] = (2*(np.cos(np.deg2rad((phC[i]-phA[i])+120))-K**2*np.cos(np.deg2rad((phC[i]-phA[i])-120))))/(1-K**2)
    C[i] = (2*(np.cos(np.deg2rad((phB[i]-phC[i])+120))-K**2*np.cos(np.deg2rad((phB[i]-phC[i])-120))))/(1-K**2)
    
    y = X**2 + Y**2 + A[i]*X + B[i]*Y + C[i]*X*Y +1# + 0.1*i
    ##a[i] = X**2 + Y**2 + A[i]*X + B[i]*Y + C[i]*X*Y + 1
    Z = 1

    
    ax.contour(X,Y,y,Z)##,colors=col[i])
    
    i = i + 1


plt.show()

##
##plt.title('This should be one?')
##plt.xlim(0.5,1.6)
##plt.ylim(0.5,1.6)#0.9,1.1)
##plt.show()
##
##fig, bx = plt.subplots(subplot_kw={"projection": "3d"})
##bx.plot_surface(X,Y,a,rstride=50,cstride=50)
##plt.title('Loci')
##bx.set_xlabel('Vbn')
##bx.set_ylabel('Vcn')              ##'Complex Argument (degrees)')
##bx.set_zlabel('Angle')
####bx.set_xlim((0.8,0.85))
####bx.set_ylim((0.9,1.05))            
####bx.set_zlim((0.9,1.05))
##plt.show()
