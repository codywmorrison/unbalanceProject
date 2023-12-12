from matplotlib import pyplot as plt
from matplotlib import style


import numpy as np
import pandas as pd
import cmath as cm
from mpl_toolkits.mplot3d import axes3d

##------------------READ PI/CSV---------------------

noDays = 0
noDaysMx = noDays * 144



siteDF23 = np.genfromtxt('DFlist.csv',dtype=str,delimiter=',',usecols=(0,1,2),skip_header=1)
introStr = "PQ Monitors\EQL\SOUTHEAST"
print(introStr)
locStr = "BRISBANE CENTRAL"
subStr = "SSADA"
fdrStr = "ADAMLS23"
txrStr = "SG11119-G"
monStr = "NM22674"
atrStr = "|CUR_A"

totalStr = "\\".join([introStr,locStr,subStr,fdrStr,txrStr,monStr]) + atrStr
print(totalStr)


##read sub and search for area that it is in? or use search? really need NODW TO FIX!!!

print(siteDF23)


data = np.genfromtxt('Book2.csv',delimiter=',',usecols=(1,2,3),skip_header=1,skip_footer=noDaysMx)
dataA = np.genfromtxt('Book2.csv',delimiter=',',usecols=(1),skip_header=1,skip_footer=noDaysMx)
dataB = np.genfromtxt('Book2.csv',delimiter=',',usecols=(2),skip_header=1,skip_footer=noDaysMx)
dataC = np.genfromtxt('Book2.csv',delimiter=',',usecols=(3),skip_header=1,skip_footer=noDaysMx)
mLVUR = np.genfromtxt('Book2.csv',delimiter=',',usecols=(4),skip_header=1,skip_footer=noDaysMx)
phData = np.deg2rad(np.genfromtxt('Book2.csv',delimiter=',',usecols=(5,6,7),skip_header=1,skip_footer=noDaysMx))
interval = np.genfromtxt('Book2.csv',delimiter=',',usecols=(8),skip_header=1,skip_footer=noDaysMx)
intvl = interval.reshape(len(interval),1)

##------------------INITIALISE COMPONENTS--------------------

V0 = 0
V1 = 0
V2 = 0
Vnom = 240

dataApu = dataA / Vnom
dataBpu = dataB / Vnom
dataCpu = dataC / Vnom


a1 = -0.5 + 0.866j          ##1<120 = -0.5 + 0.866j
a2 = np.conjugate(a1)       ##1<240 = -0.5 - 0.866j

seqCoef = np.array([[1,1,1],[1,a1,a2],[1,a2,a1]])
thirdCoef = np.array([[1/3,0,0],[0,1/3,0],[0,0,1/3]])

Vln = np.zeros((3,len(data)))
Vll = np.zeros((len(phData),3),dtype=np.complex_)
Vab = np.zeros((len(phData)),dtype=np.complex_)
Vbc = np.zeros((len(phData)),dtype=np.complex_)
Vac = np.zeros((len(phData)),dtype=np.complex_)

VseqA = np.zeros((3,len(data)))
VufNeg = np.zeros(len(data))
AVuf = np.zeros(len(data))
VufZero = np.zeros(len(data))

Carg = np.zeros(len(data))

CVufNeg = np.zeros(len(data),dtype=np.complex_)
CVufZero = np.zeros(len(data),dtype=np.complex_)


##----------------SEQUENCE CALCULATION FOR PHASE-------------------
i = 0

for i in range(len(data)):

    Vln[0][i] = dataA[i] 
    Vln[1][i] = dataB[i]
    Vln[2][i] = dataC[i] 
    
    i = i + 1

VseqA = np.matmul(thirdCoef,np.matmul(seqCoef,Vln)) 

i = 0
for i in range(len(data)):

    VufNeg[i] = (np.abs(VseqA[1][i]) / np.abs(VseqA[0][i])) * 100               ##np.sqrt((VseqA[1][i].real)**2 + (VseqA[1][i].imag)**2) / np.abs(VseqA[0][i]) * 100
    VufZero[i] = (np.abs(VseqA[2][i]) / np.abs(VseqA[0][i])) * 100              ## + 1 TO SHOW DIFF


    AVuf[i] =  np.sqrt((np.abs(VseqA[1][i])**2 + np.abs(VseqA[2][i])**2)) / np.abs(VseqA[0][i]) * 100

    
    CVufNeg[i] = VseqA[1][i] / VseqA[0][i] * 100         ## * 100) * (np.cos(np.arctan(VseqA[1][i]/VseqA[0][i])) + np.sin(np.arctan(VseqA[1][i]/VseqA[0][i])) * 1j)
    CVufZero[i] = VseqA[1][i] / VseqA[0][i] * 100

    Carg[i] = 180-np.angle(VseqA[1][i] / VseqA[0][i] * 100,deg = True)

    
    ##CVufNegReal[i] = VseqA[1][i] / VseqA[0][i] * 100 
    
    i = i + 1

print(Carg)

Vavg = np.average(data,axis=1)
Vmax = np.max(data,axis=1) - Vavg
Vmin = np.abs(np.min(data,axis=1) - Vavg)
Vmaxmin = np.transpose(np.array([Vmax,Vmin]))
Vmaxdev = np.max(Vmaxmin,axis=1)
LVUR = (Vmaxdev)*100 / (Vnom)


##-------------LVUR &  VUF PLOTTING-------------------

plt.subplot(2,2,1)
plt.plot(VufNeg,label='Neg. Seq.')
plt.plot(VufZero,label='Zero. Seq.')
plt.plot(AVuf,label='Alternate VUF')
plt.legend()
plt.title('Phase Voltage Unbalance Factor, VUF (%)')

plt.subplot(2,2,2)
plt.plot(LVUR,label='Calc. LVUR')
plt.plot(mLVUR,label='Meas. LVUR')
plt.legend()
plt.title('Line Voltage Unbalance Rate, LVUR (%)')

plt.subplot(2,2,3)
plt.plot(Carg)
plt.title('Complex Argument (degrees)')


##plt.subplot(2,2,3)
##plt.plot(Vab,label='ab')
##plt.plot(Vbc,label='bc')
##plt.plot(Vac,label='ac')
##plt.legend()

plt.subplot(2,2,4)
plt.plot(CVufNeg.imag,label='Neg. CVuf')
plt.title('Imaginary Component of CVUF')

##plt.legend()
##plt.title('Complex Phase Voltage Unbalance Factor, CVUF (%)')

plt.show()



Carg1 = Carg.reshape(len(Carg),1)
VufNeg1 = VufNeg.reshape(len(VufNeg),1)
CVufNeg1 = CVufNeg.imag.reshape(len(CVufNeg),1)




print(interval.shape)
print(interval)
testarr=np.array(CVufNeg.real)
print(testarr.shape)
print(testarr)
# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_wireframe(intvl,CVufNeg1,VufNeg1, rstride=10, cstride=10)  ## was trying complex angle (between pos n neg sequences)
plt.title('Complex Argument and Modulus')
ax.set_xlabel('Time')
ax.set_ylabel('Complex Component of CVUF')              ##'Complex Argument (degrees)')
ax.set_zlabel('PVUF')

plt.show()



dataA1 = dataApu.reshape(len(dataA),1)
dataB1 = dataBpu.reshape(len(dataB),1)
dataC1 = dataCpu.reshape(len(dataC),1)



fig, bx = plt.subplots(subplot_kw={"projection": "3d"})
bx.plot_wireframe(dataA1,dataB1,dataC1,rstride=50,cstride=50)##rstride=10, cstride=10)  ## was trying complex angle (between pos n neg sequences)
plt.title('Over/Undervoltage')
bx.set_xlabel('A')
bx.set_ylabel('B')              ##'Complex Argument (degrees)')
bx.set_zlabel('C')
##bx.set_xlim((0.8,0.85))
##bx.set_ylim((0.9,1.05))            
##bx.set_zlim((0.9,1.05))
plt.show()


plt.plot(dataApu,label='A')
plt.plot(dataBpu,label='B')
plt.plot(dataCpu,label='C')
plt.legend()
plt.title('Volt Profile')
plt.show()

##fig = plt.figure()
##ax = plt.axis(projection='3d')
##ax.plot3D(np.abs(CVufNeg),CVufZero.real,CVufZero.imag)
##plt.show()




##---------To be organised --------------------
A = np.zeros(200)
B = np.zeros(200)
C = np.zeros(200)

Vbn = np.linspace(0,2,200)
Vcn = np.linspace(0,2,200)
Van = np.linspace(0.9,1.1,100)

X,Y = np.meshgrid(Vbn,Vcn)

a = np.zeros(10)

phA = np.zeros(10)
phB = np.linspace(-120,-120,10)
phC = np.linspace(110,130,10)

##col=('blue','magenta','red','yellow','green')

K = 0.02

i = 0

for i in range(10):

    A[i] = (2*(np.cos(np.deg2rad((phA[i]-phB[i])+120))-K**2*np.cos(np.deg2rad((phA[i]-phB[i])-120))))/(1-K**2)
    B[i] = (2*(np.cos(np.deg2rad((phC[i]-phA[i])+120))-K**2*np.cos(np.deg2rad((phC[i]-phA[i])-120))))/(1-K**2)
    C[i] = (2*(np.cos(np.deg2rad((phB[i]-phC[i])+120))-K**2*np.cos(np.deg2rad((phB[i]-phC[i])-120))))/(1-K**2)
    
    y = X**2 + Y**2 + A[i]*X + B[i]*Y + C[i]*X*Y +1
   
    Z = 1

    
    plt.contour(X,Y,y,Z)##,colors=col[i])
    
    i = i + 1

plt.xlim(0.9,1.1)
plt.ylim(0.9,1.1)
plt.show()


