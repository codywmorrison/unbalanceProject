from matplotlib import pyplot as plt
from matplotlib import style
from PIconnect.PIConsts import RetrievalMode

import numpy as np
import pandas as pd
import cmath as cm
import PIconnect as PI
import seaborn as sns
import pprint

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##------------PI setup------------##

print('\nDefault Time: '+PI.PIConfig.DEFAULT_TIMEZONE)
PI.PIConfig.DEFAULT_TIMEZONE = 'Australia/Brisbane'
print('\nConfig to UTC+10 Time: '+PI.PIConfig.DEFAULT_TIMEZONE)

print('\nServer List: '+str(list(PI.PIServer.servers.keys())))
print('\nDatabase List: '+str(list(PI.PIAFDatabase.servers.keys())))
print('\n-----------------------------------------------------\n')

##------------functions------------##

def indataCheck(k,n):

    testStr = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(7),skip_header=1)
    
    while testStr[k]=='No Data' or testStr[k]=='XXXX':

        n += 1
        print('Cannot form filepath for k = {f}'.format(f=str(k)))
        print('This site has been skipped. {h} have failed to form a filepath. k = {g} will now be run:'.format(h=str(n),g=str(k+1)))

        print('\n---------------------------next monitor---------------------------\n')
        
        k += 1
        
    if testStr[k]!='No Data' or testStr[k]!='XXXX':

        print('Filepath can be formed for k = {f}'.format(f=str(k)))

        return k, n


def dataGrab(fPath):
    with PI.PIAFDatabase(database="PQ Monitors") as database:
        
        ##print("\nConnected to AFServer: "+database.server_name)
        ##print("\nConnected to AFDatabase: "+database.database_name+'\n')

        element = database.descendant(fPath)
        
        ##for attr in element.attributes:
        ##    print(element.attributes[attr])

        attvalues = iter(element.attributes.values())
        
        attQuality = next(attvalues)
        attVoltC = next(attvalues)
        attVoltB = next(attvalues)
        attVoltA = next(attvalues)

        intT = '10m'
        startT1 = '2018-01-01 00:00:00'
        endT1 = '2018-12-1 00:00:00'

        startT2 = '2023-01-01 00:00:00'
        endT2 = '2023-12-1 00:00:00'
        
        dataC = attVoltC.interpolated_values(startT1,endT1,intT)
        dataB = attVoltB.interpolated_values(startT1,endT1,intT)
        dataA = attVoltA.interpolated_values(startT1,endT1,intT)

        dataC2 = attVoltC.interpolated_values(startT2,endT2,intT)
        dataB2 = attVoltB.interpolated_values(startT2,endT2,intT)
        dataA2 = attVoltA.interpolated_values(startT2,endT2,intT)
        
        dataMatrix = [[dataA,dataB,dataC],[dataA2,dataB2,dataC2]] 
        
        return dataMatrix

def feederGrab(txrN,monN,parPath):
    with PI.PIAFDatabase(database="PQ Monitors") as database:
        
        feederSRCH = database.descendant(parPath)
        for fdr in feederSRCH.children.values():

            fdrStr = str(fdr)
            fdrStr1 = fdrStr.replace('PIAFElement(','')
            fdrStr2 = fdrStr1.replace(')','')
            
            ##print('Searching through feeder: '+fdrStr2+' at address: '+ parPath+'\\'+fdrStr2)
            
            txrSRCH = database.descendant(parPath+'\\'+fdrStr2)
            for txr in txrSRCH.children.values():
                
                txrStr = str(txr)
                txrStr1 = txrStr.replace('PIAFElement(','')
                txrStr2 = txrStr1.replace(')','')
                
                ##print("TXR Element: {R}".format(R=txrStr2))

                if txrStr2==txrN:

                    fPath = parPath+'\\'+fdrStr2+'\\'+txrStr2+'\\'+monN

                    print('Found monitor filepath: '+fPath)

                    return fPath
                

def pathGrab(k):

    siteDF23 = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(5,7,8,9,10),skip_header=1)

    eqlName = "EQL"
    regName = siteDF23[k][4]
    locName = siteDF23[k][2]
    subName = siteDF23[k][1]
    txrName = siteDF23[k][0]
    monName = siteDF23[k][3]

    oPath = "\\".join([eqlName,regName,locName,subName])
    parPath = r'{}'.format(oPath)

    print('\nFound partial filepath from CSV: '+parPath+'\n')
    
    ## Not needed -----> atrName = "|CUR_A"
    
    return txrName,parPath,monName

def extractTph(x,y,z):

    Va = np.zeros(len(x))
    Vb = np.zeros(len(Va))
    Vc = np.zeros(len(Va))
    
    appVa=[]
    appVb=[]
    appVc=[]
    
    i=0
    
    for i in range(len(Va)):

        if str(x[i]) != 'No Data' and x[i] > 0.8*240:
            Va[i]=x[i]
            appVa.append(x[i])

        if str(y[i]) != 'No Data' and x[i] > 0.8*240:
            Vb[i]=y[i]
            appVb.append(y[i])

        if str(z[i]) != 'No Data' and x[i] > 0.8*240:
            Vc[i]=z[i]
            appVc.append(z[i])

    points = round(len(Va)-len(appVa))

    
    if len(appVa) != 0: ##if = len(Va)? then fully cleansed
        
        print('\n{d} points of data were cleaned'.format(d=points))
        print('Meaning {y} of {m} days are unusable data\n'.format(y=round(points/144), m=round(len(Va)/144)))
            
    return appVa, appVb, appVc

def voltageProfile(Vx,Vy,Vz,txrN):
    
    ##plt.subplot(2,1,1)
    plt.plot(Vx,label='A')
    plt.plot(Vy,label='B')
    plt.plot(Vz,label='C')
    plt.legend()
    plt.title('Volt Profile of Txr: {}'.format(txrN))

    plt.show()

    return

def calcUnbalance(Va,Vb,Vc,k,c,txrN,tID):

    ##-----------------Line Voltage Unbalance Rate, LVUR-----------------##
    Vall = np.hstack((np.reshape(np.array(Va),(len(Va),1)),np.reshape(np.array(Vb),(len(Vb),1)),np.reshape(np.array(Vc),(len(Vc),1))))
       
    Vnom = 240
    Vavg = np.average(Vall,axis=1)
    Vmax = np.max(Vall,axis=1) - Vavg
    Vmin = np.abs(np.min(Vall,axis=1) - Vavg)

    Vmaxmin = np.transpose(np.array([Vmax,Vmin]))
    Vmaxdev = np.max(Vmaxmin,axis=1)
    LVUR = (Vmaxdev)*100 / (Vavg)

    sortedLVUR = sorted(LVUR)
    lengthLVUR = len(LVUR)

    onePercentile = round(sortedLVUR[round(0.01*lengthLVUR)],2)
    twofivePercentile = round(sortedLVUR[round(0.25*lengthLVUR)],2)
    median = round(sortedLVUR[round(0.5*lengthLVUR)],2)
    sevenfivePercentile = round(sortedLVUR[round(0.75*lengthLVUR)],2)
    nineninePercentile = round(sortedLVUR[round(0.99*lengthLVUR)],2)
    


    ##-----------------Voltage Unbalance Factor, VUF-----------------##
    
    Vln = np.zeros((3,len(Va)))
    VufNeg = np.zeros(len(Va))
    
    for j in range(len(Va)):

        Vln[0][j] = Va[j] 
        Vln[1][j] = Vb[j]
        Vln[2][j] = Vc[j] 
    
        j += 1  
        
    a1 = -0.5 + 0.866j          ##1<120 = -0.5 + 0.866j
    a2 = np.conjugate(a1)       ##1<240 = 1<-120 = -0.5 - 0.866j
    
    seqCoef = np.array([[1,1,1],[1,a1,a2],[1,a2,a1]])
    thirdCoef = np.array([[1/3,0,0],[0,1/3,0],[0,0,1/3]])

    VseqA = np.zeros((3,len(Va)))
    VseqA = np.matmul(thirdCoef,np.matmul(seqCoef,Vln))

    n = 0
    
    for n in range(len(Va)):
        VufNeg[n] = (np.abs(VseqA[1][n]) / np.abs(VseqA[0][n])) * 100
        n += 1
        
    sortedVuf = sorted(VufNeg)
    lengthVuf = len(sortedVuf)

    onePercentileVUF = round(sortedVuf[round(0.01*lengthVuf)],2)
    twofivePercentileVUF = round(sortedVuf[round(0.25*lengthVuf)],2)
    medianVUF = round(sortedVuf[round(0.5*lengthVuf)],2)
    sevenfivePercentileVUF = round(sortedVuf[round(0.75*lengthVuf)],2)
    nineninePercentileVUF = round(sortedVuf[round(0.99*lengthVuf)],2)
    
    if median > 1 or nineninePercentile > 1.8:
        unbalanceList[c+1][0] = 'No. ' + str(k) + 'a'
        if tID == 1:
            unbalanceList[c+1][0] = 'No. ' + str(k) + 'b'
        unbalanceList[c+1][1] = txrN
        unbalanceList[c+1][2] = median
##        unbalanceList[c+1][3] = median
##        unbalanceList[c+1][4] = median
        unbalanceList[c+1][3] = sevenfivePercentile
        c += 1

    ubtypeA = 'LVUR'
    ubtypeB = 'VUF'
    unbalanceDistribution(LVUR,txrN,tID,ubtypeA)
    unbalanceDistribution(VufNeg,txrN,tID,ubtypeB)
    
    return VufNeg, sortedVuf, unbalanceList, c, LVUR


def distributionProfile(Va,Vb,Vc,txrN,tID):
    Apu = np.zeros(len(Va)) ##Apu = np.zeros((len(Va),3))
    Bpu = np.zeros(len(Va))
    Cpu = np.zeros(len(Va))
    if len(Va) == 0:
        return
    for i in range(len(Va)):
        Apu[i] = Va[i]/240
        Bpu[i] = Vb[i]/240
        Cpu[i] = Vc[i]/240

    ##fig, ax = plt.figure()
    sns.kdeplot([Apu,Bpu,Cpu],legend=False)##,bins=15,kde=True) ##displot? ##clip=(0.95,1.05)

    
    
    timeEntry = 'a'
    timeEntry2 = '2018'
    if tID == 1:
            timeEntry = 'b'
            timeEntry2 = '2023'
            
    plt.title('Three-Phase Distribution of {x}'.format(x=txrN+' '+timeEntry2))
    plt.xlabel('Per-Unit Voltage (240V)')
    plt.savefig('{a}.png'.format(a=txrN+timeEntry),dpi=1200)
    plt.close()
    ##plt.show()
    
    ## Can do kind='kde' to set it so it only has the density curve, not the histogram

    return

def unbalanceDistribution(unbalanceData,txrN,tID,ubtype):
    
    sns.kdeplot(unbalanceData,legend=False)##,bins=15,kde=True) ##displot? ##clip=(0.95,1.05),

    timeEntry = 'a'
    timeEntry2 = '2018'
    if tID == 1:
            timeEntry = 'b'
            timeEntry2 = '2023'
            
    plt.title('Unbalance ({t}) Distribution of {x}'.format(t=str(ubtype),x=txrN+' '+timeEntry2))
    plt.xlabel('Factor/Percentage Unbalance')
    plt.savefig('{a}.png'.format(a=txrN+timeEntry+''+str(ubtype)),dpi=1200)
    plt.close()
    ##plt.show()

    ##axvline <--- look into

    return
    
##------------main script------------##

pCount = 0
unbalanceList = [['No. DF23','TXR Name','p50','p75'],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                 [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                 [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],
                 [0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],]##np.array((10,4))

failed = 0
i = 450
start = 0
end = 50
g=0


for start in range(end):
    i, failed = indataCheck(i,failed)


    txrName, parPath, monName = pathGrab(i)

    filePath = feederGrab(txrName,monName,parPath)

    dataMatrix = dataGrab(filePath)

    tID = 0
    
    for g in range(2):
        
        Va, Vb, Vc = extractTph(dataMatrix[g][0],dataMatrix[g][1],dataMatrix[g][2])
        
        distributionProfile(Va,Vb,Vc,txrName,tID)
        ##voltageProfile(Va,Vb,Vc,txrName)

        
        
        if len(Va) != 0:
            
            Vuf,sortedVuf,finalList,pCount,LVUR = calcUnbalance(Va,Vb,Vc,i,pCount,txrName,tID)
            
        ##distributionPercentile(finalList)  <---- to be done (distribution of the LVUR or VUF)
            
        tID += 1

    
    print('\n---------------------------next monitor---------------------------\n')
    
    i+=1

print('List of Unbalance Percentiles')

pprint.pprint(finalList)

print()

print(str(start)+' filepaths accessed')

print(str(failed)+' filepaths failed to form')

##input next (otherwise stop operation maybe?)

##maybe input for year like (18/19 or 22/23) -> could pull profile or maybe not -> could pull distribution of balance????? hmm

    

    
