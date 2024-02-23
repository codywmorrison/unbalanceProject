# Written by Cody Morrison (Vac. Student) for Energy Queensland under Trevor Gear (Principal Engineer)
#
# This script uses an interface with Osisoft's PI to pull current and voltage data from
# a list of PQ monitor sites and decompose the imbalance into systematic and random.
#
# This script was written as part of a student placement with Energy Queensland over the
# university's vacation period of 2023/2024.
#
# V3 - 19/2/23


##--- module importation---##
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
import matplotlib.pylab

from PIconnect.PIConsts import RetrievalMode
import numpy as np      ## <- numpy used for majority of mathematical operations
import polars as pd     ## <- Pandas was originally used, but changed to Polars for speed
import pyarrow as pa
import pandas as pnd
import cmath as cm
import PIconnect as PI  ## <- PIconnect is the module for PI AF SDK
import seaborn as sns   ## <- seaborn and scipy are used for statistics calculations
import pprint
import datetime

from mpl_toolkits.mplot3d import axes3d
from matplotlib.dates import DateFormatter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



##------------PI setup------------##

##print('\nDefault Time: '+PI.PIConfig.DEFAULT_TIMEZONE)
PI.PIConfig.DEFAULT_TIMEZONE = 'Australia/Brisbane'
##print('\nConfig to UTC+10 Time: '+PI.PIConfig.DEFAULT_TIMEZONE)
##
##print('\nServer List: '+str(list(PI.PIServer.servers.keys())))
##print('\nDatabase List: '+str(list(PI.PIAFDatabase.servers.keys())))
print('\n-----------------------------------------------------\n')

##------------functions------------##

def indataCheck(k,n):
    # This function checks if a filepath can be formed using the csv's data file.
    # If no filepath can be formed for a particular PQ monitor site, it is skipped
    # and the next site is tested.

    ##---checking that sub, region, and location are given in the csv list so a filepath can be formed---##
    testStr = np.genfromtxt('transformer list (first list).csv',dtype=str,delimiter=',',usecols=(1),skip_header=1)
    
    while testStr[k]=='No Data' or testStr[k]=='XXXX':

        n += 1
        print('Cannot form filepath for k = {f}'.format(f=str(k)))
        print('This site has been skipped. {h} have failed to form a filepath. k = {g} will now be run:'.format(h=str(n),g=str(k+1)))
        print('\n---------------------------next monitor---------------------------\n')
        k += 1
        
    if testStr[k]!='No Data' or testStr[k]!='XXXX':

        print('Filepath can be formed for k = {f}'.format(f=str(k)))

        return k, n


def dataGrab(fPath,i):#,i
    # This function uses PIConnect (an interface with Osisoft's PI) to pull voltage
    # and current data from Energex's PQ monitors. This function outputs a matrix with
    # all of the voltage and current data for a given period (atm 2018 and 2023 calendar years).
    
    ##---pulling data from Osisoft PI with formed filepath---##
    with PI.PIAFDatabase(database="PQ Monitors") as database:
        
        element = database.descendant(fPath)
        attvalues = iter(element.attributes.values())

        attList = [next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues)]
                   ##next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues)]

        ##---timeline to pull data---##
        intT = '10m'

        #siteDF23 = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(11),skip_header=1)

        #sliceNo = len(siteDF23[i])-4
        
        #yearInt = int(siteDF23[i][sliceNo:])
        #print(sliceNo,yearInt)

        #startT1 = '2022-07-01 00:00:00'
        #endT1 = '2023-6-30 00:00:00'

        startT1 = '2022-11-01 00:00:00'
        endT1 = '2024-1-1 00:00:00'
        

        #startT2 = '2022-07-01 00:00:00'#.format(c = yearInt-2)
        #endT2 = '2023-6-30 00:00:00'#.format(d = yearInt-1)

        #startT2 = '{e}-07-01 00:00:00'.format(e = yearInt+1)
        #endT2 = '{f}-6-30 00:00:00'.format(f = yearInt+2)
        
##        startT1 = '2018-07-02 00:00:00'
##        endT1 = '2019-6-30 00:00:00'
##
##        startT2 = '2022-07-02 00:00:00'
##        endT2 = '2023-6-30 00:00:00'


        ##---search and assign Voltage and Current data to matrix---##
        for att in range(len(attList)):

            if attList[att].name == 'VOLT_C':
                VdataCa = attList[att].interpolated_values(startT1,endT1,intT)
                #VdataCb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'VOLT_B':
                VdataBa = attList[att].interpolated_values(startT1,endT1,intT)
                #VdataBb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'VOLT_A':
                VdataAa = attList[att].interpolated_values(startT1,endT1,intT)
                #VdataAb = attList[att].interpolated_values(startT2,endT2,intT)

            if attList[att].name == 'CUR_C':
                CdataCa = attList[att].interpolated_values(startT1,endT1,intT)
                #CdataCb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_B':
                CdataBa = attList[att].interpolated_values(startT1,endT1,intT)
                #CdataBb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_A':
                CdataAa = attList[att].interpolated_values(startT1,endT1,intT)
                #CdataAb = attList[att].interpolated_values(startT2,endT2,intT)        #,[VdataCb,VdataBb,VdataAb],,[CdataCb,CdataBb,CdataAb]
        
        dataMatrix = [[VdataCa,VdataBa,VdataAa],[CdataCa,CdataBa,CdataAa]] # i did swap these
        

        return dataMatrix

def feederGrab(txrN,parPath):
    # This function searches the AF database and pulls the feeder name for the monitor site.
    # As only the transformer number/name and substation name is given, the feeder
    # name must be searched for and returned to form a full filepath for PI data.
    
    ##---searching database for transformer to get feeder, as feeder isn't given in csv---##
    with PI.PIAFDatabase(database="PQ Monitors") as database:
        
        feederSRCH = database.descendant(parPath)
        for fdr in feederSRCH.children.values():

            fdrStr = str(fdr)
            fdrStr1 = fdrStr.replace('PIAFElement(','')
            fdrStr2 = fdrStr1.replace(')','')
            print(fdrStr2)

            ##---transformer being searched---##            
            txrSRCH = database.descendant(parPath+'\\'+fdrStr2)
            for txr in txrSRCH.children.values():
                
                txrStr = str(txr)
                txrStr1 = txrStr.replace('PIAFElement(','')
                txrStr2 = txrStr1.replace(')','')

                ##---building full filepath for PI to read---##
                if txrStr2==txrN:

                    fPath = parPath+'\\'+fdrStr2+'\\'+txrStr2

                    monN = monitorGrab(txrN,fPath)

                    print('------------------------------------------------')
                    print(monN)
                    print('------------------------------------------------')

                    fPathF = parPath+'\\'+fdrStr2+'\\'+txrStr2+'\\'+monN

                    print('Found monitor filepath: '+fPathF)

                    return fPathF


def monitorGrab(txrN,parPath):
    # in the name, this searches from trx to grab monitor id
    
    ##---searching database for transformer to get feeder, as feeder isn't given in csv---##
    with PI.PIAFDatabase(database="PQ Monitors") as database:

        print('in monitorGrab')
        print(parPath)
        monitorSRCH = database.descendant(parPath)
        for mtr in monitorSRCH.children.values():

            mtrStr = str(mtr)
            mtrStr1 = mtrStr.replace('PIAFElement(','')
            mtrStr2 = mtrStr1.replace(')','')

            print(mtrStr2)

            return mtrStr2

def locationGrab(subN,locPath):
    # This function uses the subname fo search and find the location
    
    ##---searching database for transformer to get feeder, as feeder isn't given in csv---##
    with PI.PIAFDatabase(database="PQ Monitors") as database:
        
        locationSRCH = database.descendant(locPath)
        for loc in locationSRCH.children.values():

            locStr = str(loc)
            locStr1 = locStr.replace('PIAFElement(','')
            locStr2 = locStr1.replace(')','')
            print(locStr2)

            ##---transformer being searched---##            
            subSRCH = database.descendant(locPath+'\\'+locStr2)
            for sub in subSRCH.children.values():
                
                subStr = str(sub)
                subStr1 = subStr.replace('PIAFElement(','')
                subStr2 = subStr1.replace(')','')

                ##---building full filepath for PI to read---##
                if subStr2==subN:

                    return locStr2               

def pathGrab(k):
    # This function forms a partial filepath and passes it to the above function
    # (feederGrab) so that a full filepath can be passed to PIConnect (dataGrab) function.
    # This function uses the DF23 csv file to concatenate a filepath.

    ##---pulling sub, region and txr name from the csv file---##
    siteDF23 = np.genfromtxt('transformer list (first list).csv',dtype=str,delimiter=',',usecols=(0,1,3),skip_header=1) ##2018UnbalancedSites.csv, DFListV1.csv

    eqlName = "EQL"
    regName = siteDF23[k][2] #changed to 2 from 3
    #locName = siteDF23[k][2]
    subName = siteDF23[k][1]
    txrName = siteDF23[k][0]
    #monName = siteDF23[k][3]

    locPath = "\\".join([eqlName,regName])
    locName = locationGrab(subName,locPath)

    print(locName)

    ##---forming/concatenating the filepath---##
    oPath = "\\".join([eqlName,regName,locName,subName])
    parPath = r'{}'.format(oPath)

    print(parPath)
    
    #print('\nFound partial filepath from CSV: '+parPath+'\n')
    
    return txrName,parPath#,monName

def extract_3ph(x,y,z,d,e,f,genCount):
    # This function passes in x, y, and z voltage or current data to be cleaned and
    # simplfied before being passed for forming graphs and unbalance calculations.
    # This function returns and appended matrix of cleaned data (shorter than input).  

    ##---initialising attributes for calculation---##
    xDF = pnd.DataFrame(x)
    x = pd.from_pandas(xDF)
    yDF = pnd.DataFrame(y)
    y = pd.from_pandas(yDF)
    zDF = pnd.DataFrame(z)
    z = pd.from_pandas(zDF)

    dDF = pnd.DataFrame(d)
    d = pd.from_pandas(dDF)
    
    eDF = pnd.DataFrame(e)
    e = pd.from_pandas(eDF)
    fDF = pnd.DataFrame(f)
    f = pd.from_pandas(fDF)

    A = np.zeros(len(x))
    B = np.zeros(len(A))
    C = np.zeros(len(A))

    D = np.zeros(len(d))
    E = np.zeros(len(D))
    F = np.zeros(len(D))

    #print(x,y,z,d,e,f)
    
    rawDatetime = xDF.index.values.tolist()
    refinedDatetime = pnd.to_datetime(rawDatetime,unit='ns')
    refinedDatetime.columns = ['Date']
    

    
    refinedDate = pnd.DataFrame()
    refinedDate['Month']= refinedDatetime.month
    refinedDate = pd.from_pandas(refinedDate)
    refinedDatetime = pd.from_pandas(refinedDatetime)
    appA = []
    appB = []
    appC = []

    appD = []
    appE = []
    appF = []

    ##datetime = []
    date = []
    time = []
    
    i=1 ## <---i = 1 is used as the datacleaning process uses x[i-2] which is only possible if i is greater than or equal to 2.

    ##---cleaning data for calculation and graphing---##
    for i in range(len(A)-2):
        i+=1
        #print(round(x[i]-x[i-2],1))
        #print(x.row(i)[0])
        
        
        #if str(x.iloc[i]) != 'No Data' and type(x.iloc[i]) is  not np.float64 and type(x.iloc[i]) is  not float:
        #    print(i)
        #    print(type(x.iloc[i]))
        #    print(x.iloc[i])

        ##---data cleaining for current---##
        if genCount > 1:
            #if round(x.row(i)[0],1) != round(x.row(i-1)[0],1) or round(x.row(i)[0]-x.row(i-1)[0]) != round(x.row(i)[0]-x.row(i-2)[0]):
            if round(x.row(i)[0],2) != round(x.row(i-1)[0],2) or round(x.row(i)[0]-x.row(i-1)[0],2) != round(x.row(i)[0]-x.row(i-2)[0],2):#(type(x[i]) is np.float64 or type(x[i]) is float) and 
                ##A[i]=x[i]
                appA.append(x.row(i)[0])
                appD.append(d.row(i)[0])
                ##B[i]=y[i]
                appB.append(y.row(i)[0])
                appE.append(e.row(i)[0])
                ##C[i]=z[i]
                appC.append(z.row(i)[0])
                appF.append(f.row(i)[0])
                date.append(refinedDate.row(i)[0])
                time.append(str(refinedDatetime[i].time()))

        ##---data cleaning for voltage--## #(x.row(i)[0] > 120) and
        #elif round(x.row(i)[0],1) != round(x.row(i-1)[0],1) or round(x.row(i)[0]-x.row(i-1)[0]) != round(x.row(i)[0]-x.row(i-2)[0]):
        elif round(x.row(i)[0],2) != round(x.row(i-1)[0],2) or round(x.row(i)[0]-x.row(i-1)[0],2) != round(x.row(i)[0]-x.row(i-2)[0],2):#(type(x[i]) is np.float64 or type(x[i]) is float) and####(x[i] > 120) and
            ##A[i]=x[i]
            appA.append(x.row(i)[0])
            appD.append(d.row(i)[0])
            ##B[i]=y[i]
            appB.append(y.row(i)[0])
            appE.append(e.row(i)[0])
            ##C[i]=z[i]
            appC.append(z.row(i)[0])
            appF.append(f.row(i)[0])
            date.append(refinedDate.row(i)[0])
            time.append(str(refinedDatetime[i].time()))

    
##    NeutralSave = pnd.DataFrame(np.transpose([appA,appB,appC]))
##    #print(NeutralSave)
##    #unbalanceDF.columns = unbalanceDF[0]
##
##    nameDF = 'savedneutral.csv'
##
##    NeutralSave.to_csv(nameDF,',')
##
##    NeutralSave = pnd.DataFrame(np.transpose([appD,appE,appF]))
##    #print(NeutralSave)
##    #unbalanceDF.columns = unbalanceDF[0]
##
##    nameDF = 'savedvolt.csv'
##
##    NeutralSave.to_csv(nameDF,',')

            
    ##---printing data cleaning output---###
    points = round(len(A)-len(appA))
    
    if len(appA) != 0:
        
        print('\n{d} points of data were cleaned'.format(d=points))
        print('Meaning {y} of {m} days are unusable data\n'.format(y=round(points/144), m=round(len(A)/144)))

    elif len(appA) == 0:
        print('No available data. It was entirely cleaned.')

    return appA, appB, appC, appD, appE, appF, date, time

def DPIB_Calculation(Pa,Pb,Pc,dValue,a,txrName):
    print('Passed into DPIB')

    def plotDPIB(DPIB,a,txrName,titleA):
        pass
##        if a == 1:
##            time = '2018-2019'
##                
##        elif a != 1:
##            time = '2022-2023'
##        
##        figDPIB = plt.figure()
##        
##        xlim1 = 1000 #~1 week = 1000 points
##        plt.xlim([xlim1,xlim1+250])
##        
##        plt.plot(DPIB)
##
##        plt.title('{b}: {c} Sample Imbalance Profile with SIB and RIB Components - {a}'.format(a=titleA,b=txrName,c=time))
##        plt.xlabel('Time, 1 point = 10 minutes')
##        plt.ylabel('Degree of Power Imbalance (DPIB)')
##
##        #plt.savefig('at1/{b}/Decomposition Example/{a}.png'.format(a=txrName+' '+time+' Sample Imbalance',b=txrName),dpi=1200)
##        #plt.savefig('Decomp/{b}/{a}.png'.format(a=txrName+' '+time+' Sample Imbalance',b=txrName),dpi=1200)
##        plt.savefig('Decomp/A. Decomp Images/{a}.png'.format(a=txrName+' '+time+' Sample Imbalance',b=txrName),dpi=1200)
##        
##        plt.close()
##        #plt.show()  
    

    def DPIB_max(Pa,Pb,Pc):

        DPIBmax = np.zeros(len(Pa))
        
        for i in range(len(Pa)):
            DPIBmax[i] = (Pa[i]-(Pa[i]+Pb[i]+Pc[i])/3)/(Pa[i]+Pb[i]+Pc[i]) * 100

        #print(DPIBmax)

        
        return DPIBmax
          

    def DPIB_min(Pa,Pb,Pc):

        DPIBmin = np.zeros(len(Pa))

        for i in range(len(Pa)):
            DPIBmin[i] = ((Pa[i]+Pb[i]+Pc[i])/3-Pc[i])/(Pa[i]+Pb[i]+Pc[i]) * 100

        #print(DPIBmin)
            
        return DPIBmin

 
    if dValue == 1:

        DPIBres = np.zeros(len(Pa))
        
        DPIBmax = DPIB_max(Pa,Pb,Pc)
        DPIBmin = DPIB_min(Pa,Pb,Pc)
        titleA = 'Definite-Order'

        for i in range(len(DPIBmax)):
            DPIBres[i] = np.sqrt(DPIBmax[i]**2+DPIBmax[i]**2)

        DPIB = np.array([DPIBres,DPIBmax,DPIBmin]).T

    elif dValue == 2:

        DPIB = DPIB_max(Pa,Pb,Pc)
        titleA = 'Definite-Max'

    elif dValue == 3:

        DPIB = DPIB_max(Pa,Pb,Pc)
        titleA = 'Definite-Min'

    #print(DPIB)
    
    plotDPIB(DPIB,a,txrName,titleA)
    
    return DPIB


def currentImbalanceDecomposition(Pa,Pb,Pc,txrName,a):
    
    # This method is decomposes and determines whether the system is in a
    # definite-max, definite-min, defienite-order or random imbalance scenario.
    # This informs as to conditions that are effecting or causing the unbalance.

    def plotTestImbalance(SIB,RIB,Px,Py,Pz,dValue,txrName,a):

        return 
##        fig = plt.figure(figsize=(16,9))
##        time = "24-25"
##        titleA = 'Random'
##        if dValue == 1:
##            titleA = 'Definite-Order'
##        elif dValue == 2:
##            titleA = 'Definite-Max'
##        elif dValue == 3:
##            titleA = 'Definite-Min'
##        else:
##            titleA = 'Random'
##
##           
##        plt.title('{b}: Sample Power Profile with SIB and RIB Components - {a}'.format(a=titleA,b=txrName))
##
##        xlim1 = 500 #~1 week = 1000 points
##        plt.xlim([xlim1,xlim1+250]) 
##        
##        plt.plot(np.array([Px,Py,Pz]).T/1000,label=['Non-decomposed Power A','Non-decomposed Power B','Non-decomposed Power C'])
##        plt.plot(SIB/1000,label=['Systematic Imbalance A','Systematic Imbalance B','Systematic Imbalance C'])
##        plt.plot(RIB/1000,label=['Random Imbalance A','Random Imbalance B','Random Imbalance C'])
##        
##        plt.legend()
##        plt.xlabel('Time, 1 point = 10 minutes')
##        plt.ylabel('Power, in kVA')
##        
##        #plt.savefig('at1/{b}/Decomposition Example/{a}.png'.format(a=txrName+' '+time+' Sample Profile',b=txrName),dpi=1200)
##        #plt.savefig('Decomp/{b}/{a}.png'.format(a=txrName+' '+time+' Sample Profile',b=txrName),dpi=1200)
##        plt.savefig('{a}.png'.format(a=txrName+' '+time+' Sample Profile',b=txrName),dpi=1200)#Decomp/A. Decomp Images/     <--- this was removed
##                   
##        plt.close()
##        plt.show()
##
##        return a

    def decomposition(Px,Py,Pz,dValue,txrName,a):
    # Please see 'Three Phase Power Imbalance Decomposition
    # into Systematic Imbalance and Random Imbalance' by Wangwei Kong,
    # Kang Ma, and Qiuwei Wu

        
        def defOrdDecomp(Px,Py,Pz,txrName,a):
            print('passed into def defOrdDecomp()')

            ##----initialisation of SIB, RIB comps.----##
            P2s = np.zeros(len(Px))
            P3s = np.zeros(len(Px))

            P2r = np.zeros(len(Px))
            P3r = np.zeros(len(Px))

            ##----Systematic and Random Imbalance Decomposition----##
            for t in range(len(Px)):
            
                P2s[t] = min(Px[t],Py[t])
                P3s[t] = min(Px[t],Py[t],Pz[t])

                P2r[t] = max(0,Py[t]-Px[t])
                P3r[t] = max(0,Pz[t]-Px[t],Pz[t]-Py[t])

            SIB = np.array([Px,P2s,P3s]).T
            RIB = np.array([np.zeros(len(P2r)),
                            P2r,P3r]).T

            a = plotTestImbalance(SIB,RIB,Px,Py,Pz,dValue,txrName,a)
            return SIB, RIB, a

        def defMaxDecomp(Px,Py,Pz,txrName,a):
            print('passed into def defMaxDecomp()')

            ##----initialisation of SIB, RIB comps.----##
            P2s = np.zeros(len(Px))
            P3s = np.zeros(len(Px))

            P2r = np.zeros(len(Px))
            P3r = np.zeros(len(Px))

            ##----Systematic and Random Imbalance Decomposition----##
            for t in range(len(Px)):
            
                P2s[t] = min(Px[t],Py[t])
                P3s[t] = min(Px[t],Pz[t])

                P2r[t] = max(0,Py[t]-Px[t])
                P3r[t] = max(0,Pz[t]-Px[t])

            SIB = np.array([Px,P2s,P3s]).T
            RIB = np.array([np.zeros(len(P2r)),
                            P2r,P3r]).T

            a = plotTestImbalance(SIB,RIB,Px,Py,Pz,dValue,txrName,a)
            return SIB, RIB, a

        def defMinDecomp(Px,Py,Pz,dValue,txrName,a):
            print('passed into def defMinDecomp()')

            ##----initialisation of SIB, RIB comps.----##
            P3s = np.zeros(len(Px))

            P3r = np.zeros(len(Px))

            ##----Systematic and Random Imbalance Decomposition----##
            for t in range(len(Px)):
            
                P3s[t] = min(Px[t],Py[t],Pz[t])

                P3r[t] = max(0,Pz[t]-Px[t],Pz[t]-Py[t])

            SIB = np.array([Px,Py,P3s]).T
            RIB = np.array([np.zeros(len(P2r)),
                            np.zeros(len(P2r)),
                            P3r]).T
            
            
            a = plotTestImbalance(SIB,RIB,Px,Py,Pz,dValue,txrName,a)
            
            return SIB, RIB, a

       
        if dValue == 1:
            SIB, RIB, a = defOrdDecomp(Px,Py,Pz,txrName,a)

        if dValue == 2:
            SIB, RIB, a = defMaxDecomp(Px,Py,Pz,txrName,a)

        if dValue == 3:
            SIB, RIB, a = defMinDecomp(Px,Py,Pz,txrName,a)

        return SIB, RIB

    #----if elif catchers----#

    print('passed into cIB():')
          
    d = 0.02 # <- Academic paper recommends 5% measurement error
    tHold = 0.50

    mxa = 0
    mxb = 0
    mxc = 0

    mna = 0
    mnb = 0
    mnc = 0

    for i in range(len(Pa)):
    #----definite-max----#

        if Pa[i] == max(Pa[i],Pb[i]*(1+d),Pc[i]*(1+d)):
            mxa += 1
               
        elif Pb[i] == max(Pb[i],Pa[i]*(1+d),Pc[i]*(1+d)):
            mxb += 1
  
        elif Pc[i] == max(Pc[i],Pa[i]*(1+d),Pb[i]*(1+d)):
            mxc += 1

    #----definite-min----#
            
        elif Pa[i] == min(Pa[i],Pb[i]*(1+d),Pc[i]*(1+d)):
            mna += 1
            
        elif Pb[i] == min(Pb[i],Pa[i]*(1+d),Pc[i]*(1+d)):
            mnb += 1
            
        elif Pc[i] == min(Pc[i],Pa[i]*(1+d),Pb[i]*(1+d)):
            mnc += 1

    maxList = [['Max A', 'Max B', 'Max C'],[mxa, mxb, mxc],['Pa','Pb','Pc']]
    minList = [['Min A', 'Min B', 'Min C'],[mna, mnb, mnc],['Pa','Pb','Pc']]


    #maxList = [['mxa', 'mxb', 'mxc'],[50, 25, 1000],['Pa','Pb','Pc']]
    #minList = [['mna', 'mnb', 'mnc'],[152, 180, 53],['Pa','Pb','Pc']]
    #total = 1500

    print(maxList)
    print(minList)
    
    total = mxa+mxb+mxc+mna+mnb+mnc
    
    Lc = 0      # <- latch for testing
    k = 0

    Px = 0
    Py = 0
    Pz = 0

    arrP = [np.mean(Pa),np.mean(Pb),np.mean(Pc)]
    
    for m in range(3):
        for n in range(3):       
            print(str(round((maxList[1][m]+minList[1][n])/total*100,1))+'%')

    
    for m in range(3):

        #arrPmax = ['Pa','Pb','Pc']
        arrPmax = [np.mean(Pa),np.mean(Pb),np.mean(Pc)]
        arrLf = arrPmax[:m]+arrPmax[m+1:]
        arrSl = arrPmax[m]
   
        for n in range(3):

            arrPmin = [np.mean(Pa),np.mean(Pb),np.mean(Pc)]
            arrLf2 = arrPmin[:n]+arrPmin[n+1:]
            arrSl2 = arrPmin[n]
                       
            #print('sl1 '+str(arrPmax[m])+' and '+maxList[0][m])             
            #print('sl2 '+str(arrPmin[n])+' and '+minList[0][n])
          
            if (maxList[1][m]+minList[1][n])/total >= tHold and \
               arrSl == max(arrSl,(1+d)*arrLf[0],(1+d)*arrLf[1]) and \
               arrSl2 == min(arrSl2,(1-d)*arrLf2[0],(1-d)*arrLf2[1]):

                print(maxList[0][m]+' and '+minList[0][n]+' Definite-Order Scenario')
                print(str(round((maxList[1][m]+minList[1][n])/total*100,1))+'%')
                Lc = 1
                print(str(maxList[2][m]))

                if maxList[2][m] == 'Pa':
                    Px = Pa
                    if minList[2][n] == 'Pb':
                        Py = Pc
                        Pz = Pb
                    if minList[2][n] == 'Pc':
                        Py = Pb
                        Pz = Pc
                        
                if maxList[2][m] == 'Pb':
                    Px = Pb
                    if minList[2][n] == 'Pa':
                        Py = Pc
                        Pz = Pa
                    if minList[2][n] == 'Pc':
                        Py = Pa
                        Pz = Pc
                        
                if maxList[2][m] == 'Pc':
                    Px = Pc
                    if minList[2][n] == 'Pa':
                        Py = Pb
                        Pz = Pa
                    if minList[2][n] == 'Pb':
                        Py = Pa
                        Pz = Pb  

                dValue = 1
                
                maxP = round(np.median(Px),1)
                minP = round(np.median(Pz),1)

                scenario = 'Definite-Order: {a} and {b}'.format(a=maxList[0][m],b=minList[0][n])
                print(scenario)

                pFit = round((maxList[1][m]+minList[1][n])/total*100,2)
                
                    
                SIB, RIB = decomposition(Px,Py,Pz,dValue,txrName,a)
                DPIB = DPIB_Calculation(Px,Py,Pz,dValue,a,txrName)
                DPIB = DPIB.T[0]
                
                ## maybe seperate DPIB and only take the resultant?
                
                break 

    for m in range(3):

        arrPmax = [np.mean(Pa),np.mean(Pb),np.mean(Pc)]
        arrLf = arrPmax[:m]+arrPmax[m+1:]
        arrSl = arrPmax[m]
        
        for n in range(3):

            if maxList[1][m]/total >= tHold and maxList[1][m] == max(maxList[1]) and \
                       arrSl == max(arrSl,(1+d)*arrLf[0],(1+d)*arrLf[1]):
                
                if Lc == 1:
                    break

                print(maxList[0][m]+' Definite-Maximum Scenario')
                print(maxList[1][m]/total)
                Lc = 1

                dValue = 2

                if maxList[2][m] == 'Pa':
                    Px = Pa
                    Py = Pb
                    Pz = Pc

                if maxList[2][m] == 'Pb':
                    Px = Pb
                    Py = Pa
                    Pz = Pc

                if maxList[2][m] == 'Pc':
                    Px = Pc              
                    Py = Pa
                    Pz = Pb

                maxP = round(np.median(Px),1)
                minP = round(np.median(Pz),1)

                scenario = 'Definite-Max: {a}'.format(a=maxList[0][m])
                print(scenario)

                pFit = round(maxList[1][m]/total*100,1)
                    
                SIB, RIB = decomposition(Px,Py,Pz,dValue,txrName,a)
                DPIB = DPIB_Calculation(Px,Py,Pz,dValue,a,txrName)
               
            if minList[1][n]/total >= tHold and minList[1][n] == max(minList[1]) and \
                       arrSl == min(arrSl,(1-d)*arrLf[0],(1-d)*arrLf[1]):

                if Lc == 1:
                    break
            
                print(minList[0][m]+' Definite-Minimum Scenario')
                print(minList[1][n]/total)   
                Lc = 1

                dValue = 3

                if minList[2][m] == 'Pc':
                    Px = Pa
                    Py = Pb
                    Pz = Pc

                if maxList[2][m] == 'Pb':
                    Px = Pa
                    Py = Pc
                    Pz = Pb

                if maxList[2][m] == 'Pa':
                    Px = Pb              
                    Py = Pc
                    Pz = Pa

                maxP = round(np.median(Px),1)
                minP = round(np.median(Pz),1)

                scenario = 'Definite-Min: {b}'.format(b=minList[0][n])
                
                print(scenario)

                pFit = round(minList[1][n]/total*100,1)
                    
                SIB, RIB = decomposition(Px,Py,Pz,dValue,txrName,a)
                DPIB = DPIB_Calculation(Px,Py,Pz,dValue,a,txrName)

                

            else:

                if Lc == 1:
                    break
                elif k == 8:
                    print('Random scenario: Cannot be decomposed')

                    SIB = np.array([np.zeros(len(Pa)),np.zeros(len(Pa)),np.zeros(len(Pa))]).T
                    RIB = np.array([np.zeros(len(Pa)),np.zeros(len(Pa)),np.zeros(len(Pa))]).T
                    DPIB = [850653]

                    #for i in range(6):
                    #
                    minFit = np.array(['Minimum:',str(round(max(minList[1][0]/total,minList[1][1]/total,minList[1][2]/total)*100,1))])
                    maxFit = np.array(['Maximum:',str(round(max(maxList[1][0]/total,maxList[1][1]/total,maxList[1][2]/total)*100,1))])
                    ordFit = np.array(['Order:',str(round(max((maxList[1][0]-minList[1][0])/total,(maxList[1][1]-minList[1][0])/total,
                                                       (maxList[1][2]-minList[1][0])/total,(maxList[1][0]-minList[1][1])/total,
                                                       (maxList[1][1]-minList[1][1])/total,(maxList[1][2]-minList[1][1])/total,
                                                       (maxList[1][0]-minList[1][2])/total,(maxList[1][1]-minList[1][2])/total,
                                                       (maxList[1][2]-minList[1][2])/total)*100,1))])

                    fitList = np.array([minFit,maxFit,ordFit])

                    randomFit = np.array(['Maximum:','100'])
                    for h in range(3):
                        if fitList[h][1] == max(fitList.T[1]):
                            
                            randomFit[0] = fitList[h][0]
                            randomFit[1] = fitList[h][1]

                    pFit = str(randomFit[1])

                    maxP = round(max(np.median(Pa),np.median(Pb),np.median(Pc)),1)
                    minP = round(min(np.median(Pa),np.median(Pb),np.median(Pc)),1)

                    scenario = 'Random: '+randomFit[0]
                    print('random test')
                    print(a)

                    dValue = 0
                    
                    plotTestImbalance(SIB,RIB,Pa,Pb,Pc,dValue,txrName,a)

                    

                k += 1

    if DPIB[0] == 850653:
        DPIBmed = 'n/a'
        SIBmed = 'n/a'
        RIBmed = 'n/a'
        
    else:
        DPIBmed = round(np.nanmedian(DPIB),2)
        SIBmed = round(np.nanmedian(SIB),1)
        RIBmed = round(np.nanmedian(RIB),1)
        
    return SIBmed, RIBmed, DPIBmed, maxP, minP, scenario, pFit


def neutralCurrent(x,y,z):
    # This function calculates the neutral current of the system (at the PQ monitor).

    if len(A) == 0:
        In = np.zeros(len(x))
        print('Neutral calculation failed')
        return In

    ##---calculating neutral current---##
    In = np.zeros(len(x))

    for i in range(len(x)):
        In[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2 - x[i]*y[i] - y[i]*z[i] - z[i]*x[i])    ## <--- Equation assumes no non-resistive load (phase angles are 0,-120,120). For a non-resistive load, 
                                                                                            ##      phase angle measurements must be used. However, these are greatly effected by PV DER, i.e., unreliable.
    return In





##-----------------main script-----------------##
# View individual functions/methods for description

##---intialising matrix for percentiles---##
decompList = np.zeros((1,17),dtype=object)
decompList[0] = ['No. in DF23','TXR Name','Type','50th I Max Phase','50th I Min Phase','% Fit','DPIB Med. (%)','SIB Med. (A)','RIB Med. (A)','50th Ia',
                 '50th Ib','50th Ic','50th In','95th Ia','95th Ib','95th Ic','95th In']

##---intialising counters for main script---##
pCount = 0
failed = 0
i = 0
recordI = i
start = 0

valueForInput = 100

end = 2#valueForInput - i
g=0

##---main script prints for number of sites given---##
for start in range(end):
    
    i, failed = indataCheck(i,failed)

    txrName, parPath = pathGrab(i)


    try:
        filePath = feederGrab(txrName,parPath)
    except TypeError:
        failedText = 'type error'
        arrDecompList = np.array([i,txrName,failedText,failedText,failedText,failedText,failedText,failedText,failedText,failedText]).astype(str)
        print('Failed due to TypeError\n')
        i+=1
        continue
        
        
    try:
        dataMatrix = dataGrab(filePath,i)
    except AttributeError:
        print('atrribute error')
        #continue
        i+=1

   

    genCount = 0
    gCount=0
    In = 0
    try:
        ##---prints for number of rows in data matrix, aka number of tags from PI---##    
        for g in range(1):      #changed to 1 from 2 (2 did 2 iterations for first finacial yr and then second fin. yr)

            A, B, C, Ia, Ib, Ic, date, time = extract_3ph(dataMatrix[g][0],dataMatrix[g][1],dataMatrix[g][2],
                                                       dataMatrix[g+1][0],dataMatrix[g+1][1],dataMatrix[g+1][2],genCount)

            #vNi = [[Va,Vb,Vc],[Ia,Ib,Ic]]
            #print(vNi)
            #for p in range(2):        
            #print(g)
            a = g+1

                
            #if genCount >1:
            if A != []:

                # This says power, but I am actually passing current by not including the equation
                Pa = np.array(Ia) #3*np.array(A)*
                Pb = np.array(Ib)      # <--- 3*Vph*Iph = S (apparent power) is it not sqr root 3*Vph*Iph?
                Pc = np.array(Ic) #3*np.array(C)* 3*np.array(B)*
                

                SIB, RIB, DPIB, maxP, minP, scenario, pFit = currentImbalanceDecomposition(Pa,Pb,Pc,txrName,a)

                In = neutralCurrent(Pa,Pb,Pc)
            
                        
            genCount += 2

            if a == 1:
                time = '2018-2019'          ### ADD COLUMN FOR TOTAL MEDIAN LOAD TO TEST IF LOAD HAS INCREASED
                
            elif a != 1:
                time = '2022-2023'

            arrDecompList = np.array([i,txrName,scenario,maxP,minP,pFit,DPIB,SIB,RIB,
                                      np.round(np.nanmedian(Pa),1),np.round(np.nanmedian(Pb),1),np.round(np.nanmedian(Pc),1),np.round(np.nanmedian(In),1),
                                      np.round(np.nanpercentile(Pa,95),1),np.round(np.nanpercentile(Pb,95),1),np.round(np.nanpercentile(Pc,95),1),
                                      np.round(np.nanpercentile(In,95),1)]).astype(str)
            #print(arrDecompList)

            decompList = np.vstack((decompList,arrDecompList))
            print('List updated: '+str(i))
            #print(decompList)
                
        print('\n---------------------------next monitor---------------------------\n')
        
        i+=1
    except pa.lib.ArrowInvalid:
        print('Failed due to pyarrow exception\n')
        i+=1
    

##---printing summary of program---##
print('List of Decomposition')

#       decompDF = pnd.DataFrame(decompList);nameDF = 'VIJI TEST {a} to {b}.csv'.format(a=recordI,b=i);decompDF.to_csv(nameDF,',')

decompDF = pnd.DataFrame(decompList)

nameDF = 'Grujica 24-25 Component Decomposition_{a} to {b}.csv'.format(a=recordI,b=i)

decompDF.to_csv(nameDF,',')

print(decompDF)

print()

print(str(start+1)+' filepaths accessed')

print(str(failed)+' filepaths failed to form')
    
