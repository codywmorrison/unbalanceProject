# Written by Cody Morrison (Vac. Student) for Energy Queensland under Trevor Gear (Principal Engineer)
#
# This script uses an interface with Osisoft's PI to pull current and voltage data from
# a list of PQ monitor sites and calculate the PCUR.
#
# This script was written as part of a student placement with Energy Queensland over the
# university's vacation period of 2023/2024.
#
# V2 - 18/1/23


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

        siteDF23 = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(11),skip_header=1)

        sliceNo = len(siteDF23[i])-4
        
        yearInt = int(siteDF23[i][sliceNo:])
        print(sliceNo,yearInt)

        startT1 = '{c}-07-01 00:00:00'.format(c = yearInt-2)
        endT1 = '{d}-6-30 00:00:00'.format(d = yearInt-1)

        startT2 = '{e}-07-01 00:00:00'.format(e = yearInt+1)
        endT2 = '{f}-6-30 00:00:00'.format(f = yearInt+2)
        
##        startT1 = '2018-07-02 00:00:00'
##        endT1 = '2019-6-30 00:00:00'
##
##        startT2 = '2022-07-02 00:00:00'
##        endT2 = '2023-6-30 00:00:00'


        ##---search and assign Voltage and Current data to matrix---##
        for att in range(len(attList)):

            if attList[att].name == 'CUR_C':
                CdataCa = attList[att].interpolated_values(startT1,endT1,intT)
                CdataCb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_B':
                CdataBa = attList[att].interpolated_values(startT1,endT1,intT)
                CdataBb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_A':
                CdataAa = attList[att].interpolated_values(startT1,endT1,intT)
                CdataAb = attList[att].interpolated_values(startT2,endT2,intT)        
        
        dataMatrix = [[CdataCa,CdataBa,CdataAa],[CdataCb,CdataBb,CdataAb]]
        print(dataMatrix)

        return dataMatrix

def feederGrab(txrN,monN,parPath):
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

            ##---transformer being searched---##            
            txrSRCH = database.descendant(parPath+'\\'+fdrStr2)
            for txr in txrSRCH.children.values():
                
                txrStr = str(txr)
                txrStr1 = txrStr.replace('PIAFElement(','')
                txrStr2 = txrStr1.replace(')','')

                ##---building full filepath for PI to read---##
                if txrStr2==txrN:

                    fPath = parPath+'\\'+fdrStr2+'\\'+txrStr2+'\\'+monN
                    print('Found monitor filepath: '+fPath) ##add \n to format with searc print

                    return fPath
                

def pathGrab(k):
    # This function forms a partial filepath and passes it to the above function
    # (feederGrab) so that a full filepath can be passed to PIConnect (dataGrab) function.
    # This function uses the DF23 csv file to concatenate a filepath.

    ##---pulling sub, region and txr name from the csv file---##
    siteDF23 = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(5,7,8,9,10),skip_header=1) ##2018UnbalancedSites.csv, DFListV1.csv

    eqlName = "EQL"
    regName = siteDF23[k][4]
    locName = siteDF23[k][2]
    subName = siteDF23[k][1]
    txrName = siteDF23[k][0]
    monName = siteDF23[k][3]

    ##---forming/concatenating the filepath---##
    oPath = "\\".join([eqlName,regName,locName,subName])
    parPath = r'{}'.format(oPath)
    
    #print('\nFound partial filepath from CSV: '+parPath+'\n')
    
    return txrName,parPath,monName

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
        
        if round(x.row(i)[0],2) != round(x.row(i-1)[0],2) or round(x.row(i)[0]-x.row(i-1)[0],2) != round(x.row(i)[0]-x.row(i-2)[0],2):
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

def PCUR_Calculation(Ia,Ib,Ic,Id,Ie,If):
    # This function calculates the phase voltage unbalance rate, PVUR (NEMA / IEEE standard)
    # and voltage unbalance factor, VUF (IEEE 'True' standard), as well as, their distribution
    # and percentiles. It then calls distribution and profile graphing functions to print a
    # visual representation of how the unbalance in this system is acting.

    print('passed into PCUR')
    PCUR1 = np.array([[np.zeros(len(A))],[np.zeros(len(A))]])
    PCUR2 = np.array([[np.zeros(len(A))],[np.zeros(len(A))]])
    ##-----------------Phase Voltage Unbalance Rate, PVUR-----------------##
    Iall = np.hstack((np.reshape(np.array(Ia),(len(Ia),1)),np.reshape(np.array(Ib),(len(Ib),1)),np.reshape(np.array(Ic),(len(Ic),1))))
   
    Iavg = np.average(Iall,axis=1)
    Imax = np.max(Iall,axis=1) - Iavg
    Imin = np.abs(np.min(Iall,axis=1) - Iavg)

    Imaxmin = np.transpose(np.array([Imax,Imin]))
    Imaxdev = np.max(Imaxmin,axis=1)
    PCUR1 = (Imaxdev)*100 / (Iavg)

    Iall = np.hstack((np.reshape(np.array(Id),(len(Id),1)),np.reshape(np.array(Ie),(len(Ie),1)),np.reshape(np.array(If),(len(If),1))))
   
    Iavg = np.average(Iall,axis=1)
    Imax = np.max(Iall,axis=1) - Iavg
    Imin = np.abs(np.min(Iall,axis=1) - Iavg)

    Imaxmin = np.transpose(np.array([Imax,Imin]))
    Imaxdev = np.max(Imaxmin,axis=1)
    PCUR2 = (Imaxdev)*100 / (Iavg)

    print(PCUR1,PCUR2)

    return PCUR1,PCUR2

##-----------------main script-----------------##
# View individual functions/methods for description

##---intialising matrix for percentiles---##
decompList = np.zeros((1,5),dtype=object)
decompList[0] = ['No. in DF23','TXR Name','Year','Type','PCUR %']

##---intialising counters for main script---##
pCount = 0
failed = 0
i = 481
recordI = i
start = 0
end = 70
g=0

##---main script prints for number of sites given---##
for start in range(end):
    
    i, failed = indataCheck(i,failed)

    txrName, parPath, monName = pathGrab(i)

    filePath = feederGrab(txrName,monName,parPath)

    try:
        dataMatrix = dataGrab(filePath,i)
    except AttributeError:
        print('atrribute error')
        i+=1

    genCount = 0
    gCount=0
    In = 0
    try:
        ##---prints for number of rows in data matrix, aka number of tags from PI---##    
        A, B, C, D, E, F, date, time = extract_3ph(dataMatrix[0][0],dataMatrix[0][1],dataMatrix[0][2],
                                                   dataMatrix[1][0],dataMatrix[1][1],dataMatrix[1][2],genCount)

        vNi = np.array([[A,B,C],[D,E,F]])
        #print(vNi)
        #for p in range(2):        
        #print(g)
        a = g+1


        
        PCUR1,PCUR2 = PCUR_Calculation(vNi[0][0],vNi[0][1],vNi[0][2],vNi[1][0],vNi[1][1],vNi[1][2])    
    
        print('PCUR')
        print(PCUR1,PCUR2)

        PCURpre = round(np.nanmedian(PCUR1),2)
        PCURpost = round(np.nanmedian(PCUR2),2)
        
        
        time1 = '2018-2019'          ### ADD COLUMN FOR TOTAL MEDIAN LOAD TO TEST IF LOAD HAS INCREASED
        
        time2 = '2022-2023'
        
        scenario = 'PCUR'


        arrDecompList = np.array([i,txrName,time1,scenario,PCURpre]).astype(str)
        arrDecompList2 = np.array([i,txrName,time2,scenario,PCURpost]).astype(str)
        #print(arrDecompList)

        decompList = np.vstack((decompList,arrDecompList,arrDecompList2))
        print('List updated: '+str(i))
        #print(decompList)
                
        print('\n---------------------------next monitor---------------------------\n')
        
        i+=1
    except pa.lib.ArrowInvalid:
        print('Failed due to pyarrow exception\n')
        i+=1
    except:
        print('Failed due to unknown exception\n')
        i+=1


##---printing summary of program---##
print('List of Decomposition')

decompDF = pnd.DataFrame(decompList)
#unbalanceDF.columns = unbalanceDF[0]
#decompDF = decompDF[1:]

nameDF = 'V9 YearSet PCUR_{a} to {b}.csv'.format(a=recordI,b=i)

#unbalanceDF.write_csv('testingpolars.csv',separator=',')

decompDF.to_csv(nameDF,',')

print(decompDF)

print()

print(str(start+1)+' filepaths accessed')

print(str(failed)+' filepaths failed to form')
    
