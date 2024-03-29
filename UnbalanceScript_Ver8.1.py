# Written by Cody Morrison (Vac. Student) for Energy Queensland under Trevor Gear (Principal Engineer)
#
# This script uses an interface with Osisoft's PI to pull current and voltage data from
# a list of PQ monitor sites that have had a history of voltage unbalance (from 2018/2019).
# The data is processed to calculate/estimate the system's PVUR, VUF and the neutral current
# which is then graphed, printed and saved.
#
# This script was written as part of a student placement with Energy Queensland over the
# university's vacation period of 2023/2024.
#
# Neutral current is calculated under the assumption that the PF for each phase is resistive. Meaning,
# no phase angle is needed for evaluating the neutral wire's current and it is determined with assumed
# angles of 0, 120 and -120 (240) degrees.
#
# V9 - 16/1/23


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


def dataGrab(fPath,i):
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

        #print(b)
        


        startT1 = '{c}-07-01 00:00:00'.format(c = yearInt-2)
        endT1 = '{d}-6-30 00:00:00'.format(d = yearInt-1)

        startT2 = '{e}-07-01 00:00:00'.format(e = yearInt+1)
        endT2 = '{f}-6-30 00:00:00'.format(f = yearInt+2)

        
        #startT1 = '2018-07-02 00:00:00'
        # endT1 = '2019-6-30 00:00:00'
        
##        startT1 = '2017-07-02 00:00:00'
##        endT1 = '2018-6-30 00:00:00'
##
##        startT2 = '2022-07-02 00:00:00'
##        endT2 = '2023-6-30 00:00:00'


        ##---search and assign Voltage and Current data to matrix---##
        for att in range(len(attList)):

            if attList[att].name == 'VOLT_C':
                VdataCa = attList[att].interpolated_values(startT1,endT1,intT)
                VdataCb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'VOLT_B':
                VdataBa = attList[att].interpolated_values(startT1,endT1,intT)
                VdataBb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'VOLT_A':
                VdataAa = attList[att].interpolated_values(startT1,endT1,intT)
                VdataAb = attList[att].interpolated_values(startT2,endT2,intT)

            if attList[att].name == 'CUR_C':
                CdataCa = attList[att].interpolated_values(startT1,endT1,intT)
                CdataCb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_B':
                CdataBa = attList[att].interpolated_values(startT1,endT1,intT)
                CdataBb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_A':
                CdataAa = attList[att].interpolated_values(startT1,endT1,intT)
                CdataAb = attList[att].interpolated_values(startT2,endT2,intT)        
        
        dataMatrix = [[VdataCa,VdataBa,VdataAa],[VdataCb,VdataBb,VdataAb],[CdataCa,CdataBa,CdataAa],[CdataCb,CdataBb,CdataAb]]
        #[[VdataCa,VdataBa,VdataAa],[CdataCa,CdataBa,CdataAa],[VdataCb,VdataBb,VdataAb],[CdataCb,CdataBb,CdataAb]]

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
        elif round(x.row(i)[0],2) != round(x.row(i-1)[0],2) or round(x.row(i)[0]-x.row(i-1)[0],2) != round(x.row(i)[0]-x.row(i-2)[0]):#(type(x[i],2) is np.float64 or type(x[i]) is float) and####(x[i] > 120) and
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

##    if genCount > 1:
##        NeutralSave = pnd.DataFrame(np.transpose([appA,appB,appC]))
##        print(NeutralSave)
##        #unbalanceDF.columns = unbalanceDF[0]
####
##        nameDF = 'savedneutral.csv'
####
##        NeutralSave.to_csv(nameDF,',')
##    
##        NeutralSave = pnd.DataFrame(np.transpose([appD,appE,appF]))
##        print(NeutralSave)
##        #unbalanceDF.columns = unbalanceDF[0]
####
##        nameDF = 'savedvolt.csv'
####
##        NeutralSave.to_csv(nameDF,',')

            
    ##---printing data cleaning output---###
    points = round(len(A)-len(appA))
    
    if len(appA) != 0:
        
        print('\n{d} points of data were cleaned'.format(d=points))
        print('Meaning {y} of {m} days are unusable data\n'.format(y=round(points/144), m=round(len(A)/144)))

    elif len(appA) == 0:
        print('No available data. It was entirely cleaned.')
    
    ## maybe check if at any point it was because of 'No Data' or some other cleanse in the future and state what type of data was cleaned

    return appA, appB, appC, appD, appE, appF, date, time

def assign_to_unbalanceList(unbalance,c,k,txrN,ubtype,unbalanceList,monN):
    # This function assigns the calculated percenetile values to the finalList to be passed into
    # the main script to be printed at the end of the program.

    ##---intialisation and calculation---##
    percentileArr = np.zeros(7)
    percentiles = [1,5,25,50,75,95,99]
    
    for i in range(7):
        percentileArr[i] = round(np.percentile(unbalance,percentiles[i]),2)
        print(percentileArr[i])
        percentileArr[i] = round(np.nanpercentile(unbalance,percentiles[i]),2)
        print(percentileArr[i])

    ##---assigning to unbalanceList---##

    year = '2018 to 2019'
    if genCount == 1 or genCount == 3:
        year = '2022 to 2023'
    noDF = 'No. ' + str(k)

    unbalanceList = np.append(unbalanceList,[[noDF,txrN,monN,year,ubtype,percentileArr[0],percentileArr[1],percentileArr[2],
                             percentileArr[3],percentileArr[4],percentileArr[5],percentileArr[6]]],axis=0)

    
    
    median = percentileArr[3]
    
    return unbalanceList, median

def calcUnbalance(Va,Vb,Vc,k,c,txrN,gCount,genCount,unbalanceList,monN):
    # This function calculates the phase voltage unbalance rate, PVUR (NEMA / IEEE standard)
    # and voltage unbalance factor, VUF (IEEE 'True' standard), as well as, their distribution
    # and percentiles. It then calls distribution and profile graphing functions to print a
    # visual representation of how the unbalance in this system is acting.
    
    ##-----------------Phase Voltage Unbalance Rate, PVUR-----------------##
    Vall = np.hstack((np.reshape(np.array(Va),(len(Va),1)),np.reshape(np.array(Vb),(len(Vb),1)),np.reshape(np.array(Vc),(len(Vc),1))))
   
    Vnom = 230
    Vavg = np.average(Vall,axis=1)
    Vmax = np.max(Vall,axis=1) - Vavg
    Vmin = np.abs(np.min(Vall,axis=1) - Vavg)

    Vmaxmin = np.transpose(np.array([Vmax,Vmin]))
    Vmaxdev = np.max(Vmaxmin,axis=1)
    LVUR = (Vmaxdev)*100 / (Vavg)


    ##---assigning percentiles---##
    ##this was shifted to assign_to_unbalanceList()
    
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
        
    ##---assigning percentiles---##
##    onePercentileVUF = round(np.nanpercentile(VufNeg,1),2)
##    twofivePercentileVUF = round(np.nanpercentile(VufNeg,25),2)
##    medianVUF = round(np.nanpercentile(VufNeg,50),2)
##    sevenfivePercentileVUF = round(np.nanpercentile(VufNeg,75),2)
##    nineninePercentileVUF = round(np.nanpercentile(VufNeg,99),2)

    ubtypeA = 'PVUR'
    ubtypeB = 'VUF'

    ##if (median > 0.5 and gCount == 0) or gCount > 0:
    

    ##---assigning percentiles to the list for the summary---##
    if genCount > 1:
        ubtypeA = 'PCUR'

    for ii,kl in [[LVUR,ubtypeA],[VufNeg,ubtypeB]]:
        unbalanceList,median = assign_to_unbalanceList(ii,c,k,txrN,kl,unbalanceList,monN)

    unbalanceDistribution(median,LVUR,txrN,ubtypeA,genCount)
    distributionProfile(Va,Vb,Vc,txrName,genCount)
    gCount += 1
    
    return VufNeg, unbalanceList, c, LVUR,gCount


def distributionProfile(Va,Vb,Vc,txrN,genCount):
##    # This function calculates the phase voltage unbalance rate, PVUR (NEMA / IEEE standard)
##    # and voltage unbalance factor, VUF (IEEE 'true' standard), as well as, their distribution
##    # and percentiles. It then calls distribution and profile graphing functions to print a
##    # visual representation of how the unbalance in this system is acting.
##
##    ##---intialising pu data and skipping for empty arrays---##
##    Apu = np.zeros(len(Va))
##    Bpu = np.zeros(len(Va))
##    Cpu = np.zeros(len(Va))
##    
##    if len(Va) == 0:
##        return
##
##    ##---coneversion to pu for voltage data---#
##    if genCount < 2:
##        for i in range(len(Va)):
##            Apu[i] = Va[i]/230
##            Bpu[i] = Vb[i]/230
##            Cpu[i] = Vc[i]/230
##    else:
##        for i in range(len(Va)):
##            Apu[i] = Va[i]
##            Bpu[i] = Vb[i]
##            Cpu[i] = Vc[i]
##
##    ##---density curve plotting---##
##    sns.kdeplot([Apu,Bpu,Cpu],legend=False)##,bins=15,kde=True) ##displot? ##clip=(0.95,1.05),
##
##    xV = [(1.1,'r'),(1,'g'),(0.94,'r')]
##    
##    dataEntry = 'Voltage'
##    dataEntry1 = 'Per-Unit Voltage (230V)'
##    if genCount > 1:
##            dataEntry = 'Current'
##            dataEntry1 = dataEntry
##    else:
##        for a,b in xV:
##            plt.axvline(x=a,color=b,linestyle='--')
##    
##    timeEntry = 'a'
##    timeEntry2 = '18-19'
##    if genCount == 1 or genCount ==3:
##            timeEntry = 'b'
##            timeEntry2 = '22-23'
##            
##    plt.title('Three-Phase Distribution of {x}'.format(x=txrN+' '+timeEntry2))
##    plt.xlabel('{a}'.format(a=dataEntry1))
##    plt.savefig('at1/{b}/Distributions/{a}.png'.format(a=txrN+' '+timeEntry2+' '+dataEntry,b=txrN),dpi=1200)
##    plt.close()
##    ##plt.show()

    return

def unbalanceDistribution(median,unbalanceData,txrN,ubtype,genCount):
##    # This function plots and saves figures of unbalance, including PVUR, VUF and neutral
##    # current. The plot is a kernel density distribution (as a density curve, not histogram).
##    
##    ##---density curve plotting---##
##    fig, ax = plt.subplots()
##    sns.kdeplot(unbalanceData,legend=False)
##    ax.axvline(x=median,linestyle='--')
##    ax.text(median,0.99,'Median',ha='left',va='top',rotation=90,transform=ax.get_xaxis_transform())
##    xC = [(50,'r')]#,(25,'y'),(10,'g')]
##    
##    if ubtype == 'Neutral Current':
##        ubtype = 'Neutral Current'
##        dataEntry = 'Neutral Current'
##        
##        for a,b in xC:
##            ax.axvline(x=a,color=b,linestyle='dotted')
##            ax.text(a,0.99,'50A Standard',ha='left',va='top',rotation=90,transform=ax.get_xaxis_transform())
##    else:
##        dataEntry = 'Factor/Percentage Unbalance'
##        if genCount > 1:
##            dataEntry = 'Factor/Percentage Unbalance'
##            ubtype = 'PCUR'
##            ax.axvline(x=median,linestyle='--')
##    
##    timeEntry = 'a'
##    timeEntry2 = '18-19'
##    if genCount == 1 or genCount ==3:
##            timeEntry = 'b'
##            timeEntry2 = '22-23'
##            
##    plt.title('Unbalance ({t}) Distribution of {x}'.format(t=str(ubtype),x=txrN+' '+timeEntry2))
##    plt.xlabel('{a}'.format(a=dataEntry))
##    plt.savefig('at1/{b}/Distributions/{a}.png'.format(a=txrN+' '+timeEntry2+' '+str(ubtype),b=txrN),dpi=1200)
##    plt.close()
##    ##plt.show()

    return

def dataProfile(x,y,z,In,txrN,genCount):
##    # This function plots and saves figures of current and voltage data.
##    # This plots show a continous line graph for the entirety of the year.
##    
##    ##---plotting current/voltage profiles---##
##    dataEntry = 'Voltage'
##    if genCount > 1:
##            dataEntry = 'Current'
##
##    timeEntry = 'a'
##    timeEntry2 = '18-19'
##    if genCount == 1 or genCount ==3:
##            timeEntry = 'b'
##            timeEntry2 = '22-23'
##    
##    plt.plot(np.arange(0,len(x),1),x,label='A')#reshape((len(x),1))
##    plt.plot(np.arange(0,len(y),1),y,label='B')
##    plt.plot(np.arange(0,len(z),1),z,label='C')
##
##
##    plt.legend()
##    plt.title('{a} Profile of {b} '.format(a=dataEntry,b=txrN+' '+timeEntry2))
##    plt.savefig('at1/{b}/Profiles/{a}.png'.format(a=txrN+timeEntry+' '+dataEntry+' Profile',b=txrN),dpi=1200)
##    plt.close()
##
##    ##plt.show()

    return

def neutralCurrent(x,y,z,genCount,txrN,k,c,unbalanceList,date,time,monN):
    # This function calculates the neutral current of the system (at the PQ monitor).
    # The function also calcaultes the distribution and percentiles of the neutral current.
    # It then calls the unbalanceDistribution() function to print and save a distribution figure.

    if len(A) == 0:
        In = np.zeros(len(x))
        medianIn = 0
        print('Neutral calculation failed')
        return In, medianIn, unbalanceList

    ##---calculating neutral current---##
    In = np.zeros(len(x))

    for i in range(len(x)):
        In[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2 - x[i]*y[i] - y[i]*z[i] - z[i]*x[i])    ## <--- Equation assumes no non-resistive load (phase angles are 0,-120,120). For a non-resistive load, 
                                                                                            ##      phase angle measurements must be used. However, these are greatly effected by PV DER, i.e., unreliable.
    #print('type of In')
    #print(In.dtype)
    ##medianIn = round(np.percentile(In,50),2)

    ubtype = 'Neutral Current'

    unbalanceList,medianIn = assign_to_unbalanceList(In,c,k,txrN,ubtype,unbalanceList,monN)
    unbalanceDistribution(medianIn,In,txrN,ubtype,genCount)

    #plot_neutralCurrent(In,date,time,genCount,txrN)

    return In, medianIn, unbalanceList

def plot_neutralCurrent(In,date,time,genCount,txrN):
##    # This function plots a 3D graph of time, date and neutral current.
##
##    ##----Set warnings so they catch----##
##    warnings.filterwarnings("error")
##    
##    #----saving times rules----#
##    
##    timeEntry = 'a'
##    timeEntry2 = '18-19'
##    if genCount == 1 or genCount == 3:
##            timeEntry = 'b'
##            timeEntry2 = '22-23'
##
##
##    ##----intialisation and data arranging----#
##    
##    multi_arr = np.hstack((np.reshape(np.array(date),(len(date),1)).astype(float),
##                           np.reshape(np.array(time),(len(time),1)),
##                           np.reshape(np.zeros(len(time)),(len(time),1)).astype(float),
##                           np.reshape(np.array(In),(len(In),1)).astype(float)))
##    
##    df_cn = pnd.DataFrame(multi_arr)
##
##    df_cn.columns = ['Month','Time','Refined Time','Neutral Current']
##
##    for i in range(len(df_cn['Time'])):
##        df_cn['Refined Time'].iloc[i] = int(datetime.datetime.strptime(df_cn['Time'].iloc[i], '%H:%M:%S').strftime("%H%M"))
##
##    df_cn = pd.from_pandas(df_cn)
##    df_cn.drop_in_place("Time")#drop(labels='Time',axis=1)
##
##    df_cn.columns = ['Month','Time','Neutral Current']
##    
##    df_cn = df_cn.with_columns(pd.all().cast(pd.Float64, strict=False)) ## this converts to Float64
##
##    Iy = 'Neutral Current'
##    tM = 'Month'
##    tT = 'Time'
##    zlist = [1,2,3,4,5,6,7,8,9,10,11,12]
##
##    #df1 = df_cn.filter(pd.col(tM) == 1)
## 
##
##    dfList = [df_cn.filter(pd.col(tM) == 1),df_cn.filter(pd.col(tM) == 2),df_cn.filter(pd.col(tM) == 3),
##              df_cn.filter(pd.col(tM) == 4),df_cn.filter(pd.col(tM) == 5),df_cn.filter(pd.col(tM) == 6),
##              df_cn.filter(pd.col(tM) == 7),df_cn.filter(pd.col(tM) == 8),df_cn.filter(pd.col(tM) == 9),
##              df_cn.filter(pd.col(tM) == 10),df_cn.filter(pd.col(tM) == 11),df_cn.filter(pd.col(tM) == 12)]
##
##    todList = np.hstack((np.arange(1410,1460,10),np.arange(1500,1560,10),np.arange(1600,1660,10),np.arange(1600,1660,10),
##                         np.arange(1700,1760,10),np.arange(1800,1860,10),np.arange(1900,1960,10),np.arange(2000,2060,10),
##                         np.arange(2100,2160,10),np.arange(2200,2260,10),np.arange(2300,2360,10),np.arange(0,60,10),
##                         np.arange(100,160,10),np.arange(200,260,10),np.arange(300,360,10),np.arange(400,460,10),
##                         np.arange(500,560,10),np.arange(600,660,10),np.arange(700,760,10),np.arange(800,860,10),
##                         np.arange(900,960,10),np.arange(1000,1060,10),np.arange(1100,1160,10),np.arange(1200,1260,10),
##                         np.arange(1300,1360,10),1400))
##
##    #print(todList)
##    
##    todList = np.flip(np.sort(todList))#.flip
##    #print(todList)
##
##    todMedians = []
##    todGroup = np.array([])
##    for m in zlist:
####        print(zlist)
####        print(m)
##        for n in todList:
##            
##            toSelect = dfList[m-1].filter(pd.col(tT) == n)
##            
##            todMedians = np.append(todMedians,toSelect.select(pd.col(Iy)).median())
##            
##        if m-1 == 0:
##            todGroup = todMedians
##        else:
##            todGroup = np.vstack((todGroup,todMedians))
####        print('group')
####        print(todGroup)
####        print(todGroup.shape)
####
####        print('medians')
####        print(todMedians)
####        print('shapes')
####        print(todGroup.shape)
####        print(todMedians.shape)
##        todMedians = []
##
##    arrList = [dfList[0][Iy].to_numpy(),dfList[1][Iy].to_numpy(),dfList[2][Iy].to_numpy(),
##               dfList[3][Iy].to_numpy(),dfList[4][Iy].to_numpy(),dfList[5][Iy].to_numpy(),
##               dfList[6][Iy].to_numpy(),dfList[7][Iy].to_numpy(),dfList[8][Iy].to_numpy(),
##               dfList[9][Iy].to_numpy(),dfList[10][Iy].to_numpy(),dfList[11][Iy].to_numpy()]
##
##    fig = plt.figure()
##    xcolumns = []
##    ycolumns = []
##    plt.close()
##        
##    for iv,jk in [[arrList[0],1],[arrList[1],2],[arrList[2],3],[arrList[3],4],[arrList[4],5],[arrList[5],6],
##                  [arrList[6],7],[arrList[7],8],[arrList[8],9],[arrList[9],10],[arrList[10],11],[arrList[11],12]]:
##
##        try:
##            kdexy = sns.kdeplot(iv)
##            line = kdexy.lines[jk-1]
##            x,y = line.get_data()
##            
##
##            if x[140] < 1:
##                x = np.zeros(200)
##                y = np.zeros(200)
##
##            if jk-1 == 0:
##                xcolumns = x
##                ycolumns = y
##            else:
##                xcolumns = np.append(xcolumns,x)
##                ycolumns = np.append(ycolumns,y)
##        except UserWarning:
##            jk -= 1
##
##            print('warning issued')
##        ##plt.show()
##        
##
####        xcolumns = np.append(xcolumns,x)
####        
####        ycolumns = np.append(ycolumns,y)
##
##    plt.close()
##    #plt.show()
##
##    #print(xcolumns.shape)
##    #print(xcolumns)
##    #print(ycolumns.shape)
##    #print(ycolumns)
##    
##    xJan,xFeb,xMar,xApr,xMay,xJune,xJuly,xAug,xSept,xOct,xNov,xDec = np.split(xcolumns,12)
##    xMonth = np.vstack((xJan,xFeb,xMar,xApr,xMay,xJune,xJuly,xAug,xSept,xOct,xNov,xDec))
##    
##    yJan,yFeb,yMar,yApr,yMay,yJune,yJuly,yAug,ySept,yOct,yNov,yDec = np.split(ycolumns,12)
##    yMonth = np.vstack((yJan,yFeb,yMar,yApr,yMay,yJune,yJuly,yAug,ySept,yOct,yNov,yDec))
##
##    zMonth = np.array([np.ones(len(xJan)),np.ones(len(xJan))*2,np.ones(len(xJan))*3,np.ones(len(xJan))*4,np.ones(len(xJan))*5,np.ones(len(xJan))*6,
##                       np.ones(len(xJan))*7,np.ones(len(xJan))*8,np.ones(len(xJan))*9,np.ones(len(xJan))*10,np.ones(len(xJan))*11,np.ones(len(xJan))*12])
##    
##    #----plotting layered 2d graphs of neutral current----#
##    
##    ax = plt.subplot(projection='3d')
##    for i in range(12):
##        ax.plot(xMonth[i],zMonth[i],yMonth[i])
##
##    
##    monthLabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
##    
##    ax.set_yticks(zlist,labels=monthLabels)
##
##    ax.view_init(elev=25, azim=290)
##
##    plt.title('KDE of Monthy In {b} '.format(b=txrN+' '+timeEntry2))
##    plt.savefig('at1/{b}/Distributions/{a}.png'.format(a=txrN+' KDE for '+timeEntry2,b=txrN),dpi=1200)
##    
##    #plt.show()
##
##    #----plotting month against TOD median distribution----#
##   
##    for s in zlist:
##        no1 = s-1
##        
##        X = xMonth[no1]
##        Y = np.arange(0,len(todGroup[s-1]),1)
##
##        Z = np.zeros((len(X),len(Y)))
##        X, Y = np.meshgrid(X, Y)
##        
##        for i in range(len(X)):
##            for j in range(len(Y)):
##            #print(yMonth[4][i]+yMonth[6][j])
##                Z[i][j] = yMonth[no1][i]*todGroup[s-1][j]
##        
##        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
##        ax.plot_surface(X, Y, Z.T,cmap='plasma',zorder=1)
##        #ax.plot_wireframe(X, Y, Z.T,cmap='plasma',zorder=1,rstride=10, cstride=10)
##
##        ax.plot(xMonth[no1],yMonth[no1]*100,zs=0,zdir='y',color='r',zorder=10)
##        ax.plot(np.arange(0,len(todGroup[s-1]),1),todGroup[s-1]/100,zs=0,zdir='x',color='y',zorder=11)
##
##        plt.yticks(np.linspace(0,150,7), labels=todList.astype(str)[3::24])
##
##        ax.set_xlim(0,max(xMonth[no1]))
##        #ax.set_ylim(0,max(xMonth[no2]))
##        #ax.set_zlim(0,0.05)
##        
##        ax.set_xlabel('Distribution for Month')
##        ax.set_ylabel('Time of Day Median')
##        ax.set_zlabel('Neutral Current * Kernel Density')
##
##        ax.view_init(elev=25, azim=270)
##
##        plt.title('KDE of Monthy In and Time of Day In for {a} {b} '.format(a=monthLabels[s-1],b=txrN+' '+timeEntry2))
##        plt.savefig('at1/{b}/Distributions/{a}.png'.format(a=txrN+' 3D KDE for '+monthLabels[s-1]+' '+timeEntry2,b=txrN),dpi=1200)
##        plt.close()    
##        #plt.show()
##    
##    plt.close()
##    #----plotting 3d bivariate distribution----#
##    # this only plots 2 months against eachother, must choose other variable to
##    # make this more applicable
##    
####    no1 = 1
####    no2 = 9
####    
####    X = xMonth[no1]#np.arange(0,300,1.5)#
####    Y = xMonth[no2]#np.arange(0,300,1.5)#
####
####
####    Z = np.zeros((len(X),len(Y)))
####    X, Y = np.meshgrid(X, Y)
####
####    for i in range(len(X)):
####        for j in range(len(Y)):
####            Z[i][j] = yMonth[no1][i]*yMonth[no2][j]
####    
####    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
####    ax.plot_surface(X, Y, Z,cmap='plasma',zorder=1)#,rstride=10, cstride=10)
####
####    ax.plot(xMonth[no1],yMonth[no1]/100,zs=0,zdir='x',color='r',zorder=10)
####    ax.plot(xMonth[no2],yMonth[no2]/100,zs=0,zdir='y',color='y',zorder=11)
####
####    ax.set_xlim(0,max(xMonth[no1]))
####    ax.set_ylim(0,max(xMonth[no2]))
####    #ax.set_zlim(0,0.05)
####    
####    ax.set_xlabel('x-axis')
####    ax.set_ylabel('y-axis')
####    ax.set_zlabel('z-axis')
####    plt.show()
##
##    #print('passed through plot_neutralCurrent')
##    ##----Reset warnings so they don't catch----##
##    warnings.resetwarnings()
    return


##-----------------main script-----------------##
# View individual functions/methods for description

##---intialising matrix for percentiles---##
unbalanceList = np.zeros((1,12),dtype=object)
unbalanceList[0] = ['No. in DF23','TXR Name','Mon Name','Year','UF Type','p1','p5','p25','p50','p75','p95','p99']

##---intialising counters for main script---##
pCount = 0
failed = 0
i = 5
recordI = i
start = 0
end = 3
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
        for g in range(2):

            A, B, C, D, E, F, date, time = extract_3ph(dataMatrix[g][0],dataMatrix[g][1],dataMatrix[g][2],
                                                       dataMatrix[g+2][0],dataMatrix[g+2][1],dataMatrix[g+2][2],genCount)

            vNi = [[A,B,C],[D,E,F]]
            #print(A,B,C,D,E,F)
            
            #for p in range(2):
            #if genCount >1:
                
                
            In, medianIn, unbalanceList = neutralCurrent(D,E,F,genCount,txrName,i,pCount,unbalanceList,date,time,monName)
            print('medians currents')
            print(medianIn)
            print(np.median(In))

                #currentImbalanceDecomposition()
        
            dataProfile(A,B,C,In,txrName,genCount)

        #powerImbalance
        
            if len(A) != 0:
            
                Vuf,unbalanceList,pCount,LVUR,gCount = calcUnbalance(A,B,C,i,pCount,txrName,gCount,genCount,unbalanceList,monName)
            
        ##distributionPercentile(finalList)  <---- to be done (distribution of the LVUR or VUF)
            
            ## distribution of the neutral current (why was 50 A chosen; is that usually an 'unbalanced system',
            ## or is 50 A chosen because neutral wires can handle ~50 A before exceeding temperature rating,
            ## or does 50A correspond to an non-compliant neutral voltage)
            
        #print('Current Count is currently: '+str(genCount))
        #if genCount > 1:
        #    print('Current Count is greater than 1: '+str(genCount))

            genCount += 1

        
        print('\n---------------------------next monitor---------------------------\n')
        
        i+=1
    except pa.lib.ArrowInvalid:
        print('Failed due to pyarrow exception\n')
        i+=1
    #except:
    #    print('Failed due to unknown exception\m')
    #    i+=1

##---printing summary of program---##
print('List of Unbalance Percentiles')

unbalanceDF = pnd.DataFrame(unbalanceList)
#unbalanceDF.columns = unbalanceDF[0]
#unbalanceDF = unbalanceDF[1:]

nameDF = 'V9 YearSet siteUnbalance_{a} to {b}.csv'.format(a=recordI,b=i)

#unbalanceDF.write_csv('testingpolars.csv',separator=',')

unbalanceDF.to_csv(nameDF,',')

print(unbalanceDF)

print()

print(str(start)+' filepaths accessed')

print(str(failed)+' filepaths failed to form')
    
