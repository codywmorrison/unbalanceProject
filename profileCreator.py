# Written by Cody Morrison (Vac. Student) for Energy Queensland under Trevor Gear (Principal Engineer)
#
# This creates a profile?
#
# This script was written as part of a student placement with Energy Queensland over the
# university's vacation period of 2023/2024.
#
# V1 - 29/2/24


##--- module importation---##
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
import matplotlib.pylab

#from PIconnect.PIConsts import RetrievalMode
import numpy as np      ## <- numpy used for majority of mathematical operations
import polars as pl     ## <- Pandas was originally used, but changed to Polars for speed
import pyarrow as pa
import pandas as pd
import cmath as cm
import math
#import PIconnect as PI  ## <- PIconnect is the module for PI AF SDK
import seaborn as sns   ## <- seaborn and scipy are used for statistics calculations
import pprint
import datetime

from mpl_toolkits.mplot3d import axes3d
from matplotlib.dates import DateFormatter

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



##------------PI setup------------##

##print('\nDefault Time: '+PI.PIConfig.DEFAULT_TIMEZONE)
#PI.PIConfig.DEFAULT_TIMEZONE = 'Australia/Brisbane'
##print('\nConfig to UTC+10 Time: '+PI.PIConfig.DEFAULT_TIMEZONE)
##
##print('\nServer List: '+str(list(PI.PIServer.servers.keys())))
##print('\nDatabase List: '+str(list(PI.PIAFDatabase.servers.keys())))
#print('\n-----------------------------------------------------\n')

##------------functions------------##

#   ratio of energy usage to generation used to scale profile type?.... adding heavier peaks?
#   if over x is consumed and it is still high load, then peaks on evening and morning and low solar?
#   if under x is consumed and it is still high load, then only peaks on evening and morning
#   if over x is consumed and low load, then flat duck curve?
#   if under x is consumed and low load, then ??
#
#   CLASSIFY and then build profiles using customers that have available data??? sort invterval data by NMI no.
#   each customer will be classified as HU HS, HU LS, LU HS, LU LS? or break up more? also use % or ratios for degree...?


solarRankings = ["FAILED SOLAR ESTIMATION - Solar estimate is incorrect","EXTREMELY HIGH CONSUMPTION - Extremely high usage during PV hours",
                 "VERY HIGH CONSUMPTION - Very high usage during PV hours","HIGH CONSUMPTION - High usage during PV hours","MEDIUM CONSUMPTION - Medium usage during PV hours",
                 "LOW CONSUMPTION - Low usage during PV hours","VERY LOW CONSUMPTION - Very low usage during PV hours","EXTREMELY LOW CONSUMPTION - Extremely low or almost no usage during PV hours",
                 "NO CONSUMPTION - No usage during PV hours or no solar"]

#   IF: high consumption... but still high load (not native)... -> high usage outside of PV hours?         or high consumption (with high native) but low load -> low usage outside of PV hours ie less 'peaky'?






# IMPORTATION OF DATA
customerEnergyData = pl.read_csv('customerEnergyData - All Sites.csv',has_header=True,columns=(0,2,3,4,5,6,7,12,13,14,15,16,17,18,24,26,27))
#CHECKING ALL IS GRABBED
print(customerEnergyData.columns)
print("Energy data read from .csv")
customerIntervalData = pl.read_csv('customerIntervalData - All Sites.csv',has_header=True,columns=(0,1,2,3,4,5))
print("Interval data read from .csv -> no. of data points = "+ str(len(customerIntervalData)))

# THIS CREATES A DATETIME COLUMN
customerIntervalData = customerIntervalData.with_columns(pl.col("IntervalStartTime").str.slice(10,6))
customerIntervalData = customerIntervalData.with_columns(pl.concat_str(["IntervalDate","IntervalStartTime"]).alias("Datetime"))
customerIntervalData = customerIntervalData.with_columns(pl.col("Datetime").str.to_datetime(format="%Y-%m-%dT%H:%M"))

#prints average consumption for those that have solar
#print(customerEnergyData.select(["Consumption of Energy Generation"]).filter(pl.col("Consumption of Energy Generation")>0).mean(axis=0))





allNMIs = np.genfromtxt('customerEnergyData - All Sites.csv',dtype=str,delimiter=',',usecols=(0),skip_header=1).astype(str) ## <----- get a list of every single NMI (ALL SITES) and all data from all the NMIs... SP1809 etc

#print(allNMIs)
print("Number of customers with Energy Data = "+str(len(allNMIs)))


print("Number of customers with Interval Data = "+str(len(customerIntervalData.unique(subset=["NMI"],keep='first'))))

##print(customerIntervalData.unique(subset=["NMI"],keep='first'))
      

data = customerIntervalData

# CLASSIFICATION OF CUSTOMERS DO THIS IN EXCEL...?
for NMI in allNMIs:
    classData = customerEnergyData.filter((pl.col("NMI") == str(NMI)))


    #if 







###data = pd.read_csv("Txr3 2023 Customer Data.csv")    #separator=",",,infer_schema_length=200000     <---- polars term
##data = pl.read_csv('customerEnergyData.csv',has_header=True)
##
data = data.filter((pl.col("NMI") == str("3120759801"))).filter((pl.col("NMISuffix") == 'E1'))# | (pl.col("NMISuffix") == 'E2'))
print(data)

##dataSolar = data.filter((pl.col("NMISuffix") == 'B1'))# | (pl.col("NMISuffix") == 'B2'))
##
###for load
##partForFollowing = data.with_columns(DT = pl.col('Date Time').str.to_datetime("%d/%m/%Y %H:%M").cast(pl.Datetime)).sort('DT')
##datetimePartition = partForFollowing.partition_by("Day Time", maintain_order=True) # <- change to time to output 1 day...\
##
###for solar
##partForFollowingSol = dataSolar.with_columns(DT = pl.col('Date Time').str.to_datetime("%d/%m/%Y %H:%M").cast(pl.Datetime)).sort('DT')
##datetimePartitionSol = partForFollowingSol.partition_by("Day Time", maintain_order=True) # <- change to time to output 1 day...\




##
##
##
##for NMI in allNMIs:
##
##
##
##
##    
##    #   data sort to NMI no.
##    data = data.filter((pl.col("NMI") == str(NMI))).filter((pl.col("NMISuffix") == 'E1'))# | (pl.col("NMISuffix") == 'E2'))
##    dataSolar = data.filter((pl.col("NMI") == NMI)).filter((pl.col("NMISuffix") == 'B1'))# | (pl.col("NMISuffix") == 'B2'))
##
##    #   load
##    partForFollowing = data.with_columns(DT = pl.col('Date Time').str.to_datetime("%d/%m/%Y %H:%M").cast(pl.Datetime)).sort('DT')
##    datetimePartition = partForFollowing.partition_by("Day Time", maintain_order=True) # <- change to time to output 1 day...\
##    
##    #   solar
##    partForFollowingSol = dataSolar.with_columns(DT = pl.col('Date Time').str.to_datetime("%d/%m/%Y %H:%M").cast(pl.Datetime)).sort('DT')
##    datetimePartitionSol = partForFollowingSol.partition_by("Day Time", maintain_order=True) # <- change to time to output 1 day...\
##











##    
##
##
##
##def pullEnergyData(customerNo):
##    #siteDF23 = np.genfromtxt('customerEnergyData.csv',dtype=str,delimiter=',',usecols=(0,2,3),skip_header=1) ##2018UnbalancedSites.csv, DFListV1.csv
##
##    ## polars is notated as pd
##    customerList = pd.read_csv('customerEnergyData.csv',has_header=True,columns=(0,2,3,5,6,11,17,19,20))
##    
##
##
##
##def indataCheck(k,n):
##    # This function checks if a filepath can be formed using the csv's data file.
##    # If no filepath can be formed for a particular PQ monitor site, it is skipped
##    # and the next site is tested.
##
##    ##---checking that sub, region, and location are given in the csv list so a filepath can be formed---##
##    testStr = np.genfromtxt('transformer list (first list).csv',dtype=str,delimiter=',',usecols=(1),skip_header=1)
##    
##    while testStr[k]=='No Data' or testStr[k]=='XXXX':
##
##        n += 1
##        print('Cannot form filepath for k = {f}'.format(f=str(k)))
##        print('This site has been skipped. {h} have failed to form a filepath. k = {g} will now be run:'.format(h=str(n),g=str(k+1)))
##        print('\n---------------------------next monitor---------------------------\n')
##        k += 1
##        
##    if testStr[k]!='No Data' or testStr[k]!='XXXX':
##
##        print('Filepath can be formed for k = {f}'.format(f=str(k)))
##
##        return k, n
##
##
##def dataGrab(fPath,i,date,counter):#,i
##    # This function uses PIConnect (an interface with Osisoft's PI) to pull voltage
##    # and current data from Energex's PQ monitors. This function outputs a matrix with
##    # all of the voltage and current data for a given period (atm 2018 and 2023 calendar years).
##    
##    ##---pulling data from Osisoft PI with formed filepath---##
##    with PI.PIAFDatabase(database="PQ Monitors") as database:
##
##
##        inputPath = fPath#'EQL\\SOUTHEAST\\BRISBANE SOUTH\\SSH22\\LGL3A\\SP1809-G\\SP1809-G-TR1' #fPath
##        
##        element = database.descendant(inputPath)
##        attvalues = iter(element.attributes.values())
##
##        attList = [next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
##                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues)]
##
##        ##---timeline to pull data---##
##        intT = '10m'
##
##        #siteDF23 = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(11),skip_header=1)
##
##        #sliceNo = len(siteDF23[i])-4
##        
##        #yearInt = int(siteDF23[i][sliceNo:])
##        #print(sliceNo,yearInt)
##
##        #startT1 = '2022-07-01 00:00:00'
##        #endT1 = '2023-6-30 00:00:00'
##
##        print(date)
##        print(date[0])
##
##        startT1 = date[0]#'2023-11-01 00:00:00'
##        endT1 = date[1]#'2023-12-1 00:00:00'
##        
##
##        #startT2 = '2022-07-01 00:00:00'#.format(c = yearInt-2)
##        #endT2 = '2023-6-30 00:00:00'#.format(d = yearInt-1)
##
##        #startT2 = '{e}-07-01 00:00:00'.format(e = yearInt+1)
##        #endT2 = '{f}-6-30 00:00:00'.format(f = yearInt+2)
##        
####        startT1 = '2018-07-02 00:00:00'
####        endT1 = '2019-6-30 00:00:00'
####
####        startT2 = '2022-07-02 00:00:00'
####        endT2 = '2023-6-30 00:00:00'
##
##
##        ##---search and assign Voltage and Current data to matrix---##
##        for att in range(len(attList)):
##
##            
##
##            if attList[att].name == 'CUR_C':
##                CdataCa = attList[att].interpolated_values(startT1,endT1,intT)
##                #CdataCb = attList[att].interpolated_values(startT2,endT2,intT)
##            if attList[att].name == 'CUR_B':
##                CdataBa = attList[att].interpolated_values(startT1,endT1,intT)
##                #CdataBb = attList[att].interpolated_values(startT2,endT2,intT)
##            if attList[att].name == 'CUR_A':
##                CdataAa = attList[att].interpolated_values(startT1,endT1,intT)
##                #CdataAb = attList[att].interpolated_values(startT2,endT2,intT)        #,[VdataCb,VdataBb,VdataAb],,[CdataCb,CdataBb,CdataAb]
##
##
##            if attList[att].name == 'P_C':
##                PdataC = attList[att].interpolated_values(startT1,endT1,intT)
##                
##            if attList[att].name == 'P_B':
##                PdataB = attList[att].interpolated_values(startT1,endT1,intT)
##                
##            if attList[att].name == 'P_A':
##                PdataA = attList[att].interpolated_values(startT1,endT1,intT)
##
##
##            if attList[att].name == 'Q_C':
##                QdataC = attList[att].interpolated_values(startT1,endT1,intT)
##                
##            if attList[att].name == 'Q_B':
##                QdataB = attList[att].interpolated_values(startT1,endT1,intT)
##                
##            if attList[att].name == 'Q_A':
##                QdataA = attList[att].interpolated_values(startT1,endT1,intT)         
##
##        
##        #dataMatrix = [[CdataAa,CdataBa,CdataCa],[PdataA,PdataB,PdataC],[QdataA,QdataB,QdataC]] # i did swap these
##        dataMatrix = [[CdataAa,CdataBa,CdataCa],[CdataAa,CdataBa,CdataCa],[CdataAa,CdataBa,CdataCa]] # i did swap thes
##
##
##        return dataMatrix
##
##def feederGrab(txrN,parPath):
##    # This function searches the AF database and pulls the feeder name for the monitor site.
##    # As only the transformer number/name and substation name is given, the feeder
##    # name must be searched for and returned to form a full filepath for PI data.
##    
##    ##---searching database for transformer to get feeder, as feeder isn't given in csv---##
##    with PI.PIAFDatabase(database="PQ Monitors") as database:
##        
##        feederSRCH = database.descendant(parPath)
##        for fdr in feederSRCH.children.values():
##
##            fdrStr = str(fdr)
##            fdrStr1 = fdrStr.replace('PIAFElement(','')
##            fdrStr2 = fdrStr1.replace(')','')
##            print(fdrStr2)
##
##            ##---transformer being searched---##            
##            txrSRCH = database.descendant(parPath+'\\'+fdrStr2)
##            for txr in txrSRCH.children.values():
##                
##                txrStr = str(txr)
##                txrStr1 = txrStr.replace('PIAFElement(','')
##                txrStr2 = txrStr1.replace(')','')
##
##                ##---building full filepath for PI to read---##
##                if txrStr2==txrN:
##
##                    fPath = parPath+'\\'+fdrStr2+'\\'+txrStr2
##
##                    monN = monitorGrab(txrN,fPath)
##
##                    print('------------------------------------------------')
##                    print(monN)
##                    print('------------------------------------------------')
##
##                    fPathF = parPath+'\\'+fdrStr2+'\\'+txrStr2+'\\'+monN
##
##                    print('Found monitor filepath: '+fPathF)
##
##                    return fPathF
##
##
##def monitorGrab(txrN,parPath):
##    # in the name, this searches from trx to grab monitor id
##    
##    ##---searching database for transformer to get feeder, as feeder isn't given in csv---##
##    with PI.PIAFDatabase(database="PQ Monitors") as database:
##
##        print('in monitorGrab')
##        print(parPath)
##        monitorSRCH = database.descendant(parPath)
##        for mtr in monitorSRCH.children.values():
##
##            mtrStr = str(mtr)
##            mtrStr1 = mtrStr.replace('PIAFElement(','')
##            mtrStr2 = mtrStr1.replace(')','')
##
##            print(mtrStr2)
##
##            return mtrStr2
##
##def locationGrab(subN,locPath):
##    # This function uses the subname fo search and find the location
##    
##    ##---searching database for transformer to get feeder, as feeder isn't given in csv---##
##    with PI.PIAFDatabase(database="PQ Monitors") as database:
##        
##        locationSRCH = database.descendant(locPath)
##        for loc in locationSRCH.children.values():
##
##            locStr = str(loc)
##            locStr1 = locStr.replace('PIAFElement(','')
##            locStr2 = locStr1.replace(')','')
##            print(locStr2)
##
##            ##---transformer being searched---##            
##            subSRCH = database.descendant(locPath+'\\'+locStr2)
##            for sub in subSRCH.children.values():
##                
##                subStr = str(sub)
##                subStr1 = subStr.replace('PIAFElement(','')
##                subStr2 = subStr1.replace(')','')
##
##                ##---building full filepath for PI to read---##
##                if subStr2==subN:
##
##                    return locStr2               
##
##def pathGrab(k):
##    # This function forms a partial filepath and passes it to the above function
##    # (feederGrab) so that a full filepath can be passed to PIConnect (dataGrab) function.
##    # This function uses the DF23 csv file to concatenate a filepath.
##
##    ##---pulling sub, region and txr name from the csv file---##
##    siteDF23 = np.genfromtxt('GeebungTxrs.csv',dtype=str,delimiter=',',usecols=(0,1,3),skip_header=1) ##2018UnbalancedSites.csv, DFListV1.csv
##
##    eqlName = "EQL"
##    regName = siteDF23[k][2] #changed to 2 from 3
##    #locName = siteDF23[k][2]
##    subName = siteDF23[k][1]
##    txrName = siteDF23[k][0]
##    #monName = siteDF23[k][3]
##
##    locPath = "\\".join([eqlName,regName])
##    locName = locationGrab(subName,locPath)
##
##    print(locName)
##
##    ##---forming/concatenating the filepath---##
##    oPath = "\\".join([eqlName,regName,locName,subName])
##    parPath = r'{}'.format(oPath)
##
##    print(parPath)
##    
##    #print('\nFound partial filepath from CSV: '+parPath+'\n')
##    
##    return txrName,parPath#,monName
##
##def extract_3ph(d,e,f,x,y,z,j,k,l,genCount):
##    # This function passes in x, y, and z voltage or current data to be cleaned and
##    # simplfied before being passed for forming graphs and unbalance calculations.
##    # This function returns and appended matrix of cleaned data (shorter than input).  
##
##    ##---initialising attributes for calculation---##
##    xDF = pnd.DataFrame(x)
##    x = pd.from_pandas(xDF)
##    yDF = pnd.DataFrame(y)
##    y = pd.from_pandas(yDF)
##    zDF = pnd.DataFrame(z)
##    z = pd.from_pandas(zDF)
##
##    jDF = pnd.DataFrame(j)
##    j = pd.from_pandas(jDF)
##    kDF = pnd.DataFrame(k)
##    k = pd.from_pandas(kDF)
##    lDF = pnd.DataFrame(l)
##    l = pd.from_pandas(lDF)
##
##    
##
##    dDF = pnd.DataFrame(d)
##    d = pd.from_pandas(dDF)
##    
##    eDF = pnd.DataFrame(e)
##    e = pd.from_pandas(eDF)
##    fDF = pnd.DataFrame(f)
##    f = pd.from_pandas(fDF)
##
##    A = np.zeros(len(x))
##    B = np.zeros(len(A))
##    C = np.zeros(len(A))
##
##    J = np.zeros(len(j))
##    K = np.zeros(len(J))
##    L = np.zeros(len(J))
##
##    D = np.zeros(len(d))
##    E = np.zeros(len(D))
##    F = np.zeros(len(D))
##
##    #print(x,y,z,d,e,f)
##    
##    rawDatetime = xDF.index.values.tolist()
##    refinedDatetime = pnd.to_datetime(rawDatetime,unit='ns')
##    refinedDatetime.columns = ['Date']
##    
##
##    
##    refinedDate = pnd.DataFrame()
##    refinedDate['Date']= refinedDatetime.date
##    refinedDate = pd.from_pandas(refinedDate)
##    
##
##
##    refinedDay = pnd.DataFrame()
##    refinedDay['Day']= refinedDatetime.day
##    refinedDay = pd.from_pandas(refinedDay)
##
##    refinedMonth = pnd.DataFrame()
##    refinedMonth['Month']= refinedDatetime.month
##    refinedMonth = pd.from_pandas(refinedMonth)
##
##    refinedYear = pnd.DataFrame()
##    refinedYear['Year']= refinedDatetime.year
##    refinedYear = pd.from_pandas(refinedYear)
##
##    refinedDatetime = pd.from_pandas(refinedDatetime)
##
##
##    appA = []
##    appB = []
##    appC = []
##
##    appD = []
##    appE = []
##    appF = []
##
##    appJ = []
##    appK = []
##    appL = []
##
##    ##datetime = []
##    date = []
##    time = []
##    year = []
##    day = []
##    month = []
##    
##    i=1 ## <---i = 1 is used as the datacleaning process uses x[i-2] which is only possible if i is greater than or equal to 2.
##    #plt.plot(d)
##    #plt.show()
##    ##---cleaning data for calculation and graphing---##
##    for i in range(len(A)-2):
##        i+=1
##        #print(round(x[i]-x[i-2],1))
##        #print(x.row(i)[0])
##        
##        
##        #if str(x.iloc[i]) != 'No Data' and type(x.iloc[i]) is  not np.float64 and type(x.iloc[i]) is  not float:
##        #    print(i)
##        #    print(type(x.iloc[i]))
##        #    print(x.iloc[i])
##        
##        ##---data cleaining for current---##
##       
####        if round(d.row(i)[0],3) != round(d.row(i-1)[0],3) or round(d.row(i)[0]-d.row(i-1)[0],3) != round(d.row(i-1)[0]-d.row(i-2)[0],3):#(type(x[i]) is np.float64 or type(x[i]) is float) and 
##            #print('passing')
##        ##A[i]=x[i]
##        appA.append(x.row(i)[0])
##        appD.append(d.row(i)[0])
##        appJ.append(j.row(i)[0])
##        ##B[i]=y[i]
##        appB.append(y.row(i)[0])
##        appE.append(e.row(i)[0])
##        appK.append(k.row(i)[0])
##        ##C[i]=z[i]
##        appC.append(z.row(i)[0])
##        appF.append(f.row(i)[0])
##        appL.append(l.row(i)[0])
##        
##        date.append(str(refinedDate.row(i)[0]))#[0]
##        year.append(refinedYear.row(i)[0])#[0]
##        month.append(refinedMonth.row(i)[0])#[0]
##        day.append(refinedDay.row(i)[0])#[0]
##        time.append(str(refinedDatetime[i].time()))
##
##            
##    ##---printing data cleaning output---###
##    points = round(len(A)-len(appA))
##    
##    if len(appA) != 0:
##        
##        print('\n{d} points of data were cleaned'.format(d=points))
##        print('Meaning {y} of {m} days are unusable data\n'.format(y=round(points/144), m=round(len(A)/144)))
##
##    elif len(appA) == 0:
##        print('No available data. It was entirely cleaned.')
##
##
##    return appD, appE, appF, appA, appB, appC, appJ, appK, appL, date, time, year, month, day
##
##def neutralCurrent(x,y,z,xP,yP,zP):
##    # This function calculates the neutral current of the system (at the PQ monitor).
##
##    if len(Ia) == 0:
##        In = np.zeros(len(x))
##        
##        print('Neutral calculation failed')
##        return In
####
####    vectorAx = np.zeros(len(x))
####    vectorAy = np.zeros(len(vectorAx))
####    vectorBx = np.zeros(len(vectorAx))
####    vectorBy = np.zeros(len(vectorAx))
####    vectorCx = np.zeros(len(vectorAx))
####    vectorCy = np.zeros(len(vectorAx))
####    vectorNx = np.zeros(len(vectorAx))
####    vectorNy = np.zeros(len(vectorAx))
####    PfIn = np.zeros(len(vectorAx))
####
####
####    for i in range(len(x)):
####
####        #print(xP[i],x[i],math.cos(180/math.pi))
####        vectorAx[i] = math.cos(math.pi/180*xP[i])*x[i]
####        vectorAy[i] = math.sin(math.pi/180*xP[i])*x[i]
####
####        vectorBx[i] = math.cos(math.pi/180*yP[i])*y[i]
####        vectorBy[i] = math.sin(math.pi/180*yP[i])*y[i]
####
####        vectorCx[i] = math.cos(math.pi/180*zP[i])*z[i]
####        vectorCy[i] = math.sin(math.pi/180*zP[i])*z[i]
####
####    #vectorNx[i] = np.len(
##
##    
##
##    ##---calculating neutral current---##
##    In = np.zeros(len(x))
##
##    for i in range(len(x)):
##
####        vectorNx[i] = vectorAx[i] + vectorBx[i] + vectorCx[i]
####        vectorNy[i] = vectorAy[i] + vectorBy[i] + vectorCy[i]
####
####        In[i] = np.sqrt(vectorNx[i]**2 + vectorNy[i]**2)
####
####        PfIn[i] = math.atan((vectorNy[i]/vectorNx[i]))*180/math.pi  #math.atan2(vectorNy[i],vectorNx[i])*180/math.pi
####
####        #if PfIn[i] < 0:
##        #    In[i] = In[i]*-1
##            
##
##
##
##        #print(vectorNx[i],vectorNy[i])
##
##
##
##
##        In[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2 - x[i]*y[i] - y[i]*z[i] - z[i]*x[i])    ## <--- Equation assumes no non-resistive load (phase angles are 0,-120,120). For a non-resistive load, 
##        #                                                                                    ##      phase angle measurements must be used. However, these are greatly effected by PV DER, i.e., unreliable.
##
##    print(In)
####    print(PfIn)
##
##    #plt.plot(PfIn,label='A')
##    #plt.plot(In,label='B')
##    #plt.legend()
##    #plt.show()
##    
##    return In#,PfIn
##
##
##def polarsData(In,Ia,Ib,Ic,PfIn,Pfa, Pfb, Pfc,date,time, year, month, day,dayCOUNTER):
##    ## Polars is set in pd not pl...
##    
##    print("--IN POLARS--")
##
##    data = {"Date": date, "Time": time, "Year": year, "Month": month, "Day": day,
##            "Neutral Current": In, "A Current": Ia, "B Current": Ib, "C Current": Ic,
##            "N Power Factor": PfIn, "A Power Factor": Pfa, "B Power Factor": Pfb,
##            "C Power Factor": Pfc}
##
##    
##    polarsDF = pd.DataFrame(data)
##
##    uniqueDate = polarsDF.select(["Date"]).unique(keep='first',maintain_order=True)#.to_numpy()#.select(["Date"])subset=["Date"],
##    #uniqueDate = polarsDF.select(["Month"]).unique(keep='first',maintain_order=True)#.to_numpy()#.select(["Date"])subset=["Date"],
##
##    print(uniqueDate)
##
##    listDate = uniqueDate.to_numpy()
##    
##    print(listDate[10][0])
##    print(str(listDate[10][0]))
##
##    count = 1
##
##    polarsDay = polarsDF.filter(pd.col("Date") == str(listDate[10][0]))
##    #polarsDay = polarsDF.filter(pd.col("Month") == listDate[10][0])
##    print(polarsDay)
##    maxNeutral = polarsDay.select(["Neutral Current"]).max()             ## we want max neutral current and phase currents for that max, not max of all 4 currents.
##    print(maxNeutral.to_numpy())
##    
##    maxRow = polarsDay.filter(pd.col("Neutral Current") == maxNeutral.to_numpy())
##
##    maxValues = maxRow.select(["Neutral Current","A Current","B Current","C Current","N Power Factor","A Power Factor","B Power Factor","C Power Factor"]) 
##    
##    #print(maxValues)
##    i = 0
##    PA_thresh = 120
##
##
##
##    
##    for dateT in listDate:
##        #print('iterate')
##        #print(str(dateT))
##
##        
##        polarsDay = polarsDF.filter(pd.col("Date") == str(dateT[0]))
##        #polarsDay = polarsDF.filter(pd.col("Month") == dateT[0])
##
##        maxNeutral = polarsDay.select(["Neutral Current"]).max()             ## we want max neutral current and phase currents for that max, not max of all 4 currents.
##        #print(maxNeutral.to_numpy())
##    
##        maxRow = polarsDay.filter(pd.col("Neutral Current") == maxNeutral.to_numpy())
##
##        maxValues = maxRow.select(["Time","Neutral Current","A Current","B Current","C Current","N Power Factor","A Power Factor","B Power Factor","C Power Factor"]) #,"A Current","B Current","C Current"]).max()
##        #print(maxValues) ## PF has been added, but should multiple current by -1 if it is within a certain range (close to reversed PF)
##
##        
##        
##    ## MAYBE SET UP THE IF AND ELIF INTO A FUNCTION?
##        if count == 1:
##            maxValuesList = np.hstack([[dateT],maxValues.to_numpy()])
##            count = 2
##
##            if maxValuesList[i][7] > PA_thresh or maxValuesList[i][7] < -PA_thresh:
##                maxValuesList[i][3] = maxValuesList[i][3]*-1
##
##            if maxValuesList[i][8] > PA_thresh+120 or maxValuesList[i][8] < -PA_thresh+120:
##                maxValuesList[i][4] = maxValuesList[i][4]*-1
##
##                
##            if maxValuesList[i][9] > PA_thresh-120 or maxValuesList[i][9] < -PA_thresh-120:
##                maxValuesList[i][5] = maxValuesList[i][5]*-1
##
##            if maxValuesList[i][6] < -90 or maxValuesList[i][6] > 90:
##                maxValuesList[i][2] = maxValuesList[i][2]*-1
##
##
##            
##        else:
##            maxValuesList = np.vstack((maxValuesList,np.hstack([[dateT],maxValues.to_numpy()])))
##
##            if maxValuesList[i][7] > PA_thresh or maxValuesList[i][7] < -PA_thresh:
##                maxValuesList[i][3] = maxValuesList[i][3]*-1
##
##            if maxValuesList[i][8] > PA_thresh+120 or maxValuesList[i][8] < -PA_thresh+120:
##                maxValuesList[i][4] = maxValuesList[i][4]*-1
##
##                
##            if maxValuesList[i][9] > PA_thresh-120 or maxValuesList[i][9] < -PA_thresh-120:
##                maxValuesList[i][5] = maxValuesList[i][5]*-1
##
##            if maxValuesList[i][6] < -90 or maxValuesList[i][6] > 90:
##                maxValuesList[i][2] = maxValuesList[i][2]*-1
##
##        i += 1
##
##    #print(maxValues)
##    
##    print(maxValuesList)
##    fig = plt.figure(figsize=(26,18))
##
##    maxValuesListStr = maxValuesList.astype(str)
##    print(maxValuesListStr)
##    #x_axis = maxValuesList.T[0]
##    #x_axis = list(zip(maxValuesListStr.T[0],maxValuesListStr.T[1]))
##    x_axis = maxValuesList.T[0]+' '+maxValuesList.T[1]
##    x_length = maxValuesList.T[0]
##    print(x_axis)
##    
##    y_n = maxValuesList.T[2]
##    y_a = maxValuesList.T[3]
##    y_b = maxValuesList.T[4]
##    y_c = maxValuesList.T[5]
##
##    x_ticks = np.arange(len(x_length))
##
##    plt.bar(x_ticks - 0.3,y_n,0.2,label = 'n Current',zorder = 6,color='dimgrey')
##    plt.bar(x_ticks - 0.1,y_a,0.2,label = 'a Current',zorder = 6,color='orangered')
##    plt.bar(x_ticks + 0.1,y_b,0.2,label = 'b Current',zorder = 6,color='gold')
##    plt.bar(x_ticks + 0.3,y_c,0.2,label = 'c Current',zorder = 6,color='royalblue')
##
##    plt.title("Maximum Phase and Neutral Currents per Day")
##    plt.xlabel("Day")
##    plt.ylabel("Current (A)")
##    plt.xticks(x_ticks,x_axis,rotation=90) #tick by month not day
##    plt.legend()
##    plt.grid(zorder = -1)
##    plt.savefig('images 2.0/TXR (no reverse) - daily - {a}.png'.format(a=str(dayCOUNTER)),dpi=800)
##    plt.close()
##    #plt.show()
##
##
##    print(dayCOUNTER)
##    print(dayCOUNTER)
##    print(dayCOUNTER)
##    print(dayCOUNTER)
##    print(dayCOUNTER)
##    
##
##
##    
##
##    return 
##
##def powerFactorCalc(PPfa, PPfb, PPfc, QPfa, QPfb, QPfc,Ia,Ib,Ic):
##
##    ##something does seem right
##
##    
##
##    
##    print(PPfb,QPfa)
##    Pfa = np.zeros(len(PPfa))
##    Pfb = np.zeros(len(Pfa))
##    Pfc = np.zeros(len(Pfa))
##
##    for i in range(len(PPfa)):
##
##        if QPfa[i] == 0:
##            Pfa[i] = 0
##        else:
##            Pfa[i] = math.acos(PPfa[i]/(np.sqrt(PPfa[i]**2 + QPfa[i]**2)))*180/math.pi
##
##
##        if QPfb[i] == 0:
##            Pfb[i] = 120
##        else:
##            Pfb[i] = math.acos(PPfb[i]/(np.sqrt(PPfb[i]**2 + QPfb[i]**2)))*180/math.pi + 120
##
##        if QPfc[i] == 0:
##            Pfc[i] = -120
##        else:
##            Pfc[i] = math.acos(PPfc[i]/(np.sqrt(PPfc[i]**2 + QPfc[i]**2)))*180/math.pi - 120
##
##    
##
##
##    print('pfa and pfb and pfc')
##    print(Pfa,Pfb,Pfc)
##    print('no pfb?')
##    #print(max(Pfa),min(Pfa))
##    #print(max(Pfb),min(Pfb))
##    #print(max(Pfc),min(Pfc))
##
##    #return Pfa,Pfb,Pfc
##
##    PA_thresh = 160
##
####    for i in range(len(Ia)):#
####        if Pfa[i] > PA_thresh or Pfa[i] < -PA_thresh:
####            Ia[i] = Ia[i]*-1
####
####        if Pfb[i] > PA_thresh+120 or Pfb[i] < -PA_thresh+120:
####            Ib[i] = Ib[i]*-1
####
####        if Pfc[i] > PA_thresh-120 or Pfc[i] < -PA_thresh-120:
####            Ic[i] = Ic[i]*-1
##
####    print(Pfa)
####    print(Pfb)
####    print(Pfc)
####
####    plt.plot(Pfa,label='A')
####    plt.plot(Pfb,label='B')
####    plt.plot(Pfc,label='C')
####    plt.legend()
####    plt.show()
####
##    plt.plot(Ia,label='A')
##    plt.plot(Ib,label='B')
##    plt.plot(Ic,label='C')
##    plt.legend()
##    plt.show()
####
##    return Pfa,Pfb,Pfc,Ia,Ib,Ic
##
##
####-----------------main script-----------------##
### View individual functions/methods for description
##
####---intialising matrix for percentiles---##
##decompList = np.zeros((1,17),dtype=object)
##decompList[0] = ['No. in DF23','TXR Name','Type','50th I Max Phase','50th I Min Phase','% Fit','DPIB Med. (%)','SIB Med. (A)','RIB Med. (A)','50th Ia',
##                 '50th Ib','50th Ic','50th In','95th Ia','95th Ib','95th Ic','95th In']
##
####---intialising counters for main script---##
##pCount = 0
##failed = 0
##i = 0
##recordI = i
##start = 0
##
##valueForInput = 100
##
##end = 1#valueForInput - i
##g=0
##
###['2023-1-01 00:00:00','2023-1-31 00:00:00'],['2023-2-01 00:00:00','2023-2-28 00:00:00'],['2023-3-01 00:00:00','2023-3-31 00:00:00'],
##        #['2023-4-01 00:00:00','2023-4-30 00:00:00'],['2023-5-01 00:00:00','2023-5-31 00:00:00'],
####Days = [['2023-6-01 00:00:00','2023-6-30 00:00:00'],
####        ['2023-7-01 00:00:00','2023-7-31 00:00:00'],['2023-8-01 00:00:00','2023-8-31 00:00:00'],['2023-9-01 00:00:00','2023-9-30 00:00:00'],
####        ['2023-10-01 00:00:00','2023-10-31 00:00:00'],['2023-11-01 00:00:00','2023-11-30 00:00:00'],['2023-12-01 00:00:00','2023-12-31 00:00:00'],
####        ['2024-1-01 00:00:00','2024-1-31 00:00:00'],['2024-2-01 00:00:00','2024-2-28 00:00:00'],['2024-3-01 00:00:00','2024-3-31 00:00:00'],
####        ['2024-4-01 00:00:00','2024-4-30 00:00:00']]
##Days = [['2020-1-02 00:00:00','2021-1-01 00:00:00'],['2021-1-02 00:00:00','2022-1-01 00:00:00'],['2022-1-02 00:00:00','2023-1-01 00:00:00'],['2023-1-02 00:00:00','2024-1-01 00:00:00']]
##
####---main script prints for number of sites given---##
##dayCOUNTER = 1
##for dIterate in Days:
##    #for start in range(end):
##
##    
##
##
##    
##    i, failed = indataCheck(i,failed)
##
##    txrName, parPath = pathGrab(i)
##
##
##    try:
##        filePath = feederGrab(txrName,parPath)
##    except TypeError:
##        failedText = 'type error'
##        arrDecompList = np.array([i,txrName,failedText,failedText,failedText,failedText,failedText,failedText,failedText,failedText]).astype(str)
##        print('Failed due to TypeError\n')
##        i+=1
##        dayCOUNTER += 1
##        continue
##        
##        
##    try:
##        dataMatrix = dataGrab(filePath,i,dIterate,dayCOUNTER)
##    
##    except AttributeError:
##        print('atrribute error')
##        dayCOUNTER += 1
##        i+=1
##        continue
##
##    #dayCOUNTER += 1
##
##    genCount = 0
##    gCount=0
##    In = 0
##    #try:
##    ##---prints for number of rows in data matrix, aka number of tags from PI---##    
##    #for g in range(1):      #changed to 1 from 2 (2 did 2 iterations for first finacial yr and then second fin. yr)
##
##    try:
##        Ia, Ib, Ic, PPfa, PPfb, PPfc, QPfa, QPfb, QPfc, date, time, year, month, day = extract_3ph(dataMatrix[g][0],dataMatrix[g][1],dataMatrix[g][2],
##                                                                              dataMatrix[g+1][0],dataMatrix[g+1][1],dataMatrix[g+1][2],
##                                                                              dataMatrix[g+2][0],dataMatrix[g+2][1],dataMatrix[g+2][2],genCount)
##    except:
##        print('arrow error')
##        dayCOUNTER += 1
##        i+=1
##        continue
##    
####    Pfa, Pfb, Pfc,Ha,Hb,Hc = powerFactorCalc(PPfa, PPfb, PPfc, QPfa, QPfb, QPfc,Ia,Ib,Ic)
##    
##    
##
##    #vNi = [[Va,Vb,Vc],[Ia,Ib,Ic]]
##    #print(vNi)
##    #for p in range(2):        
##    #print(g)
##    a = g+1
##
##        
##    #if genCount >1:
##    if Ia != []:
##
##        # This says power, but I am actually passing current by not including the equation
####        Pa = np.array(Ha) #3*np.array(A)*
####        Pb = np.array(Hb)      # <--- 3*Vph*Iph = S (apparent power) is it not sqr root 3*Vph*Iph?
####        Pc = np.array(Hc) #3*np.array(C)* 3*np.array(B)*
##        Pa = np.array(Ia) #3*np.array(A)*
##        Pb = np.array(Ib)      # <--- 3*Vph*Iph = S (apparent power) is it not sqr root 3*Vph*Iph?
##        Pc = np.array(Ic)
##
##        #SIB, RIB, DPIB, maxP, minP, scenario, pFit = currentImbalanceDecomposition(Pa,Pb,Pc,txrName,a)
##
####        In,PfIn = neutralCurrent(Ia,Ib,Ic,Pfa, Pfb, Pfc)
##        Pfa = np.zeros(len(Ia))
##        Pfb = np.zeros(len(Ia))
##        Pfc = np.zeros(len(Ia))
##        PfIn = np.zeros(len(Ia))
##        In = neutralCurrent(Ia,Ib,Ic,Pfa, Pfb, Pfc)
##
##    
##
##        polarsData(In,Ia,Ib,Ic,PfIn,Pfa,Pfb,Pfc,date,time,year,month,day,dayCOUNTER)
##    
##                
##    genCount += 2
##
##    if a == 1:
##        time = '2018-2019'          ### ADD COLUMN FOR TOTAL MEDIAN LOAD TO TEST IF LOAD HAS INCREASED
##        
##    elif a != 1:
##        time = '2022-2023'
##
##    #arrDecompList = np.array([i,txrName,scenario,maxP,minP,pFit,DPIB,SIB,RIB,
##    #                          np.round(np.nanmedian(Pa),1),np.round(np.nanmedian(Pb),1),np.round(np.nanmedian(Pc),1),np.round(np.nanmedian(In),1),
##    #                          np.round(np.nanpercentile(Pa,95),1),np.round(np.nanpercentile(Pb,95),1),np.round(np.nanpercentile(Pc,95),1),
##    #                          np.round(np.nanpercentile(In,95),1)]).astype(str)
##    #print(arrDecompList)
##
##    #decompList = np.vstack((decompList,arrDecompList))
##    print('List updated: '+str(i))
##    #print(decompList)
##            
##    print('\n---------------------------next day?---------------------------\n')
##    
##    i+=1
##    #except pa.lib.ArrowInvalid:
##    #    print('Failed due to pyarrow exception\n')
##    #    i+=1
##    dayCOUNTER += 1
##    
##
####---printing summary of program---##
##print('List of Decomposition')
##
###       decompDF = pnd.DataFrame(decompList);nameDF = 'VIJI TEST {a} to {b}.csv'.format(a=recordI,b=i);decompDF.to_csv(nameDF,',')
##
##decompDF = pnd.DataFrame(decompList)
##
##nameDF = 'ok test 123 Component Decomposition_{a} to {b}.csv'.format(a=recordI,b=i)
##
##decompDF.to_csv(nameDF,',')
##
##print(decompDF)
##
##print()
##
##print(str(start+1)+' filepaths accessed')
##
##print(str(failed)+' filepaths failed to form')
##    
