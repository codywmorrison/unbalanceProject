# Written by Cody Morrison (Vac. Student) for Energy Queensland under Trevor Gear (Principal Engineer)
#
# This script generates a bar graph
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

from PIconnect.PIConsts import RetrievalMode
import numpy as np      ## <- numpy used for majority of mathematical operations
import polars as pd     ## <- Pandas was originally used, but changed to Polars for speed
import pyarrow as pa
import pandas as pnd
import cmath as cm
import math
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
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
                   next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues)]

        ##---timeline to pull data---##
        intT = '10m'

        #siteDF23 = np.genfromtxt('DFListV1.csv',dtype=str,delimiter=',',usecols=(11),skip_header=1)

        #sliceNo = len(siteDF23[i])-4
        
        #yearInt = int(siteDF23[i][sliceNo:])
        #print(sliceNo,yearInt)

        #startT1 = '2022-07-01 00:00:00'
        #endT1 = '2023-6-30 00:00:00'

        startT1 = '2023-11-01 00:00:00'
        endT1 = '2023-12-1 00:00:00'
        

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

            

            if attList[att].name == 'CUR_C':
                CdataCa = attList[att].interpolated_values(startT1,endT1,intT)
                #CdataCb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_B':
                CdataBa = attList[att].interpolated_values(startT1,endT1,intT)
                #CdataBb = attList[att].interpolated_values(startT2,endT2,intT)
            if attList[att].name == 'CUR_A':
                CdataAa = attList[att].interpolated_values(startT1,endT1,intT)
                #CdataAb = attList[att].interpolated_values(startT2,endT2,intT)        #,[VdataCb,VdataBb,VdataAb],,[CdataCb,CdataBb,CdataAb]


            if attList[att].name == 'P_C':
                PdataC = attList[att].interpolated_values(startT1,endT1,intT)
                
            if attList[att].name == 'P_B':
                PdataB = attList[att].interpolated_values(startT1,endT1,intT)
                
            if attList[att].name == 'P_A':
                PdataA = attList[att].interpolated_values(startT1,endT1,intT)


            if attList[att].name == 'Q_C':
                QdataC = attList[att].interpolated_values(startT1,endT1,intT)
                
            if attList[att].name == 'Q_B':
                QdataB = attList[att].interpolated_values(startT1,endT1,intT)
                
            if attList[att].name == 'Q_A':
                QdataA = attList[att].interpolated_values(startT1,endT1,intT)         

        
        dataMatrix = [[CdataCa,CdataBa,CdataAa],[PdataC,PdataB,PdataA],[QdataA,QdataB,QdataC]] # i did swap these

        print(PdataC)
        

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

def extract_3ph(d,e,f,x,y,z,j,k,l,genCount):
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

    jDF = pnd.DataFrame(j)
    j = pd.from_pandas(jDF)
    kDF = pnd.DataFrame(k)
    k = pd.from_pandas(kDF)
    lDF = pnd.DataFrame(l)
    l = pd.from_pandas(lDF)

    

    dDF = pnd.DataFrame(d)
    d = pd.from_pandas(dDF)
    
    eDF = pnd.DataFrame(e)
    e = pd.from_pandas(eDF)
    fDF = pnd.DataFrame(f)
    f = pd.from_pandas(fDF)

    A = np.zeros(len(x))
    B = np.zeros(len(A))
    C = np.zeros(len(A))

    J = np.zeros(len(j))
    K = np.zeros(len(J))
    L = np.zeros(len(J))

    D = np.zeros(len(d))
    E = np.zeros(len(D))
    F = np.zeros(len(D))

    #print(x,y,z,d,e,f)
    
    rawDatetime = xDF.index.values.tolist()
    refinedDatetime = pnd.to_datetime(rawDatetime,unit='ns')
    refinedDatetime.columns = ['Date']
    

    
    refinedDate = pnd.DataFrame()
    refinedDate['Date']= refinedDatetime.date
    refinedDate = pd.from_pandas(refinedDate)
    


    refinedDay = pnd.DataFrame()
    refinedDay['Day']= refinedDatetime.day
    refinedDay = pd.from_pandas(refinedDay)

    refinedMonth = pnd.DataFrame()
    refinedMonth['Month']= refinedDatetime.month
    refinedMonth = pd.from_pandas(refinedMonth)

    refinedYear = pnd.DataFrame()
    refinedYear['Year']= refinedDatetime.year
    refinedYear = pd.from_pandas(refinedYear)

    refinedDatetime = pd.from_pandas(refinedDatetime)


    appA = []
    appB = []
    appC = []

    appD = []
    appE = []
    appF = []

    appJ = []
    appK = []
    appL = []

    ##datetime = []
    date = []
    time = []
    year = []
    day = []
    month = []
    
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
                appJ.append(j.row(i)[0])
                ##B[i]=y[i]
                appB.append(y.row(i)[0])
                appE.append(e.row(i)[0])
                appK.append(k.row(i)[0])
                ##C[i]=z[i]
                appC.append(z.row(i)[0])
                appF.append(f.row(i)[0])
                appL.append(l.row(i)[0])
                
                date.append(str(refinedDate.row(i)[0]))#[0]
                year.append(refinedYear.row(i)[0])#[0]
                month.append(refinedMonth.row(i)[0])#[0]
                day.append(refinedDay.row(i)[0])#[0]
                time.append(str(refinedDatetime[i].time()))

        ##---data cleaning for voltage--## #(x.row(i)[0] > 120) and
        #elif round(x.row(i)[0],1) != round(x.row(i-1)[0],1) or round(x.row(i)[0]-x.row(i-1)[0]) != round(x.row(i)[0]-x.row(i-2)[0]):
        elif round(x.row(i)[0],2) != round(x.row(i-1)[0],2) or round(x.row(i)[0]-x.row(i-1)[0],2) != round(x.row(i)[0]-x.row(i-2)[0],2):#(type(x[i]) is np.float64 or type(x[i]) is float) and####(x[i] > 120) and
            ##A[i]=x[i]
            appA.append(x.row(i)[0])
            appD.append(d.row(i)[0])
            appJ.append(j.row(i)[0])
            ##B[i]=y[i]
            appB.append(y.row(i)[0])
            appE.append(e.row(i)[0])
            appK.append(k.row(i)[0])
            ##C[i]=z[i]
            appC.append(z.row(i)[0])
            appF.append(f.row(i)[0])
            appL.append(l.row(i)[0])
            
            date.append(str(refinedDate.row(i)[0]))
            year.append(refinedYear.row(i)[0])#[0]
            month.append(refinedMonth.row(i)[0])#[0]
            day.append(refinedDay.row(i)[0])#[0]
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




    

    

    return appD, appE, appF, appA, appB, appC, appJ, appK, appL, date, time, year, month, day

def neutralCurrent(x,y,z):
    # This function calculates the neutral current of the system (at the PQ monitor).

    if len(Ia) == 0:
        In = np.zeros(len(x))
        print('Neutral calculation failed')
        return In

    ##---calculating neutral current---##
    In = np.zeros(len(x))

    for i in range(len(x)):
        In[i] = np.sqrt(x[i]**2 + y[i]**2 + z[i]**2 - x[i]*y[i] - y[i]*z[i] - z[i]*x[i])    ## <--- Equation assumes no non-resistive load (phase angles are 0,-120,120). For a non-resistive load, 
                                                                                            ##      phase angle measurements must be used. However, these are greatly effected by PV DER, i.e., unreliable.
    return In


def polarsData(In,Ia,Ib,Ic,Pfa, Pfb, Pfc,date,time, year, month, day):
    ## Polars is set in pd not pl...
    
    print("in polars")

    data = {"Date": date, "Time": time, "Year": year, "Month": month, "Day": day,
            "Neutral Current": In, "A Current": Ia, "B Current": Ib, "C Current": Ic,
            "A Power Factor": Pfa, "B Power Factor": Pfb, "C Power Factor": Pfc}

    
    polarsDF = pd.DataFrame(data)

    uniqueDate = polarsDF.select(["Date"]).unique(keep='first',maintain_order=True)#.to_numpy()#.select(["Date"])subset=["Date"],

    #print(uniqueDate)

    listDate = uniqueDate.to_numpy()
    print(listDate[10][0])
    print(str(listDate[10][0]))

    count = 1

    polarsDay = polarsDF.filter(pd.col("Date") == str(listDate[10][0]))
    maxNeutral = polarsDay.select(["Neutral Current"]).max()             ## we want max neutral current and phase currents for that max, not max of all 4 currents.
    print(maxNeutral.to_numpy())
    
    maxRow = polarsDay.filter(pd.col("Neutral Current") == maxNeutral.to_numpy())

    maxValues = maxRow.select(["Neutral Current","A Current","B Current","C Current","A Power Factor","B Power Factor","C Power Factor"]) 
    
    #print(maxValues)
    i = 0
    PA_thresh = 120



    
    for dateT in listDate:
        #print('iterate')
        #print(str(dateT))

        
        polarsDay = polarsDF.filter(pd.col("Date") == str(dateT[0]))

        maxNeutral = polarsDay.select(["Neutral Current"]).max()             ## we want max neutral current and phase currents for that max, not max of all 4 currents.
        #print(maxNeutral.to_numpy())
    
        maxRow = polarsDay.filter(pd.col("Neutral Current") == maxNeutral.to_numpy())

        maxValues = maxRow.select(["Time","Neutral Current","A Current","B Current","C Current","A Power Factor","B Power Factor","C Power Factor"]) #,"A Current","B Current","C Current"]).max()
        print(maxValues) ## PF has been added, but should multiple current by -1 if it is within a certain range (close to reversed PF)

        
        
    ## MAYBE SET UP THE IF AND ELIF INTO A FUNCTION?
        if count == 1:
            maxValuesList = np.hstack([[dateT],maxValues.to_numpy()])
            count = 2

            if maxValuesList[i][6] > PA_thresh or maxValuesList[i][6] < -PA_thresh:
                maxValuesList[i][3] = maxValuesList[i][3]*-1

            elif maxValuesList[i][7] > PA_thresh+120 or maxValuesList[i][7] < -PA_thresh+120:
                maxValuesList[i][4] = maxValuesList[i][4]*-1

                
            elif maxValuesList[i][8] > PA_thresh-120 or maxValuesList[i][8] < -PA_thresh-120:
                maxValuesList[i][5] = maxValuesList[i][5]*-1


            
        else:
            maxValuesList = np.vstack((maxValuesList,np.hstack([[dateT],maxValues.to_numpy()])))

            if maxValuesList[i][6] > PA_thresh or maxValuesList[i][6] < -PA_thresh:
                maxValuesList[i][3] = maxValuesList[i][3]*-1

            elif maxValuesList[i][7] > PA_thresh+120 or maxValuesList[i][7] < -PA_thresh+120:
                maxValuesList[i][4] = maxValuesList[i][4]*-1

                
            elif maxValuesList[i][8] > PA_thresh-120 or maxValuesList[i][8] < -PA_thresh-120:
                maxValuesList[i][5] = maxValuesList[i][5]*-1

        i += 1

    #print(maxValues)
    
    #print(maxValuesList)

    x_axis = maxValuesList.T[0]
    print(x_axis)
    
    y_n = maxValuesList.T[2]
    y_a = maxValuesList.T[3]
    y_b = maxValuesList.T[4]
    y_c = maxValuesList.T[5]

    x_ticks = np.arange(len(x_axis))

    plt.bar(x_ticks - 0.3,y_n,0.2,label = 'n Current',zorder = 6)
    plt.bar(x_ticks - 0.1,y_a,0.2,label = 'a Current',zorder = 6)
    plt.bar(x_ticks + 0.1,y_b,0.2,label = 'b Current',zorder = 6)
    plt.bar(x_ticks + 0.3,y_c,0.2,label = 'c Current',zorder = 6)

    plt.title("Maximum Phase and Neutral Currents per Day")
    plt.xlabel("Day")
    plt.ylabel("Current (A)")
    plt.xticks(x_ticks,x_axis,rotation=90) #tick by month not day
    plt.legend()
    plt.grid(zorder = -1)
    plt.show()
    
        


    

    print(maxIn)


    print(maxIn2)
    
    #print(polarsDay)



    

    #print(polarsDF)

    return 

def powerFactorCalc(PPfa, PPfb, PPfc, QPfa, QPfb, QPfc):

    ##something does seem right

    Pfa = np.zeros(len(PPfa))
    Pfb = np.zeros(len(Pfa))
    Pfc = np.zeros(len(Pfa))

    for i in range(len(PPfa)):
        Pfa[i] = math.acos(PPfa[i]/(np.sqrt(PPfa[i]**2 + QPfa[i]**2)))*180/math.pi
        Pfb[i] = math.acos(PPfb[i]/(np.sqrt(PPfb[i]**2 + QPfb[i]**2)))*180/math.pi + 120
        Pfc[i] = math.acos(PPfc[i]/(np.sqrt(PPfc[i]**2 + QPfc[i]**2)))*180/math.pi - 120

    print(Pfa,Pfb,Pfc)
    print(max(Pfa),min(Pfa))
    print(max(Pfb),min(Pfb))
    print(max(Pfc),min(Pfc))

    return Pfa,Pfb,Pfc



##-----------------main script-----------------##
# View individual functions/methods for description

##---intialising matrix for percentiles---##
decompList = np.zeros((1,17),dtype=object)
decompList[0] = ['No. in DF23','TXR Name','Type','50th I Max Phase','50th I Min Phase','% Fit','DPIB Med. (%)','SIB Med. (A)','RIB Med. (A)','50th Ia',
                 '50th Ib','50th Ic','50th In','95th Ia','95th Ib','95th Ic','95th In']

##---intialising counters for main script---##
pCount = 0
failed = 0
i = 10
recordI = i
start = 0

valueForInput = 100

end = 1#valueForInput - i
g=0


#Days

##---main script prints for number of sites given---##
#for dIterate in Days:
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
    #try:
    ##---prints for number of rows in data matrix, aka number of tags from PI---##    
    for g in range(1):      #changed to 1 from 2 (2 did 2 iterations for first finacial yr and then second fin. yr)

        Ia, Ib, Ic, PPfa, PPfb, PPfc, QPfa, QPfb, QPfc, date, time, year, month, day = extract_3ph(dataMatrix[g][0],dataMatrix[g][1],dataMatrix[g][2],
                                                                              dataMatrix[g+1][0],dataMatrix[g+1][1],dataMatrix[g+1][2],
                                                                              dataMatrix[g+2][0],dataMatrix[g+2][1],dataMatrix[g+2][2],genCount)

        
        Pfa, Pfb, Pfc = powerFactorCalc(PPfa, PPfb, PPfc, QPfa, QPfb, QPfc)
        
        

        #vNi = [[Va,Vb,Vc],[Ia,Ib,Ic]]
        #print(vNi)
        #for p in range(2):        
        #print(g)
        a = g+1

            
        #if genCount >1:
        if Ia != []:

            # This says power, but I am actually passing current by not including the equation
            Pa = np.array(Ia) #3*np.array(A)*
            Pb = np.array(Ib)      # <--- 3*Vph*Iph = S (apparent power) is it not sqr root 3*Vph*Iph?
            Pc = np.array(Ic) #3*np.array(C)* 3*np.array(B)*
            

            #SIB, RIB, DPIB, maxP, minP, scenario, pFit = currentImbalanceDecomposition(Pa,Pb,Pc,txrName,a)

            In = neutralCurrent(Pa,Pb,Pc)


            polarsData(In,Ia,Ib,Ic,Pfa,Pfb,Pfc,date,time,year,month,day)
        
                    
        genCount += 2

        if a == 1:
            time = '2018-2019'          ### ADD COLUMN FOR TOTAL MEDIAN LOAD TO TEST IF LOAD HAS INCREASED
            
        elif a != 1:
            time = '2022-2023'

        #arrDecompList = np.array([i,txrName,scenario,maxP,minP,pFit,DPIB,SIB,RIB,
        #                          np.round(np.nanmedian(Pa),1),np.round(np.nanmedian(Pb),1),np.round(np.nanmedian(Pc),1),np.round(np.nanmedian(In),1),
        #                          np.round(np.nanpercentile(Pa,95),1),np.round(np.nanpercentile(Pb,95),1),np.round(np.nanpercentile(Pc,95),1),
        #                          np.round(np.nanpercentile(In,95),1)]).astype(str)
        #print(arrDecompList)

        #decompList = np.vstack((decompList,arrDecompList))
        print('List updated: '+str(i))
        #print(decompList)
            
    print('\n---------------------------next monitor---------------------------\n')
    
    i+=1
    #except pa.lib.ArrowInvalid:
    #    print('Failed due to pyarrow exception\n')
    #    i+=1
    

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
    
