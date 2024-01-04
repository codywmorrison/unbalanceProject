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
# V5 - 19/12/23


##--- module importation---##
from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
import matplotlib.pylab as pl

from PIconnect.PIConsts import RetrievalMode
import numpy as np
import pandas as pd
import cmath as cm
import PIconnect as PI
import seaborn as sns
import pprint
import datetime

from mpl_toolkits.mplot3d import axes3d
from matplotlib.dates import DateFormatter

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


def dataGrab(fPath):
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
        
        startT1 = '2018-07-02 00:00:00'
        endT1 = '2019-6-30 00:00:00'

        startT2 = '2022-07-02 00:00:00'
        endT2 = '2023-6-30 00:00:00'


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
    
    print('\nFound partial filepath from CSV: '+parPath+'\n')
    
    return txrName,parPath,monName

def extract_3ph(x,y,z,genCount):
    # This function passes in x, y, and z voltage or current data to be cleaned and
    # simplfied before being passed for forming graphs and unbalance calculations.
    # This function returns and appended matrix of cleaned data (shorter than input).  

    ##---initialising attributes for calculation---##
    A = np.zeros(len(x))
    B = np.zeros(len(A))
    C = np.zeros(len(A))

    xDF = pd.DataFrame(x)
    rawDatetime = xDF.index.values.tolist()
    refinedDatetime = pd.to_datetime(rawDatetime,unit='ns')
    refinedDatetime.columns = ['Date']
    
    refinedDate = pd.DataFrame()
    refinedDate['Month']= refinedDatetime.month
    
    appA = []
    appB = []
    appC = []

    ##datetime = []
    date = []
    time = []
    
    i=1 ## <---i = 1 is used as the datacleaning process uses x[i-2] which is only possible if i is greater than or equal to 2.

    ##---cleaning data for calculation and graphing---##
    for i in range(len(A)-2):
        i+=1

        #if str(x.iloc[i]) != 'No Data' and type(x.iloc[i]) is  not np.float64 and type(x.iloc[i]) is  not float:
        #    print(i)
        #    print(type(x.iloc[i]))
        #    print(x.iloc[i])

        ##---data cleaining for current---##
        if genCount > 1:
            if (type(x.iloc[i]) is np.float64 or type(x.iloc[i]) is float) and (round(x[i],1) != round(x[i-1],1) or round(x[i]-x[i-1]) != round(x[i]-x[i-2])):
                ##A[i]=x[i]
                appA.append(x[i])

                ##B[i]=y[i]
                appB.append(y[i])

                ##C[i]=z[i]
                appC.append(z[i])

                ##datetime.append(refinedDatetime[i])
                date.append(refinedDate.iloc[i])
                time.append(str(refinedDatetime[i].time()))

        ##---data cleaning for voltage--##
        else:
            if ((type(x.iloc[i]) is np.float64 or type(x.iloc[i]) is float) and x[i] > 0.5*240) and (round(x[i],1) != round(x[i-1],1) or round(x[i]-x[i-1]) != round(x[i]-x[i-2])):
                ##A[i]=x[i]
                appA.append(x[i])

                ##B[i]=y[i]
                appB.append(y[i])

                ##C[i]=z[i]
                appC.append(z[i])

                date.append(refinedDate.iloc[i])
                time.append(refinedDatetime[i].time())

    ##---printing data cleaning output---###
    points = round(len(A)-len(appA))
    
    if len(appA) != 0:
        
        print('\n{d} points of data were cleaned'.format(d=points))
        print('Meaning {y} of {m} days are unusable data\n'.format(y=round(points/144), m=round(len(A)/144)))

    elif len(appA) == 0:
        print('No available data. It was entirely cleaned.')
    
    ## maybe check if at any point it was because of 'No Data' or some other cleanse in the future and state what type of data was cleaned

    return appA, appB, appC, date, time

def assign_to_unbalanceList(unbalance,c,k,txrN,ubtype,unbalanceList):
    # This function assigns the calculated percenetile values to the finalList to be passed into
    # the main script to be printed at the end of the program.

    ##---intialisation and calculation---##
    percentileArr = np.zeros(7)
    percentiles = [1,5,25,50,75,95,99]
    
    for i in range(7):
        percentileArr[i] = round(np.percentile(unbalance,percentiles[i]),2) 

    ##---assigning to unbalanceList---##

    year = '2018 to 2019'
    if genCount == 1 or genCount == 3:
        year = '2022 to 2023'
    noDF = 'No. ' + str(k)

    unbalanceList = np.append(unbalanceList,[[noDF,txrN,year,ubtype,percentileArr[0],percentileArr[1],percentileArr[2],
                             percentileArr[3],percentileArr[4],percentileArr[5],percentileArr[6]]],axis=0)
    
    median = percentileArr[3]
    
    return unbalanceList, median

def calcUnbalance(Va,Vb,Vc,k,c,txrN,gCount,genCount,unbalanceList):
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
    onePercentileVUF = round(np.percentile(VufNeg,1),2)
    twofivePercentileVUF = round(np.percentile(VufNeg,25),2)
    medianVUF = round(np.percentile(VufNeg,50),2)
    sevenfivePercentileVUF = round(np.percentile(VufNeg,75),2)
    nineninePercentileVUF = round(np.percentile(VufNeg,99),2)

    ubtypeA = 'PVUR'
    ubtypeB = 'VUF'

    ##if (median > 0.5 and gCount == 0) or gCount > 0:
    

    ##---assigning percentiles to the list for the summary---##
    if genCount > 1:
        ubtypeA = 'PCUR'
    unbalanceList,median = assign_to_unbalanceList(LVUR,c,k,txrN,ubtypeA,unbalanceList)

    unbalanceDistribution(median,LVUR,txrN,ubtypeA,genCount)
    distributionProfile(Va,Vb,Vc,txrName,genCount)
    gCount += 1
    
    return VufNeg, unbalanceList, c, LVUR,gCount


def distributionProfile(Va,Vb,Vc,txrN,genCount):
    # This function calculates the phase voltage unbalance rate, PVUR (NEMA / IEEE standard)
    # and voltage unbalance factor, VUF (IEEE 'true' standard), as well as, their distribution
    # and percentiles. It then calls distribution and profile graphing functions to print a
    # visual representation of how the unbalance in this system is acting.

    ##---intialising pu data and skipping for empty arrays---##
    Apu = np.zeros(len(Va))
    Bpu = np.zeros(len(Va))
    Cpu = np.zeros(len(Va))
    
    if len(Va) == 0:
        return

    ##---coneversion to pu for voltage data---#
    if genCount < 2:
        for i in range(len(Va)):
            Apu[i] = Va[i]/230
            Bpu[i] = Vb[i]/230
            Cpu[i] = Vc[i]/230
    else:
        for i in range(len(Va)):
            Apu[i] = Va[i]
            Bpu[i] = Vb[i]
            Cpu[i] = Vc[i]

    ##---density curve plotting---##
    sns.kdeplot([Apu,Bpu,Cpu],legend=False)##,bins=15,kde=True) ##displot? ##clip=(0.95,1.05),

    xV = [(1.1,'r'),(1,'g'),(0.94,'r')]
    
    dataEntry = 'Voltage'
    dataEntry1 = 'Per-Unit Voltage (230V)'
    if genCount > 1:
            dataEntry = 'Current'
            dataEntry1 = dataEntry
    else:
        for a,b in xV:
            plt.axvline(x=a,color=b,linestyle='--')
    
    timeEntry = 'a'
    timeEntry2 = '18-19'
    if genCount == 1 or genCount ==3:
            timeEntry = 'b'
            timeEntry2 = '22-23'
            
    plt.title('Three-Phase Distribution of {x}'.format(x=txrN+' '+timeEntry2))
    plt.xlabel('{a}'.format(a=dataEntry1))
    plt.savefig('at1/{b}/Distributions/{a}.png'.format(a=txrN+' '+timeEntry2+' '+dataEntry,b=txrN),dpi=1200)
    plt.close()
    ##plt.show()

    return

def unbalanceDistribution(median,unbalanceData,txrN,ubtype,genCount):
    # This function plots and saves figures of unbalance, including PVUR, VUF and neutral
    # current. The plot is a kernel density distribution (as a density curve, not histogram).
    
    ##---density curve plotting---##
    fig, ax = plt.subplots()
    sns.kdeplot(unbalanceData,legend=False)
    ax.axvline(x=median,linestyle='--')
    ax.text(median,0.99,'Median',ha='left',va='top',rotation=90,transform=ax.get_xaxis_transform())
    xC = [(50,'r')]#,(25,'y'),(10,'g')]
    
    if ubtype == 'Neutral Current':
        ubtype = 'Neutral Current'
        dataEntry = 'Neutral Current'
        
        for a,b in xC:
            ax.axvline(x=a,color=b,linestyle='dotted')
            ax.text(a,0.99,'50A Standard',ha='left',va='top',rotation=90,transform=ax.get_xaxis_transform())
    else:
        dataEntry = 'Factor/Percentage Unbalance'
        if genCount > 1:
            dataEntry = 'Factor/Percentage Unbalance'
            ubtype = 'PCUR'
            ax.axvline(x=median,linestyle='--')
    
    timeEntry = 'a'
    timeEntry2 = '18-19'
    if genCount == 1 or genCount ==3:
            timeEntry = 'b'
            timeEntry2 = '22-23'
            
    plt.title('Unbalance ({t}) Distribution of {x}'.format(t=str(ubtype),x=txrN+' '+timeEntry2))
    plt.xlabel('{a}'.format(a=dataEntry))
    plt.savefig('at1/{b}/Distributions/{a}.png'.format(a=txrN+' '+timeEntry2+' '+str(ubtype),b=txrN),dpi=1200)
    plt.close()
    ##plt.show()

    return

def dataProfile(x,y,z,In,txrN,genCount):
    # This function plots and saves figures of current and voltage data.
    # This plots show a continous line graph for the entirety of the year.
    
    ##---plotting current/voltage profiles---##
    dataEntry = 'Voltage'
    if genCount > 1:
            dataEntry = 'Current'

    timeEntry = 'a'
    timeEntry2 = '18-19'
    if genCount == 1 or genCount ==3:
            timeEntry = 'b'
            timeEntry2 = '22-23'
    
    ##plt.subplot(2,1,1)
    plt.plot(x,label='A')
    plt.plot(y,label='B')
    plt.plot(z,label='C')
    ##plt.subplot(2,2,1)
    ##plt.plot(In,label='C')
    plt.legend()
    plt.title('{a} Profile of {b} '.format(a=dataEntry,b=txrN+''+timeEntry2))
    plt.savefig('at1/{b}/Profiles/{a}.png'.format(a=txrN+timeEntry+' '+dataEntry+' Profile',b=txrN),dpi=1200)
    plt.close()

    ##plt.show()

    return

def neutralCurrent(x,y,z,genCount,txrN,k,c,unbalanceList,date,time):
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
                                                                                            ##      phase angle measurements must be used. However, these are greatly effected by PV DER.
    
    ##medianIn = round(np.percentile(In,50),2)

    ubtype = 'Neutral Current'

    unbalanceList,medianIn = assign_to_unbalanceList(In,c,k,txrN,ubtype,unbalanceList)
    unbalanceDistribution(medianIn,In,txrN,ubtype,genCount)

    plot_neutralCurrent(In,date,time,genCount)

    return In, medianIn, unbalanceList

def plot_neutralCurrent(In,date,time,genCount):
    # This function plots a 3D graph of time, date and current.
    
    multi_arr = np.hstack((np.reshape(np.array(date),(len(date),1)).astype(float),
                           np.reshape(np.array(time),(len(time),1)),
                           np.reshape(np.zeros(len(time)),(len(time),1)).astype(float),
                           np.reshape(np.array(In),(len(In),1)).astype(float)))
    print("MULTI ARRAY")
    print(multi_arr)
    
    df_cn = pd.DataFrame(multi_arr)

    df_cn.columns = ['Month','Time','Refined Time','Neutral Current']

    


    for i in range(len(df_cn['Time'])):
        df_cn['Refined Time'].iloc[i] = int(datetime.datetime.strptime(df_cn['Time'].iloc[i], '%H:%M:%S').strftime("%H%M"))

    
    df_cn = df_cn.drop(labels='Time',axis=1)

    df_cn.columns = ['Month','Time','Neutral Current']

    print(df_cn)

    def printgroups(grp):
        return grp.median()

    df_cn = df_cn.astype(float)
    ##df_cn = df_cn.sort_values(by=['Month'])





    
##    df_group = df_cn.groupby(['Month','Time'],sort=False,as_index=False)['Neutral Current'].apply(printgroups)#['Neutral Current'].median()
##    print('DF_GROUP')
##    print(df_group)
##
##    df_time = df_cn.groupby(['Time'],sort=False,as_index=False)['Neutral Current'].apply(printgroups)#['Neutral Current'].median()
##    print('DF_TIME')
##    print(df_time)
##    
##    df_group_month = df_group.groupby(['Month'],sort=False,as_index=False)['Neutral Current'].median()
##    print('DF_MONTH')
##    print(df_group_month)

    
    #df_matrix = np.zeros((53000,25))
    ##for ii,ki in [[7,0],[8,1],[9,2],[10,3],[11,4],[12,5],[1,6],[2,7],[3,8],[4,9],[5,10],[6,11]]:
    #    df_matrix[ki] = df_cn[df_cn['Month'] == ii]
        

    
    df1 = df_cn[df_cn['Month'] == 1]
    df2 = df_cn[df_cn['Month'] == 2]
    df3 = df_cn[df_cn['Month'] == 3]
    df4 = df_cn[df_cn['Month'] == 4]
    df5 = df_cn[df_cn['Month'] == 5]
    df6 = df_cn[df_cn['Month'] == 6]
    df7 = df_cn[df_cn['Month'] == 7]
    df8 = df_cn[df_cn['Month'] == 8]
    df9 = df_cn[df_cn['Month'] == 9]
    df10 = df_cn[df_cn['Month'] == 10]
    df11 = df_cn[df_cn['Month'] == 11]
    df12 = df_cn[df_cn['Month'] == 12]

    arr1 = df1['Neutral Current'].to_numpy()
    arr2 = df2['Neutral Current'].to_numpy()
    arr3 = df3['Neutral Current'].to_numpy()
    arr4 = df4['Neutral Current'].to_numpy()
    arr5 = df5['Neutral Current'].to_numpy()
    arr6 = df6['Neutral Current'].to_numpy()
    arr7 = df7['Neutral Current'].to_numpy()
    arr8 = df8['Neutral Current'].to_numpy()
    arr9 = df9['Neutral Current'].to_numpy()
    arr10 = df10['Neutral Current'].to_numpy()
    arr11 = df11['Neutral Current'].to_numpy()
    arr12 = df12['Neutral Current'].to_numpy()

    
    

    fig = plt.figure()

    #ax = plt.subplot(projection='3d')

    zlist = [1,2,3,4,5,6,7,8,9,10,11,12]
    #dflist = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12]
    arrlist = [arr1,arr2,arr3,arr4,arr5,arr6,arr7,arr8,arr9,arr10,arr11,arr12]

    for hh in arrlist:
        print(hh)
    
    #for z in zs:

    saveV = []    
    xsave = []
    ysave = [[],[],[]]

    xcolumns = []#np.empty(shape=(1,200))#.reshape(1,200)
    print('1st x col')
    print(xcolumns)
    ycolumns = []#np.arange(200).reshape(1,200)
    zcolumns = []#np.arange(200).reshape(1,200)

    
    for iv,jk in [[arr1,1],[arr2,2],[arr3,3],[arr4,4],[arr5,5],[arr6,6],[arr7,7],[arr8,8],[arr9,9],[arr10,10],[arr11,11],[arr12,12]]:   #list(zip(arrlist,zlist)):    #
    #for iv in arrlist:
        #kdexy = sns.kdeplot(iv)#iv['Neutral Current'])
        kdexy = sns.kdeplot(iv)
        
        line = kdexy.lines[jk-1]
        x,y = line.get_data()
        
        xcolumns = np.append(xcolumns,x)
        
        ycolumns = np.append(ycolumns,y)

    plt.show()

    xJan,xFeb,xMar,xApr,xMay,xJune,xJuly,xAug,xSept,xOct,xNov,xDec = np.split(xcolumns,12)
    xMonth = np.vstack((xJan,xFeb,xMar,xApr,xMay,xJune,xJuly,xAug,xSept,xOct,xNov,xDec))
    
    yJan,yFeb,yMar,yApr,yMay,yJune,yJuly,yAug,ySept,yOct,yNov,yDec = np.split(ycolumns,12)
    yMonth = np.vstack((yJan,yFeb,yMar,yApr,yMay,yJune,yJuly,yAug,ySept,yOct,yNov,yDec))

    zMonth = np.array([np.ones(len(xJan)),np.ones(len(xJan))*2,np.ones(len(xJan))*3,np.ones(len(xJan))*4,np.ones(len(xJan))*5,np.ones(len(xJan))*6,
                       np.ones(len(xJan))*7,np.ones(len(xJan))*8,np.ones(len(xJan))*9,np.ones(len(xJan))*10,np.ones(len(xJan))*11,np.ones(len(xJan))*12])

    print(xMonth.shape)
    print(yMonth.shape)
    print(zMonth.shape)
    print(zMonth)
    
    
    ax = plt.subplot(projection='3d')
    for i in range(12):
        ax.plot(xMonth[i],zMonth[i],yMonth[i])


    monthLabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    
    ax.set_yticks(zlist,labels=monthLabels) 
    
    plt.show()
    
    #X = xMonth[6]#np.arange(0,300,1.5)#
    #Y = xMonth[9]#np.arange(0,300,1.5)#

    X = np.interp(np.arange(0,300,1.5),xMonth[6],yMonth[6])
    print('XMONTH6')
    print(xMonth[6])
    print(X)
    
    Y = np.interp(np.arange(0,300,1.5),xMonth[9],yMonth[9])
    print('XMONTH9')
    print(xMonth[9],label='month9')
    print(Y,label='Y')

    plt.plot(xMonth[9])
    plt.plot(Y)
    plt.legend()

    plt.show()
    
    
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(X**2+Y**2)*yMonth[6]*yMonth[9]#+Y*xMonth[9]*yMonth[6]            #np.array([yMonth[6], xMonth[9]])
    #plt.plot(np.arange(0,10,0.05),yMonth[6])
    #plt.plot(np.arange(0,10,0.05),yMonth[9])
    #plt.plot(xMonth[6],yMonth[6])
    #plt.plot(xMonth[9],yMonth[9])
    print(X)
    print(Y)
    print(Z)
    plt.show()
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(X, Y, Z,cmap='plasma',rstride=10, cstride=10)

    ax.plot(np.interp(np.arange(0,300,1.5),xMonth[6],yMonth[6]),yMonth[6],zs=0,zdir='y',color='r')
    ax.plot(np.interp(np.arange(0,300,1.5),xMonth[9],yMonth[9]),yMonth[9],zs=0,zdir='x',color='y')
    
    #ax.plot(xMonth[6],yMonth[6],zs=0,zdir='y',color='r')
    #ax.plot(xMonth[9],yMonth[9],zs=0,zdir='x',color='y')

    ax.set_xlim(0,300)
    ax.set_ylim(0,300)
    ax.set_zlim(0,0.05)
    
    ax.set_xlabel('x-axis')
    ax.set_ylabel('y-axis')
    ax.set_zlabel('z-axis')
    plt.show()

## THIS IS FOR HAVING TWO DISTRIBUTIONS AND Z IS THE DENSITY OF BOTH. SEE KDEPLOT3.py!!!!!
##    X = [ 507, 1100, 1105, 1080, 378, 398, 373]
##    Y = [1047,  838,  821,  838, 644, 644, 659]
##    Z = [ 300,   55,   15,   15,  55,  15,  15]

##    kernel = stats.gaussian_kde(np.array([arr1, arr2]))#, weights=(zMonth[3])) ## <---- this was a practice
##
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection="3d")
##    xs, ys = np.mgrid[0:300:30j, 0:300:30j] #[0:1500:30j, 0:1500:30j]##check these axis lol
##    zs = kernel(np.array([xs.ravel(), ys.ravel()])).reshape(xs.shape)
##    ax.plot_surface(xs, ys, zs, cmap="hot_r", lw=0.5, rstride=1, cstride=1, ec='k')
##    plt.show()





##    kernel = stats.gaussian_kde(np.array([xMonth[4], yMonth[4]]), weights=(zMonth[3])) ## <---- this was a practice
##
##    fig = plt.figure()
##    ax = fig.add_subplot(111, projection="3d")
##    xs, ys = np.mgrid[0:300:30j, 0:300:30j] #[0:1500:30j, 0:1500:30j]##check these axis lol
##    zs = kernel(np.array([xs.ravel(), ys.ravel()])).reshape(xs.shape)
##    ax.plot_surface(xs, ys, zs, cmap="hot_r", lw=0.5, rstride=1, cstride=1, ec='k')
##    plt.show()






    ax = plt.subplot(projection='3d')
    ax.plot_trisurf(xMonth[4], yMonth[4], xMonth[3])
    plt.show()



    
    listMonths = np.linspace(1,12,12)
    listTimes = np.linspace(0,2350,236)

    #arr_cn = df_cn.to_numpy()
    #print(arr_cn)
    arr_time = []
    arr_month = []
    

    
    ##df_month = pd.DataFrame(np.zeros(10000))#arr_month)
    ##df_month.columns = ['Neutral Current']
    ##df_time = pd.DataFrame(np.zeros(10000))#arr_time)
    ##df_time.columns = ['Neutral Current']
    

    
##    for i in range(len(time)):
##            
##        for j in listMonths:
##            #print('J is intiaited')
##            if int(arr_cn[i][0]) == j:
##                arr_month = np.append(arr_month,arr_cn[i][2]) ##df_month['Neutral Current'].loc[i] = arr_cn[i][2]
##                #print('J worked')
##                pass
##        for k in listTimes:
##            #print('k')
##            if arr_cn[i][1] == k:
##                arr_time = np.append(arr_time,arr_cn[i][2])
##                
##                #print('k worked')
##            ## this doesnt make sense its just creating one big  list and sorting it by time...?
##            #if df_cn['Time'].iloc[i] == k:
##            #    df_time['Neutral Current'].loc[i] = df_cn['Neutral Current'].loc[i]

    


            
    #arr_t_float = arr_time.astype(float)
    #arr_m_float = arr_month.astype(float)
    
    #print(arr_cn)
    #print(arr_m_float)
    #print(arr_t_float)


    
    
    ##np.meshgrid(df_cn['Time'],df_cn['Month'])
    ##sns.kdeplot(df_cn['Neutral Current'],df_cn['Time'])



##    fig = plt.figure()
##    ax = plt.axes(projection ='3d')
## 
##    # Creating plot
##    ax.plot_surface(df_cn['Counter'], df_cn['Month'], df_cn['Neutral Current'])
## 
##    # show plot
##    plt.show()







    
    
    fig = plt.figure()





    x = df_group['Month']
    y = df_group['Neutral Current']
    X, Y = np.meshgrid(x,y)
    Z = df_group

    index = df_group.index
    columns = df_group.columns

    x, y = np.meshgrid(df_group['Neutral Current'],df_group_month['Neutral Current'])#np.arange(12),np.arange(len(df_group['Time'])))
    z =np.arange(len(df_group['Time']))# pd.DataFrame([df_group['Neutral Current'] * df_group_month['Neutral Current']])        #np.array([[df_group[c][i] for c in columns] for i in index])


    #for j in range(
    
    print(z)
    
    
    
    axes = fig.add_subplot(projection='3d')
    axes.plot_surface(x, y, z)
    plt.show()

##    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
##    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, cmap=cm.Blues)
##
##    ax.set(xticklabels=[],
##           yticklabels=[],
##           zticklabels=[])
##
##    plt.show()
    


    



    sns.kdeplot(df_group['Neutral Current'],df_group_month['Neutral Current'])
    plt.show()



    fig = plt.figure()



    sns.kdeplot(df_time['Neutral Current'],df_month['Neutral Current'])
    plt.show()



    fig = plt.figure()
    ax = plt.axes(projection ='3d')


    sns.kdeplot(df_time['Neutral Current'],df_month['Neutral Current'])
    plt.show()

    
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')


    #hh_mm = DateFormatter('%H:%M')
##    fig = plt.figure()
##    ax = fig.add_subplot(projection='3d')
##    
##    for z in df_cn['Month']:
##        x = df_cn['RefinedTime']
##        y = df_cn['Neutral Current']
##        
##        ax.plot(x, y, zs=z, zdir='z')
##        obj1 = ax.fill_between(x, 0.5, y, step='pre', alpha=0.1) 
##        ax.add_collection3d(obj1, zs = z, zdir = 'z')
##
##    ax.scatter(df_cn['Month'], df_cn['RefinedTime'], df_cn['Neutral Current'], c='y', marker='o')
##
##    ax.set_xlabel('X Label')
##    ax.set_ylabel('Y Label')
##    ax.set_zlabel('Z Label')
##
##    plt.show()

    #ax.plot_wireframe(In.reshape(12,12).T, date.reshape(12,12).T, time.reshape(12,12).T)
    #plt.show
    

    print('passed into plot_neutralCurrent')
    return
    


##-----------------main script-----------------##
# View individual functions/methods for description

##---intialising matrix for percentiles---##
unbalanceList = np.zeros((1,11),dtype=object)
unbalanceList[0] = ['No. in DF23','TXR Name','Year','UF Type','p1','p5','p25','p50','p75','p95','p99']

##---intialising counters for main script---##
pCount = 0
failed = 0
i = 567
recordI = i
start = 0
end = 1
g=0

##---main script prints for number of sites given---##
for start in range(end):
    
    i, failed = indataCheck(i,failed)

    txrName, parPath, monName = pathGrab(i)

    filePath = feederGrab(txrName,monName,parPath)

    dataMatrix = dataGrab(filePath)

    genCount = 0
    gCount=0
    In = 0

    ##---prints for number of rows in data matrix, aka number of tags from PI---##    
    for g in range(len(dataMatrix)):

        A, B, C, date, time = extract_3ph(dataMatrix[g][0],dataMatrix[g][1],dataMatrix[g][2],genCount)

        if genCount >1:
            In, medianIn, unbalanceList = neutralCurrent(A,B,C,genCount,txrName,i,pCount,unbalanceList,date,time)
        
        dataProfile(A,B,C,In,txrName,genCount)
        
        if len(A) != 0:
            
            Vuf,unbalanceList,pCount,LVUR,gCount = calcUnbalance(A,B,C,i,pCount,txrName,gCount,genCount,unbalanceList)
            
        ##distributionPercentile(finalList)  <---- to be done (distribution of the LVUR or VUF)
            
            ## distribution of the neutral current (why was 50 A chosen; is that usually an 'unbalanced system',
            ## or is 50 A chosen because neutral wires can handle ~50 A before exceeding temperature rating,
            ## or does 50A correspond to an non-compliant neutral voltage)
            
        print('Current Count is currently: '+str(genCount))
        if genCount > 1:
            print('Current Count is greater than 1: '+str(genCount))

        genCount += 1

    
    print('\n---------------------------next monitor---------------------------\n')
    
    i+=1


##---printing summary of program---##
print('List of Unbalance Percentiles')

unbalanceDF = pd.DataFrame(unbalanceList)
unbalanceDF.columns = unbalanceDF.iloc[0]
unbalanceDF = unbalanceDF[1:]

nameDF = 'siteUnbalance_{a} to {b}.csv'.format(a=recordI,b=i)

unbalanceDF.to_csv(nameDF,',')

print(unbalanceDF)

print()

print(str(start)+' filepaths accessed')

print(str(failed)+' filepaths failed to form')

##input next (otherwise stop operation maybe?)

##maybe input for year like (18/19 or 22/23) -> could pull profile or maybe not -> could pull distribution of balance????? hmm

##finalList[

    
