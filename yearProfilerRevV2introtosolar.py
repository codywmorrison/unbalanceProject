# Written by Cody Slade (Part-time Student Engineer) for Energy Queensland
# under Trevor Gear (Principal Engineer)
#
# This script estimates load profiles that will be used in PowerFactory for modelling
# unbalance in LV networks (downstream of a distribution transformer)
#
# This script was written as part of a continued student placement with Energy
# Queensland during part-time work 2024. Thankyou. Cody.
#
# V2.1 - 27/3/24


from matplotlib import pyplot as plt
import matplotlib as mpl
from scipy import stats
from scipy.stats.stats import pearsonr
import matplotlib.pylab

from PIconnect.PIConsts import RetrievalMode
import numpy as np      ## <- numpy used for majority of mathematical operations
import polars as pl     ## <- Pandas was originally used, but changed to Polars for speed
import pyarrow as pa
import pandas as pd
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

# --- main ---


data = pl.read_csv("customerDataTX2.csv",separator=",",infer_schema_length=200000)

dataSolar = data.filter((pl.col("NMISuffix") == 'B1'))# | (pl.col("NMISuffix") == 'B2'))

data = data.filter(pl.col("NMISuffix") == 'E1')


#for load
partForFollowing = data.with_columns(DT = pl.col('Date Time').str.to_datetime().cast(pl.Datetime)).sort('DT')
datetimePartition = partForFollowing.partition_by("Day Time", maintain_order=True) # <- change to time to output 1 day...\

#for solar
partForFollowingSol = dataSolar.with_columns(DT = pl.col('Date Time').str.to_datetime().cast(pl.Datetime)).sort('DT')
datetimePartitionSol = partForFollowingSol.partition_by("Day Time", maintain_order=True) # <- change to time to output 1 day...\

numpyEnergy = np.zeros(len(datetimePartition))
numpyEnergyLoad = np.zeros(len(datetimePartition))
numpyEnergySolar = np.zeros(len(datetimePartition))
numpyDateTime = np.zeros(len(datetimePartition)).astype(str)


with PI.PIAFDatabase(database="PQ Monitors") as database:
        
    element = database.descendant('EQL\\SOUTHEAST\\BRISBANE SOUTH\\SSKSN\\KSN15\\SP54878-B\\NM17178')    #'EQL\\SOUTHEAST\\BRISBANE SOUTH\\SSH22\LGL3A\\SP1809-G\\SP1809-G-TR1' TX1
    attvalues = iter(element.attributes.values())                       #'EQL\\SOUTHEAST\\BRISBANE SOUTH\\SSKSN\\KSN15\\SP54878-B\\NM17178'# THIS IS ON EXISTING TRANSFORMER.... it had reverse power flow sp1809-g

    attList = [next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues),
               next(attvalues),next(attvalues),next(attvalues),next(attvalues),next(attvalues)]

    ##---timeline to pull data---##
    intT = '30m'


    #startT2 = '2023-05-29 00:00:00'
    #endT2 = '2023-06-30 00:00:00'
    startT1 = '2023-07-1 00:00:00'
    endT1 = '2024-3-1 00:00:00'
    

    ##---search and assign Voltage and Current data to matrix---##
    for att in range(len(attList)):

        if attList[att].name == 'P': #change to P and add the two values?   #change to P? ---- changed to P when added solar (whats B2 and B1... is B1 solar and B2 generator or smthn?)
            S = attList[att].interpolated_values(startT1,endT1,intT)


sPandas = pd.DataFrame(S)


### ----- Percentile Profiling ----- ###

print('\nExtrapolated Profiling of Customer Load')
print('\nDirectioned Apparent Power Estimation... Lmlt = load coefficient, Smlt = solar coefficient and C = y-axis adjustment\n')
print("Purpose: Deviation between the estimated curve and historical data's profile (|C1y - C2y|) will be minimised through iterative comparison\n with change in parameters; Lmlt, Slmt and C. A Pearson correlation coefficient will be at with the chosen 'minimised' curve parameters")

# Set below resolution type to edit the precision of the estimation. A 'HIGH' resolution will be extremely slow.

countType = 'LOW'       #resolution type for iterations

if countType == 'LOW':
    pRange = 60
    lCount = 500
    sCount = 40
    cCount = 25
elif countType == 'HIGH':
    pRange = 90
    lCount = 750
    sCount = 100
    cCount = 50


S = sPandas.to_numpy()/1000
sPolars = pl.from_pandas(sPandas)/1000

S[S<0] = 0


rValues = []
rMax = []

st = 0
en = st + len(S)  #4000   #len(S)

strongThreshold = 0.775

minList = []
minMax = []
minMlt = []
minPerc = []
minSMlt = []
minAdd = []

for percentile in range(pRange):    #pRange is adjusted by resolution
    percentileRank = percentile + (100 - pRange)/2             #percentile
    percentileRankSol = 99

    for DT in range(len(datetimePartition)):
        numpyEnergyLoad[DT] = np.nanpercentile(datetimePartition[DT].select(["Value (Energy)"]).to_numpy(),percentileRank)
        numpyEnergySolar[DT] = np.abs(np.nanpercentile(datetimePartitionSol[DT].select(["Value (Energy)"]).to_numpy(),percentileRankSol))

    S = sPandas.to_numpy()/1000
    S[S<0] = 0
    
    S = S[st:en].reshape(en-st,1).T
    
    nEl = numpyEnergyLoad[st:en].reshape(en-st,1).T
    nEs = numpyEnergySolar[st:en].reshape(en-st,1).T

    dMax = []
    dAvg = []
    dMlt = []
    sdMlt = []
    uAdd3=[]
    dPerc = []
    dValues = []
    
    mltMain = 200       # mltMain is used to scale up the load coefficient

    for mlt in np.linspace(0.01,30,lCount): #500 for lCount  

        #mlt = 150
        sMax=[]
        sMlt = []
        uAdd2 = []
        
        #dAvg = []
        #dValues=[]

        #this slows down the process significantly... by factors... Do not increase no. of iterations unless time is available
        for mltS in np.linspace(0.05,50,sCount): #40 for sCount
            
            uAdd = []
            uMax = []
            
            for uNd in np.linspace(-35,35,cCount): #20 for cCount
                nEc = np.subtract(nEl[0].T*mlt*mltMain, nEs[0].T*mltS) + uNd##(can add)
                nEc[nEc<0] = 0

                deviation = np.abs(np.subtract(nEc,S[0]))
                max_dev = np.nanmedian(deviation)      
                
                uMax = np.hstack((uMax,max_dev))
                uAdd = np.hstack((uAdd,uNd))


            uMaxIndex = np.where(uMax==min(uMax))

            
            sMax = np.hstack((sMax,uMax[uMaxIndex]))
            sMlt = np.hstack((sMlt,mltS))
            uAdd2 = np.hstack((uAdd2,uAdd[uMaxIndex]))    

        
##        if max_dev < 10:        # 10 is a desirable median max-deviation...
##            plt.plot(nEc,label='subtracted')
##            plt.plot(S[0],label='S PI')
##            plt.legend()
##            plt.show()


        sMaxIndex = np.where(sMax==min(sMax))

        dMax = np.hstack((dMax,sMax[sMaxIndex]))
        dMlt = np.hstack((dMlt,mlt*mltMain))
        dPerc = np.hstack((dPerc,percentileRank))
        sdMlt = np.hstack((sdMlt,sMlt[sMaxIndex]))
        uAdd3 = np.hstack((uAdd3,uAdd2[sMaxIndex]))
    
    dMaxIndex = np.where(dMax==min(dMax))
    
    print('Output for '+str(round(dPerc[dMaxIndex][0]))+'th P. Curve: Median Deviation = ' +str(round(dMax[dMaxIndex][0],2))+' at Lmlt = '+str(round(dMlt[dMaxIndex][0],2))+', Smlt = ' + str(round(sdMlt[dMaxIndex][0],2))+' and C = ' + str(round(uAdd3[dMaxIndex][0],2)))


    minAdd = np.hstack((minAdd,uAdd3[dMaxIndex][0]))
    minMlt = np.hstack((minMlt,dMlt[dMaxIndex][0]))
    minSMlt = np.hstack((minSMlt,sdMlt[dMaxIndex][0]))
    minPerc = np.hstack((minPerc,dPerc[dMaxIndex][0]))
    
    minMax = np.hstack((minMax,min(dMax)))



minMaxIndex = np.where(minMax==min(minMax))


### --- printed output of estimation --- ###

print('\n\nOutput of Directioned Apparent Power Estimation...\n')

print('Percentile = '+str(int(minPerc[minMaxIndex])))
print('Median Max-Deviation '+str(round(minMax[minMaxIndex][0]),2))
print('Lmlt = '+str(round(minMlt[minMaxIndex][0],2)))
print('Smlt = ' + str(round(minSMlt[minMaxIndex][0],2)))
print('C = '+str(round(minAdd[minMaxIndex][0],2)))


for DT in range(len(datetimePartition)):
    numpyEnergyLoad[DT] = np.nanpercentile(datetimePartition[DT].select(["Value (Energy)"]).to_numpy(),percentileRank)
    numpyEnergySolar[DT] = np.abs(np.nanpercentile(datetimePartitionSol[DT].select(["Value (Energy)"]).to_numpy(),percentileRankSol))

numpyEnergy = np.subtract(numpyEnergyLoad*minMlt[minMaxIndex][0],numpyEnergySolar*minSMlt[minMaxIndex][0]) + minAdd[minMaxIndex][0]
numpyEnergy[numpyEnergy<0] = 0


### --- pearson correlation coefficient, this may be expanded upon --- ###

pearsonR = np.corrcoef(numpyEnergy[0:len(S[0])],S[0])

if pearsonR[1,0] > 0.75:
    print('Very Strong Pearson Correlation:\n'+str(pearsonR[1,0]))
elif pearsonR[1,0] > 0.7:
    print('Strong Pearson Correlation:\n'+str(pearsonR[1,0]))
else:
    print('Pearson Correlation Coefficient of:\n'+str(pearsonR[1,0]))

percentileValues = pl.from_numpy(numpyEnergy,schema=['Energy Usage (kVA scaled) x{a} @ {b}'.format(a=str(round(minMlt[minMaxIndex][0],2)),b=str(int(minPerc[minMaxIndex][0])))])
dtValues = pl.from_numpy(numpyDateTime,schema=['Date Time'])

outputData = dtValues.hstack(percentileValues)

print(outputData)#.with_columns(DT = pl.col('Date Time').str.to_datetime().cast(pl.Datetime)).sort('DT'))

outputData.write_csv('Yearly Profile V2.1 (Profiled with Deviation) for TX2.csv')


fig = plt.subplots()
plt.plot(S[0],label='Power from PI')
plt.plot(np.subtract(numpyEnergyLoad*minMlt[minMaxIndex][0],numpyEnergySolar*minSMlt[minMaxIndex][0] + minAdd[minMaxIndex][0]),label='Calculated Profile')
plt.legend()
plt.title(str(int(minPerc[minMaxIndex][0])) + 'th Profile and Measured Apparent Power from PI (x'+str(round(minMlt[minMaxIndex][0],2))+')')
plt.xlabel('Time (1 point = 30min)')
plt.ylabel('Apparent Power (kVA)')
plt.show()




