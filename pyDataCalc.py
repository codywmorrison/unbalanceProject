import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv("datasheet summer.csv")

print(df)

PaQ = np.zeros(48)
PbQ = np.zeros(48)
PcQ = np.zeros(48)
#hourCount = np.arange(1,49,1)
#print(hourList)



##for hHour in np.arange(1,49,1):
##    #print(hHour)
##    group = df.loc[df['Hour'] == hHour]
##    #powerV = group.Pa.reset_index().Pa.values
##    powerV = np.vstack((group.Pa.reset_index().Pa.values,
##                        group.Pb.reset_index().Pb.values,
##                        group.Pc.reset_index().Pc.values))
##    #print(powerV)
##    #print('1 iteration')
###
##    if hHour == 1:
##        hourList = powerV
##    else:
##        hourList = np.vstack((hourList,powerV))
##
##
##    
##print(hourList)
###fig = subplot()
###plt.show()

h1 = df.loc[df['Hour'] == 1]

h2 = df.loc[df['Hour'] == 2]
h3 = df.loc[df['Hour'] == 3]
h4 = df.loc[df['Hour'] == 4]
h5 = df.loc[df['Hour'] == 5]
h6 = df.loc[df['Hour'] == 6]
h7 = df.loc[df['Hour'] == 7]
h8 = df.loc[df['Hour'] == 8]
h9 = df.loc[df['Hour'] == 9]
h10 = df.loc[df['Hour'] == 10]
h11 = df.loc[df['Hour'] == 11]
h12 = df.loc[df['Hour'] == 12]
h13 = df.loc[df['Hour'] == 13]
h14 = df.loc[df['Hour'] == 14]
h15 = df.loc[df['Hour'] == 15]
h16 = df.loc[df['Hour'] == 16]
h17 = df.loc[df['Hour'] == 17]
h18 = df.loc[df['Hour'] == 18]
h19 = df.loc[df['Hour'] == 19]
h20 = df.loc[df['Hour'] == 20]
h21 = df.loc[df['Hour'] == 21]
h22 = df.loc[df['Hour'] == 22]
h23 = df.loc[df['Hour'] == 23]
h24 = df.loc[df['Hour'] == 24]
h25 = df.loc[df['Hour'] == 25]
h26 = df.loc[df['Hour'] == 26]
h27 = df.loc[df['Hour'] == 27]
h28 = df.loc[df['Hour'] == 28]
h29 = df.loc[df['Hour'] == 29]
h30 = df.loc[df['Hour'] == 30]
h31 = df.loc[df['Hour'] == 31]
h32 = df.loc[df['Hour'] == 32]
h33 = df.loc[df['Hour'] == 33]
h34 = df.loc[df['Hour'] == 34]
h35 = df.loc[df['Hour'] == 35]
h36 = df.loc[df['Hour'] == 36]
h37 = df.loc[df['Hour'] == 37]
h38 = df.loc[df['Hour'] == 38]
h39 = df.loc[df['Hour'] == 39]
h40 = df.loc[df['Hour'] == 40]
h41 = df.loc[df['Hour'] == 41]
h42 = df.loc[df['Hour'] == 42]
h43 = df.loc[df['Hour'] == 43]
h44 = df.loc[df['Hour'] == 44]
h45 = df.loc[df['Hour'] == 45]
h46 = df.loc[df['Hour'] == 46]
h47 = df.loc[df['Hour'] == 47]
h48 = df.loc[df['Hour'] == 48]

listofhours = [h1,h2,h3,h4,h5,h6,h7,h8,h9,h10,h11,h12,h13,h14,h15,h16,
               h17,h18,h19,h20,h21,h22,h23,h24,h25,h26,h27,h28,h29,h30,
               h31,h32,h33,h34,h35,h36,h37,h38,h39,h40,h41,h42,h43,h44,
               h45,h46,h47,h48]


print(h1)



perc = 0.95
#listofhours
#hourList
for count in range(48):
    #for hour in listofhours:#for count in range(48):
        #print(count) 
        
    PaQ[count] = listofhours[count].Pa.quantile(perc)
    PbQ[count] = listofhours[count].Pb.quantile(perc)
    PcQ[count] = listofhours[count].Pc.quantile(perc)

    
print(PaQ,PbQ,PcQ)


data = np.vstack((PaQ,PbQ,PcQ)).T

columns = ['Pa','Pb','Pc']

data1 = pd.DataFrame(data,columns=columns)

print(data1)

colours = ['red','blue','yellow']
data1.plot(kind='line',color=colours)

plt.title('{a}th Power Profile of Transformer'.format(a=int(perc*100)))
plt.xticks(np.arange(0,48,2))
plt.yticks(np.arange(-5,65,5))
plt.grid()
plt.xlabel('Time, 1 point = 30 mins')
plt.ylabel('Apparent Power, kVA')
#data1.plot(kind='line',y='Pb',color='blue')
#data1.plot(kind='line',y='Pc',color='yellow')
plt.show()


nameDF = 'Percentile Profile for TX.csv'

data1.to_csv(nameDF,',')


