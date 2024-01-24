import numpy as np
import seaborn as sns
import math
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter


importData = np.genfromtxt('PythonDist.csv',dtype=str,delimiter=',',usecols=(0,1),skip_header=1)

print(importData)

randomData = importData.T[0]

randomData1 = np.array([i for i in randomData if i])



orderData = importData.T[1]


randomData = randomData1.astype(float)
orderData = orderData.astype(float)

randomDataNeg = randomData[randomData<0]
randomDataPos = randomData[randomData>0]

orderDataNeg = orderData[orderData<0]
orderDataPos = orderData[orderData>0]


fig, ax = plt.subplots(figsize=(14,8))

bw = 0.65
leg = ['Ran: Unsuccessful POW','Ran: Successful POW','Ord: Unsuccessful POW','Ord: Successful POW']
#Increase ΔIn

sns.kdeplot([randomDataNeg,randomDataPos],legend=False,bw_adjust=bw,cut=0,palette=['red','orange'])

sns.kdeplot([orderDataNeg,orderDataPos],legend=False,bw_adjust=bw,cut=0,palette=['purple','blue'])

xV = [(np.median(randomDataNeg),'red','r-Neg Median:  {a}%'.format(a=round(np.median(randomDataNeg)))),
       (np.median(randomDataPos),'orange','r-Pos Median:  {a}%'.format(a=round(np.median(randomDataPos)))),
      (np.median(orderDataNeg),'purple','o-Neg Median:  {a}%'.format(a=round(np.median(orderDataNeg)))),
       (np.median(orderDataPos),'blue','o-Pos Median:  {a}%'.format(a=round(np.median(orderDataPos))))]

plt.axvline(x=0,color='black',linestyle='dotted')

for a,b,c in xV:
    plt.axvline(x=a,color=b,linestyle='--')
    ax.text(a-1,0.99,c,ha='right',va='top',rotation=90,size=13,transform=ax.get_xaxis_transform())


##ax2 = ax1.twinx()
##sns.kdeplot(data=penguins, x="flipper_length_mm", color="k", label="kde density", ls=':', lw=5, ax=ax2)
ax.set_ylim(0, ax.get_ylim()[1] / 0.75)  # similir limits on the y-axis to align the plots
#ax.set_yticklabels(np.arange(0,550,50))
#ax.yaxis.set_major_formatter(ScalarFormatter(2))#-1 * 550))  # show axis such that 1/binwidth corresponds to 100%
ax.set_ylabel(f'Density Estimation with binwidth {bw}')

ax.set_xlim([-100,100])
plt.figtext( .15,.8,"o-Neg: 264 Sites")
plt.figtext( .15,.775, "o-Pos: 44 Sites")
plt.figtext( .15,.75, "Success rate = 86%")
plt.figtext( .15,.70, "r-Neg: 28 Sites")
plt.figtext( .15,.675, "r-Pos: 27 Sites")
plt.figtext( .15,.65, "Success rate = 52%")

#ax.set_ylim([0,])

plt.title('Distribution of Decrease and Increase in Neutral Current (ΔIn as %)')
plt.legend(labels=leg)
#sns.displot([randomData,randomDataNeg,randomDataPos],bins=np.arange(-100,100,2.5),legend=True,kde=True)#,bw_adjust=bw)##,bins=15,kde=True)
ax.set_xlabel('Neutral Current Change (%) - ΔIn / In')
plt.savefig('Change in Neutral Current.png',dpi=1200)
plt.show()



fig, ax = plt.subplots(figsize=(14,8))

bw = 0.65
leg = ['Ord: Unsuccessful POW','Ord: Successful POW']
#Increase ΔIn

#sns.kdeplot([randomDataNeg,randomDataPos],legend=False,bw_adjust=bw,cut=0,palette=['red','orange'])

sns.kdeplot([orderDataNeg,orderDataPos],legend=False,bw_adjust=bw,cut=0,palette=['purple','blue'])

xV = [(np.nanpercentile(orderDataNeg,25),'purple','o-Neg 75th:  {a}%'.format(a=round(np.nanpercentile(orderDataNeg,25)))),
      (np.nanmedian(orderDataNeg),'purple','o-Neg Median:  {a}%'.format(a=round(np.nanmedian(orderDataNeg)))),
      (np.nanpercentile(orderDataNeg,75),'purple','o-Neg 25th:  {a}%'.format(a=round(np.nanpercentile(orderDataNeg,75)))),
      (np.nanpercentile(orderDataPos,25),'blue','o-Pos 25th:  {a}%'.format(a=round(np.nanpercentile(orderDataPos,25)))),
      (np.nanmedian(orderDataPos),'blue','o-Pos Median:  {a}%'.format(a=round(np.nanmedian(orderDataPos)))),
      (np.nanpercentile(orderDataPos,75),'blue','o-Pos 75th:  {a}%'.format(a=round(np.nanpercentile(orderDataPos,75))))]

plt.axvline(x=0,color='black',linestyle='dotted')

for a,b,c in xV:
    plt.axvline(x=a,color=b,linestyle='--')
    ax.text(a-1,0.99,c,ha='right',va='top',rotation=90,size=13,transform=ax.get_xaxis_transform())


x = ax.lines[0].get_xdata()
y = ax.lines[0].get_ydata()
#round(np.median(orderDataPos))

x1 = ax.lines[1].get_xdata()
y1 = ax.lines[1].get_ydata()


ax.fill_between(x, y, where=x < np.nanpercentile(orderDataPos,75), color='gold', alpha=0.3, zorder=10)
ax.fill_between(x, y, where=x < np.nanmedian(orderDataPos), color='yellow', alpha=0.5,zorder=1)
ax.fill_between(x, y,  color='grey', alpha=0.3,zorder=1)

ax.fill_between(x1, y1, where=x1 > np.nanpercentile(orderDataNeg,25), color='red', alpha=0.3, zorder=10)
ax.fill_between(x1, y1, where=x1 > np.nanmedian(orderDataNeg), color='blue', alpha=0.3,zorder=1)
ax.fill_between(x1, y1,  color='grey', alpha=0.3,zorder=1)#where=x1 > np.nanmedian(orderDataNeg),

#ax.set_ylim(0,0.05)
ax.set_ylim(0, ax.get_ylim()[1] / 0.75)  # similir limits on the y-axis to align the plots

ax.set_ylabel(f'Density Estimation with binwidth {bw}')
ax.set_xlabel('Neutral Current Change (%) - ΔIn / In')

ax.set_xlim([-100,100])
plt.figtext( .15,.8,"o-Neg: 264 Sites")
plt.figtext( .15,.775, "o-Pos: 44 Sites")
plt.figtext( .15,.75, "Success rate = 86%")

plt.figtext( .4,.375, "Purple:")
plt.figtext( .4,.35, "132")
plt.figtext( .4,.325, "Sites")
plt.figtext( .35,.375, "Red:")
plt.figtext( .35,.35, "+66")
plt.figtext( .35,.325, "Sites")
plt.figtext( .3,.375, "Grey:")
plt.figtext( .3,.35, "+66")
plt.figtext( .3,.325, "Sites")

plt.figtext( .53,.275, "Yellow:")
plt.figtext( .53,.25, "22")
plt.figtext( .53,.225, "Sites")
plt.figtext( .58,.275, "Gold:")
plt.figtext( .58,.25, "+11")
plt.figtext( .58,.225, "Sites")
plt.figtext( .65,.225, "Grey:")
plt.figtext( .65,.2, "+11")
plt.figtext( .65,.175, "Sites")




#ax.set_ylim([0,])

plt.title('Distribution of Decrease and Increase in Neutral Current for Ordered Characteristic (ΔIn as %)')
plt.legend(labels=leg)


plt.savefig('Change in Neutral Current (Ordered).png',dpi=1200)
plt.show()

fig, ax = plt.subplots(figsize=(14,8))

bw = 0.65
leg = ['Ran: Unsuccessful POW','Ran: Successful POW']
#Increase ΔIn

sns.kdeplot([randomDataNeg,randomDataPos],legend=False,bw_adjust=bw,cut=0,palette=['red','orange'])

#sns.kdeplot([orderDataNeg,orderDataPos],legend=False,bw_adjust=bw,cut=0,palette=['purple','blue'])

xV = [(np.nanpercentile(randomDataNeg,25),'purple','r-Neg 75th:  {a}%'.format(a=round(np.nanpercentile(randomDataNeg,25)))),
      (np.nanmedian(randomDataNeg),'purple','r-Neg Median:  {a}%'.format(a=round(np.nanmedian(randomDataNeg)))),
      (np.nanpercentile(randomDataNeg,75),'purple','r-Neg 25th:  {a}%'.format(a=round(np.nanpercentile(randomDataNeg,75)))),
      (np.nanpercentile(randomDataPos,25),'blue','r-Pos 25th:  {a}%'.format(a=round(np.nanpercentile(randomDataPos,25)))),
      (np.nanmedian(randomDataPos),'blue','r-Pos Median:  {a}%'.format(a=round(np.nanmedian(randomDataPos)))),
      (np.nanpercentile(randomDataPos,75),'blue','r-Pos 75th:  {a}%'.format(a=round(np.nanpercentile(randomDataPos,75))))]

plt.axvline(x=0,color='black',linestyle='dotted')

for a,b,c in xV:
    plt.axvline(x=a,color=b,linestyle='--')
    ax.text(a-1,0.99,c,ha='right',va='top',rotation=90,size=13,transform=ax.get_xaxis_transform())


x = ax.lines[0].get_xdata()
y = ax.lines[0].get_ydata()
#round(np.median(orderDataPos))

x1 = ax.lines[1].get_xdata()
y1 = ax.lines[1].get_ydata()


ax.fill_between(x, y, where=x < np.nanpercentile(randomDataPos,75), color='gold', alpha=0.3, zorder=10)
ax.fill_between(x, y, where=x < np.nanmedian(randomDataPos), color='yellow', alpha=0.5,zorder=1)
ax.fill_between(x, y,  color='grey', alpha=0.3,zorder=1)

ax.fill_between(x1, y1, where=x1 > np.nanpercentile(randomDataNeg,25), color='red', alpha=0.3, zorder=10)
ax.fill_between(x1, y1, where=x1 > np.nanmedian(randomDataNeg), color='blue', alpha=0.3,zorder=1)
ax.fill_between(x1, y1,  color='grey', alpha=0.3,zorder=1)#where=x1 > np.nanmedian(orderDataNeg),

#ax.set_ylim(0,0.05)
ax.set_ylim(0, ax.get_ylim()[1] / 0.75)  # similir limits on the y-axis to align the plots

ax.set_ylabel(f'Density Estimation with binwidth {bw}')
ax.set_xlabel('Neutral Current Change (%) - ΔIn / In')

ax.set_xlim([-75,75])
plt.figtext( .15,.8,"r-Neg: 28 Sites")
plt.figtext( .15,.775, "r-Pos: 27 Sites")
plt.figtext( .15,.75, "Success rate = 52%")

plt.figtext( .475,.375, "Purple:")
plt.figtext( .475,.35, "14")
plt.figtext( .475,.325, "Sites")
plt.figtext( .425,.375, "Red:")
plt.figtext( .425,.35, "+7")
plt.figtext( .425,.325, "Sites")
plt.figtext( .3,.275, "Grey:")
plt.figtext( .3,.25, "+7")
plt.figtext( .3,.225, "Sites")

plt.figtext( .54,.275, "Yellow:")
plt.figtext( .54,.25, "14")
plt.figtext( .54,.225, "Sites")
plt.figtext( .63,.275, "Gold:")
plt.figtext( .63,.25, "+7")
plt.figtext( .63,.225, "Sites")
plt.figtext( .76,.275, "Grey:")
plt.figtext( .76,.25, "+~6")
plt.figtext( .76,.225, "Sites")



#ax.set_ylim([0,])

plt.title('Distribution of Decrease and Increase in Neutral Current for Random Characteristic (ΔIn as %)')
plt.legend(labels=leg)


plt.savefig('Change in Neutral Current (Random).png',dpi=1200)
plt.show()
