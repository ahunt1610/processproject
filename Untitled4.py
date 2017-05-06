
# coding: utf-8

# In[11]:

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'matplotlib notebook')

from pylab import *
import matlab
import math as math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import control.matlab as control



# In[8]:

#parameters
Kc = .52    #controller gain
td = 0      #controller time constant, realized this was not needed for this model
Tf = 1      #throughput time delay, from when ordered to when arrived
d = 5770    #demand
r = 11270    #production schedule to leave 5500 in inventory at all time

taus = .000000000001           #used for 'f' to set s very small so it doesnt matter
f= control.tf([1],[taus,1])    #transfer function to get the units to balance, basically equals 1 yest gets the 's' values to be equal on num and denom

# transfer functions
K = Kc*control.tf([td,1],[1])         #PD controller transfer function

num,den= control.pade(Tf,3)
P = control.tf(num,den)              #time delay transfer function

Kv = control.tf([1],[1,0])             #step response transfer function

Hud = ((K*Kv)/(1+K*P*Kv))             #Transfer function from the production to the demand (disturbance) 
Hur = (K*f/(1+K*P*Kv))                #transfer function from the production to set point

Ys = ((P*K*Kv)/(1+P*K*Kv))*r- (1/(1+P*K*Kv))*d         #Change in production
Us = ((K*Kv)/(1+K*P*Kv))*r - (K*Kv/(1+K*P*Kv))*d       #change in inventory

Hud2 = (Hud)*(Kv/(1+P*K*Kv))                         #production to demand of second node

t= np.linspace(0,50,1000)
yd,t = control.step(Hud2,t)
plt.plot(t,yd)
plt.xlabel('Time (day)')
plt.ylabel('Inventory')
plt.title('Change in inventory Vs. Time at Time delay 1 and gain .52')

print(K)
print(P)
print(Kv)
print(Us)
print(Ys)


# In[9]:

t= np.linspace(0,50,1000)
plt.figure(figsize=(11,10))
plt.subplot(2,2,1)
yu,t = control.step(Hud,t)
plt.plot(t,yu)
plt.xlabel('Time (day)')
plt.ylabel('Inventory')
plt.title('Change in inventory Vs. Time at Time delay 1 and gain .52')

plt.subplot(2,2,2)
y,t = control.step(Hur,t)
plt.plot(t,y)
plt.xlabel('Time (day)')
plt.ylabel('Production change')
plt.title('Change in production Vs. Time at Time delay 1 and gain .52')

plt.subplot(2,2,3)
y,t = control.step(Us,t)
plt.plot(t,y)
plt.xlabel('Time (day)')
plt.ylabel('Total inventory')
plt.title('Inventory Vs. Time at Time delay 1 and gain .52')

plt.subplot(2,2,4)
y,t = control.step(Ys,t)
plt.plot(t,y)
plt.xlabel('Time (day)')
plt.ylabel('Produced')
plt.title('Produced Vs. Time at Time delay 1 and gain .52')


# In[10]:

omegamax=0
plt.figure(figsize=(6,6))
plt.subplot(2,2,1)
w= np.logspace(-2,3)
mag,phase,omega = control.bode(Hud2,w)
plt.tight_layout()

# find the cross-over frequency and gain at cross-over
wc = np.interp(-180.0,np.flipud(phase),np.flipud(omega))
gc = np.interp(wc,omega,mag)

# get the subplots axes
ax1,ax2 = plt.gcf().axes

# add features to the magnitude plot
plt.sca(ax1)
plt.plot([omega[0],omega[-1]],[gc,gc],'r--')
[gmin,gmax] = plt.ylim()
plt.plot([wc,wc],[gmin,gmax],'r--')
plt.title("Time constant 3 Gain of .52")

# add features to the phase plot
plt.sca(ax2)
plt.plot([omega[0],omega[-1]],[-180,-180],'r--')
[pmin,pmax] = plt.ylim()
plt.plot([wc,wc],[pmin,pmax],'r--')
plt.title("Frequency ")

j=np.argmax(mag)
omegamax= omega[j] 
magmax = mag[j]
print "Magnitude is ",magmax," and frequency is ",omegamax


# In[ ]:



