# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:36:38 2019

@author: Remedios HUA
"""
#sell order mechanism(higher price)

#import csv
import sys
sys.path.append("D:/python/python_project/Neural Network")
import pandas as pd
import os,glob
import numpy as np
from optimal_function import *
from all_cost import *

order_files = []
mess_files = []
os.chdir("D:/NUS-MFE/Machine learning/_data_dwn_30_198__ATVI_2018-11-01_2019-01-31_5")
for file in glob.glob("*orderbook_5.csv"):
    order_files.append(file)
for file in glob.glob("*message_5.csv"):
    mess_files.append(file)
    
V = 1000
H = 60*2
T = 4       #H/T is the length of episode
I = 100     #unit of inventory = 10
unit = V/I
startTrad   = int(9.5*60*60)
endTrad     = int(16*60*60)
bounds = range(startTrad,endTrad+H,H) #set interval bounds
b1 = len(bounds)
action = range(200,-300,-50)
inventory = range(I,-1,-1)



boundIdx = np.zeros([b1,1])
num_file = len(order_files)

for inum in range(num_file):
    spread = []
    vol_dif = []
    year = mess_files[inum][5:9]
    mon = mess_files[inum][10:12]
    day = mess_files[inum][13:15]
    
    demoDate = np.array([int(year),int(mon),int(day)])
    with open(mess_files[inum],mode = 'r') as csv_file:
        MB = pd.read_csv(csv_file, header=None)
    with open(order_files[inum],mode = 'r') as csv_file:
        OB = pd.read_csv(csv_file, header=None) 
    
    assert len(MB)==len(OB)    #if the size not match
    shape = MB.shape
    
    for d in range(0,int((MB.iat[shape[0]-1,0]-MB.iat[0,0])/H)): #d is the number of interval
        
        cost = np.zeros([len(inventory),len(action)])      #define the cost matrix,action*inventory
        private_variable = dict()
        
        for t in range(T,0, -1):#t is the idx of episode

#            sum_cost = np.zeros([len(inventory),len(action)])                           
            interval_up = (d+1)*H + startTrad
            interval_low = d*H + startTrad
            upper = interval_up-(T-t)*(H/T)
            lower = interval_up-(T-t+1)*(H/T)
            max_remain = np.zeros([T+1,I])
            
            period_m = []
            period_o = []
            period_m = MB[(MB[0] >= lower) & (MB[0] <=upper)]
            period_o = OB[(MB[0] >= lower) & (MB[0] <=upper)]
            
            period_m = np.array(period_m)
            period_o = np.array(period_o)
            msize = period_o.shape
        
            mid_spread = ((period_o[:,0] + period_o[:,2])/2)[0]
#            if t==T :
#                for i in range(0,I-1)
            if t == T:
                
                for i in range(0, I+1):
                    #market order->bid price1, bid price2, ...bid price5
                    remain = i * V / I
                    cost[i,:],remain = cost_T(period_o,i,mid_spread,remain)
                    
                    if remain%unit < unit/2:
                        max_remain[t,i] = I-i
                    else:
                        max_remain[t,i] = T-i+1
                        
                
            else:
#                cost = dict()
                for i in range(0, I+1):   
                    
                    remain = i*V/I 
                    #the maximum shares can be executed in this stage, based on previous stage
                    if i!=0:
                        cost[i,:],remain = cost_other(period_m,period_o,remain,mid_spread,i,action,V,I,msize)
                        temp = divmod(remain,unit)
#                        executed = i-temp[0] + (temp[1]>unit/2) #real i being executed in this episode
#                        
#                        remain = I-executed  #have to be executed in next episode
#                        cost[str(j)] = cost[str(j)] + private_variable[str(t+1)][remain]
                        
                            
            private_variable[str(t)] = cost
        optimal_out = optimal(cost)
        
        file_name = "optimal_"+str(year)+str(mon)+str(day)+"_"+str(t+d*T)+".csv"
            
        order_csv = optimal_out
        np.savetxt(file_name, order_csv, delimiter=",")
    
    
    
    
    
    
        