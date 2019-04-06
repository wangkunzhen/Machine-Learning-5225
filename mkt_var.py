# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 23:45:43 2019

@author: Remedios HUA
"""
import glob,os,csv
import numpy as np
import pandas as pd

order_files = []
mess_files = []
#define the path of files,put all orderbook and messagebook in list separately
os.chdir("D:/NUS-MFE/Machine learning/_data_dwn_30_198__ATVI_2018-11-01_2019-01-31_5")
for file in glob.glob("*orderbook_5.csv"):
    order_files.append(file)
for file in glob.glob("*message_5.csv"):
    mess_files.append(file)
    

frequency = int(2*60)       #predefine interval length    
startTrad   = int(9.5*60*60)
endTrad     = int(16*60*60)
level_interval = 30    #moving average input
T = 4
trend_thre = 0.03      #threshold for trend level
level = 6              #num of level for each market variable
bounds = range(startTrad,endTrad+frequency,frequency) #set interval bounds
b1 = len(bounds)


num_file = len(order_files)
for inum in range(46,num_file):
    
    boundIdx = np.zeros([b1,1])
    order_file = order_files[inum]
    mess_file = mess_files[inum]
    year = order_file[5:9]
    mon = order_file[10:12]
    day = order_file[13:15]
    demoDate = np.array([int(year),int(mon),int(day)])
    
    with open(mess_file,mode = 'r') as csv_file:
        mess = np.array(list(csv.reader(csv_file)))

    with open(order_file,mode = 'r') as csv_file:
        order = np.array(list(csv.reader(csv_file)))
    
    line = len(mess)   
    timeIdx = []
    for i in range(line):    
        timeIdx.append(float(mess[i,0]) >= startTrad and float(mess[i,0]) <= endTrad)
    
    #extract data from startTrad to endTrad
    mess = mess[timeIdx,0:6]
    mess = mess.astype(float)
    templist = list(mess[:,1])
    order = order[timeIdx]
    order = order.astype(float)
    
    order[:,0::2] = np.divide(order[:,0::2],10000)   #convert price 
    k1 = 0
    for k2 in range(line):
        if mess[k2,0] >= bounds[k1]:
            boundIdx[k1] = k2
            k1 += 1
    boundIdx[-1] = line   

#    for k2 in range(line):
#        if k1 == 1:
#            idx = (mess[:,1]>=bounds[k1] and mess[:,1]<=bounds[k1+1])
#            if not mess[idx,0]:
#                boundIdx[k1] = boundIdx[k1-1];
#                k1 = k1+1
#            else:
#                if mess[k2,0] >= bounds[k1]:
#                    boundIdx[k1] = k2
#                    k1 += 1
#        else:
#            idx = (mess[:,1]>=bounds[k1-1]) and (mess[:,1]<=bounds[k1])
#            if not mess[idx,0]:
#                boundIdx[k1] = 0;
#                k1 = k1+1
#            else:
#                if mess[k2,0] >= bounds[k1]:
#                    boundIdx[k1] = k2
#                    k1 += 1
            
#    boundIdx[-1] = line 
    
    #calculate market variable
    #part one: mid_spread 
    mid_spread = np.divide(order[:,0]+order[:,2],2)
    #part two: price spread
    spread = order[:,0]-order[:,2]
    #part three: volume mis-balance
    vol_mis = abs(order[:,1]-order[:,3])
    
    
    #part four: level
    moving_average = np.zeros([line,1])
    trend = np.zeros([line,1])
    for iline in range(line):
        if iline < level_interval:
            continue
        else:
            moving_average[iline] = np.mean(order[iline-30:iline-1,0])
            
    moving_average[0:level_interval-1] = moving_average[level_interval]
    
    #part five: price trend
    trend = order[:,0] - moving_average.T
    
    idx1 = np.where(abs(trend)<trend_thre) 
    idx2 = np.where(trend>trend_thre)
    idx3 = np.where(trend<-trend_thre)
    
    trend[idx1] = 1 #no obvious upward or downward trend
    trend[idx2] = 2 #obvious upward trend
    trend[idx3] = 0 #obvious downward trend
    
    #blurry variables
    for idx in range(b1-1):
      tmp_low = int(boundIdx[idx][0])
      tmp_up = int(boundIdx[idx+1][0])
      if tmp_up:
          num = tmp_up-tmp_low
          mlist = {}
          mlist["tmp_avg"] = moving_average[tmp_low:tmp_up]
          mlist["tmp_spread"] = spread[tmp_low:tmp_up] 
          mlist["tmp_mid"] = mid_spread[tmp_low:tmp_up]
          mlist["tmp_mis"] = vol_mis[tmp_low:tmp_up]
          divide_factor = [0,int(num/6)-1,int(num/6*2)-1,int(num/6*3)-1,int(num/6*4)-1,int(num/6*5)-1,num-1]
      
      for ilist in mlist:
          blurry = np.zeros([num,1])
          tmp = np.sort(mlist[ilist])
          factor = tmp[divide_factor]
          blurry[np.where(mlist[ilist]<=factor[6])] = 6
          blurry[np.where(mlist[ilist]<=factor[5])] = 5
          blurry[np.where(mlist[ilist]<=factor[4])] = 4 
          blurry[np.where(mlist[ilist]<=factor[3])] = 3
          blurry[np.where(mlist[ilist]<=factor[2])] = 2
          blurry[np.where(mlist[ilist]<=factor[1])] = 1
          mlist[ilist] = blurry
      
      moving_average[tmp_low:tmp_up] = mlist["tmp_avg"]
      spread[tmp_low:tmp_up] = mlist["tmp_spread"].T
      mid_spread[tmp_low:tmp_up] = mlist["tmp_mid"].T
      vol_mis[tmp_low:tmp_up] = mlist["tmp_mis"].T
      
    
    mess = np.column_stack((mess,spread))
    mess = np.column_stack((mess,vol_mis))
    mess = np.column_stack((mess,moving_average))
    mess = np.column_stack((mess,trend.T))
    
    market_variable = np.zeros([b1-1,4*T])
    for d in range(0,b1-1):
        interval_up = (d+1)*frequency + startTrad
        
        for t in range(T):
            lower = interval_up-(T-t)*level_interval
            idx = np.argwhere(mess[:,0]>=lower)[0]
            market_variable[d,4*t:4*(t+1)] = mess[idx,-5:-1]
            
            
#    market = {}
#    market["mid_spread"] = mid_spread
#    market["spread"] = spread
#    market["moving_average"] = moving_average
#    market["vol_mis"] = vol_mis
#    market["trend"] = trend
    
    #seperate csv file 

    file_name = "market_"+str(year)+str(mon)+str(day)+".csv"

    np.savetxt(file_name, market_variable, delimiter=",")
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        