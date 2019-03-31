# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 21:54:50 2019

@author: Remedios HUA
"""

def cost_T(period_o,i,mid_spread,remain):
    
    cost = 0
    m_remain = remain
    for m in range(0, 4):
        if m_remain > period_o[0,m*4+3]:
            cost = cost + period_o[0,m*4+2]*period_o[0,m*4+3]
            m_remain=m_remain-period_o[0,m*4+3]
        else:
            cost = cost + period_o[0,m*4+2]*m_remain
            m_remain=max(0,m_remain-period_o[0,m*4+3])
            
    if m_remain>0:
        print("remain order cannot be fully executed")
        cost = cost + m_remain*period_o[-1,-1] 
        m_remain = 0
        #use lowest bid price to execute all shares
         
    cost = mid_spread*remain-cost
        
    return cost,m_remain

def cost_other(period_m,period_o,remain,mid_spread,j,action,V,I,msize):
    
    lps = 0
    non_exe = 0
    cost = np.zeros(np.array(action).size)
    remain = np.repeat(remain,np.array(action).size)
    for a in range(np.array(action).size):
        
        iaction = np.array(action)[a]
        #remain = i * V / I                 #initial order volume at this episode
        order_price = period_o[0,0] - iaction   
        #our sell price should be around ask price
        
        if order_price <= period_o[0,2]:
            #our price less than the bid price
            counter = 0
            temp=0
            print("order executed immediately")
            
            while order_price <= period_o[0,2+counter*4] and counter<5 and remain[a]>0: 
                #search through the buy book, for price with which our order can be executed
                if remain[a] > period_o[0,3+counter*4]:
                    remain[a] = remain[a]-period_o[0,3+counter*4]
                    temp += period_o[0,3+counter*4]
                    cost[a] = cost[a]+order_price*period_o[0,3+counter*4]
                    counter += 1
                else:
                    cost[a] = cost[a] + order_price*remain[a]
                    remain[a] = 0
#                      
            remain[a] = max(remain[a],0)
            if remain[a] == 0:
                print("order fully executed")
                cost[a] = mid_spread*(j*V/I)-cost[a]
                continue
            
            else:
                count3 = 0
                lps = temp 
                lowest_price = order_price
                # lps means the size of order has not been executed, 
                # look through whole sell book,to search

                for idx,val in enumerate(range(0, msize[0] - 1)):
                            
                    if remain[a] == 0:
                        cost[a] = mid_spread*((j*V/I)-remain[a])-cost[a]
                    
                    if period_m[idx,1]==1 and period_m[idx,5]==-1:
                        #submit a sell limit order, which price lower than us
                        if period_m[idx,4] < order_price:
                            lps += period_m[idx,3]
                            non_exe += period_m[idx,3]
                            
                    elif period_m[idx,1] == 4 and period_m[idx,5]==1 and period_m[idx,4] > order_price:
                        #execution of a buy order with higher price
                        if  period_m[idx,3] > non_exe:
                            tmp = period_m[idx,3]-non_exe            #size of our order being executed
                            cost[a] += tmp*period_m[idx,4]
                            remain[a] = max(remain[a]-tmp,0)
                            lps = max(0,lps-period_m[idx,3])
                            non_exe = 0
                        else:
                            lps = lps-period_m[idx,3]
                            non_exe = non_exe-period_m[idx,3]
                            
                    elif (period_m[idx,1] == 2 or period_m[idx,1] == 3) and period_m[idx,5] == -1:
                    #cancel or delete an order
                        if period_m[idx,4] < order_price:
                            lps = lps-period_m[idx,3]
                            non_exe = non_exe-period_m[idx,3]
                            
                        elif period_m[idx,4] == order_price:
   # update lps if limit sell order with price lower than our sell price is cancelled
   # note in the case cancellation of new sell order is exactly the same price as our order,
   # we need to check if the initial sell order is placed after start of time,
   # by checking if the ID appear once or twice.
                
                            label = period_m[idx,2]
                            if len(np.where(period_m[:,2]==label)) == 1: #the order is submitted before us
                                lps = lps-period_m[idx,3]
                                non_exe = non_exe-period_m[idx,3]
                                
                    elif period_m[idx,1] == 4 and period_m[idx,5] == -1 and period_m[idx,4] < order_price:
                        #a sell order with lower price is executed
                        
                        lps = max(lps - period_m[idx,3],0) 
                        non_exe = max(non_exe-period_m[idx,3],0)
                            
                    elif period_m[idx,1] == 4 and period_m[idx,5] == -1 and period_m[idx,4] >= order_price:
                        if period_m[idx,3] > lps:
                            tmp = min(period_m[idx,3]-lps,remain[a])
                            cost[a] += period_m[idx,4]*tmp
                            lps = 0
                            remain[a] = max(0,remain[a]-tmp)
                            
                        else:
                            lps = lps - period_m[idx,3]
        
        else:   
            count3 = 0
            lps = 0  # lps means the size of order has not been executed
            lowest_price = order_price
            while count3 < 5 and period_o[0, count3 * 4] <= order_price:
                lps = lps + period_o[0, count3 * 4 + 1]
                lowest_price = min(lowest_price,period_o[0,count3*4])
                count3 = count3 + 1
                
            lps=lps
            
            for idx,val in enumerate(range(0, msize[0] - 1)):
                
                if remain[a] == 0:
                    cost[a] = mid_spread[0]*((j*V/I)-remain[a])-cost[a]
                    return cost,remain[a]
                
                if period_m[idx,1]==1 and period_m[idx,5]==-1:
                    #submit a sell limit order, which price lower than us
                    if period_m[idx,4] < order_price:
                        lps += period_m[idx,3]
                        lowest_price = min(lowest_price,period_m[idx,4])        
                        
                elif period_m[idx,1] == 4 and period_m[idx,5]==1 and period_m[idx,4] > order_price:
                    #execution of a buy order with higher price
                    if  period_m[idx,3] > lps:
                        tmp = period_m[idx,3]-lps                #size of our order being executed
                        cost[a] += tmp*period_m[idx,4]
                        remain[a] = max(remain[a]-tmp,0)
                        lps = 0
                        
                    else:
                        lps = lps-period_m[idx,3]
                        
                        
                elif (period_m[idx,1] == 2 or period_m[idx,1] == 3) and period_m[idx,5] == -1:
                #cancel or delete a sell order
                    if period_m[idx,4] < order_price:
                        lps = lps-period_m[idx,3]

                    elif period_m[idx,4] == order_price:
# update lps if limit sell order with price lower than our sell price is cancelled
# note in the case cancellation of new sell order is exactly the same price as our order,
# we need to check if the initial sell order is placed after start of time,
# by checking if the ID appear once or twice.
        #search = period_m[period_m[:,0]<=time & period_m[:,0]>= lower]
                        label = period_m[idx,2]
                        if len(np.where(period_m[:,2]==label)) == 1: 
                            #the order is submitted before us
                            lps = lps-period_m[idx,3]
                            
                elif period_m[idx,1] == 4 and period_m[idx,5] == -1 and period_m[idx,4] < order_price:
                    #a sell order with lower price is executed
                    
                    lps = max(lps - period_m[idx,3],0) 
                   
                elif period_m[idx,1] == 4 and period_m[idx,5] == -1 and period_m[idx,4] >= order_price:
                    if period_m[idx,3] > lps:
                        tmp = min(period_m[idx,3]-lps,remain[a])
                        cost[a] += period_m[idx,4]*tmp
                        lps = 0
                        remain[a] = max(0,remain[a]-tmp)
                        
                    else:
                        lps = lps - period_m[idx,3]
    return cost, remain