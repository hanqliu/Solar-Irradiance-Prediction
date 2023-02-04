import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from tqdm.notebook import tqdm
from time import sleep
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import heapq

da = pd.read_csv(r'D:\e\南开光伏课题数据集及说明\NK2_GF\评测数据\气象数据\Station_1.csv')
da = np.array(da)
da
c1 = da[0:10000, 2]
c2=da[0:10000,3]
c3=da[0:10000,4]
c4=da[0:10000,5]
c5=da[0:10000,6]
#ax,ay=get_AFAGPS(c1,h=0.2)
#max_ay = max(ay)
#ay = [y / max_ay for y in ay]
#plt.figure(figsize=(40,10))
#plt.plot(ax,ay)</div><i class="fa fa-lightbulb-o "></i>

def get_AFAGPS(X, tag='D', h=0.2):
    '''
    Parameters
    ----------
    X: list
        需要计算功率谱的数据
    tag: str
    default: D
        C: continuous Fourier Transform
        D: Discrete Fourier Transfrom
    h: float
        指定AFAGPS算法中过滤系数

    Examples
    --------
    ref to test.py
    '''
    X, N = [float(i) for i in X], len(X)
    # normalized
    mean, std = np.mean(X), np.std(X)
    X = [(i - mean) / std for i in X]
    # reverse X
    X_reverse = X[::-1]
    # add zeros
    X_reverse.extend([0] * N)
    X=X_reverse[::-1]
	
    # circular convolution, the same as matlab function "cconv"
    Rn = listConvolve(X, X_reverse)
    Rn = [1.0 / (N-i) * Rn[i] for i in range(1000)]   ######change1
    # filt
    for idx, i in enumerate(Rn):
        if abs(i / Rn[0]) <= h:
            Rn[idx] = 0
    # add zeros
    Rn.extend(len(Rn) * [0])    ########change2

    plist = [2 * value.real - Rn[0] for value in np.fft.fft(Rn)]
    m = max([abs(i) for i in plist])
    plist = [abs(i) / m for i in plist]

    if tag == 'C':
        """CFT"""
        yy = [i + j for i,
              j in zip(plist[len(plist) // 2 + 1:], plist[:len(plist) // 2 + 1])]
    elif tag == 'D':
        """DFT"""
        yy = plist
    '''
    else:
        import sys
        print("Error!")
        sys.exit(1)
        '''
    xx = []
    for i in range(len(yy)):
        xx.append(i * np.pi / len(yy))
    return xx, yy

def sfgps(X):
    X, N = [float(i) for i in X], len(X)
    # normalized
    mean, std = np.mean(X), np.std(X)
    X = [(i - mean) / std for i in X]
    yy=(abs(np.fft.fft(X)))**2
    ave=1/(N+1)*sum(X)
    y=list()
    for m in yy:
        if m>ave+std:
            y.append(m)
        else:
            y.append(std**2)
    return y


def recover(c):
    da=getsfgps(c)
    #da=getafa(c)
    ind=[da.index(a) for a in da if a >0.2]
    A=[]
    for i in ind:
        A.append(da[i])
    index=ind[0:math.ceil(len(ind)/2)]
    x=np.linspace(0,208*np.pi,9984)
    y=[]
    yy=[0 for i in range(9984)]
    for i in range(0,math.ceil(len(ind)/2)):
        y.append(np.sqrt(A[i])*np.cos(index[i]/10000*x))
    #for m in y:
    #   yy=yy+m
    #return yy
    return y

#show figure
def seegap(c):
    m=np.linspace(0,6.28,99)
    gaap=[]
    for i in m:
        print(i)
        A,index=getda(c)
        yy=getre(A,index,i)
        me=np.mean(yy)
        st=np.std(yy)
        yy=(yy-me)/st
        cme=np.mean(c)
        cst=np.std(c)
        c=(c-cme)/cst
        gap=sum([abs(yy[n]-c[n]) for n in range(len(yy)-1)])
        gaap.append(gap)
    mi=min(gaap)
    print(np.where(gaap==mi))
    y=getre(A,index,m[np.where(gaap==mi)][0])
    #plt.plot(c)
    #plt.plot(y)
    #plt.show()
    return y

#return certain gap
def findgap(c,i):
    A,index=getda(c)
    yy=getre(A,index,i)
    me=np.mean(yy)
    st=np.std(yy)
    yy=(yy-me)/st
    cme=np.mean(c)
    cst=np.std(c)
    c=(c-cme)/cst
    gap=sum([abs(yy[n]-c[n]) for n in range(len(yy)-1)])
    return gap


def stdlise(x):
    me,std=np.mean(x),np.std(x)
    x=(x-me)/std
    return x



def minimize(c):
    A,index=getda(c)
    m=np.linspace(0,6.28,99)
    x=np.linspace(0,208*np.pi,9984)
    Y=[0 for i in range(len(x))]
    for i in range(0,len(index)-1):
        gaap=[]
        for theta in m:
            the=np.array([theta for i in range(len(x))])
            y=(np.sqrt(A[i])*np.cos((index[i]+1)/96*x+the)).astype(list)
            test=[m+n for m,n in zip(Y,y)]
            gap=sum([abs(test[n]-c1[n]) for n in range(len(Y)-1)])
            gaap.append(gap)
        mk=np.array([m[gaap.index(min(gaap))] for i in range(len(x))])
        yfori=(np.sqrt(A[i])*np.cos((index[i]+1)/96*x+mk)).astype(list)
        Y=[m+n for m,n in zip(Y,yfori)]
    return Y
'''       
for m in y:
    yy=yy+m
print('getre done')

gaap=[]
    y=[]
    yy=[0 for i in range(9984)]

    
    for i in m:
        print(i)
        A,index=getda(c)
        yy=getre(A,index,i)
        me=np.mean(yy)
        st=np.std(yy)
        yy=(yy-me)/st
        cme=np.mean(c)
        cst=np.std(c)
        c=(c-cme)/cst
        gap=sum([abs(yy[n]-c[n]) for n in range(len(yy)-1)])
        gaap.append(gap)
    mi=min(gaap)
    print(np.where(gaap==mi))
    y=getre(A,index,m[np.where(gaap==mi)][0])
'''


#return A,index
def getda(c):
    da=getsfgps(c)
    #da=getafa(c,0.2)
    ind = map(da.index, heapq.nlargest(12, da)) 
    ind=list(ind)
    ind=list(set(filter(lambda x: x<5000, ind)))
    A=[]
    for i in ind:
        A.append(da[i])
    #index=ind[0:math.ceil(len(ind)/2)]
    print('getda done')
    return A,ind

def mininew(c):
    A,index=getda(c)
    m=np.linspace(0,6.28,99)
    x=np.linspace(0,208*np.pi,99840)
    Y=[0 for i in range(len(x))]
    for i in range(0,len(index)-1):
        gaap=[]
        for theta in m:
            the=np.array([theta for i in range(len(x))])
            y=(np.sqrt(A[i])*np.cos((index[i]+1)/96*x+the)).astype(list)
            test=[m+n for m,n in zip(Y,y)]
            gap=sum([abs(test[n*10]-c1[n]) for n in range(9984)])
            gaap.append(gap)
        print(m[gaap.index(min(gaap))])
        mk=np.array([m[gaap.index(min(gaap))] for i in range(len(x))])
        yfori=(np.sqrt(A[i])*np.cos((index[i]+1)/96*x+mk)).astype(list)
        Y=[m+n for m,n in zip(Y,yfori)]
    return Y