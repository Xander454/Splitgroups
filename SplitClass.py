#!/usr/bin/python
import sys
import numpy as np
from sklearn import svm
import math
import random
import Neuralnetprob
import utils
import Neuralnet
import collections
import itertools
import os
import pickle

def get_iterable(x):
    if isinstance(x, collections.Iterable):
        return x
    else:
        return (x,)



global Data
global Y
Data,Y=utils.LoadData()

TrainKeys,TestKeys=utils.GetRandomSetKeys(np.arange(len(Y)),DataFrac=0.5,TrainFrac=0.5)


def TrainSplit(TrainKeys,TestKeys,Grouping):

	Xtrain=Data[TrainKeys]
	Xtest=Data[TestKeys]
	ytrain=Y[TrainKeys]
	ytest=Y[TestKeys]

	ytrain=ChangeGroups(ytrain,Grouping)
	ytest=ChangeGroups(ytest,Grouping)


	print 'Training with '+str(np.bincount(ytrain)[1:])
	
        clf=utils.ChooseSVM(Xtrain,ytrain,Xtest,ytest)
	return clf


def ChangeGroups(y,Grouping):
	for g in Grouping:
		        y[np.where(y==g)]=0
        y[np.where(y!=0)]=2
        y[np.where(y==0)]=1
	return y


def MakeGroups(y,Groups,Grouping):
	yout=np.copy(y)
	for G,Gi in zip(Groups,Grouping):
		yout[np.where(y==G)]=Gi
	return yout

def EvaluateSplit(clf,TestKeys,Grouping):
        Xtest=Data[TestKeys]
        ytest=Y[TestKeys]
	ytest=ChangeGroups(ytest,Grouping)
	
	Proba=clf.predict_proba(Xtest)
	LogLoss=0
        for p,y in zip(Proba,ytest):
                LogLoss=LogLoss-math.log(p[y-1])

        return LogLoss/len(ytest)

def Write(FileName,LogLoss,grouping):
        if (os.path.exists(FileName)):
                f=open(FileName,'rb')
                [Done,Score,Groupings]=pickle.load(f)
                f.close()
                index=FindSpot(grouping,Groupings)
                if (grouping == Groupings[index]).all() and Done[index]!=2:
                        Score[index]=LogLoss
                        Done[index]=Done[index]+1

        else:
		sys.exit('No file FileName exists')
        f=open(FileName,'wb')
        pickle.dump([Done,Score,Groupings],f)
        f.close()


def Read(FileName):
        if (os.path.exists(FileName)):
                f=open(FileName,'rb')
                [Done,Score,Groupings]=pickle.load(f)
                f.close()
		i=FindZero(Done)
                if Done[i] == 0 or Done[i] == 1:
			gnext=Groupings[i]
			Done[i]=Done[i]+1
        else:
                sys.exit('No file FileName exists')
        f=open(FileName,'wb')
        pickle.dump([Done,Score,Groupings],f)
        f.close()
	return gnext

def FindSpot(grouping,Groupings):
	for i,g in zip(np.arange(len(Groupings)),Groupings):
		if (grouping==g).all():
			break
	return i
	
def FindZero(Done):
	for i,d in zip(np.arange(len(Done)),Done):
		if d==0:
			break	
	
	if Done[i]==2:
		for i,d in zip(np.arange(len(Done)),Done):
                	if d==1:
                        	break

	return i
		

			
def GetComb(N):
	return np.array(list(itertools.combinations('123456789',N)),dtype=np.int)

def GetCombPerm(N):
	Comb=np.ones(9)*2
        Comb[np.array(list(itertools.combinations('123456789',N)),dtype=np.int)]=1
        return Comb.astype('int')
	


FileName=sys.argv[2]

Groupings=GetComb(int(sys.argv[1]))
Done=np.zeros(len(Groupings))
Score=np.ones(len(Groupings))*10

print 'Shape of Groupings'+str(Groupings.shape)

if not (os.path.exists(FileName)):
	f=open(FileName,'wb')
        pickle.dump([Done,Score,Groupings],f)
        f.close()	

while 1:
	Grouping = Read(FileName)
	Grouping=get_iterable(Grouping)
	trainkeys,testkeys=utils.GetRandomSetKeys(TrainKeys,DataFrac=1,TrainFrac=0.5)
	clf=TrainSplit(trainkeys,testkeys,Grouping)
	LogLoss=EvaluateSplit(clf,TestKeys,Grouping)
	Write(FileName,LogLoss,Grouping)
	












