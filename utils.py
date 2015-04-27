#!/usr/bin/python
import sys
import numpy as np
from sklearn import svm
import math
import random

def LoadData(Name='train.csv'):
	Data=np.loadtxt(open(Name,"rb"),delimiter=",",skiprows=1,usecols=np.arange(1,94)) # possibly uint16

	Data=np.divide(Data,(np.sum(Data,axis=1)).reshape(len(np.sum(Data,axis=1)),1))

	Labels = np.genfromtxt(open(Name,"rb"),dtype='str',delimiter=",",skiprows=1,usecols=94)
	Y=[]
	for y in Labels:
       		Y=np.append(Y,np.int(y[-1]))

	Y=Y.astype(int)
	return Data,Y

def ChooseSVM(Xtrain,ytrain,Xtest,ytest):
        C = [1e-2, 1, 1e2]
        Gamma = [1e-1, 1, 1e1]
        Score = 0
        for c in C:
                for gamma in Gamma:
                        clf=svm.SVC(cache_size=6000,probability=True,kernel='rbf',C=c,gamma=gamma)
                        clf.fit(Xtrain,ytrain)
                        score=clf.score(Xtest,ytest)
                        if score > Score:
                                Score = score
                                Cbest = c
                                gammabest = gamma
				bestclf=clf
        print Score
        return bestclf



def RunClassification(CLF,Xtest,UpToLayer):
        k=0
        NLastLayer=0
        Proba=np.array([], dtype=np.int64).reshape(0,9)
        for testdata in Xtest:
                if testdata[0]!=testdata[0]:
                        testdata=np.zeros(93)

                i=0
                logproba=1
                while logproba>0.1 and i<UpToLayer:
                        proba=CLF[i].predict_proba(testdata)
                        logproba=-np.sum(proba*np.log(proba),axis=1)
                        i=i+1
                        if i==UpToLayer:
                                NLastLayer=NLastLayer+1
                Proba=np.vstack((Proba,proba))
                k=k+1
        return Proba



def GetRandomSetKeys(Keys,DataFrac,TrainFrac):
        np.random.shuffle(Keys)
        Keys = Keys[0:int(math.ceil(DataFrac*len(Keys)))]
        Split = np.split(Keys,[math.ceil(len(Keys)*TrainFrac),len(Keys)])
        TrainKeys=Split[0]
        TestKeys=Split[1]
        return np.array(TrainKeys),np.array(TestKeys)


def GetSample(N,Keys,Group):
	KeysGroup=np.where(Y[Keys]==Group)
		
		

def CheckProba(Proba,ytest):
	LogLoss=0
	for p,y in zip(Proba,ytest):
		LogLoss=LogLoss-math.log(p[y-1])
	
	return LogLoss/len(ytest)





	




