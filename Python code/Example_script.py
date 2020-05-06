"""
This code accompanies the manuscript "Rapid diagnosis of hereditary 
haemolytic anaemias using automated rheoscopy and supervised machine learning" 
and provides the functions necessary to perform
sample analysis, classifier training and sample classification.

The following script provides an example of how to read and process the data
for classifier training and subsequent sample classification.

This example uses one sample from each disease type to train a classifier
and then classifies all remaining samples.
 
@author: Pedro L. Moura, Timothy J. Satchwell, Ashley M. Toye
"""
from Base_Functions import *


Ctrl1 = ReadARCA('Ctrl1.csv')
Ctrl2 = ReadARCA('Ctrl2.csv')
Ctrl3 = ReadARCA('Ctrl3.csv')
Ctrl4 = ReadARCA('Ctrl4.csv')
Ctrl5 = ReadARCA('Ctrl5.csv')
Ctrl6 = ReadARCA('Ctrl6.csv')
CDAII1 = ReadARCA('CDAII1.csv')
CDAII2 = ReadARCA('CDAII2.csv')
CDAII3 = ReadARCA('CDAII3.csv')
CDAII4 = ReadARCA('CDAII4.csv')
CDAII5 = ReadARCA('CDAII5.csv')
CDAII6 = ReadARCA('CDAII6.csv')
CDAII7 = ReadARCA('CDAII7.csv')
CDAII8 = ReadARCA('CDAII8.csv')
CDAII9 = ReadARCA('CDAII9.csv')
GX1 = ReadARCA('GX1.csv')
GX2 = ReadARCA('GX2.csv')
GX3 = ReadARCA('GX3.csv')
HS1 = ReadARCA('HS1.csv')
HS2 = ReadARCA('HS2.csv')
HS3 = ReadARCA('HS3.csv')
HS4 = ReadARCA('HS4.csv')
HS5 = ReadARCA('HS5.csv')
HS6 = ReadARCA('HS6.csv')
HS7 = ReadARCA('HS7.csv')
HS8 = ReadARCA('HS8.csv')
HS9 = ReadARCA('HS9.csv')
HS10 = ReadARCA('HS10.csv')
HS11 = ReadARCA('HS11.csv')
HS12 = ReadARCA('HS12.csv')
HS13 = ReadARCA('HS13.csv')
PiezoHX1 = ReadARCA('PiezoHX1.csv')
PiezoHX2 = ReadARCA('PiezoHX2.csv')
PiezoHX3 = ReadARCA('PiezoHX3.csv')
PiezoHX4 = ReadARCA('PiezoHX4.csv')
PiezoHX5 = ReadARCA('PiezoHX5.csv')
PiezoHX6 = ReadARCA('PiezoHX6.csv')
PiezoHX7 = ReadARCA('PiezoHX7.csv')
PiezoHX8= ReadARCA('PiezoHX8.csv')
PiezoHX9 = ReadARCA('PiezoHX9.csv')
PiezoHX10 = ReadARCA('PiezoHX10.csv')
PKD1 = ReadARCA('PKD1.csv')
PKD2 = ReadARCA('PKD2.csv')
PKD3 = ReadARCA('PKD3.csv')
PKD4 = ReadARCA('PKD4.csv')
PKD5 = ReadARCA('PKD5.csv')
PKD6 = ReadARCA('PKD6.csv')

CtrlTrain=ARCADataAugmenter(Ctrl1)
CDAIITrain=ARCADataAugmenter(CDAII1)
HSTrain=ARCADataAugmenter(HS1)
PiezoHXTrain=ARCADataAugmenter(PiezoHX1)
PKDTrain=ARCADataAugmenter(PKD1)

TrainingSet=np.concatenate((CtrlTrain[0],CDAIITrain[0],HSTrain[0],PiezoHXTrain[0],PKDTrain[0]))
#ZeroTrainSet=np.concatenate((CtrlTrain[0],CtrlTrain[0],CtrlTrain[0],CtrlTrain[0],CtrlTrain[0]))
TestingSet=np.concatenate((CtrlTrain[1],CDAIITrain[1],HSTrain[1],PiezoHXTrain[1],PKDTrain[1]))
#ZeroTrainTest=np.concatenate((CtrlTrain[1],CtrlTrain[1],CtrlTrain[1],CtrlTrain[1],CtrlTrain[1]))
SampleAnnotationSet=np.concatenate((CtrlTrain[2],CDAIITrain[2],HSTrain[2],PiezoHXTrain[2],PKDTrain[2]))

ARCAClassifier= ClassifierTrain(TrainingSet,TestingSet,SampleAnnotationSet)
#ARCAClassifier= ClassifierTrain(ZeroTrainSet,ZeroTrainTest,SampleAnnotationSet)

print('Ctrl')
c= Counter(RepeatedARCAClassify(ARCAClassifier,Ctrl2,Ctrl3,Ctrl4,Ctrl5,Ctrl6)[0])
d=MakeFinalAnnotation(c)
print('Total samples: ' + str(sum(d.values())))

print('CDAII')
c= Counter(RepeatedARCAClassify(ARCAClassifier,CDAII2,CDAII3,CDAII4,CDAII5,CDAII6,CDAII7,CDAII8,CDAII9)[0])
d=MakeFinalAnnotation(c)
print('Total samples: ' + str(sum(d.values())))

print('HS')
c= Counter(RepeatedARCAClassify(ARCAClassifier,HS2,HS3,HS4,HS5,HS6,HS7,HS8,HS9,HS10,HS11,HS12,HS13)[0])
d=MakeFinalAnnotation(c)
print('Total samples: ' + str(sum(d.values())))

print('HX')
c= Counter(RepeatedARCAClassify(ARCAClassifier,PiezoHX2,PiezoHX3,PiezoHX4,PiezoHX5,PiezoHX6,PiezoHX7,PiezoHX8,PiezoHX9,PiezoHX10)[0])
d=MakeFinalAnnotation(c)
print('Total samples: ' + str(sum(d.values())))

print('PKD')
c= Counter(RepeatedARCAClassify(ARCAClassifier,PKD2,PKD3,PKD4,PKD5,PKD6)[0])
d=MakeFinalAnnotation(c)
print('Total samples: ' + str(sum(d.values())))
