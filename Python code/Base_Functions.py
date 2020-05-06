"""
This code accompanies the manuscript "Rapid diagnosis of hereditary 
haemolytic anaemias using automated rheoscopy and supervised machine learning" 
and provides the functions necessary to perform
sample analysis, classifier training and sample classification.

@author: Pedro L. Moura, Timothy J. Satchwell, Ashley M. Toye
"""
import matplotlib
import os
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs
import seaborn as sns; sns.set(color_codes=True)
sns.set_style("white")
from matplotlib import rcParams
from sklearn import svm, metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from timeit import default_timer as timer
from collections import Counter

#These parameters alter the generated graphs.
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
params = {'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
          'axes.labelsize': 'xx-large'}
rcParams.update(params)

#This command removes the temporary file created for sample classification
#in order to avoid re-classification of previous samples
if os.path.isfile("varstore.dat"):
    os.remove("varstore.dat")

#This function reads an ARCA raw data file into memory and extracts
#cross-sectional area and deformability metrics
def ReadARCA(filename):
    import pandas as pd
    df = pd.read_csv(filename, usecols=(13,16,19))
    cols = df.columns
    cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, bytes)) else x)
    df.columns = cols
    df=df[df._Error_code == 0]
    Deformability = df.as_matrix(columns=['_A/B'])
    Area = df.as_matrix(columns=['_Area_um2'])
    Deformability = np.transpose(Deformability)
    Deformability = Deformability[0]
    Area = np.transpose(Area)
    Area = Area[0]
    print('Median area = ' + str(round(np.median(Area),3)))
    print('Median deformability = ' + str(round(np.median(Deformability),3)))
    return [Area, Deformability]

#This function generates error bars when joining multiple datasets
#(important for visualization)
def GetErrorBars(*args):
    i=0
    AreaEB = []
    DefEB = []
    TempAreaEB1=[]
    TempDefEB1=[]
    for arg in args:
        AreaHistogram = np.histogram(arg[0], bins=np.arange(0,141,5))
        DefHistogram = np.histogram(arg[1], bins=np.arange(1,3.31,0.1))
        TotalCells = sum(AreaHistogram[0])
        AreaProbDist = AreaHistogram[0]/TotalCells
        DefProbDist = DefHistogram[0]/TotalCells
        TempAreaEB1.append(AreaProbDist)
        TempDefEB1.append(DefProbDist)
    for i in range(len(TempAreaEB1[0])):
        TempAreaEB2=[]
        for j in range(len(TempAreaEB1)):
            TempAreaEB2.append(TempAreaEB1[j][i])
        AreaEB.append(np.std(TempAreaEB2))
    for i in range(len(TempDefEB1[0])):
        TempDefEB2=[]
        for j in range(len(TempDefEB1)):
            TempDefEB2.append(TempDefEB1[j][i])
        DefEB.append(np.std(TempDefEB2))
    return [AreaEB, DefEB]

#This function attributes the same weight to the datasets (despite differing
#cell numbers) and must be run before joining and visualization
def ReshapeForJoin(*args):
    minimum = 20000
    for arg in args:
        array_size = len(arg[0])
        if array_size < minimum:
            minimum = array_size
    print (minimum)
    rng_state = np.random.get_state()
    Argument_List = []
    for arg in args:
        np.random.set_state(rng_state)
        np.random.shuffle(arg[0])
        np.random.set_state(rng_state)
        np.random.shuffle(arg[1])
        AreaSet  = arg[0][-minimum:]
        DeformabilitySet  = arg[1][-minimum:]
        Argument_List.append([AreaSet,DeformabilitySet])
    return Argument_List

#This function joins multiple datasets into one and calls the error bar function
#to create error bars for visualization
def JoinARCA(*args):
    Areas = []
    Deformabilities = []
    AvgAreas = []
    AvgDeformabilities = []
    for arg in args:
        Areas = np.concatenate((Areas, arg[0]))
        Deformabilities = np.concatenate((Deformabilities, arg[1]))
        AvgAreas.append(np.median(arg[0]))
        AvgDeformabilities.append(np.median(arg[1]))
    ErrorBars = GetErrorBars(*args)
    print('Median area = ' + str(round(np.median(Areas),3))+' ± '+(str(round(np.std(AvgAreas),3))))
    print((np.percentile(Areas, 75) - np.percentile(Areas, 25)))
    print('Median deformability = ' + str(round(np.median(Deformabilities),3))+' ± '+(str(round(np.std(AvgDeformabilities),3))))
    print((np.percentile(Deformabilities, 75) - np.percentile(Deformabilities, 25)))
    return [Areas, Deformabilities, ErrorBars[0],ErrorBars[1]]

#This function generates a 2D contour plot of cross-sectional area and DI
def ContourARCA(*args):
    f, ax = plt.subplots()
    plt.xlabel('Cross-sectional area (µm²)')
    plt.ylabel('Deformability Index (Length/Width)')
    ax.set(xlim=(20, 120), ylim=(1, 3))
    cmap_list = ["Reds","Blues","Greys","Greens","Purples","copper","winter","Oranges"]
    i=0
    for arg in args:
        ax = sns.kdeplot(arg[0], arg[1], cmap=cmap_list[i])
        i=i+1
        
#This function generates a 2D scatter plot of cross-sectional area and DI
def ScatterARCA(*args):
    f, ax = plt.subplots()
    plt.xlabel('Cross-sectional area (µm²)')
    plt.ylabel('Deformability Index (Length/Width)')
    ax.set(xlim=(20, 120), ylim=(1, 3))
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    i=0
    for arg in args:
        ax = sns.regplot(arg[0], arg[1], color = c_list[i], ci=None,fit_reg=False,scatter_kws={'s':4})
        i=i+1

#This function generates a histogram plot of cross-sectional area
def AreaHistogramARCA(*args, ErrorBars = True, FillBetween = False):
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    i=0
    f, ax = plt.subplots()
    ymax=[]
    plt.xlabel('Cross-sectional area (µm²)')
    plt.ylabel('Probability')
    for arg in args:
        histogram = np.histogram(arg[0], bins=np.arange(0,141,5))
        TotalCells = sum(histogram[0])
        ProbDist = histogram[0]/TotalCells
        ymax.append(max(ProbDist))
    for arg in args:
        histogram = np.histogram(arg[0], bins=np.arange(0,141,5))
        TotalCells = sum(histogram[0])
        ProbDist = histogram[0]/TotalCells
        ax.set(xlim=(0, 140),ylim=(0,(max(ymax))+(max(ymax))/4))
        BinToUse = np.arange(2.5,142,5)
        if len(arg) == 2:
            ax1 = plt.plot(BinToUse,ProbDist,color=c_list[i])
        else:
            if ErrorBars == True:
                ax1 = plt.errorbar(BinToUse,ProbDist,arg[2],color=c_list[i], elinewidth=1, #ecolor='black', 
                               capsize=4, capthick=1)
            else:
                ax1 = plt.plot(BinToUse,ProbDist,color=c_list[i])
            if FillBetween == True:
                ax1 = plt.fill_between(BinToUse,ProbDist-arg[2],ProbDist+arg[2], color=c_list[i], alpha=.2)
        i=i+1

#This function generates a histogram plot of deformability index
def DefHistogramARCA(*args, ErrorBars = True, FillBetween = False):
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    i=0
    f, ax = plt.subplots()
    ymax=[]
    plt.xlabel('Deformability Index (Length/Width)')
    plt.ylabel('Probability')
    for arg in args:
        histogram = np.histogram(arg[1], bins=np.arange(1,3.31,0.1))
        TotalCells = sum(histogram[0])
        ProbDist = histogram[0]/TotalCells
        ymax.append(max(ProbDist))
    for arg in args:
        histogram = np.histogram(arg[1], bins=np.arange(1,3.31,0.1))
        TotalCells = sum(histogram[0])
        ProbDist = histogram[0]/TotalCells
        ax.set(xlim=(1, 3.5),ylim=(0,(max(ymax))+(max(ymax))/4))
        BinToUse = np.arange(1.05,3.35,0.1)
        if len(arg) == 2:
            ax1 = plt.plot(BinToUse,ProbDist,color=c_list[i])
        else:
            if ErrorBars == True:
                ax1 = plt.errorbar(BinToUse,ProbDist,arg[3],color=c_list[i], elinewidth=1, #ecolor='black', 
                               capsize=4, capthick=1)
            else:
                ax1 = plt.plot(BinToUse,ProbDist,color=c_list[i]) 
            if FillBetween == True:
                ax1 = plt.fill_between(BinToUse,ProbDist-arg[3],ProbDist+arg[3], color=c_list[i], alpha=.2)
        i=i+1
        
#This function generates a KDE estimation of the deformability index histogram
def ContDefHistogramARCA(*args):
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    i=0
    f, ax = plt.subplots()
    ymax=[]
    plt.xlabel('Deformability Index (Length/Width)')
    plt.ylabel('Probability')
    for arg in args:
        kde = sm.nonparametric.KDEUnivariate(arg[1])
        kde.fit()
        kde.density = kde.density/sum(kde.density)
        ymax.append(max(kde.density))
    for arg in args:
        kde = sm.nonparametric.KDEUnivariate(arg[1])
        kde.fit()
        kde.density = kde.density/sum(kde.density)
        ax.set(xlim=(1, 3.5),ylim=(0,(max(ymax))+(max(ymax))/10))
        ax1 = plt.plot(kde.support,kde.density,color=c_list[i])
        i=i+1

#This function generates a KDE estimation of the cross-sectional area histogram
def ContAreaHistogramARCA(*args):
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    i=0
    f, ax = plt.subplots()
    ymax=[]
    plt.xlabel('Cross-sectional area (µm²)')
    plt.ylabel('Probability')
    for arg in args:
        kde = sm.nonparametric.KDEUnivariate(arg[0])
        kde.fit()
        kde.density = kde.density/sum(kde.density)
        ymax.append(max(kde.density))
    for arg in args:
        kde = sm.nonparametric.KDEUnivariate(arg[0])
        kde.fit()
        kde.density = kde.density/sum(kde.density)
        ax.set(xlim=(0, 140),ylim=(0,(max(ymax))+(max(ymax))/10))
        ax1 = plt.plot(kde.support,kde.density,color=c_list[i])
        i=i+1

#This is a wrapper function that calls all of the previous visualization functions
def EasyARCA(*args):
    i = 0
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    ContourARCA(*args)
    ScatterARCA(*args)
    AreaHistogramARCA(*args)
    DefHistogramARCA(*args)
    print('In the order given by the input datasets, the colours are:')
    for arg in args:
        print(c_list[i])
        i = i + 1
        
#This function generates a 2D contour plot and accompanying KDE histograms
def JointARCA(*args):
    cmap_list = ["Reds","Blues","Greys","Greens","Purples","copper","winter","Oranges"]
    c_list = ["red","blue","black","limegreen","magenta","gold","cyan","orange"]
    i = 0
    g = sns.JointGrid(x=args[0][0], y = args[0][1],  xlim=(20, 120), ylim=(1, 2.5))
    g.set_axis_labels('Cross-sectional area (µm²)', 'Deformability Index (Length/Width)', fontsize=16)
    for arg in args:
        sns.kdeplot(arg[0], arg[1], cmap=cmap_list[i],  ax=g.ax_joint)
        sns.distplot(arg[0], kde=True, hist=False, color=c_list[i], ax=g.ax_marg_x)
        sns.distplot(arg[1], kde=True, hist=False, color=c_list[i], ax=g.ax_marg_y, vertical=True)
        i = i+1
    plt.show()

#This function gets the number of times the temporary file has been accessed and adds to the counter
def get_var_value(filename="varstore.dat"):
    with open(filename, "a+") as f:
        f.seek(0)
        val = int(f.read() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(val))
        return val
    
#This function augments the data for training set generation through randomized subsetting
def ARCADataAugmenter(*args):
    h=get_var_value()-1
    num_samples = 10000
    TrainingSetCombined = []
    TestingSetCombined = []
    CombinedClassifiers = []
    for arg in args:
        rng_state = np.random.get_state()
        np.random.shuffle(arg[0])
        np.random.set_state(rng_state)
        np.random.shuffle(arg[1])
        AreaFVTestingSet  = arg[0][-500:]
        AreaFVTestingSet = AreaFVTestingSet.reshape(1,-1)
        AreaFVTestingSet = AreaFVTestingSet/140
        AreaFVTestingAvg = np.mean(AreaFVTestingSet)
        AreaFVTestingStd = np.std(AreaFVTestingSet)
        DeformabilityFVTestingSet  = arg[1][-500:]
        DeformabilityFVTestingSet = DeformabilityFVTestingSet.reshape(1,-1)
        DeformabilityFVTestingSet = DeformabilityFVTestingSet/3.3
        DeformabilityFVTestingAvg = np.mean(DeformabilityFVTestingSet)
        DeformabilityFVTestingStd = np.std(DeformabilityFVTestingSet)
        AreaFVTrainingSet = arg[0][:-500]
        AreaFVTrainingSet = AreaFVTrainingSet.reshape(1,-1)
        AreaFVTrainingSet = AreaFVTrainingSet/140
        DeformabilityFVTrainingSet = arg[1][:-500]
        DeformabilityFVTrainingSet = DeformabilityFVTrainingSet.reshape(1,-1)
        DeformabilityFVTrainingSet = DeformabilityFVTrainingSet/3.3
        idx = np.random.randint(0,len(AreaFVTrainingSet[0]),size=(num_samples,500))
        AreaTrainingSet = AreaFVTrainingSet[0][idx]
        DefTrainingSet = DeformabilityFVTrainingSet[0][idx]
        TrainingFVs = []
        for i in range(0,num_samples):
            LocalFV = []
            LocalFV.append(np.mean(AreaTrainingSet[i]))
            LocalFV.append(np.std(AreaTrainingSet[i]))
            LocalFV.append(np.mean(DefTrainingSet[i]))
            LocalFV.append(np.std(DefTrainingSet[i]))
            TrainingFVs.append(LocalFV)
        TestingSet = [AreaFVTestingAvg,AreaFVTestingStd,DeformabilityFVTestingAvg,DeformabilityFVTestingStd]
        TrainingSetCombined.append(TrainingFVs)
        TestingSetCombined.append(TestingSet)
        ClassifierInfo = np.zeros(num_samples)+h
        CombinedClassifiers.append(ClassifierInfo)
    TrainingSetCombined = tuple(TrainingSetCombined)
    TestingSetCombined = tuple(TestingSetCombined)
    CombinedClassifiers = tuple(CombinedClassifiers)
    FinalTrainingSet = np.vstack(TrainingSetCombined)
    FinalTestingSet = np.vstack(TestingSetCombined)
    FinalClassifiers = np.hstack(CombinedClassifiers)
    return [FinalTrainingSet,FinalTestingSet,FinalClassifiers]

#This is a wrapper function that calls sci-kit learn classifiers and trains them with the training/testing sets and classifier information
#Change the uncommented classifiers to modify the classifier in use
def ClassifierTrain(FinalTrainingSet,FinalTestingSet,FinalClassifiers):
    clf = KNeighborsClassifier()
    #clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    #clf = AdaBoostClassifier(algorithm="SAMME",n_estimators=50)
    #clf = RandomForestClassifier()
    #clf = GradientBoostingClassifier()
    #clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
    clf.fit(FinalTrainingSet, FinalClassifiers)
    print(clf.predict(FinalTestingSet))
    #print(clf.predict_proba(FinalTestingSet))
    return clf

#This function classifies a dataset (stochastically)
def ARCAClassify(knn, *args):
    TestingSetCombined = []
    for arg in args:
        rng_state = np.random.get_state()
        np.random.shuffle(arg[0])
        np.random.set_state(rng_state)
        np.random.shuffle(arg[1])
        AreaFVTestingSet = arg[0][-500:]
        AreaFVTestingSet = AreaFVTestingSet.reshape(1,-1)
        AreaFVTestingSet = AreaFVTestingSet/140
        AreaFVTestingAvg = np.mean(AreaFVTestingSet)
        AreaFVTestingStd = np.std(AreaFVTestingSet)
        DeformabilityFVTestingSet = arg[1][-500:]
        DeformabilityFVTestingSet = DeformabilityFVTestingSet.reshape(1,-1)
        DeformabilityFVTestingSet = DeformabilityFVTestingSet/3.3
        DeformabilityFVTestingAvg = np.mean(DeformabilityFVTestingSet)
        DeformabilityFVTestingStd = np.std(DeformabilityFVTestingSet)
        TestingSet = [AreaFVTestingAvg,AreaFVTestingStd,DeformabilityFVTestingAvg,DeformabilityFVTestingStd]
        TestingSetCombined.append(TestingSet)
    TestingSetCombined = tuple(TestingSetCombined)
    FinalTestingSet = np.vstack(TestingSetCombined)
    return knn.predict(FinalTestingSet)
    
#This function calls the ARCAClassify function repeatedly to provide higher certainty to the classification
def RepeatedARCAClassify(knn, *args):
    ListOfClassificationLists = []
    ListOfModes = []
    ListOfCertainties = []
    n_repeats=10000
    for i in range(0,n_repeats):
        ClassificationList = ARCAClassify(knn,*args)
        ListOfClassificationLists.append([ClassificationList])
    LCLArray = np.array(ListOfClassificationLists)
    for j in range(len(args)):
        ListOfSamplewiseClassifications = []
        for k in range(0,n_repeats):
            ListOfSamplewiseClassifications.append(LCLArray[k][:,j])
        LocalMode = stats.mode(ListOfSamplewiseClassifications)
        ListOfModes.append(LocalMode[0][0])
        PercentCertainty = LocalMode[1][0]*100/n_repeats
        ListOfCertainties.append(PercentCertainty)
    ListOfModes=np.reshape(ListOfModes,(1,-1))
    ListOfCertainties=np.reshape(ListOfCertainties,(1,-1))
    return (ListOfModes)

#This function takes in the output of RepeatedARCAClassify and provides a list of classified samples
def MakeFinalAnnotation(CountedSample):
    x = {}
    x['Ctrl']= CountedSample[0]
    x['CDAII']= CountedSample[1]
    x['HS']= CountedSample[2]
    x['HX']= CountedSample[3]
    x['PKD']= CountedSample[4]
    print(x)
    return(x)
    