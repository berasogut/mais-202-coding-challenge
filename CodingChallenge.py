import sys
# You will need to install the SciPy platform and the libraries to run this code
# The next line is to add a path for the libraries to look for and if you don't already have the libraries in sys.path you should append too.
# sys.path.append("/anaconda3/lib/python3.7/site-packages")
import scipy
import numpy
import matplotlib
import pandas
import sklearn
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import certifi
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.datasets import make_moons
url = "https://raw.githubusercontent.com/berasogut/mais-202-coding-challenge/master/data.csv"
dataset = pandas.read_csv(url)
ct = {}
sum = {}
avg = {}
for i in range(dataset.shape[0]):
    ct[dataset['purpose'][i]] = 0
    sum[dataset['purpose'][i]] = 0
for i in range(dataset.shape[0]):
    ct[dataset['purpose'][i]] += 1
    sum[dataset['purpose'][i]] += dataset['int_rate'][i]
    avg[dataset['purpose'][i]] = sum[dataset['purpose'][i]]/ct[dataset['purpose'][i]]
list1 = []
list2 = []
for (k,v) in avg.items():
    list1.append(k)
    list2.append(v)
    
import numpy as np
y_pos = np.arange(len(list1))
 
# Bars and colors
plt.bar(y_pos, list2, color=['cyan',(0.5,0.1,0.5,0.6)], edgecolor='black')
 
# Axis names
plt.xlabel('Purpose')
plt.ylabel('Mean (int_rate)')
 
# Limits for the y axis
plt.ylim(0,18.5)
 
plt.xticks(y_pos, list1, rotation=90, color='brown')
plt.yticks(color='brown')

plt.subplots_adjust(bottom=0.4, top=0.99)

plt.show()

