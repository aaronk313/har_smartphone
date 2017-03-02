#
# The objective of this dataset is to use gyroscopic data to predict
# the type of human activity out of the following list of possible activities
#
# 1 WALKING
# 2 WALKING_UPSTAIRS
# 3 WALKING_DOWNSTAIRS
# 4 SITTING
# 5 STANDING
# 6 LAYING
#

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report


if sys.platform=='win32':
    path = 'Z:\OneDrive\datascience\projects\har_smartphone\\'
elif sys.platform=='darwin':
    path = '/extstore/FILECABINET/OneDrive/datascience/projects/har_smartphone/'

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]

X_train_raw = []

with open(path+'train/X_train.txt') as f:
    reader = csv.reader(f,delimiter=" ")
    X_train_list = list(reader)

for line in X_train_list:
    X_train_raw.append(map(float,filter(None,line)))

X_train = pd.DataFrame(X_train_raw)
X = X_train.values


with open(path+'train/y_train.txt') as f:
    reader = csv.reader(f,delimiter=" ")
    y_train_list = list(reader)

y_train = []

for label in y_train_list:
    y_train.append(map(int,label))

y_t = np.array(y_train)
y = np.squeeze(y_t[:,np.newaxis, np.newaxis])


lr = LogisticRegression()
lr.fit(X,y)

def report(s_model,x_data,y_data):
    scores = cross_val_score(s_model,x_data, y_data, scoring='accuracy', cv=10)
    print('INITIAL MODEL ACCURACY (NO Cross Validation: ', s_model.score(x_data, y_data))
    print('CV 10-FOLD Scores: ', scores)
    print('CV 10-FOLD Mean Accuracy', scores.mean() )
    print('CV 10-FOLD Standard Deviation of Accuracy', scores.std() )

report(lr, X, y)


X_test_raw = []

with open(path+'test/X_test.txt') as f:
    reader = csv.reader(f,delimiter=" ")
    X_test_list = list(reader)

for line in X_test_list:
    X_test_raw.append(map(float,filter(None,line)))
    
X_test = pd.DataFrame(X_test_raw)
Xt = X_test.values

y_test = []

with open(path+'test/y_test.txt') as f:
    reader = csv.reader(f,delimiter=" ")
    y_test_list = list(reader)

for label in y_test_list:
    y_test.append(map(int,label))

y_tst = np.array(y_test)
yt = np.squeeze(y_tst[:,np.newaxis, np.newaxis])
y_pred = lr.predict(Xt)

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

print "LOGISTIC REGRESSTION ========================"
print "Accuracy: ",accuracy_score(yt,y_pred)
print "Precision: ", precision_score(yt,y_pred, average="weighted")
print "Recall: ", recall_score(yt,y_pred, average="weighted")
print "F1: ",f1_score(yt,y_pred, average="weighted")
print confusion_matrix(yt, y_pred)
print "============================================="


hmplog = plt.axes()
sns.heatmap(confusion_matrix(yt, y_pred), yticklabels=LABELS, xticklabels=LABELS, ax=hmplog)
hmplog.set_title('Logistic Regression')
hmplog.show()



# RandomForest 

from sklearn.ensemble import RandomForestClassifier

rfclf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=2, random_state=0)
rfclf.fit(X,y)

yrf_pred = rfclf.predict(Xt)

rf_scores_mean = cross_val_score(rfclf, X, y, cv=5).mean()
rf_accuracy = accuracy_score(yt,yrf_pred)
rf_prec = classification_report(yt, yrf_pred).split('total')[1].split('   ')[2].strip()
rf_rec = classification_report(yt, yrf_pred).split('total')[1].split('   ')[4]
    
rf_Yall_score = rfclf.predict_proba(Xt)[:,1]

print "RANDOM FOREST ========================"
print "RT Accuracy: ",accuracy_score(yt,yrf_pred)
print "RT Precision: ", precision_score(yt,yrf_pred, average="weighted")
print "RT Recall: ", recall_score(yt,yrf_pred, average="weighted")
print "RT F1: ",f1_score(yt,yrf_pred, average="weighted")
print confusion_matrix(yt, yrf_pred)
print "======================================"

hmprt = plt.axes()
sns.heatmap(confusion_matrix(yt, yrf_pred), yticklabels=LABELS, xticklabels=LABELS, ax=hmprt)
hmprt.set_title('Random Forest')
hmprt.show()

