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
import sklearn as sk
import seaborn as sns
import numpy as np
import csv

path = '/extstore/FILECABINET/OneDrive/datascience/projects/har_smartphone/'

with open(path+'train/X_train.txt') as f:
    reader = csv.reader(f,delimiter=" ")
    X_train_list = list(reader)
    
X_train_raw = []

for line in X_train_list:
    X_train_raw.append(filter(None,line))

X_train = pd.DataFrame(X_train_raw)

