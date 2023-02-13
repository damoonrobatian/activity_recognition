import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings("ignore")
import os

root_path = "/home/damoon/Dropbox/programming/python/projects/1_activity_recognition/"
os.chdir(root_path)
#%% Load the data
train = pd.read_csv(root_path + "data/train-1.csv")
test = pd.read_csv(root_path + "data/test.csv")
#%% Get to know the data
train.shape
test.shape
# Are the variables the same?
(train.columns == test.columns).sum() # yes

# Compare the values of variables in the training and test sets
# Are the values of 'Activity' are the same in the training and test?
train["Activity"].unique() == test["Activity"].unique()  # yes

# Are the values 
train['subject'].sort_values().unique() 
test['subject'].sort_values().unique() 


train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
# both['subject'] = '#' + both['subject'].astype(str)

uniq_subjects = both["subject"].sort_values().unique()

#%%
rows_per_subject = {}
t = []
for d in both.groupby('subject'):
    rows_per_subject[d[0]] = d[1].shape[0]
    t.append([d[0], d[1].shape[0]])    

rows_per_subject 
t
pd.DataFrame(t)
