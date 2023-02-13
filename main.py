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

## Compare the values of variables in the training and test sets

# 1. Are the values of 'Activity' are the same in the training and test?
train["Activity"].unique() == test["Activity"].unique()  # yes
#%%
# For ease mix the training and test sets
train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
#%%
xtick_loc = np.arange(len(both["Activity"].unique()))
bar_width = .2
# fig, ax = plt.subplot()
plt.bar(x=xtick_loc - .2, height=both["Activity"].value_counts(), width = bar_width, label='Both')
plt.bar(x=xtick_loc , height=train["Activity"].value_counts(),  width = bar_width, label='Train')
plt.bar(x=xtick_loc + .2, height=test["Activity"].value_counts(),  width = bar_width, label='Test')
plt.title("Distribution of 'Activity'")
plt.xticks(ticks = xtick_loc, labels = both["Activity"].value_counts().index, rotation = 45, 
           size = 7,
           ha = 'right', rotation_mode='anchor')
plt.legend()
#%%
ax[1].bar(x=train["Activity"].value_counts().index, height=train["Activity"].value_counts(), label='Train')
ax[1].set_title("Training Set")

ax[2].bar(x=test["Activity"].value_counts().index, height=test["Activity"].value_counts(), label='Test')
ax[2].set_title("Test Set")
ax[2].set_xticklabels(rotation = 45, labels = train["Activity"].value_counts().index)
#%%
x_location = [_+1 for _ in range(len(both["Activity"].unique()))]
plt.bar(x=x_location - .2, height=both["Activity"].value_counts(), label='Both')
plt.bar(x=x_location, height=train["Activity"].value_counts(), label='Train')
plt.bar(x=x_location + .2, height=test["Activity"].value_counts(), label='Test')
both["Activity"].value_counts().index

X_axis = 
X_axis + 1
#%%    
# 2. Values of 'subject' in the training and test sets 
train['subject'].sort_values().unique() 
test['subject'].sort_values().unique() 

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
