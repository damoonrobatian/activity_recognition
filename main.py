import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import warnings
warnings.filterwarnings("ignore")
import os

root_path = input("Enter the working directory's path:\n")
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
           ha = 'right', rotation_mode='anchor') # ha='right' is not enough to visually align labels with ticks.
                                                 # For rotation=45, use both ha='right' and rotation_mode='anchor'
                                                 # For other angles, use a ScaledTranslation() instead

plt.legend()
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
