import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
train_subjects = train['subject'].sort_values().unique() 
test_subjects = test['subject'].sort_values().unique() 
#%%
rows_per_subject = []
for subject, sub_df in both.groupby('subject'):
    if subject in train_subjects:
        rows_per_subject.append([str(subject), sub_df.shape[0], "Train"])  # str(subject) to be able to sort the plot later
    else:
        rows_per_subject.append([str(subject), sub_df.shape[0], "Test"])
    
rows_per_subject = pd.DataFrame(rows_per_subject, columns = ['subject', 'num_of_rows', 'label'])
rows_per_subject_sorted = rows_per_subject.sort_values('num_of_rows')
rows_per_subject_sorted
#%% For some reason values are not sorted in the plot!!!
colors1 = ['red' if i == 'Test' else 'blue' for i in rows_per_subject_sorted['label']] 
labels = ['Train', 'Test']
handles = [plt.Rectangle((0,0), 1, 1, color='blue'),
           plt.Rectangle((0,0), 1, 1, color='red')]

plt.bar(x='subject', height = 'num_of_rows', data = rows_per_subject_sorted, color = colors1)
plt.title("Total Number of the Rows for Every Subject")
plt.xticks(ticks = np.arange(30), labels = rows_per_subject_sorted['subject'], size = 7)
plt.legend(handles, labels)

#%%
rows_per_subject.min()
rows_per_subject.max()
rows_per_subject.mean()
rows_per_subject.sort_values('num_of_rows')
#%%














