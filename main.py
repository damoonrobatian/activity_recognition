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
#%%
## Compare the values of variables in the training and test sets

# 1. Are the values of 'Activity' are the same in the training and test?
train["Activity"].unique() == test["Activity"].unique()  # yes
#%%
# For ease mix the training and test sets
train['Data'] = 'Train'
test['Data'] = 'Test'
both = pd.concat([train, test], axis=0).reset_index(drop=True)
#%%
both.describe()
both.info()
both.nunique()    # number of unique values of each variable
both.dtypes
both.dtypes.unique()
#%% Missingness in data must be explored:
both.isna().sum().sum()    # 0 na but another symbol could have been used as na
both.isnull().sum().sum()
both.mode(numeric_only=True)
both.values
#%% Should check if the distribution among different values of 'Activity' are balanced
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
# plt.style.use("seaborn")
plt.style.use('ggplot')
colors1 = ['red' if i == 'Test' else 'blue' for i in rows_per_subject_sorted['label']] 
labels = ['Belonging to Train Set', 'Belonging to Test Set']
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
#%% 
'''
First Model: Random Forest
--------------------------
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score, accuracy_score
#%%
both_factorized = both.copy()
both_factorized['Activity'] = both_factorized['Activity'].factorize()[0]

X_train = both_factorized[both_factorized['Data'] == 'Train'].drop(columns=['subject', 'Activity', 'Data'])
y_train = both_factorized[both_factorized['Data'] == 'Train']['Activity']

X_test = both_factorized[both_factorized['Data'] == 'Test'].drop(columns=['subject', 'Activity', 'Data'])
y_test = both_factorized[both_factorized['Data'] == 'Test']['Activity']
#%%

rfc = RandomForestClassifier() 
# The following gives the default set of RF tuning parameters.
rfc.get_params()

# Below, we create a grid of RF tuning parameters.
n_estimators = [100, 500, 1000]
# max_features = ['log2', 'sqrt']   # sqrt is the default
max_depth = [int(x) for x in np.linspace(10, 20, num = 3)]
max_depth.append(None)  # None is the default value
min_samples_split = [2, 5, 10]
min_samples_leaf = [2, 4, 8]
# bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
              # 'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf}
              # 'bootstrap': bootstrap}

param_grid
#%%
# scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}
scoring = ["accuracy_score", "f1"]
#%%
rfc_random = RandomizedSearchCV(estimator = rfc,
                                param_distributions = param_grid, 
                                n_iter = 3, # how many combinations to choose from the grid
                                cv = 3, 
                                scoring = ["accuracy", "f1_macro"],
                                refit = "accuracy",
                                verbose=2, 
                                n_jobs = -1)
#%%
# Fit the model
rfc_random.fit(X_train, y_train)
# Get the all params (not really useful)
rfc_random.get_params()
# Get the best params
rfc_random.best_params_
# Get the model with the best parameters (this can be used for test-performance or prediction)
rfc_final = rfc_random.best_estimator_

y_test_pred = rfc_final.predict(X_test)
# Now, evaluate the performance using the metric you want
f1_score(y_test, y_test_pred)
accuracy_score(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))






