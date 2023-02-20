import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import os
#%% Provide the path to working directory
root_path = input("Enter the working directory's path:\n")
#%% Set the working directory
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
#%% Activity distribution for each subject
# sns.set(font_scale = .8)   # for some reason breaks the color
sns.countplot(y='subject',hue='Activity', data = both)
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
#%% Alternative visualization
sns.set(font_scale=.6)
sns.barplot(data = rows_per_subject_sorted, x = 'subject', y = "num_of_rows", hue='label')
# sns.barplot(data = rows_per_subject_sorted, x = "num_of_rows", y='subject', hue='label')
#%% Distribution of Activity levels
plt.title('No of Datapoints per Activity', fontsize=15)
sns.countplot(data = both, x = "Activity")
plt.xticks(rotation=90)
#%%
rows_per_subject.min()
rows_per_subject.max()
rows_per_subject.mean()
#%%
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(both, hue='Activity', aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()

plt.annotate("Stationary Activities", xy=(-0.956,12), xytext=(-0.8, 16), size=10, va='center', ha='left',
             arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=10, va='center', ha='left',
             arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=0.1"))
#%% 
features_to_plot = ["tBodyAccMag-mean()", "tGravityAccMag-mean()", "tBodyAccJerkMag-mean()",
                    "tBodyGyroMag-mean()", "tBodyGyroJerkMag-mean()", "fBodyAccMag-mean()"]

sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(both, hue='Activity', aspect=2)
# facetgrid.map(sns.distplot, features_to_plot[0], hist=False).add_legend()
# facetgrid.map(sns.distplot, features_to_plot[1], hist=False).add_legend()
facetgrid.map(sns.distplot, features_to_plot[2], hist=False).add_legend()
#%%
fig, ax = plt.subplots(3,2, sharex=True, sharey=True)
# ax[0]
#%% Checking the densities of some variables
for feature in features_to_plot:
    both[feature].plot.density()
plt.title("PDF of 6 Features")
# There is apparetnly a pattern of bi-modality in each of the variables.
#%%
both[features_to_plot[1]].plot.density(color = "green")
both[features_to_plot[2]].plot.density(color = "red")

#%%
import plotly.express as px

df = px.data.iris()
features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

fig = px.scatter_matrix(
    df,
    dimensions=features,
    color="species"
)
fig.update_traces(diagonal_visible=False)
fig.show()
#%%

    
    
    
    


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
n_estimators = [100, 500, 1000, 1200, 1500]
# max_features = ['log2', 'sqrt']   # sqrt is the default
max_depth = [int(x) for x in np.linspace(5, 20, num = 3)]
max_depth.append(None)  # None is the default value
min_samples_split = [10, 15, 30]
min_samples_leaf = [10, 20, 30]
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
scoring = ["accuracy", "f1_macro", "f1_micro"]
#%% Set the CV seach
n_iter_ = 20
rfc_random = RandomizedSearchCV(estimator = rfc,
                                param_distributions = param_grid, 
                                n_iter = n_iter_, # how many combinations to choose from the grid
                                cv = 5, 
                                scoring = scoring,
                                refit = "accuracy", # has to be set for refitting the model over entire training set
                                verbose=2, 
                                error_score=np.NaN, # makes the procedure robust to fit failure (see "Robustness to  Failure"
                                                    # on https://scikit-learn.org/stable/modules/grid_search.html#multimetric-grid-search)
                                return_train_score=True,
                                n_jobs = -1)
#%% Fit the model
rfc_random.fit(X_train, y_train)
#%% Search results
'''
The following is the list of possible outputs to create. However, some of them might be not very useful.

  Before fitting
  --------------
    rfc_random.get_params()     ---> Get the all params (not really useful). Can be called before fitting the model

  After fitting
  -------------
    rfc_random.best_index_      ---> Index of the best set of params (not useful)
    rfc_random.best_score_      ---> Mean test cross-validated score (the one in 'refit') of the best_estimator 
    rfc_random.best_params_     ---> Best set of the params based on mean test score used for refit
    rfc_random.best_estimator_  ---> Best model after refitting on the entire train data (?) This is the final model to be tested
                                     on the test set.
    
And finally, the most useful output:
    rfc_random.cv_results_      ---> Dictionary of all CV results, which can be directly imported to a dataframe (very useful)
'''
cv_results = pd.DataFrame(rfc_random.cv_results_)
# Some informative columns of the cv_results
col_to_show = ['params', 'mean_train_accuracy', 'std_train_accuracy', 'mean_test_accuracy', 'std_test_accuracy',
                         'mean_train_f1_macro', 'std_train_f1_macro', 'mean_test_f1_macro', 'std_test_f1_macro']
to_show = cv_results[col_to_show]
#%% Get the model with the best parameters (this can be used for test-performance or prediction)
rfc_final = rfc_random.best_estimator_
y_test_pred = rfc_final.predict(X_test)
#%% Now, evaluate the performance using the metric you want
f1_score(y_test, y_test_pred, average = 'micro')
accuracy_score(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))
print(confusion_matrix(y_test, y_test_pred))
#%% Visualization
fig, ax = plt.subplots(2, sharex = True, sharey = True)
ax[0].plot(np.arange(n_iter_), 'mean_train_accuracy', data = to_show, marker = '.', linestyle = '--', color = 'b', linewidth = .5, label = 'Train')
ax[0].plot(np.arange(n_iter_), 'mean_test_accuracy', data = to_show, marker = '*', linestyle = '-.', color = 'r', linewidth = .5, label = 'Test')
ax[0].set_title("Accuracy")

ax[1].plot(np.arange(n_iter_), 'mean_train_f1_macro', data = to_show, marker = '.', linestyle = '--', color = 'b', linewidth = .5, label = 'Train')
ax[1].plot(np.arange(n_iter_), 'mean_test_f1_macro', data = to_show, marker = '*', linestyle = '--', color = 'r', linewidth = .5, label = 'Test')
ax[1].set_title("F1_macro")
ax[1].set_xlabel("Parameters")
# ax[1].set_xticklabels(np.arange(n_iter_), labels = np.arange(n_iter_))
plt.xticks(ticks = np.arange(n_iter_), labels = np.arange(n_iter_) + 1)
plt.legend()
#%%









