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
#%% 
'''
Exploratory Data Analysis (EDA)
-------------------------------
In the current section, we try to understand the data better. EDA can significantly improve the quality of model selection in later
steps. We focus on the distribution of variables provided in the dataset and visualize the results for an easier inference. Since
analyzing all variables might not be practical using additional information provided outside of the dataset will help detect variables
with more value. 
'''
#%% Distribution of Activity levels (This is number of the rows per activity value in entire data.)
plt.title('Number of Datapoints per Activity', fontsize=12)
sns.countplot(data = both, x = "Activity")
plt.xticks(rotation=45, fontsize = 7, 
           ha = 'right', rotation_mode = 'anchor') # ha='right' is not enough to visually align labels with ticks.
                                                   # For rotation=45, use both ha='right' and rotation_mode='anchor'
                                                   # For other angles, use a ScaledTranslation() instead
#%% Should check if the distribution among different values of 'Activity' are balanced
xtick_loc = np.arange(len(both["Activity"].unique()))
bar_width = .2

plt.bar(x=xtick_loc - .2, height=both["Activity"].value_counts(), width = bar_width, label='Both')
plt.bar(x=xtick_loc , height=train["Activity"].value_counts(),  width = bar_width, label='Train')
plt.bar(x=xtick_loc + .2, height=test["Activity"].value_counts(),  width = bar_width, label='Test')
plt.title("Distribution of Activity Values", fontsize = 12)
plt.xticks(ticks = xtick_loc, labels = both["Activity"].value_counts().index, rotation = 45, 
           size = 7,
           ha = 'right', rotation_mode='anchor') # ha='right' is not enough to visually align labels with ticks.
                                                 # For rotation=45, use both ha='right' and rotation_mode='anchor'
                                                 # For other angles, use a ScaledTranslation() instead

plt.legend()
#%% Activity distribution for each subject (Not very useful)
# sns.set(font_scale = .8)   # for some reason breaks the color
plt.figure(figsize=(15,30))
sns.countplot(y='subject',hue='Activity', data = both)
#%% 
# 2. Values of 'subject' in the training and test sets 
train_subjects = train['subject'].sort_values().unique() 
test_subjects = test['subject'].sort_values().unique() 
#%% Number of the rows per subject
rows_per_subject = []
for subject, sub_df in both.groupby('subject'):
    if subject in train_subjects:
        rows_per_subject.append([str(subject), sub_df.shape[0], "Train"])  # str(subject) to be able to sort the plot later
    else:
        rows_per_subject.append([str(subject), sub_df.shape[0], "Test"])
    
rows_per_subject = pd.DataFrame(rows_per_subject, columns = ['subject', 'num_of_rows', 'label'])
rows_per_subject_sorted = rows_per_subject.sort_values('num_of_rows')
rows_per_subject_sorted
#%% Visualize number of the rows per subject
# plt.style.use("seaborn")
plt.style.use('ggplot')
colors1 = ['red' if i == 'Test' else 'blue' for i in rows_per_subject_sorted['label']] 
labels = ['Belonging to Train Set', 'Belonging to Test Set']
handles = [plt.Rectangle((0,0), 1, 1, color='blue'),
           plt.Rectangle((0,0), 1, 1, color='red')]

plt.bar(x='subject', height = 'num_of_rows', data = rows_per_subject_sorted, color = colors1)
plt.title("Total Number of the Rows for Every Subject")
plt.xlabel("Subject")
plt.ylabel("Number of Rows")
plt.xticks(ticks = np.arange(30), labels = rows_per_subject_sorted['subject'], size = 7)
plt.legend(handles, labels)
#%% Alternative visualization for rows per subject
sns.set(font_scale=.6)   # Affects all font sizes. Title should be resized manually
ax = sns.barplot(data = rows_per_subject_sorted, x = 'subject', y = "num_of_rows", hue='label', dodge=False)  # dodge=False adjusts the bars' width and distance
ax.set_title("Total Number of the Rows for Every Subject", fontsize = 12)
ax.set_ylabel("Number of Rows")
#%% Summary of rows per subject
rows_per_subject.describe().astype(int)
#%% Let's focus on the following 6 columns (These columns were chosen based on the information provided in features_info.txt file.)
features_of_interest1 = ["tBodyAccMag-mean()", "tGravityAccMag-mean()", "tBodyAccJerkMag-mean()",
                    "tBodyGyroMag-mean()", "tBodyGyroJerkMag-mean()", "fBodyAccMag-mean()"]

colors1 = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharex=True, sharey=True)

# loop through the variables and plot the density plot in each subplot
for i, var in enumerate(features_of_interest1):
    row = i // 2   # floor
    col = i % 2
    sns.kdeplot(both[var], ax=axes[row, col], color = colors1[i], fill=True)     # fill and shade seem to do the same thing
    axes[row, col].set_title(var)

# adjust the spacing between subplots
plt.tight_layout()
'''
There is apparetnly a pattern of bi-modality in each of the variables. An interesting task would be to inverstigate
if there is any relation between this bi-modal pattern and the activity being performed. Before that, let's check 
how correlated these features are with each other.
'''
#%% Similarity of the features_of_interest1 quartiles
df_of_interest1 = both[features_of_interest1].describe().round(2)
# min and max are the same for all variables of interest, so remove them for visualization
idx_of_interest1 = set(df_of_interest1.index).difference(['min', 'max', 'count'])

plt.xticks(rotation = 45, ha = 'right', rotation_mode = 'anchor')
for row in idx_of_interest1:
    data = df_of_interest1.loc[row]
    sns.lineplot(data=data, marker = 'o', linestyle = '--', linewidth = .6 ,label = row)
plt.ylabel("")
plt.title("Quartiles and Standard Deviation of Features of Interest", fontsize = 12)
#%% Correlation (features_of_interest1)
sns.pairplot(data = both[features_of_interest1]) # Seems that 'tBodyAccMag-mean()' and 'tGravityAccMag-mean()' are equal!!!
#%% Are 'tBodyAccMag-mean()' and 'tGravityAccMag-mean()' the same? 
both['tBodyAccMag-mean()'].equals(both['tGravityAccMag-mean()'])   # Yes, they are the same
'''
What if there are other duplicate columns?
------------------------------------------
This will be checked below. We will use the pd.DataFrame.duplicated() method, applied to the columns.
'''
both.T.duplicated().sum() # 21 columns are duplicates of other columns!
#%%
def detect_duplicate_columns(df):
    '''
    Parameters
    ----------
    df : pandas DataFrame
        pandas dataframe whose columns must be checked for duplicates

    Returns
    -------
    tuple : 
        A tuple with 2 entries "output" and "cols_to_drop"
        
        output: Dictionary
                A dictionary containing columns (key) and a list of other columns (value) that are identical with the key
        
        cols_to_drop: List
                A list containing all values (columns) of the "output" dictionary, for easy removal
    '''

    output = {}
    # Duplicate cols will be stored for later easy removal
    cols_to_drop = []
    
    while df.shape[1] > 1:
        first_col_name = df.columns[0]
        rest_cols = df.columns[1:]
        duplicates = []
        
        for second_col_name in rest_cols:
            if df[first_col_name].equals(df[second_col_name]):
                duplicates.append(second_col_name)
                
        if len(duplicates) > 0:
            output[first_col_name] = duplicates
            df = df.drop(columns = duplicates)
            cols_to_drop += duplicates
            
        df = df.drop(columns = first_col_name)
        
        # df = df.drop(columns = to_drop)
        
    return output, cols_to_drop

duplicate_columns = detect_duplicate_columns(both.drop(columns = ['subject', 'Activity', 'Data']))
#%% Did we find 21 duplicates?
len(duplicate_columns[1])     # yes 21 
#%% All duplicate columns need to be removed (I will store the original df just in case)
both_without_duplicate_cols = both.drop(columns = duplicate_columns[1]) 
both_without_duplicate_cols
#%% Recall features_of_interest1; one of the features was removed.  Now, create features_of_interest2 for further exploration.
features_of_interest2 = []

for col in features_of_interest1:
    print(col, ": ", col in both_without_duplicate_cols.columns.to_list())
    if col in both_without_duplicate_cols.columns.to_list():
        features_of_interest2.append(col)

features_of_interest2
#%% Updated correlation (features_of_interest2)
sns.pairplot(data = both_without_duplicate_cols[features_of_interest2]) 
#%% Each of the features_of_interest2 per activity
fig, ax = plt.subplots(nrows=len(features_of_interest2), 
                       ncols = both_without_duplicate_cols['Activity'].nunique(), 
                       sharex = True, sharey = True,
                       figsize=(20, 24))

fig.suptitle("Density of features_of_interest2 Separated by Activities", fontsize = 18)

# iterate through columns and plot densities for each group
for i, column in enumerate(features_of_interest2):
    for j, activity in enumerate(both_without_duplicate_cols['Activity'].unique()):
        sns.kdeplot(data = both_without_duplicate_cols[both_without_duplicate_cols['Activity'] == activity], 
                    x=column, ax=ax[i, j], 
                    color = colors1[j], fill = True, label = activity)
        
        ax[i, j].set_ylabel(column, size = 14)
        ax[i, j].set_xlabel(activity, fontsize = 14)
# adjust the spacing between subplots
plt.tight_layout()
'''
Apparently, there is a distinction between "passive" and "active" activities in all features_of_interest2. That is, in the 
passive activities, i.e., standing, sitting, and laying, a sharp peak happens at the beginning of the variables' range, while
for the active ones, i.e., walking, walking downstairs, and walking upstairs, a blunt peak occurs at larger values of the 
variable. To conclude this topic, we can plot a similar thing but for active and passive activities as two separate categories. 
'''
#%%
passiv_activ = [['SITTING', 'STANDING', 'LAYING'], ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS']]

fig, ax = plt.subplots(nrows=5, ncols = 2, sharex = True, sharey = True, figsize=(10, 14))
fig.suptitle("Density of features_of_interest2 Separated by Passive and Active Categories", fontsize = 14)

# iterate through columns and plot densities for each group
for i, column in enumerate(features_of_interest2):
    for j in range(len(passiv_activ)):
        sns.kdeplot(data = both_without_duplicate_cols[both_without_duplicate_cols['Activity'].isin(passiv_activ[j])], 
                    x=column, ax=ax[i, j], 
                    color = colors1[j], fill = True)
        
        ax[i, j].set_ylabel(column, size = 12)
        ax[i, j].set_xlabel(['Passive', 'Active'][j], fontsize = 14)
# adjust the spacing between subplots
plt.tight_layout()
#%%
sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(both, hue='Activity', aspect=2)
facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()

plt.annotate("Stationary Activities", xy=(-0.956,12), xytext=(-0.8, 16), size=10, va='center', ha='left',
             arrowprops=dict(arrowstyle="simple",connectionstyle="arc3,rad=0.1"))

plt.annotate("Moving Activities", xy=(0,3), xytext=(0.2, 9), size=10, va='center', ha='left',
             arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=0.1"))
#%% 

sns.set_palette("Set1", desat=0.80)
facetgrid = sns.FacetGrid(both, hue='Activity', aspect=2)
# facetgrid.map(sns.distplot, features_of_interest1[0], hist=False).add_legend()
# facetgrid.map(sns.distplot, features_of_interest1[1], hist=False).add_legend()
facetgrid.map(sns.distplot, features_of_interest1[2], hist=False).add_legend()
#%%
# import plotly.express as px

# df = px.data.iris()
# features = ["sepal_width", "sepal_length", "petal_width", "petal_length"]

# fig = px.scatter_matrix(
#     df,
#     dimensions=features,
#     color="species"
# )
# fig.update_traces(diagonal_visible=False)
# fig.show()
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









