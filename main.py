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

