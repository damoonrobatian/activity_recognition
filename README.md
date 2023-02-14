# Human Activity Recognition using Smart Wearable Data

> Note: <br> *This is a project in progress and will be completed gradually. This README.md file was prepared using the information provided in the original [README.txt](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones#) file. The main purpose of creating the present file is to provide a reader-friendlier document.* 

  
## Introduction
The pool of prticipants of the experiment contains 30 volunteers within an age range of 19 to 48 years old. Each person was assigned an ID number between 1 and 30 and performed six *Activities of Daily Living (ADL)*, while wearing a smartphone (Samsung Galaxy S II) on the waist. The activities performed were `STANDING`, `SITTING`, `LAYING`, `WALKING`, `WALKING_DOWNSTAIRS`, `WALKING_UPSTAIRS`. Using the phone's embedded accelerometer and gyroscope, 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz were recorded. The data were labeled manually and divided into two parts randomly, i.e., the train and test sets. The training set contains 70% of the participants (21 subjects). The test set includes the remaining 30% of the volunteers (9 subjects). A list of participant IDs in each set is provided below. The corresponding feature in the datasets is `subject`:


| Set    | Participant ID (`subject`)                                                          |
| ------ |-------------------------------------------------------------------------------------| 
| Train  | $1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30$ |
| Test   | $2,  4,  9, 10, 12, 13, 18, 20, 24$                                                 |  


 experiments have been video-recorded to label the data manually. The obtained dataset has been randomly partitioned into two sets, where 70% of the volunteers was selected for generating the training data and 30% the test data. 

The sensor signals (accelerometer and gyroscope) were pre-processed by applying noise filters and then sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window). The sensor acceleration signal, which has gravitational and body motion components, was separated using a Butterworth low-pass filter into body acceleration and gravity. The gravitational force is assumed to have only low frequency components, therefore a filter with 0.3 Hz cutoff frequency was used. From each window, a vector of features was obtained by calculating variables from the time and frequency domain. See 'features_info.txt' for more details. 

## Get To Know the Data
`subject` is an integer in $\[1, 30\]$. That is, the data were collected from $30$ subjects. 
 
`Activity` can take one of the following values: 
  
### Training Set
`subject` values are 1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30

### Test Set
`subject` values are 2,  4,  9, 10, 12, 13, 18, 20, 24

## Exploratory Data Analysis

## Models and Evaluation

## Results

## Conclusion



## Related Links
  + [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones?select=train.csv)
  + [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
  + [Video](https://www.youtube.com/watch?v=XOEN9W05_4A)
  
## References
  1. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 
  2. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz.  Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic. Journal of Universal Computer Science. Special Issue in Ambient Assisted Living: Home Care.   Volume 19, Issue 9. May 2013
  3. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. 4th International Workshop of Ambient Assited Living, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings. Lecture Notes in Computer Science 2012, pp 216-223. 
  4. Jorge Luis Reyes-Ortiz, Alessandro Ghio, Xavier Parra-Llanas, Davide Anguita, Joan Cabestany, Andreu Català. Human Activity and Motion Disorder Recognition: Towards Smarter Interactive Cognitive Environments. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.  

