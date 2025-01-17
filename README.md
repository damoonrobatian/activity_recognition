# Human Activity Recognition using Smart Wearable Data

> Note: <br> *This is a project in progress and will be completed gradually. This README.md file was prepared using the information provided in the original [README.txt](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones#) file. The main purpose of creating the present file is to provide a reader-friendlier document.* 

  
## Summary of the Experiment
The pool of participants of the experiment contains 30 volunteers within an age range of 19 to 48 years old. Each person was assigned an ID number between 1 and 30 and performed six *Activities of Daily Living (ADL)*while wearing a smartphone (Samsung Galaxy S II) on the waist. The activities performed were `STANDING`, `SITTING`, `LAYING`, `WALKING`, `WALKING_DOWNSTAIRS`, and `WALKING_UPSTAIRS`. Using the phone's embedded accelerometer and gyroscope, 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz were recorded. The data were labeled manually and divided into two parts randomly, i.e., the train and test sets. The training set contains 70% of the participants (21 subjects). The test set includes the remaining 30% of the volunteers (9 subjects). A list of participant IDs in each set is provided below. The corresponding feature in the datasets is `subject`:


| Set    | Participant ID (`subject`)                                                          |
| ------ |-------------------------------------------------------------------------------------| 
| Train  | $1,  3,  5,  6,  7,  8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30$ |
| Test   | $2,  4,  9, 10, 12, 13, 18, 20, 24$                                                 |  

The collected data were pre-processed by noise filtering applied to a sequence of *overlapped*, *fixed-length* time windows. A vector of features was created for every window. For more details, refer to the original [README.txt](https://github.com/damoonrobatian/activity_recognition/blob/b13a2b7384f49e3754c59bc259ffa73e98f31e14/data/UCI%20HAR%20Dataset/README.txt) and [features_info.txt](https://github.com/damoonrobatian/activity_recognition/blob/b13a2b7384f49e3754c59bc259ffa73e98f31e14/data/UCI%20HAR%20Dataset/README.txt). 

The experiments have been video-recorded, a sample of which can be found [here](https://www.youtube.com/watch?v=XOEN9W05_4A). 
 

## Get To Know the Data
### Training Set

### Test Set

## Exploratory Data Analysis

## Models and Evaluation

## Results

## Conclusion



## Related Links
  + [Kaggle](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones?select=train.csv)
  + [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
  + [Video](https://www.youtube.com/watch?v=XOEN9W05_4A)
  
## References
  1. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013. 
  2. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra, Jorge L. Reyes-Ortiz.  Energy Efficient Smartphone-Based Activity Recognition using Fixed-Point Arithmetic. Journal of Universal Computer Science. Special Issue in Ambient Assisted Living: Home Care.   Volume 19, Issue 9. May 2013
  3. Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. Human Activity Recognition on Smartphones using a Multiclass Hardware-Friendly Support Vector Machine. 4th International Workshop of Ambient Assisted Living, IWAAL 2012, Vitoria-Gasteiz, Spain, December 3-5, 2012. Proceedings. Lecture Notes in Computer Science 2012, pp 216-223. 
  4. Jorge Luis Reyes-Ortiz, Alessandro Ghio, Xavier Parra-Llanas, Davide Anguita, Joan Cabestany, Andreu Català. Human Activity and Motion Disorder Recognition: Towards Smarter Interactive Cognitive Environments. 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.  

