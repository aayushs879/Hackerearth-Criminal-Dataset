# Hackerearth-Criminal-Dataset
Problem Statement There has been a surge in crimes committed in recent years, making crime a top cause of concern for law enforcement. If we are able to estimate whether someone is going to commit a crime in the future, we can take precautions and be prepared. You are given a dataset containing answers to various questions concerning the professional and private lives of several people. A few of them have been arrested for various small and large crimes in the past. Use the given data to predict if the people in the test data will commit a crime. The train data consists of 45718 rows, while the test data consists of 11430 rows.
Description of Dataset - 
The dataset consisted of 45000 labeled samples persons along with their person id and their personal details as features, 71 features in total.
Implementation-
Firstly, visualized dataset and dropped totally irrelevant features using plots, performed feature selection using sklearn on the basis of chi squared test on remaining features.
Model- First, Implemented a multi layered perceptron using sklearn in pytho, tuning all the parameters several times got a score of 85% on the submission script, The accuracy score was provided by hackerearth itself.
Second- Implemented a support vector classifier using a gaussian kernel again using sklearn on python, after tuning all the parameters got an accuracy score of 90% on cross validation set extracted from the training set itself, the submission script this time got an accuracy score of 93% from hackerearth.
#after using cv set from training data i used all the data to train the classifier itself.
