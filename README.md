# Reds_ContactPhysicsSimulation

Sam Rizzuto

Cincinnati Reds

Technical Assessment: 2.5

7 February 2022

The attached folder, ```22-ds``` contains 20 files. Thirteen of those are visualizations (in .png format) that I created to further display the predicted values of Exit Velocity, Launch Angle, and Direction and help illustrate my findings. Two files are the original ```train.csv``` and ```test.csv``` files that were provided. Along with those are two more csv's called ```myTrainDF.csv``` and ```myTest.csv``` - these contains all transformations that were performed and carried out on the original data sets. In ```myTrainDF.csv```, I carried out five different statistical/machine learning methods to calculate the predicted values of the three variables (Exit Velocity, Launch Angle, and Direction). The columns for these models are all labeled with the same structure: ```{variable}_{ModelType}```, with the implemented models being: Random Forest, SVM, logistic regression (GLM), GAM, and KNN. Each of the 5 models predicted the three variables, so there are 15 extra columns in ```myTrainDF.csv```. 

OPS and Handedness were also added into the model to factor in the on-base/slugging ability of the batter and matchup-specific at bats, respectively.

In ```myTestDF.csv``` are the original given columns, along with 9 extra total columns displaying the Exit Velocity, Launch Angle, and Direction for each of the GAM, GLM, and KNN models.

Total Time: 3h 57m
