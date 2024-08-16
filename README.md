# KNN_Impl

Implementing KNN from scratch in Python, using numpy, pandas and matplotlib for visualization.

The dataset used is the [Diabetes Dataset For Beginners](https://www.kaggle.com/code/melikedilekci/diabetes-dataset-for-beginners)  from Kaggle. This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The selected dataset contains 9 columns and 768 entries. The columns are as follows:
1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
3. BloodPressure: Diastolic blood pressure (mm Hg)
4. SkinThickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. DiabetesPedigreeFunction: Diabetes pedigree function
8. Age: Age (years)
9. Outcome: Class variable (0 or 1), where 1 is positive for diabetes and 0 is negative for diabetes

The dataset is split into 75% training and 25% testing data. The KNN algorithm is implemented to predict the outcome of the testing data, having its base k value set to 5, using the Euclidean distance as the distance metric and brute force as the algorithm to find the nearest neighbors.

The final accuracy of the model is between 67% and 78%, depending on the random seed used to split the data and cross-validation, as well as the k value used.

The model also runs a loop to find the best k value for the model and best cross-validation split. The best results overall are obtained with a k value of 5 and a cross-validation split of 5.
